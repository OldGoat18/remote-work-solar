"""Step 11 — storage PHYSICAL layer L0 / L1 (NO price). Built strictly on the C1 mesh × date × slot net-load
layer (netload_mesh_date); never reads scenario_aggregate, never a system representative-day, never JEPX.

Ported (numerics verbatim) from the source `configured asset` (C2).

RED LINE #3: C2 is pure physics. Surplus energy / low-valley charge opportunity / evening deficit / shiftable
energy / capacity-saturation curve only. JEPX price is multiplied on ONLY in a separate price-translation layer.

L0 (physical envelope, per scenario × PV):
  peakshave_energy_GWh = sum over real days of  sum_slot max(net - daily_mean, 0) * 0.5h
  local_surplus_GWh    = sum_mesh-day local PV surplus (the nonlinear mesh-level reverse the system sum hides)
  evening_deficit_GWh / n_negative_net_days
  WFH EFFECT: no_wfh vs S1..S10 -- WFH fills the midday valley, so it ERODES both the local surplus and the
     system peak-shave swing (WFH and storage are partial SUBSTITUTES at midday).

L1 (capacity-saturation curve, system grain): shifted_energy(E) = sum_days min(E, day_peakshave_energy).

Spatial (R3): local surplus + peak-shave energy aggregated by prefecture.

Inputs : derived_path("netload_mesh_date", "{netload_system_date_slot,storage_primitives_mesh}.parquet")
Outputs: out_path("scorecards", "storage_physics_scorecard.csv")  (L0 per scenario × PV + WFH effect)
         out_path("scorecards", "storage_saturation_curve.csv")   (L1 shifted_energy(E))
         derived_path("storage", "storage_by_pref.csv")            (spatial)
6 GATES: source-purity / peakshave>0 / saturation monotone+saturates / WFH erodes local surplus (S1<no_wfh,
  future) / WFH erodes system peak-shave swing (future) / spatial sums match.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os

import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path, out_path

SYS = derived_path(_asset_location('asset_0063'), "netload_system_date_slot.parquet")
PRIM = derived_path(_asset_location('asset_0063'), "storage_primitives_mesh.parquet")
SC_OUT = out_path(_asset_location('asset_0058'), "storage_physics_scorecard.csv")
SAT_OUT = out_path(_asset_location('asset_0058'), "storage_saturation_curve.csv")
PREF_OUT = derived_path(_asset_location('asset_0064'), "storage_by_pref.csv")
DT = 0.5  # hours per 30-min slot
E_SWEEP = [0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]  # GWh battery sizes for the saturation curve


def l0_and_l1(sysdf):
    """per (scenario, pv): L0 system peak-shave + L1 saturation curve, from the date-resolved net curve."""
    l0_rows, sat_rows = [], []
    for (sid, pv), g in sysdf.groupby(["scenario_id", "pv_scenario"]):
        wide = g.pivot_table(index="date", columns="slot", values="net_load_MW").reindex(columns=range(48))
        net = wide.values                                   # (n_days, 48) MW
        daymean = np.nanmean(net, axis=1, keepdims=True)
        peakshave_day_MWh = np.nansum(np.maximum(net - daymean, 0.0), axis=1) * DT   # per day MWh
        # required power = peak 30-min charge/discharge magnitude (|net - daymean| max), in MW
        pwr_MW = float(np.nanmax(np.abs(net - daymean)))
        local_surplus_MWh = float(g.local_surplus_MWh.sum())   # sum mesh-grain surplus over all (date,slot)
        evening = g[g.slot.between(34, 44)]
        evening_deficit_MWh = float(np.maximum(evening.net_load_MW, 0).sum() * DT)
        n_neg = int((np.nanmin(net, axis=1) < 0).sum())
        l0_rows.append(dict(scenario_id=sid, pv_scenario=pv,
                            peakshave_energy_GWh=round(peakshave_day_MWh.sum() / 1000.0, 2),
                            peakshave_per_day_GWh=round(peakshave_day_MWh.mean() / 1000.0, 3),
                            required_power_GW=round(pwr_MW / 1000.0, 2),
                            local_surplus_GWh=round(local_surplus_MWh / 1000.0, 1),
                            evening_deficit_GWh=round(evening_deficit_MWh / 1000.0, 1),
                            n_negative_net_days=n_neg, n_days=int(net.shape[0])))
        # L1 saturation: shifted(E) = sum_days min(E_MWh, day_peakshave_MWh)
        for E in E_SWEEP:
            shifted = float(np.minimum(E * 1000.0, peakshave_day_MWh).sum() / 1000.0)
            sat_rows.append(dict(scenario_id=sid, pv_scenario=pv, storage_GWh=E,
                                 shifted_energy_GWh=round(shifted, 2),
                                 cycles_per_window=round(shifted / E, 2) if E > 0 else 0.0))
    return pd.DataFrame(l0_rows), pd.DataFrame(sat_rows)


def spatial(primdf):
    """local surplus + peak-shave energy by prefecture (storage hot-prefectures) for R3."""
    g = primdf.groupby(["scenario_id", "pv_scenario", "pref"]).agg(
        local_surplus_GWh=("local_surplus_MWh", lambda x: round(x.sum() / 1000.0, 1)),
        peakshave_energy_GWh=("peakshave_energy_MWh", lambda x: round(x.sum() / 1000.0, 1)),
        n_mesh=("mesh", "nunique")).reset_index()
    g.to_csv(PREF_OUT, index=False)
    return g


def run_gates(l0, sat, sp):
    fails = []

    def ck(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            fails.append(name)

    # source purity: the data INPUTS are the C1 layer only; jepx/scenario_aggregate/rep-day appear only in the
    # negative-context docstring, never as a read path.
    ck("1 source purity (C1 netload_mesh_date only; no jepx / scenario_aggregate / rep-day input)",
       ("jepx" not in (SYS + PRIM).lower()) and ("scenario_aggregate" not in (SYS + PRIM).lower())
       and (_asset_location('asset_0063') in SYS), "inputs = netload_system_date_slot + storage_primitives_mesh")
    ck("2 peak-shave energy > 0 (all arms)", bool((l0.peakshave_energy_GWh > 0).all()),
       f"min {l0.peakshave_energy_GWh.min():.1f} GWh")
    # 3 saturation curve monotone in E and saturating (cycles fall as E grows)
    mono = True
    for (sid, pv), g in sat.groupby(["scenario_id", "pv_scenario"]):
        g = g.sort_values("storage_GWh")
        mono &= bool(np.all(np.diff(g.shifted_energy_GWh) >= -1e-6)) and bool(np.all(np.diff(g.cycles_per_window) <= 1e-6))
    ck("3 L1 saturation monotone in E + cycles fall (saturates)", mono)
    # 4 WFH erodes mesh local surplus under future PV (S1 < no_wfh)
    def ls(sid):
        return float(l0[(l0.scenario_id == sid) & (l0.pv_scenario == "future_pv_ai")].local_surplus_GWh.iloc[0])
    ck("4 WFH erodes local surplus (S1 < no_wfh, future)", ls("S1") < ls("no_wfh"),
       f"S1 {ls('S1'):.0f} < no_wfh {ls('no_wfh'):.0f} GWh")
    # 5 WFH erodes the system peak-shave swing under future PV (S1 < no_wfh) -- WFH flattens midday
    def ps(sid):
        return float(l0[(l0.scenario_id == sid) & (l0.pv_scenario == "future_pv_ai")].peakshave_energy_GWh.iloc[0])
    ck("5 WFH erodes system peak-shave (S1 < no_wfh, future)", ps("S1") < ps("no_wfh"),
       f"S1 {ps('S1'):.1f} < no_wfh {ps('no_wfh'):.1f} GWh")
    # 6 spatial sums reconcile to the L0 local surplus
    sp_tot = float(sp[(sp.scenario_id == "S1") & (sp.pv_scenario == "future_pv_ai")].local_surplus_GWh.sum())
    l0_tot = ls("S1")
    ck("6 spatial pref-sum reconciles L0 local surplus (S1/future)", abs(sp_tot - l0_tot) < 1.0,
       f"pref-sum {sp_tot:.0f} vs L0 {l0_tot:.0f} GWh")
    if fails:
        raise SystemExit("storage_physics: GATE FAILS -> " + ", ".join(fails))


def main():
    os.makedirs(os.path.dirname(PREF_OUT), exist_ok=True)
    sysdf = pd.read_parquet(SYS)
    primdf = pd.read_parquet(PRIM)
    l0, sat = l0_and_l1(sysdf)
    sp = spatial(primdf)
    l0.to_csv(SC_OUT, index=False); sat.to_csv(SAT_OUT, index=False)
    print(f"[saved] {SC_OUT}  ({len(l0)} rows)")
    print(f"[saved] {SAT_OUT}  ({len(sat)} rows)")
    print(f"[saved] {PREF_OUT}  ({len(sp)} rows)")
    print("=== GATES ===")
    run_gates(l0, sat, sp)
    print("11_storage_physics (C2): ALL GATES PASS (L0/L1 physical; no price)")
    # headline echo
    for sid in ["no_wfh", "S1", "S10"]:
        r = l0[(l0.scenario_id == sid) & (l0.pv_scenario == "future_pv_ai")].iloc[0]
        print(f"  {sid:6s}/future: peak-shave {r.peakshave_energy_GWh:6.1f} GWh | local surplus {r.local_surplus_GWh:6.0f} GWh "
              f"| neg-days {r.n_negative_net_days:2d} | req power {r.required_power_GW:.1f} GW")


if __name__ == "__main__":
    main()
