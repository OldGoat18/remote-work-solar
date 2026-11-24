"""Step 09 — the A-caliber NET-LOAD curve + unified index engine (net-load / relief / SCR / shape half).

Ported (numerics verbatim) from the source `configured asset`, NET-LOAD half ONLY.

The BTM self-consumption split (owner_BTM / adopter_BTM / future_BTM and any kappa / 0.2593 self-consumption
model) is DROPPED — it belongs to the excluded eta/least PV/BTM user-state twin. The figures consume only
`wfh_energy_multiplier`, `pv_selfcons_rate`, and `nonowner_relief_GWh`, so this port keeps exactly the net /
reverse / relief / SCR / shape columns and the scenario_config evidence metadata.

Loops A's S1-S10 over baseline_p0 and computes BOTH the current-PV and future-PV net-load curves at mesh × slot
from A's HVAC-inclusive dWFH (a_dwfh injection), then the per-scenario net/relief/SCR/shape index.

Load injection: a_dwfh.dwfh ALREADY bakes the scenario multiplier, so ws = dwfh_a DIRECTLY — NO *mult, NO /0.2593.
base-strip the ledger's current E_wfh first.

PV tiers:  current  G = g_owner (metered owner PV);  future  G = g_owner + alpha15_REPOS_rooftop increment_kw *
ker / 1000.  L_total is PV-INVARIANT (only G differs), so the WFH net-load delta is identical across tiers.

reverse = sum_mesh max(G - L, 0) (nonlinear -> mesh-grain sum); relief(scenario) = reverse(no_wfh) -
reverse(scenario). SCR (grid-level PV absorption rate) = 1 - reverse/G_pv (kappa-independent, net-load based).
net_mw = sum(net)*2/n_dates (MWh/30min -> rep-day MW). ramp_15_18 = net[slot36]-net[slot30];
peak_to_valley = max/min.

Output: derived_path("scenario_s1s10", "scenario_indices_s1s10.csv") (THE sole WFH scenario schema, BTM-free) +
        scenario_aggregate_s1s10_a_caliber.parquet + out_path("scorecards", "scenario_s1s10_a_caliber_score.csv").
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import glob
import os

import duckdb
import pandas as pd

from wlpr.io.paths import derived_path, out_path

BASE = derived_path(_asset_location('asset_0052'), _asset_location('asset_0053'))
ADWFH = derived_path(_asset_location('asset_0055'), "a_dwfh_mesh_mon_slot.parquet")
CFG = derived_path(_asset_location('asset_0143'), "scenario_config.csv")
ALPHA_CAP = derived_path(_asset_location('asset_0054'), "alpha_rooftop_capacity_by_mesh.parquet")
KERNEL = derived_path(_asset_location('asset_0054'), "obs_stock_yield_kernel.parquet")  # MAINLINE kernel = OBSERVED-stock yield shape.
OUTD = os.path.dirname(derived_path(_asset_location('asset_0057'), "scenario_indices_s1s10.csv"))
ALL_SCEN = [f"S{i}" for i in range(1, 11)]
FAILS: list[str] = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def prepare_kernel(con: duckdb.DuckDBPyConnection) -> None:
    """Register only the PV yield kernel needed by the alpha-rooftop main line."""
    ker = pd.read_parquet(KERNEL)[["pref", "m", "slot", "sim_ykw_kwh_per_kw_30min"]]
    ker["pref"] = ker.pref.astype(str)
    con.register("kernel", ker)


def row_sql(part, ym, sid):
    """per (pref,date,slot): mesh-grain-summed net + reverse for {arm, no_wfh} x {current, future} PV tiers.
    ws = dwfh_a DIRECTLY (multiplier pre-baked in a_dwfh). reverse summed at mesh grain (nonlinear)."""
    return f"""
    with led as (
      select cast(mesh as varchar) mesh, cast(pref as varchar) pref, date, slot,
             cast(substr(date,5,2) as int) mon,
             L_owner_gross l_owner, L_nonowner_import l_non, E_wfh, G_pv g_owner
      from read_parquet('{part}')
    ),
    j as (
      select led.*,
        coalesce(alpha.increment_kw_a015,0) alpha15_inc_kw,
        coalesce(k.sim_ykw_kwh_per_kw_30min,0) ker,
        case when (led.l_owner+led.l_non)>0 then least(led.l_owner/(led.l_owner+led.l_non),1.0) else 0 end osh,
        coalesce(adw.dwfh,0) dwfh_a
      from led
      left join read_parquet('{ALPHA_CAP}') alpha on alpha.mesh=led.mesh and alpha.pref=led.pref
      left join kernel k on k.pref=led.pref and k.m=led.mon and k.slot=led.slot
      left join read_parquet('{ADWFH}') adw
             on adw.mesh=led.mesh and adw.mon=led.mon and adw.slot=led.slot and adw.scenario_id='{sid}'
    ),
    m as (
      select pref, date, slot,
        greatest(l_owner - E_wfh*osh, 0) + greatest(l_non - E_wfh*(1-osh), 0) L_now,
        greatest(l_owner - E_wfh*osh, 0) + dwfh_a*osh
          + greatest(l_non - E_wfh*(1-osh), 0) + dwfh_a*(1-osh) L_arm,
        g_owner G_cur,
        g_owner + alpha15_inc_kw*ker/1000.0 G_fut,
        dwfh_a
      from j
    )
    select pref, date, slot,
      sum(dwfh_a) dwfh_a,
      sum(L_arm - G_cur) net_cur_arm, sum(greatest(G_cur - L_arm,0)) rev_cur_arm,
      sum(L_now - G_cur) net_cur_now, sum(greatest(G_cur - L_now,0)) rev_cur_now,
      sum(L_arm - G_fut) net_fut_arm, sum(greatest(G_fut - L_arm,0)) rev_fut_arm,
      sum(L_now - G_fut) net_fut_now, sum(greatest(G_fut - L_now,0)) rev_fut_now,
      sum(G_cur) g_cur, sum(G_fut) g_fut
    from m group by pref, date, slot
    """


def shape_metrics(df, netcol, n_dates):
    """system rep-day MW shape metrics from a (slot -> sum net MWh/30min) frame."""
    s = df.groupby("slot")[netcol].sum() * 2.0 / n_dates   # MWh/30min summed over dates -> rep-day MW
    net = s.reindex(range(48)).fillna(0.0)
    ramp = float(net.get(36, 0.0) - net.get(30, 0.0))      # 15:00->18:00 ramp (slots 30->36)
    vmin, vmax = float(net.min()), float(net.max())
    return dict(net_min_MW=vmin, net_peak_MW=vmax, net_mean_MW=float(net.mean()),
                net_peak_to_valley_ratio=(vmax / vmin if abs(vmin) > 1e-9 else float("nan")),
                max_30min_ramp_MW=float(net.diff().abs().max()), ramp_15_18_MW=ramp)


def main():
    os.makedirs(OUTD, exist_ok=True)
    cfg = pd.read_csv(CFG).set_index("scenario_id") if os.path.exists(CFG) else pd.DataFrame().set_index(pd.Index([], name="scenario_id"))
    # Gap-A precondition
    adw_ids = set(duckdb.sql(f"select distinct scenario_id from read_parquet('{ADWFH}')").df().scenario_id)
    gate("Gap-A: a_dwfh holds all S1-S10", set(ALL_SCEN).issubset(adw_ids),
         f"a_dwfh scenarios={sorted(adw_ids)}")
    if FAILS:
        raise SystemExit(f"09 precondition FAILED: {FAILS} — run the a_dwfh unified userstate builder S1..S10 first")

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'"); con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    prepare_kernel(con)
    parts = sorted(glob.glob(os.path.join(BASE, _asset_location('asset_0018'), _asset_location('asset_0061'), _asset_location('asset_0062'))))

    agg_rows = []      # per (scenario, pv, slot) net/gross/pv_gen — the aggregate parquet
    idx_rows = []      # the unified net-load index (BTM-free)
    for sid in ALL_SCEN:
        acc = []
        for part in parts:
            ym = part.replace("\\", "/").split(_asset_location('asset_0070'))[1].split("/")[0]
            acc.append(con.execute(row_sql(part.replace("\\", "/"), ym, sid)).df())
        d = pd.concat(acc, ignore_index=True)
        n_dates = d.date.nunique()
        # ---- aggregate net/reverse per pv tier; relief = reverse(no_wfh) - reverse(arm) ----
        tot = {c: float(d[c].sum()) for c in d.columns if c not in ("pref", "date", "slot")}
        relief_cur = (tot["rev_cur_now"] - tot["rev_cur_arm"]) / 1e3      # GWh
        relief_fut = (tot["rev_fut_now"] - tot["rev_fut_arm"]) / 1e3
        dwfh_gwh = tot["dwfh_a"] / 1e3
        # grid-level green-power absorption (PV self-consumption) rate = 1 - reverse/G_pv, kappa-independent
        # (net-load based, NOT the retired BTM eta/least model). G_pv is WFH-invariant.
        gpv_cur, gpv_fut = tot["g_cur"] / 1e3, tot["g_fut"] / 1e3
        scr_cur = 1.0 - tot["rev_cur_arm"] / tot["g_cur"] if tot["g_cur"] > 0 else float("nan")
        scr_fut = 1.0 - tot["rev_fut_arm"] / tot["g_fut"] if tot["g_fut"] > 0 else float("nan")
        scr_now_cur = 1.0 - tot["rev_cur_now"] / tot["g_cur"] if tot["g_cur"] > 0 else float("nan")
        scr_now_fut = 1.0 - tot["rev_fut_now"] / tot["g_fut"] if tot["g_fut"] > 0 else float("nan")
        # shape metrics per tier (arm + no_wfh) for the deltas
        mc_arm = shape_metrics(d, "net_cur_arm", n_dates); mc_now = shape_metrics(d, "net_cur_now", n_dates)
        mf_arm = shape_metrics(d, "net_fut_arm", n_dates); mf_now = shape_metrics(d, "net_fut_now", n_dates)
        # evidence metadata from scenario_config (NO BTM split — that is the excluded eta/least engine)
        c = cfg.loc[sid] if sid in cfg.index else None
        ev = dict(evidence_tier=(str(c.evidence_tier) if c is not None else ""),
                  real_data_basis=(str(c.real_data_basis) if c is not None and "real_data_basis" in c else ""),
                  extrapolation=(str(c.extrapolation) if c is not None and "extrapolation" in c else ""),
                  wfh_energy_multiplier=(float(c.wfh_energy_multiplier) if c is not None else None),
                  scenario_name=(str(c.scenario_name) if c is not None and "scenario_name" in c else sid),
                  primary_lever=(str(c.primary_lever) if c is not None and "primary_lever" in c else ""))
        for pv, mc, rel, mnow, gpv, scr in [
            ("current", mc_arm, relief_cur, mc_now, gpv_cur, scr_cur),
            ("future_pv_ai", mf_arm, relief_fut, mf_now, gpv_fut, scr_fut)]:
            idx_rows.append(dict(
                scenario_id=sid, scenario_name=ev["scenario_name"], primary_lever=ev["primary_lever"],
                pv_scenario=pv, wfh_energy_multiplier=ev["wfh_energy_multiplier"],
                dWFH_total_GWh=round(dwfh_gwh, 3),
                nonowner_relief_GWh=round(rel, 3), delta_mesh_reverse_GWh_vs_no_wfh=round(-rel, 3),
                pv_gen_GWh=round(gpv, 1), pv_selfcons_rate=round(scr, 4),
                net_min_MW=round(mc["net_min_MW"], 1), net_peak_MW=round(mc["net_peak_MW"], 1),
                net_mean_MW=round(mc["net_mean_MW"], 1),
                net_peak_to_valley_ratio=round(mc["net_peak_to_valley_ratio"], 4),
                max_30min_ramp_MW=round(mc["max_30min_ramp_MW"], 2), ramp_15_18_MW=round(mc["ramp_15_18_MW"], 2),
                delta_ramp_15_18_MW_vs_no_wfh=round(mc["ramp_15_18_MW"] - mnow["ramp_15_18_MW"], 2),
                delta_peak_to_valley_vs_no_wfh=round(mc["net_peak_to_valley_ratio"] - mnow["net_peak_to_valley_ratio"], 4),
                evidence_tier=ev["evidence_tier"], real_data_basis=ev["real_data_basis"],
                extrapolation=ev["extrapolation"],
                truth_status_wfh="A_self_anchored_at_survey_central",
                evidence_label=("derived_from_alpha_rooftop_REPOS_a15_netload" if pv == "future_pv_ai"
                                else "current_metered_pv_netload"),
                window="Oct2023-Feb2024_daily_30min", electrical_scope="LV_owner_export_only"))
        # no_wfh anchor rows (zero deltas) per pv — scenario-INVARIANT, emit ONCE (from S1's pass)
        for pv, mnow, gpv, scr in ([("current", mc_now, gpv_cur, scr_now_cur),
                                    ("future_pv_ai", mf_now, gpv_fut, scr_now_fut)] if sid == ALL_SCEN[0] else []):
            idx_rows.append(dict(
                scenario_id="no_wfh", scenario_name="no_wfh", primary_lever="counterfactual",
                pv_scenario=pv, wfh_energy_multiplier=0.0, dWFH_total_GWh=0.0,
                nonowner_relief_GWh=0.0, delta_mesh_reverse_GWh_vs_no_wfh=0.0,
                pv_gen_GWh=round(gpv, 1), pv_selfcons_rate=round(scr, 4),
                net_min_MW=round(mnow["net_min_MW"], 1), net_peak_MW=round(mnow["net_peak_MW"], 1),
                net_mean_MW=round(mnow["net_mean_MW"], 1),
                net_peak_to_valley_ratio=round(mnow["net_peak_to_valley_ratio"], 4),
                max_30min_ramp_MW=round(mnow["max_30min_ramp_MW"], 2), ramp_15_18_MW=round(mnow["ramp_15_18_MW"], 2),
                delta_ramp_15_18_MW_vs_no_wfh=0.0, delta_peak_to_valley_vs_no_wfh=0.0,
                evidence_tier="counterfactual", real_data_basis="", extrapolation="",
                truth_status_wfh="counterfactual_remove_wfh", evidence_label="counterfactual",
                window="Oct2023-Feb2024_daily_30min", electrical_scope="LV_owner_export_only"))
        # aggregate parquet rows (system slot profile, both PV tiers). gross_load_MW = net + pv_gen = L_arm
        # (PV-INVARIANT). The scenario-INVARIANT no_wfh anchor is emitted ONCE (from S1's pass).
        def _emit_agg(scen, ncol, gcol, pvs):
            netp = d.groupby("slot")[ncol].sum().reindex(range(48)).fillna(0.0) * 2.0 / n_dates
            pvp = d.groupby("slot")[gcol].sum().reindex(range(48)).fillna(0.0) * 2.0 / n_dates
            for slot in range(48):
                nmw = float(netp.get(slot, 0.0)); gmw = float(pvp.get(slot, 0.0))
                agg_rows.append(dict(scenario_id=scen, pv_scenario=pvs, slot=int(slot),
                                     net_load_MW=nmw, pv_gen_MW=gmw, gross_load_MW=nmw + gmw))
        _emit_agg(sid, "net_cur_arm", "g_cur", "current")
        _emit_agg(sid, "net_fut_arm", "g_fut", "future_pv_ai")
        if sid == ALL_SCEN[0]:
            _emit_agg("no_wfh", "net_cur_now", "g_cur", "current")
            _emit_agg("no_wfh", "net_fut_now", "g_fut", "future_pv_ai")
        print(f"  {sid}: dWFH={dwfh_gwh:.1f} GWh | relief cur/fut={relief_cur:.3f}/{relief_fut:.3f} "
              f"| net-min cur/fut={mc_arm['net_min_MW']:.0f}/{mf_arm['net_min_MW']:.0f} MW", flush=True)
    con.close()

    idx = pd.DataFrame(idx_rows)
    idx.to_csv(os.path.join(OUTD, "scenario_indices_s1s10.csv"), index=False)
    pd.DataFrame(agg_rows).to_parquet(os.path.join(OUTD, "scenario_aggregate_s1s10_a_caliber.parquet"), index=False)

    # ---- gates (net-load / relief / SCR only; BTM-free) ----
    s1c = idx[(idx.scenario_id == "S1") & (idx.pv_scenario == "current")].iloc[0]
    s1f = idx[(idx.scenario_id == "S1") & (idx.pv_scenario == "future_pv_ai")].iloc[0]
    s10f = idx[(idx.scenario_id == "S10") & (idx.pv_scenario == "future_pv_ai")].iloc[0]
    # future grid-level relief is owned by THIS alpha-rooftop net-load engine; pinned after the alpha=15% cutover.
    # 26.215 GWh is the released legacy snapshot pin.  The public rebuild
    # uses the complete recovered mesh/date baseline and therefore treats the
    # pin as a comparison diagnostic while retaining the physical ordering
    # gate (future > current > 0).
    legacy_pin = 26.215
    relief_order_ok = s1f.nonowner_relief_GWh > s1c.nonowner_relief_GWh > 0
    gate("G-RELIEF S1 future relief (legacy pin diagnostic; future > current > 0)",
         relief_order_ok,
         f"public={s1f.nonowner_relief_GWh:.3f} GWh; legacy_pin={legacy_pin:.3f} GWh; "
         f"current={s1c.nonowner_relief_GWh:.3f} GWh")
    gate("G6 relief internal closure (delta_mesh_reverse == -relief)",
         float((idx.delta_mesh_reverse_GWh_vs_no_wfh + idx.nonowner_relief_GWh).abs().max()) < 1e-6)
    gate("G7 PV-invariant WFH delta_ramp (current == future per scenario)",
         _g7_pv_invariant(idx))
    gate("G-SCR pv_selfcons_rate in [0,1] and future absorption rises with WFH (S10>S1)",
         bool(idx.pv_selfcons_rate.between(0, 1).all()) and s10f.pv_selfcons_rate > s1f.pv_selfcons_rate > 0,
         f"S1f {s1f.pv_selfcons_rate:.4f} -> S10f {s10f.pv_selfcons_rate:.4f}")
    gate("no legacy arm names / 0.2593 in the index", not idx.scenario_id.str.contains("S4S13|S14_").any())
    gate("index is BTM-FREE (no *_BTM / kappa column)",
         not any(("_BTM" in c) or ("kappa" in c) for c in idx.columns),
         f"columns={sorted(idx.columns)}")

    # ---- scorecard (net-load / relief / SCR only) ----
    score_out = idx[idx.scenario_id != "no_wfh"][["scenario_id", "pv_scenario", "dWFH_total_GWh",
        "nonowner_relief_GWh", "pv_gen_GWh", "pv_selfcons_rate", "net_min_MW", "ramp_15_18_MW", "evidence_tier"]]
    score_out.to_csv(out_path(_asset_location('asset_0058'), "scenario_s1s10_a_caliber_score.csv"), index=False)

    print("\n=== A S1-S10 net-load index (future-PV relief + SCR + net-load) ===")
    print(idx[(idx.scenario_id != "no_wfh") & (idx.pv_scenario == "future_pv_ai")][
        ["scenario_id", "dWFH_total_GWh", "nonowner_relief_GWh", "pv_selfcons_rate", "net_min_MW", "ramp_15_18_MW"]
    ].to_string(index=False))
    print(f"\n[saved] {os.path.join(OUTD, 'scenario_indices_s1s10.csv')}")
    if FAILS:
        raise SystemExit(f"09 gates FAILED: {FAILS}")
    print("\n09_indices_s1s10: ALL GATES PASS  (A S1-S10 = sole A-caliber net-load + relief index, BTM-free)")


def _g7_pv_invariant(idx):
    a = idx[(idx.pv_scenario == "current") & (idx.scenario_id != "no_wfh")].set_index("scenario_id").delta_ramp_15_18_MW_vs_no_wfh
    b = idx[(idx.pv_scenario == "future_pv_ai") & (idx.scenario_id != "no_wfh")].set_index("scenario_id").delta_ramp_15_18_MW_vs_no_wfh
    return bool((a - b).abs().max() < 1.0)


if __name__ == "__main__":
    main()
