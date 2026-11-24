"""R3 DECISIVE TEST — is the 3.8% local absorption limited by SPATIAL MISMATCH or by MAGNITUDE?
(ported from build_spatial_reallocation_counterfactual.py)

The R3 descriptive layer found WFH absorbs only ~3.8% of the future-PV midday surplus. That number alone does
NOT prove a spatial story. This builder settles it with a counterfactual at mesh x date x slot (red-line
compliant):

  A_actual(date,slot)  = sum_mesh min(dWFH, surplus_no_wfh)         WFH absorbed where it ACTUALLY sits
  A_spatial(date,slot) = min( sum_mesh dWFH , sum_mesh surplus )    WFH budget that slot, FREELY re-placed onto surplus
  D(date,slot)         = sum_mesh surplus_no_wfh                    the surplus to absorb

Same per-slot WFH budget, same surplus field; the ONLY thing relaxed is WHERE the WFH load sits. So:
  match_actual   = sum A_actual / sum D        (~the 3.8%)
  match_spatial  = sum A_spatial / sum D       (the ceiling if WFH were optimally placed in space)
  spatial_gain   = match_spatial - match_actual  -> how much a perfect spatial match would BUY  (the spatial story)
  magnitude_wall = 1 - match_spatial             -> unabsorbable even with perfect placement (budget/timing)

A_spatial uses the metro-wide (4-pref) per-slot budget -> the MOST generous spatial freedom, so it is a hard
upper bound. The verdict is printed; gates only lock internal consistency + cross-checks.

Output: configured asset
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import glob
import os
import sys

sys.dont_write_bytecode = True

import duckdb
import pandas as pd

from wlpr.io.paths import derived_path, out_path

BASE = derived_path(_asset_location('asset_0052'), _asset_location('asset_0053'))
ALPHA = derived_path(_asset_location('asset_0054'), "alpha_rooftop_capacity_by_mesh.parquet")
KER = derived_path(_asset_location('asset_0054'), "obs_stock_yield_kernel.parquet")
ADW = derived_path(_asset_location('asset_0055'), "a_dwfh_mesh_mon_slot.parquet")
OUTS = out_path(_asset_location('asset_0058'), "r3_spatial_reallocation_scorecard.csv")
MIDDAY = (20, 30)


def _con():
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.register("kernel", con.execute(
        f"select cast(pref as varchar) pref, m, slot, sim_ykw_kwh_per_kw_30min ker from read_parquet('{KER}')").df())
    return con


def _slot_sql(part):
    """per (date,slot) metro-pooled: surplus S, WFH budget B, actual absorption A (future PV, S1 arm)."""
    return f"""
    with led as (select cast(mesh as varchar) mesh, cast(pref as varchar) pref, date, slot,
                   cast(substr(date,5,2) as int) mon,
                   L_owner_gross l_owner, L_nonowner_import l_non, E_wfh, G_pv g_owner
                 from read_parquet('{part}')),
    j as (select led.*, coalesce(alpha.increment_kw_a015,0) inc, coalesce(k.ker,0) ker,
            case when (l_owner+l_non)>0 then least(l_owner/(l_owner+l_non),1.0) else 0 end osh,
            greatest(coalesce(adw.dwfh,0),0) dwfh
          from led
            left join read_parquet('{ALPHA}') alpha on alpha.mesh=led.mesh and alpha.pref=led.pref
            left join kernel k on k.pref=led.pref and k.m=led.mon and k.slot=led.slot
            left join read_parquet('{ADW}') adw on adw.mesh=led.mesh and adw.mon=led.mon and adw.slot=led.slot and adw.scenario_id='S1'),
    m as (select date, slot,
            greatest(l_owner-E_wfh*osh,0)+greatest(l_non-E_wfh*(1-osh),0) L_no,
            g_owner + inc*ker/1000.0 G_fut, dwfh
          from j)
    select date, slot,
           sum(greatest(G_fut - L_no, 0)) S,
           sum(dwfh) B,
           sum(least(dwfh, greatest(G_fut - L_no, 0))) A_actual
    from m group by date, slot"""


def build():
    parts = [p.replace("\\", "/") for p in glob.glob(os.path.join(BASE, _asset_location('asset_0018'), _asset_location('asset_0061'), _asset_location('asset_0062')))]
    if not parts:
        raise SystemExit(f"baseline_p0 missing under {BASE}")
    con = _con()
    acc = [con.execute(_slot_sql(part)).df() for part in parts]
    con.close()
    df = pd.concat(acc).groupby(["date", "slot"], as_index=False).agg(
        S=("S", "sum"), B=("B", "sum"), A_actual=("A_actual", "sum"))     # metro-pool prefs before min(B,S)
    df["A_spatial"] = df[["B", "S"]].min(axis=1)                          # WFH freely re-placed onto surplus that slot

    rows = []
    for scope, mask in [("all_slots", df.slot.notna()),
                        ("midday_20_30", df.slot.between(*MIDDAY))]:
        d = df[mask]
        D, Aa, Asp, Bsum = d.S.sum(), d.A_actual.sum(), d.A_spatial.sum(), d.B.sum()
        rows.append(dict(
            scope=scope,
            surplus_D_GWh=round(D / 1000, 1),
            wfh_budget_B_GWh=round(Bsum / 1000, 1),
            absorbed_actual_GWh=round(Aa / 1000, 1),
            absorbed_spatial_opt_GWh=round(Asp / 1000, 1),
            match_actual=round(Aa / D, 4) if D > 0 else float("nan"),
            match_spatial_opt=round(Asp / D, 4) if D > 0 else float("nan"),
            spatial_gain=round((Asp - Aa) / D, 4) if D > 0 else float("nan"),       # what perfect placement BUYS
            magnitude_wall=round(1 - Asp / D, 4) if D > 0 else float("nan"),        # unabsorbable even if placed perfectly
            budget_vs_surplus=round(Bsum / D, 4) if D > 0 else float("nan"),
        ))
    sc = pd.DataFrame(rows)
    sc.to_csv(OUTS, index=False)
    print(f"[saved] {OUTS}")
    print(sc.to_string(index=False))
    return sc


def verdict(sc):
    a = sc[sc.scope == "all_slots"].iloc[0]
    # ratio of the loss that a perfect SPATIAL match could recover, vs the loss that is fundamentally magnitude
    recoverable = a.spatial_gain                       # match_spatial - match_actual
    print("\n=== VERDICT (all slots) ===")
    print(f"  actual local absorption        {a.match_actual:.1%}")
    print(f"  ceiling if WFH placed perfectly {a.match_spatial_opt:.1%}")
    print(f"  -> spatial mismatch could buy   {recoverable:.1%}  (actual {a.match_actual:.1%} -> ceiling {a.match_spatial_opt:.1%})")
    print(f"  -> magnitude wall (unreachable) {a.magnitude_wall:.1%}  (WFH budget is {a.budget_vs_surplus:.0%} of surplus)")
    if a.match_spatial_opt >= 3 * max(a.match_actual, 1e-9) and a.spatial_gain >= 0.10:
        print("  READING: SPATIAL mismatch BINDS -- perfect placement would multiply absorption; the spatial story has teeth.")
    elif a.match_spatial_opt <= 0.12:
        print("  READING: MAGNITUDE wall dominates -- even perfect placement absorbs little; the spatial framing is oversold.")
    else:
        print("  READING: MIXED -- spatial placement helps but a magnitude wall remains; report both, claim neither alone.")


def run_gates(sc):
    fails = []

    def ck(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
        if not ok:
            fails.append(n)

    a = sc[sc.scope == "all_slots"].iloc[0]
    # Legacy snapshot pins are comparison values only in the public rebuild.
    ck("1 actual absorption is positive (legacy R3 pin diagnostic)", a.absorbed_actual_GWh > 0,
       f"public={a.absorbed_actual_GWh:.1f} GWh; legacy_pin=26.2 GWh")
    ck("2 surplus is positive (legacy storage pin diagnostic)", a.surplus_D_GWh > 0,
       f"public={a.surplus_D_GWh:.1f} GWh; legacy_pin=688.9 GWh")
    ck("3 ordering actual <= spatial-optimal <= total surplus", a.absorbed_actual_GWh <= a.absorbed_spatial_opt_GWh + 1e-6
       and a.match_spatial_opt <= 1.0 + 1e-9,
       f"actual {a.absorbed_actual_GWh} <= spatial {a.absorbed_spatial_opt_GWh} <= D {a.surplus_D_GWh}")
    ck("4 decomposition is well-formed (spatial_gain>=0, magnitude_wall in [0,1])",
       a.spatial_gain >= -1e-9 and 0 <= a.magnitude_wall <= 1, f"gain {a.spatial_gain} wall {a.magnitude_wall}")
    if fails:
        raise SystemExit("reallocation: GATE FAILS -> " + ", ".join(fails))
    print("reallocation: ALL GATES PASS")


def main():
    sc = build()
    verdict(sc)
    print("=== GATES ===")
    run_gates(sc)


if __name__ == "__main__":
    main()
