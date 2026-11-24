"""Step 09b — the A-caliber MESH-GRAIN net-load / reverse summary.

Ported (numerics verbatim) from the source `configured asset`.

The mesh-grain twin of indices_s1s10: the SAME a_dwfh injection engine (ws = dwfh_a DIRECTLY, multiplier
pre-baked), the SAME owner-export-only reverse max(G - L, 0) evaluated at mesh × slot grain — but grouped to
MESH (window-summed) instead of (pref, date, slot). Feeds the R3 social-distribution layer + figures.

owner_selfcons is the SI-boundary CENTRAL no-WFH self-consumption (ingested from the userstate_mesh_summary
product, arm=central — selfcons-INVARIANT to the WFH arm). PV-tier-correct: current = selfcons_current_owner
(<= current owner G_pv per mesh); future = selfcons_total (owner + new adopter). The only arm-varying quantities
are L_all_gross_equiv (= base load + A's dWFH) and the derived mesh reverse; G_pv is arm-invariant per PV tier.

Output: derived_path("scenario_s1s10", "scenario_mesh_summary_s1s10_a_caliber.parquet")
        (mesh, pref, wfh_level, pv_scenario, L_all_import, L_all_gross_equiv, G_pv,
         owner_selfcons, owner_export, mesh_reverse_owner_export_only)  -- no_wfh + S1-S10, current + future.
Gates: G3 reverse non-negative; G-RELIEF mesh relief reproduces scenario_indices_s1s10 (< 50 MWh);
       G-DWFH Σ_mesh ΔL_gross == dWFH total per scenario.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import glob
import os

import duckdb
import pandas as pd

from wlpr.io.paths import derived_path

BASE = derived_path(_asset_location('asset_0052'), _asset_location('asset_0053'))
ADWFH = derived_path(_asset_location('asset_0055'), "a_dwfh_mesh_mon_slot.parquet")
MESHSUM = derived_path(_asset_location('asset_0055'), "userstate_mesh_summary.parquet")  # central no-WFH selfcons (owner panel)
IDX = derived_path(_asset_location('asset_0057'), "scenario_indices_s1s10.csv")     # 09 relief, per scenario
ALPHA_CAP = derived_path(_asset_location('asset_0054'), "alpha_rooftop_capacity_by_mesh.parquet")
KERNEL = derived_path(_asset_location('asset_0054'), "obs_stock_yield_kernel.parquet")  # MAINLINE kernel = OBSERVED-stock yield.
OUTP = derived_path(_asset_location('asset_0057'), "scenario_mesh_summary_s1s10_a_caliber.parquet")
OUTD = os.path.dirname(OUTP)
ALL_SCEN = [f"S{i}" for i in range(1, 11)]
FAILS: list[str] = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def prepare_kernel(con: duckdb.DuckDBPyConnection) -> None:
    """Register only the observed-stock PV yield kernel."""
    ker = pd.read_parquet(KERNEL)[["pref", "m", "slot", "sim_ykw_kwh_per_kw_30min"]]
    ker["pref"] = ker.pref.astype(str)
    con.register("kernel", ker)


def row_sql(part, ym, sid):
    """per mesh: window-summed gross load + G + owner-export-only reverse for {arm, no_wfh} x {current,
    future} PV tiers. reverse = Σ_(date,slot) max(G - L, 0). ws = dwfh_a DIRECTLY (multiplier pre-baked)."""
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
    r as (
      select mesh, pref,
        greatest(l_owner - E_wfh*osh, 0) + dwfh_a*osh
          + greatest(l_non - E_wfh*(1-osh), 0) + dwfh_a*(1-osh) L_arm,
        greatest(l_owner - E_wfh*osh, 0) + greatest(l_non - E_wfh*(1-osh), 0) L_now,
        g_owner G_cur,
        g_owner + alpha15_inc_kw*ker/1000.0 G_fut,
        dwfh_a
      from j
    )
    select mesh, any_value(pref) pref,
      sum(L_arm) L_arm, sum(L_now) L_now, sum(G_cur) g_cur, sum(G_fut) g_fut, sum(dwfh_a) dwfh_a,
      sum(greatest(G_cur - L_arm,0)) rev_cur_arm, sum(greatest(G_cur - L_now,0)) rev_cur_now,
      sum(greatest(G_fut - L_arm,0)) rev_fut_arm, sum(greatest(G_fut - L_now,0)) rev_fut_now
    from r group by mesh
    """


def main():
    os.makedirs(OUTD, exist_ok=True)
    # central no-WFH owner self-consumption per mesh (a behind-meter SI-boundary split from the userstate
    # panel, a product of the EXCLUDED engine). If it is not staged, owner_selfcons defaults to 0 — the
    # net-load core (L_arm, G_pv, reverse) is computed independently here and is unaffected.
    if os.path.exists(MESHSUM):
        ms = pd.read_parquet(MESHSUM)
        ms = ms[ms.arm == "central"][["mesh", "selfcons_current_owner", "selfcons_total"]].copy()
        ms["mesh"] = ms.mesh.astype(str)
        sc_cur_map = ms.set_index("mesh").selfcons_current_owner
        sc_fut_map = ms.set_index("mesh").selfcons_total
    else:
        sc_cur_map = pd.Series(dtype=float); sc_fut_map = pd.Series(dtype=float)
    idx = pd.read_csv(IDX)

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'"); con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    prepare_kernel(con)
    parts = sorted(glob.glob(os.path.join(BASE, _asset_location('asset_0018'), _asset_location('asset_0061'), _asset_location('asset_0062'))))

    rows = []          # the mesh summary (no_wfh + S1-S10) x (current, future)
    no_wfh_done = False
    for sid in ALL_SCEN:
        acc = []
        for part in parts:
            ym = part.replace("\\", "/").split(_asset_location('asset_0070'))[1].split("/")[0]
            acc.append(con.execute(row_sql(part.replace("\\", "/"), ym, sid)).df())
        d = (pd.concat(acc, ignore_index=True).groupby("mesh", as_index=False)
             .agg(pref=("pref", "first"), L_arm=("L_arm", "sum"), L_now=("L_now", "sum"),
                  g_cur=("g_cur", "sum"), g_fut=("g_fut", "sum"), dwfh_a=("dwfh_a", "sum"),
                  rev_cur_arm=("rev_cur_arm", "sum"), rev_cur_now=("rev_cur_now", "sum"),
                  rev_fut_arm=("rev_fut_arm", "sum"), rev_fut_now=("rev_fut_now", "sum")))
        d["mesh"] = d.mesh.astype(str)
        d["sc_cur"] = d.mesh.map(sc_cur_map).fillna(0.0)
        d["sc_fut"] = d.mesh.map(sc_fut_map).fillna(0.0)
        for pv, gcol, larm, rarm, sccol in [("current", "g_cur", "L_arm", "rev_cur_arm", "sc_cur"),
                                            ("future_pv_ai", "g_fut", "L_arm", "rev_fut_arm", "sc_fut")]:
            sub = pd.DataFrame(dict(
                mesh=d.mesh, pref=d.pref, wfh_level=sid, pv_scenario=pv,
                L_all_gross_equiv=d[larm], G_pv=d[gcol], owner_selfcons=d[sccol],
                owner_export=(d[gcol] - d[sccol]).clip(lower=0.0),
                L_all_import=(d[larm] - d[sccol]).clip(lower=0.0),
                mesh_reverse_owner_export_only=d[rarm]))
            rows.append(sub)
        if not no_wfh_done:   # no_wfh is scenario-invariant — emit once from S1's pass
            for pv, gcol, rnow, sccol in [("current", "g_cur", "rev_cur_now", "sc_cur"),
                                          ("future_pv_ai", "g_fut", "rev_fut_now", "sc_fut")]:
                sub = pd.DataFrame(dict(
                    mesh=d.mesh, pref=d.pref, wfh_level="no_wfh", pv_scenario=pv,
                    L_all_gross_equiv=d.L_now, G_pv=d[gcol], owner_selfcons=d[sccol],
                    owner_export=(d[gcol] - d[sccol]).clip(lower=0.0),
                    L_all_import=(d.L_now - d[sccol]).clip(lower=0.0),
                    mesh_reverse_owner_export_only=d[rnow]))
                rows.append(sub)
            no_wfh_done = True
        relief_cur = (d.rev_cur_now.sum() - d.rev_cur_arm.sum()) / 1e3
        relief_fut = (d.rev_fut_now.sum() - d.rev_fut_arm.sum()) / 1e3
        print(f"  {sid}: dWFH={d.dwfh_a.sum()/1e3:.1f} GWh | mesh-relief cur/fut={relief_cur:.3f}/{relief_fut:.3f} "
              f"| reverse(no_wfh) cur/fut={d.rev_cur_now.sum()/1e3:.1f}/{d.rev_fut_now.sum()/1e3:.1f} GWh", flush=True)
    con.close()

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUTP, index=False)

    # ---- gates ----
    cur = out[out.pv_scenario == "current"]
    sc_excess = (cur.owner_selfcons - cur.G_pv).clip(lower=0.0).sum() / 1e3   # GWh of selfcons over current G
    gate("G3 reverse non-negative + current selfcons ~<= G_pv (twin-vs-ledger G agg drift only)",
         bool((out.mesh_reverse_owner_export_only >= -1e-9).all()) and sc_excess < 1.0,
         f"min reverse={out.mesh_reverse_owner_export_only.min():.2e}, selfcons-over-G mass={sc_excess:.3f} GWh")
    # mesh relief(arm) = reverse(no_wfh) - reverse(arm) reproduces 09 scenario_indices relief
    worst = 0.0; checked = 0
    for sid in ALL_SCEN:
        for pv in ["current", "future_pv_ai"]:
            rev_arm = out[(out.wfh_level == sid) & (out.pv_scenario == pv)].mesh_reverse_owner_export_only.sum()
            rev_now = out[(out.wfh_level == "no_wfh") & (out.pv_scenario == pv)].mesh_reverse_owner_export_only.sum()
            mesh_relief = (rev_now - rev_arm) / 1e3
            irow = idx[(idx.scenario_id == sid) & (idx.pv_scenario == pv)]
            if len(irow):
                worst = max(worst, abs(mesh_relief - float(irow.iloc[0].nonowner_relief_GWh)))
                checked += 1
    gate("G-RELIEF mesh relief reproduces 09 scenario_indices (all S1-S10 x current/future)",
         worst < 0.05, f"max |Δrelief|={worst*1e3:.2f} MWh over {checked} (scenario,pv)")
    # Σ_mesh ΔL_gross(arm vs no_wfh, current) == dWFH total per scenario
    worst_dw = 0.0
    base_now = out[(out.wfh_level == "no_wfh") & (out.pv_scenario == "current")].set_index("mesh").L_all_gross_equiv
    for sid in ALL_SCEN:
        arm = out[(out.wfh_level == sid) & (out.pv_scenario == "current")].set_index("mesh").L_all_gross_equiv
        dL = (arm - base_now).sum() / 1e3
        dwfh = float(idx[(idx.scenario_id == sid) & (idx.pv_scenario == "current")].iloc[0].dWFH_total_GWh)
        worst_dw = max(worst_dw, abs(dL - dwfh))
    gate("G-DWFH Σ_mesh ΔL_gross == dWFH total per scenario", worst_dw < 0.5, f"max |ΔdWFH|={worst_dw:.3f} GWh")
    gate("no legacy arm names", not out.wfh_level.str.contains("S4S13|S14_|current_presence").any())

    print(f"\n[saved] {OUTP}  ({out.mesh.nunique()} meshes x {out.wfh_level.nunique()} arms x 2 PV tiers, {len(out)} rows)")
    if FAILS:
        raise SystemExit(f"09b gates FAILED: {FAILS}")
    print("\n09b_mesh_summary: ALL GATES PASS  (A S1-S10 = sole mesh-grain net-load/reverse summary)")


if __name__ == "__main__":
    main()
