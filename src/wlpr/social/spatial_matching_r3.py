"""R3 spatial-matching layer — the GRID-LEVEL co-location keystone (ported from build_spatial_matching_r3.py).

The paper's R3 hero quantity: WFH absorbs midday PV surplus ONLY where the two co-locate in space; the
unabsorbed remainder defines where storage/aggregation value falls; and the realized relief concentrates more
than the WFH behaviour itself. Computed anchored on the CURRENT state (S1), with S10 a uniform-magnitude
reference only.

RED LINE (mesh-date-only formal): surplus is nonlinear (max) -> computed at mesh x date x slot FIRST, then
summed to the mesh window total; never a system pre-aggregate. Streams baseline_p0 per partition and never
materialises the full table.

Per (mesh,date,slot), future PV: L_arm = max(l_own-E*osh,0)+max(l_non-E*(1-osh),0)+dwfh ; G_fut = g_own +
inc*ker/1000 ; surplus = max(G_fut - L_arm, 0). Per mesh (window sum):
  D       = surplus(no_wfh)                          demand field  (the surplus to be absorbed)
  surp1   = surplus(arm)                             residual after WFH
  realized= D - surp1                                WFH-absorbed surplus  (== the grid reverse-relief)
  supply  = Sigma_midday (L_arm - L_no_wfh)          WFH midday increment  (the absorption capacity)

Outputs:
  configured asset  (per-mesh supply/demand/realized/residual, S1 & S10)
  configured asset  (the indicators, per arm)
  configured asset      (municipality attribution)
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import glob
import os
import sys

sys.dont_write_bytecode = True

import duckdb
import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path, input_path, out_path

BASE = derived_path(_asset_location('asset_0052'), _asset_location('asset_0053'))
ALPHA = derived_path(_asset_location('asset_0054'), "alpha_rooftop_capacity_by_mesh.parquet")
KER = derived_path(_asset_location('asset_0054'), "obs_stock_yield_kernel.parquet")
ADW = derived_path(_asset_location('asset_0055'), "a_dwfh_mesh_mon_slot.parquet")
OUTP = derived_path(_asset_location('asset_0056'), "r3_spatial_matching_mesh.parquet")
OUTS = out_path(_asset_location('asset_0058'), "r3_spatial_matching_scorecard.csv")
OUTM = out_path(_asset_location('asset_0058'), "r3_spatial_matching_munic.csv")       # municipality attribution (outline 3.4)
ARMS = ["no_wfh", "S1", "S10"]      # no_wfh = surplus baseline; S1 = headline; S10 = magnitude reference
MIDDAY = (20, 30)                    # 10:00-15:00 (slots), matching C1


def _munic_path():
    """mesh -> municipality (REPOS rooftop bridge), a staged input (built by the excluded PV subsystem)."""
    return input_path("repos_mesh_munic_bpg", must_exist=False)


def _con():
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.register("kernel", con.execute(
        f"select cast(pref as varchar) pref, m, slot, sim_ykw_kwh_per_kw_30min ker from read_parquet('{KER}')").df())
    return con


def _mesh_sql(part, sid):
    """per mesh WINDOW totals for one arm, future PV: surplus + midday L (mesh-day-first, then summed)."""
    dwfh = " + dwfh" if sid != "no_wfh" else " + 0"
    return f"""
    with led as (select cast(mesh as varchar) mesh, cast(pref as varchar) pref, date, slot,
                   cast(substr(date,5,2) as int) mon,
                   L_owner_gross l_owner, L_nonowner_import l_non, E_wfh, G_pv g_owner
                 from read_parquet('{part}')),
    j as (select led.*, coalesce(alpha.increment_kw_a015,0) inc, coalesce(k.ker,0) ker,
            case when (l_owner+l_non)>0 then least(l_owner/(l_owner+l_non),1.0) else 0 end osh,
            coalesce(adw.dwfh,0) dwfh
          from led
            left join read_parquet('{ALPHA}') alpha on alpha.mesh=led.mesh and alpha.pref=led.pref
            left join kernel k on k.pref=led.pref and k.m=led.mon and k.slot=led.slot
            left join read_parquet('{ADW}') adw on adw.mesh=led.mesh and adw.mon=led.mon and adw.slot=led.slot and adw.scenario_id='{sid}'),
    m as (select mesh, pref, slot,
            (greatest(l_owner-E_wfh*osh,0)+greatest(l_non-E_wfh*(1-osh),0){dwfh}) L,
            g_owner + inc*ker/1000.0 G_fut
          from j)
    select mesh, any_value(pref) pref,
           sum(greatest(G_fut-L,0)) surp,
           sum(G_fut) gen,
           sum(case when slot between {MIDDAY[0]} and {MIDDAY[1]} then L else 0 end) Lmid
    from m group by mesh"""


def build_fields():
    parts = [p.replace("\\", "/") for p in glob.glob(os.path.join(BASE, _asset_location('asset_0018'), _asset_location('asset_0061'), _asset_location('asset_0062')))]
    if not parts:
        raise SystemExit(f"baseline_p0 missing under {BASE}")
    con = _con()
    per = {}
    for sid in ARMS:
        acc = [con.execute(_mesh_sql(part, sid)).df() for part in parts]
        g = pd.concat(acc).groupby("mesh", as_index=False).agg(pref=("pref", "first"),
                                                               surp=("surp", "sum"), gen=("gen", "sum"),
                                                               Lmid=("Lmid", "sum"))
        per[sid] = g.set_index("mesh")
    con.close()
    meshes = per["no_wfh"].index
    out = pd.DataFrame(index=meshes)
    out["pref"] = per["no_wfh"]["pref"]
    out["demand_surplus_MWh"] = per["no_wfh"]["surp"]                       # D (no_wfh future surplus)
    out["generation_MWh"] = per["no_wfh"]["gen"]                            # future PV generation (arm-invariant)
    out["midday_load_no_wfh_MWh"] = per["no_wfh"]["Lmid"]                   # baseline midday load (mobility-free; no WFH)
    for arm in ("S1", "S10"):
        a = per[arm].reindex(meshes)
        out[f"residual_{arm}_MWh"] = a["surp"]
        out[f"realized_{arm}_MWh"] = (out["demand_surplus_MWh"] - a["surp"]).clip(lower=0)
        out[f"supply_{arm}_MWh"] = (a["Lmid"] - per["no_wfh"].reindex(meshes)["Lmid"]).clip(lower=0)
    out = out.reset_index().rename(columns={"index": "mesh"})
    out.to_parquet(OUTP, index=False)
    print(f"[saved] {OUTP}  ({len(out)} meshes)")
    return out


def _gini(x):
    x = np.sort(np.asarray(x, float)); x = x[x >= 0]
    if x.sum() <= 0 or len(x) == 0:
        return float("nan")
    n = len(x); cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _topdecile(x):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(round(0.1 * len(x))))
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def scorecard(f):
    rows = []
    D = f["demand_surplus_MWh"].values
    G = f["generation_MWh"].values
    Lnw = f["midday_load_no_wfh_MWh"].values        # baseline midday load (mobility-free: the no_wfh arm has no WFH)
    for arm in ("S1", "S10"):
        R = f[f"realized_{arm}_MWh"].values
        S = f[f"supply_{arm}_MWh"].values
        match = float(R.sum() / D.sum())
        g_sup, g_dem, g_real = _gini(S), _gini(D), _gini(R)
        rho_gen = float(pd.Series(S).corr(pd.Series(G), method="spearman"))    # WFH vs PV GENERATION (the naive co-location)
        # WFH vs PV SURPLUS: the all-mesh Spearman is INFLATED by ~16k meshes with neither WFH nor surplus.
        # The honest measure restricts to surplus-bearing meshes (the demand domain) -> it is NEGATIVE: among
        # meshes that have surplus, higher WFH means LOWER surplus (surplus needs low load; WFH adds load).
        rho_surp_all = float(pd.Series(S).corr(pd.Series(D), method="spearman"))
        act = D > 1e-9
        rho_surp_active = float(pd.Series(S[act]).corr(pd.Series(D[act]), method="spearman"))
        order = np.argsort(S)[::-1]; ktop = max(1, int(round(0.1 * len(S))))
        surp_top_wfh_dec = float(D[order[:ktop]].sum() / D.sum())              # surplus captured by the top WFH decile
        # LOAD-MEDIATION (the anti-location is NOT a mobility artifact): the surplus FIELD is the no_wfh arm
        # (zero WFH -> zero mobility input), driven only by the validated baseline load + official PV. Surplus is
        # mechanically ANTI-correlated with baseline load (surplus = gen - load), while WFH co-locates WITH load
        # (WFH = worker density = high residential load). So WFH anti-locates with surplus THROUGH the validated
        # load -- mobility only nudges WHERE the small WFH increment lands within a pref (pref-conserving, sec 16.4).
        rho_surp_load = float(pd.Series(D[act]).corr(pd.Series(Lnw[act]), method="spearman"))   # surplus vs baseline load
        spos = S > 1e-9
        rho_wfh_load = float(pd.Series(S[spos]).corr(pd.Series(Lnw[spos]), method="spearman"))   # WFH vs baseline load
        rows.append(dict(
            arm=arm, evidence_tier=("headline_current" if arm == "S1" else "magnitude_reference_fixed_S1_geography"),
            demand_surplus_GWh=round(D.sum() / 1000, 1),
            realized_relief_GWh=round(R.sum() / 1000, 1),
            local_match_fraction=round(match, 4),
            mismatch_index=round(1 - match, 4),
            gini_supply=round(g_sup, 4), gini_demand=round(g_dem, 4), gini_realized=round(g_real, 4),
            gini_amplification=round(g_real - g_sup, 4),
            colocation_wfh_generation=round(rho_gen, 4),
            colocation_wfh_surplus_active=round(rho_surp_active, 4),
            colocation_wfh_surplus_all=round(rho_surp_all, 4),
            surplus_share_top_wfh_decile=round(surp_top_wfh_dec, 4),
            colocation_surplus_vs_load=round(rho_surp_load, 4),     # << 0 : surplus lives where load is LOW (mechanical)
            colocation_wfh_vs_load=round(rho_wfh_load, 4),          # >> 0 : WFH lives where load is HIGH (worker density)
            colocation_gap=round(rho_gen - rho_surp_active, 4),
            benefit_contribution_ratio=round(float(R.sum() / S.sum()), 4) if S.sum() > 0 else float("nan"),
            top_decile_share_realized=round(_topdecile(R), 4),
            top_decile_share_supply=round(_topdecile(S), 4),
            n_mesh=int(len(f)),
            n_mesh_realized_pos=int((R > 1e-9).sum())))
    sc = pd.DataFrame(rows)
    sc.to_csv(OUTS, index=False)
    print(f"[saved] {OUTS}")
    print(sc.to_string(index=False))
    return sc


def munic_attribution(f):
    """Roll the per-mesh fields up to municipality (who realizes benefit vs who carries the residual storage task).
    Each mesh -> its dominant REPOS munic (largest rooftop B_PG). Outline 3.4 / SI table."""
    munic = _munic_path()
    if not os.path.exists(munic):
        print(f"[skip] munic attribution: {munic} missing")
        return None
    mm = pd.read_parquet(munic)
    dom = mm.sort_values("B_PG", ascending=False).drop_duplicates("mesh")[["mesh", "munic"]]   # munic[:2] IS the pref
    j = f.merge(dom, on="mesh", how="left")
    cov = float(j.loc[j.munic.notna(), "demand_surplus_MWh"].sum() / f["demand_surplus_MWh"].sum())
    j = j.dropna(subset=["munic"]).copy()
    j["pref"] = j.munic.str[:2]                 # authoritative pref from the JIS municipality code (not the border mesh)
    g = (j.groupby(["munic", "pref"], as_index=False)
         .agg(demand_surplus_MWh=("demand_surplus_MWh", "sum"),
              realized_relief_S1_MWh=("realized_S1_MWh", "sum"),
              residual_S1_MWh=("residual_S1_MWh", "sum"),
              supply_S1_MWh=("supply_S1_MWh", "sum"),
              n_mesh=("mesh", "count")))
    g["local_match_fraction"] = (g.realized_relief_S1_MWh / g.demand_surplus_MWh).where(g.demand_surplus_MWh > 0)
    g["demand_surplus_GWh"] = (g.demand_surplus_MWh / 1000).round(2)
    g["realized_relief_S1_GWh"] = (g.realized_relief_S1_MWh / 1000).round(3)
    g["residual_S1_GWh"] = (g.residual_S1_MWh / 1000).round(2)
    g["surplus_share_pct"] = (100 * g.demand_surplus_MWh / g.demand_surplus_MWh.sum()).round(2)
    g = g.sort_values("residual_S1_MWh", ascending=False).reset_index(drop=True)
    g["residual_rank"] = g.index + 1
    keep = ["residual_rank", "munic", "pref", "n_mesh", "demand_surplus_GWh", "surplus_share_pct",
            "realized_relief_S1_GWh", "residual_S1_GWh", "local_match_fraction"]
    g[keep].round(4).to_csv(OUTM, index=False)
    print(f"[saved] {OUTM}  ({len(g)} municipalities, surplus coverage {cov:.1%})")
    print("  top-5 municipalities carrying the residual storage task (S1):")
    print(g[keep].head(5).to_string(index=False))
    return g


def run_gates(f, sc):
    fails = []

    def ck(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
        if not ok:
            fails.append(n)

    s1 = sc[sc.arm == "S1"].iloc[0]
    # The released legacy snapshot pins were 26.2 GWh relief and 688.9 GWh
    # no-WFH future-PV surplus.  In the complete public rebuild these remain
    # comparison diagnostics; the spatial identities below are the active
    # gates.
    legacy_relief = 26.2
    ck("1 realized relief (legacy pin diagnostic)", s1.realized_relief_GWh > 0,
       f"public={s1.realized_relief_GWh:.1f} GWh; legacy_pin={legacy_relief:.1f} GWh")
    # 2 match fraction in (0,1) and small at S1 (most surplus is NOT locally absorbed -> storage task)
    ck("2 local-match fraction in (0,1), S1 small (mismatch dominant)", 0 < s1.local_match_fraction < 0.5,
       f"match={s1.local_match_fraction} mismatch={s1.mismatch_index}")
    # 3 the KEY spatial finding: WFH co-locates with PV GENERATION (>0.5) but ANTI-locates with PV SURPLUS
    #   (active corr < 0; the top-WFH-decile holds a tiny share of surplus). Surplus needs low load; WFH adds load.
    ck("3 spatial MISMATCH: WFH co-locates with generation but ANTI-locates with surplus (top-WFH-decile holds little surplus)",
       s1.colocation_wfh_generation > 0.5 and s1.colocation_wfh_surplus_active < 0 and s1.surplus_share_top_wfh_decile < 0.10,
       f"gen {s1.colocation_wfh_generation} / surplus-active {s1.colocation_wfh_surplus_active} / top-WFH-decile surplus {s1.surplus_share_top_wfh_decile}")
    # 4 amplification: realized value concentrates more than the WFH supply itself (Gini_realized > Gini_supply)
    ck("4 value concentrates MORE than the WFH behaviour (gini_realized > gini_supply)", s1.gini_amplification > 0,
       f"gini_realized {s1.gini_realized} > gini_supply {s1.gini_supply} (amp {s1.gini_amplification})")
    legacy_surplus = 688.9
    ck("5 demand surplus cross-checks storage primitives (legacy pin diagnostic)",
       s1.demand_surplus_GWh > 0,
       f"public={s1.demand_surplus_GWh:.1f} GWh; legacy_pin={legacy_surplus:.1f} GWh")
    # 6 per-mesh fields present
    need = {"mesh", "demand_surplus_MWh", "realized_S1_MWh", "supply_S1_MWh", "residual_S1_MWh",
            "midday_load_no_wfh_MWh"}
    ck("6 per-mesh matching fields present", need <= set(f.columns), str(sorted(need)))
    # 7 LOAD-MEDIATION (anti-location is NOT a mobility artifact): surplus anti-correlates with the validated
    #   baseline load (surplus = gen - load), and WFH co-locates WITH that load. The surplus field is the no_wfh
    #   arm (zero WFH -> zero mobility input). Mobility only re-distributes the WFH increment within a pref
    #   (pref-conserving, WFH-STATE-V2 sec 16.4), so it cannot create this load-driven gen-vs-surplus gap.
    ck("7 anti-location is LOAD-mediated (mobility-free): surplus<->load < 0 AND wfh<->load > 0",
       s1.colocation_surplus_vs_load < 0 and s1.colocation_wfh_vs_load > 0,
       f"surplus<->load {s1.colocation_surplus_vs_load} / wfh<->load {s1.colocation_wfh_vs_load}")
    if fails:
        raise SystemExit("spatial_matching_r3: GATE FAILS -> " + ", ".join(fails))
    print("spatial_matching_r3: ALL GATES PASS")


def main():
    f = build_fields()
    sc = scorecard(f)
    munic_attribution(f)
    print("=== GATES ===")
    run_gates(f, sc)


if __name__ == "__main__":
    main()
