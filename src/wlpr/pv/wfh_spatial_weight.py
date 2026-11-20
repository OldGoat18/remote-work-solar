"""Step 05c — WFH spatial weight (ported from _maintainer/builders/build_wfh_spatial_weight.py).

WFH spatial weight = 4-layer composition (supervisor-ratified). Mobility does NOT replace the survey; survey
gives the RATE, mobility gives SPACE, census gives the labor/age structure that filters out non-workers
(retirees/children) — because night presence alone is a RESIDENTIAL DENOMINATOR, not WFH truth.

  WFH_mesh = night_residential_pop x labor/age structure x telework participation rate(pref) x weekday daytime-home wt
For the WITHIN-pref SPATIAL allocation of E_wfh, telework_rate(pref) is constant.
  working_pop(mesh)        = pop * wage_share                          (census; excludes retirees/non-workers)
  weight_current_presence  ~ working_pop * max(wd_mid / we_mid, 0.2)   (current WFH state: weekday presence)
  weight_scenario_potential~ working_pop * (0.2 + 0.8*outflow_tilt)    (future conversion potential: commuter outflow)

Reads frozen demographic_mesh_panel + derived mesh_mobility_presence + derived baseline_p0 ledger.
Writes data/derived/pv/wfh_spatial_weight.parquet (+ data/derived/pv/wfh_spatial_weight_scorecard.csv).
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path, input_path

MOB = derived_path("pv", "mesh_mobility_presence.parquet")
LED = derived_path("ledger", "baseline_p0") + "/*/*/*.parquet"
OUT = derived_path("pv", "wfh_spatial_weight.parquet")
SC = derived_path("pv", "wfh_spatial_weight_scorecard.csv")
PN = {"11": "Saitama", "12": "Chiba", "13": "Tokyo", "14": "Kanagawa"}
FLOOR = 0.2


def build_wfh_spatial_weight() -> dict:
    DEMO = input_path("demographic_mesh_panel")
    c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute("SET threads=3")
    demo = c.execute(f"select cast(mesh as varchar) mesh, cast(pref as varchar) pref, pop, wage_share, eld_pop_share from read_parquet('{DEMO}')").df()
    mob = c.execute(f"select cast(mesh as varchar) mesh, wd_mid,wd_night,we_mid,we_night from read_parquet('{MOB}')").df()
    # study meshes = the universe needing a weight (from baseline ledger, with pref)
    study = c.execute(f"select cast(mesh as varchar) mesh, arg_max(cast(pref as varchar),tot) pref, sum(tot) dem from (select cast(mesh as varchar) mesh, cast(pref as varchar) pref, sum(L_all_gross_equiv) tot from read_parquet('{LED}') group by 1,2) group by 1").df()
    c.close()
    # start from ALL study meshes; left-join demographic + mobility
    d = study.merge(demo.drop(columns=["pref"]), on="mesh", how="left").merge(mob, on="mesh", how="left")
    d = d[d.pref.isin(["11", "12", "13", "14"])].copy()
    # fill census gaps: wage_share -> pref mean; pop -> use mesh demand as proxy where missing
    for col in ["wage_share", "eld_pop_share"]:
        d[col] = d.groupby("pref")[col].transform(lambda s: s.fillna(s.mean()))
    d["pop"] = d["pop"].fillna(0.0); d["demo_missing"] = (d["pop"] <= 0).astype(int)
    # layer 1x2: working population (census pop * wage_share) -- filters retirees/non-workers; fallback = demand-proxy
    d["working_pop"] = d["pop"] * d.wage_share
    # for meshes with no census pop, fallback working_pop = pref-median working_pop scaled by mesh demand share
    for pf, x in d.groupby("pref"):
        med = x.loc[x.demo_missing == 0, "working_pop"].median()
        d.loc[(d.pref == pf) & (d.demo_missing == 1), "working_pop"] = med if np.isfinite(med) else 1.0
    d["working_pop"] = d.working_pop.clip(lower=0).fillna(0.0)
    # layer 4a: current weekday presence (mobility). stay = how much weekend-midday residential presence remains on weekdays.
    d["mobility_missing"] = ((d.we_mid.isna()) | (d.wd_mid.isna()) | (d.we_mid <= 1e-9)).astype(int)
    d["stay_raw"] = np.where(d.mobility_missing == 0, d.wd_mid / d.we_mid.clip(lower=1e-9), np.nan)
    d["stay_raw"] = np.clip(d.stay_raw, 0, 1)
    for pf, x in d.groupby("pref"):
        med = x.loc[x.mobility_missing == 0, "stay_raw"].median()
        if not np.isfinite(med):
            med = FLOOR
        d.loc[(d.pref == pf) & (d.mobility_missing == 1), "stay_raw"] = med
    d["stay"] = d.stay_raw.clip(lower=0, upper=1).fillna(FLOOR)
    d["current_presence"] = np.maximum(d.stay, FLOOR)

    # layer 4b: scenario conversion potential (mobility outflow). This is the old formula, renamed per D11.
    d["resid"] = d.we_night.where(d.we_night.notna(), d["pop"]).clip(lower=1e-9)
    d["wd_spec_outflow"] = (d.wd_night - d.wd_mid) - (d.we_night - d.we_mid)
    d["tilt"] = np.clip(d.wd_spec_outflow / d.resid, 0, 1).fillna(0.0)
    d["scenario_potential"] = FLOOR + (1 - FLOOR) * d.tilt
    # weights (within-pref spatial; survey rate applied at pref level via E_wfh anchor)
    d["weight_current_presence"] = d.working_pop * d.current_presence
    d["weight_scenario_potential"] = d.working_pop * d.scenario_potential
    d["weight"] = d.weight_current_presence  # backward-compatible alias: current-state ledgers use the D11 presence weight.
    d["weight_nightonly"] = d.resid * d.current_presence   # night-presence comparator under the current-state presence definition.
    d = d[["mesh", "pref", "pop", "wage_share", "eld_pop_share", "working_pop", "stay", "current_presence", "scenario_potential",
           "weight_current_presence", "weight_scenario_potential", "weight", "weight_nightonly", "mobility_missing"]]
    d.to_parquet(OUT, index=False)

    # scorecard: per-pref + how much the labor-structure filter MOVES allocation vs night-only (proves it matters)
    rows = []
    for pf, x in d.groupby("pref"):
        wn = x.weight_current_presence / x.weight_current_presence.sum(); wo = x.weight_nightonly / x.weight_nightonly.sum()
        moved = float(np.abs(wn - wo).sum() / 2 * 100)  # % of WFH reallocated by adding the labor/age filter
        # correlation of per-capita WFH weight with elderly share (should be NEGATIVE: high-elderly -> less WFH/head)
        xx = x[(x["pop"] > 0) & x.eld_pop_share.notna()]
        cc = np.corrcoef(xx.weight_current_presence / xx["pop"].clip(lower=1), xx.eld_pop_share)[0, 1] if len(xx) > 10 else np.nan
        corr_current_factor_stay = x[["current_presence", "stay"]].corr(method="spearman").iloc[0, 1]
        corr_potential_factor_out = x.assign(one_minus_stay=1 - x.stay)[["scenario_potential", "one_minus_stay"]].corr(method="spearman").iloc[0, 1]
        corr_current_weight_stay = x[["weight_current_presence", "stay"]].corr(method="spearman").iloc[0, 1]
        corr_potential_weight_out = x.assign(one_minus_stay=1 - x.stay)[["weight_scenario_potential", "one_minus_stay"]].corr(method="spearman").iloc[0, 1]
        corr_current_potential = x[["weight_current_presence", "weight_scenario_potential"]].corr(method="spearman").iloc[0, 1]
        rows.append(dict(pref=PN[pf], n_mesh=len(x), wage_share_mean=round(x.wage_share.mean(), 3),
            labor_filter_reallocates_pct=round(moved, 1), corr_weight_vs_elderly=round(cc, 3),
            mobility_missing_pct=round(100 * x.mobility_missing.mean(), 1),
            corr_current_factor_vs_stay=round(corr_current_factor_stay, 3),
            corr_scenario_factor_vs_one_minus_stay=round(corr_potential_factor_out, 3),
            corr_current_weight_vs_stay=round(corr_current_weight_stay, 3),
            corr_scenario_weight_vs_one_minus_stay=round(corr_potential_weight_out, 3),
            corr_current_vs_scenario_weight=round(corr_current_potential, 3)))
    scd = pd.DataFrame(rows); scd.to_csv(SC, index=False)
    pd.set_option("display.width", 200)
    print("=== WFH SPATIAL WEIGHT (D11: current presence + scenario potential; survey rate at pref) ===")
    print(scd.to_string(index=False))
    print("\nlabor_filter_reallocates_pct = how much WFH moves vs night-presence-only (proves retiree/age structure matters)")
    print("corr_weight_vs_elderly < 0 => high-elderly meshes correctly get LESS WFH (the A retiree vs B worker distinction)")
    print(f"\nmeshes {len(d)} | weight>=0 {bool((d.weight>=0).all())}")
    print(f"[saved] {OUT}\n[saved] {SC}")
    return {"n_mesh": len(d), "weight_nonneg": bool((d.weight >= 0).all())}


def run_gates() -> dict:
    out = build_wfh_spatial_weight()
    assert out["weight_nonneg"], "spatial weight must be non-negative"
    return out


def main() -> None:
    out = run_gates()
    print(f"[05c] wfh_spatial_weight: {out['n_mesh']:,} meshes | weight>=0 {out['weight_nonneg']}")
    print("05c_wfh_spatial_weight: ALL GATES PASS")


if __name__ == "__main__":
    main()
