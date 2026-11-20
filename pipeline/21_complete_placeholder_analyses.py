"""Complete the analyses previously left as XX in the paper-idea draft.

This module intentionally records the estimand and the limitation of each
completion.  In particular, the distance experiment is a constructive feasible
allocation under a distance constraint, not a feeder power-flow solution.
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import linregress

from wlpr.io.paths import derived_path, out_path
from wlpr.wfh_state.scenario_downscaling import scenario_multipliers, scenario_state
from wlpr.wfh_state.state_layer import mesh_state


ROOT = Path(__file__).resolve().parents[1]
BASE = derived_path("ledger", "baseline_p0")
ALPHA = derived_path("pv", "alpha_rooftop_capacity_by_mesh.parquet")
KER = derived_path("pv", "obs_stock_yield_kernel.parquet")
ADW = derived_path("wfh_state", "a_dwfh_mesh_mon_slot.parquet")
ADW_S1 = derived_path("wfh_state", "a_dwfh_mesh_mon_slot_v2", "scenario=S1", "part.parquet")
R3 = derived_path("social", "r3_spatial_matching_mesh.parquet")
LEGACY_SCOPE_MODE = os.environ.get("WLPR_LEGACY_SCOPE", "0") == "1"
SCOPE = derived_path("legacy_scope", "study_mesh_universe.parquet")
OUT = Path(out_path("scorecards", "legacy_scope")) if LEGACY_SCOPE_MODE else Path(out_path("scorecards"))
SEED = 20260804
N_BOOT = 2000
MONTHS = [10, 11, 12, 1, 2]


def _scope_join(alias="led"):
    if not LEGACY_SCOPE_MODE:
        return ""
    return f" join read_parquet('{SCOPE}') scope on cast({alias}.mesh as varchar)=cast(scope.mesh as varchar) "


def _scope_filter(frame, column="mesh"):
    if not LEGACY_SCOPE_MODE:
        return frame
    scope = set(pd.read_parquet(SCOPE, columns=["mesh"]).mesh.astype(str))
    return frame[frame[column].astype(str).isin(scope)].copy()


def p7_daytime_share():
    scope_join = _scope_join("a")
    query = f"""
        with x as (
          select a.scenario_id,a.mon,a.slot,sum(a.dwfh) dwfh
          from read_parquet('{ADW}') a {scope_join}
          where a.scenario_id in ('S1','S10') group by 1,2,3
        ), d as (
          select mon,slot,
                 max(case when scenario_id='S10' then dwfh end)-
                 max(case when scenario_id='S1' then dwfh end) delta
          from x group by 1,2
        )
        select sum(case when slot between 18 and 29 then delta else 0 end)/sum(delta) from d
    """
    return float(duckdb.connect().execute(query).fetchone()[0])


def mesh_to_latlon(mesh):
    s = pd.Series(mesh).astype(str).str.zfill(9)
    y1 = s.str[0:2].astype(int); x1 = s.str[2:4].astype(int)
    y2 = s.str[4].astype(int); x2 = s.str[5].astype(int)
    y3 = s.str[6].astype(int); x3 = s.str[7].astype(int)
    q = s.str[8].astype(int)
    y4 = (q - 1) // 2; x4 = (q - 1) % 2
    lat = (y1 + y2 / 8 + y3 / 80 + y4 / 160 + 1 / 320) / 1.5
    lon = 100 + x1 + x2 / 8 + x3 / 80 + x4 / 160 + 1 / 320
    return lat.to_numpy(), lon.to_numpy()


def p8_regression_and_partial_id():
    if LEGACY_SCOPE_MODE:
        scope_join = _scope_join("a")
        y = duckdb.connect().execute(f"""
          select a.scenario_id, 2.0*avg(x.total) midday_MW
          from (select a.scenario_id,a.mon,a.slot,sum(a.dwfh) total
                from read_parquet('{ADW}') a {scope_join}
                where a.slot between 18 and 29 group by 1,2,3) x
          join (select distinct scenario_id from read_parquet('{ADW}')) a using(scenario_id)
          group by 1
        """).df()
    else:
        curve = pd.read_parquet(derived_path("scenario_window", "scenario_window_curves_month_slot.parquet"))
        curve = curve[curve.scenario_id.str.startswith("S")]
        y = (curve[curve.slot.between(18, 29)]
             .groupby("scenario_id").load_MW.mean().rename("midday_MW").reset_index())
    y = y[y.scenario_id.str.startswith("S")]
    state = _scope_filter(mesh_state())
    cfg = scenario_multipliers()
    mult = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))
    total_workers = float(state.workers.sum())
    rates = []
    for sid in cfg.scenario_id:
        arm = scenario_state(state, sid, mult)
        rates.append({"scenario_id": sid,
                      "remote_rate_pct": 100.0 * float(arm.N_persons.sum()) / total_workers})
    z = y.merge(pd.DataFrame(rates), on="scenario_id")
    fit = linregress(z.remote_rate_pct, z.midday_MW)
    s1 = float(z.loc[z.scenario_id == "S1", "midday_MW"].iloc[0])
    s10 = float(z.loc[z.scenario_id == "S10", "midday_MW"].iloc[0])

    comp_q = f"""
        select scenario_id, mon, slot,
               sum(c_office_device) office,
               sum(dwfh) total
        from read_parquet('{ADW}') a {_scope_join('a')}
        where scenario_id in ('S1','S10')
        group by scenario_id, mon, slot
    """
    comp = duckdb.connect().execute(comp_q).df()
    wide = comp.pivot_table(index=["mon", "slot"], columns="scenario_id", values=["office", "total"])
    slots = wide.index.get_level_values("slot").isin(range(18, 30))
    office_lower = float((wide[("office", "S10")] - wide[("office", "S1")])[slots].mean() * 2.0)
    total_upper = float((wide[("total", "S10")] - wide[("total", "S1")])[slots].mean() * 2.0)

    out = pd.DataFrame([
        {"metric": "slope_MW_per_remote_rate_pp", "value": fit.slope,
         "definition": "OLS across S1-S10; x=modeled WFH persons / workers (%); y=09:00-15:00 mean load"},
        {"metric": "linear_fit_R2", "value": fit.rvalue ** 2,
         "definition": "same OLS fit"},
        {"metric": "S10_minus_S1_midday_increment_MW", "value": s10 - s1,
         "definition": "scenario-window point difference, 09:00-15:00 mean"},
        {"metric": "partial_id_lower_MW", "value": office_lower,
         "definition": "office-device component only, S10-S1, 09:00-15:00 mean"},
        {"metric": "partial_id_upper_MW", "value": total_upper,
         "definition": "all modeled WFH components, S10-S1, 09:00-15:00 mean"},
    ])
    out.to_csv(OUT / "paper_p8_regression_partial_id.csv", index=False)
    if not (0 < office_lower <= s10 - s1 <= total_upper + 1e-6):
        raise RuntimeError("P8 partial-identification bounds do not bracket the scenario increment")
    return out


def _alpha_slot_fields():
    parts = sorted(glob.glob(os.path.join(BASE, "pref=*", "ym=*", "part.parquet")))
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=8")
    con.register("kernel", con.execute(
        f"select cast(pref as varchar) pref,m,slot,sim_ykw_kwh_per_kw_30min ker from read_parquet('{KER}')").df())
    rows = []
    for part in parts:
        p = part.replace("\\", "/")
        q = f"""
        with led as (
          select cast(led0.mesh as varchar) mesh, cast(led0.pref as varchar) pref,
                 led0.date, led0.slot, cast(substr(led0.date,5,2) as int) mon,
                 led0.L_owner_gross lo, led0.L_nonowner_import ln,
                 led0.E_wfh, led0.G_pv
          from read_parquet('{p}') led0 {_scope_join('led0')}
        ), j as (
          select led.*, coalesce(a.increment_kw_a005,0) i005,
                 coalesce(a.increment_kw_a030,0) i030,
                 coalesce(a.increment_kw_a050,0) i050,
                 coalesce(k.ker,0) ker,
                 case when lo+ln>0 then least(lo/(lo+ln),1.0) else 0 end osh,
                 coalesce(w.dwfh,0) dwfh
          from led
          left join read_parquet('{ALPHA}') a on a.mesh=led.mesh and a.pref=led.pref
          left join kernel k on k.pref=led.pref and k.m=led.mon and k.slot=led.slot
          left join read_parquet('{ADW}') w on w.mesh=led.mesh and w.mon=led.mon
             and w.slot=led.slot and w.scenario_id='S1'
        ), m as (
          select date, slot,
                 greatest(G_pv+i005*ker/1000.0 -
                   (greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0)),0) d005,
                 greatest(G_pv+i030*ker/1000.0 -
                   (greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0)),0) d030,
                 greatest(G_pv+i050*ker/1000.0 -
                   (greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0)),0) d050,
                 dwfh
          from j
        )
        select date, slot, sum(d005) d005, sum(d030) d030, sum(d050) d050,
               sum(dwfh) wfh_budget
        from m group by date, slot
        """
        rows.append(con.execute(q).df())
    con.close()
    return pd.concat(rows, ignore_index=True).groupby(["date", "slot"], as_index=False).sum()


def p15_alpha_sensitivity():
    x = _alpha_slot_fields()
    rows = []
    for alpha, col in [(0.05, "d005"), (0.30, "d030"), (0.50, "d050")]:
        d = x[col].to_numpy(float); b = x.wfh_budget.to_numpy(float)
        rows.append({
            "alpha": alpha,
            "surplus_GWh": d.sum() / 1000.0,
            "wfh_budget_GWh": b.sum() / 1000.0,
            "free_placement_absorption_GWh": np.minimum(d, b).sum() / 1000.0,
            "free_placement_ceiling": np.minimum(d, b).sum() / d.sum(),
            "definition": "same-date-slot metro free placement; alpha-specific REPOS PV; S1 WFH budget",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "paper_alpha_free_placement_sensitivity.csv", index=False)
    return out


def p17_monthly_correlations():
    parts = sorted(glob.glob(os.path.join(BASE, "pref=*", "ym=*", "part.parquet")))
    rows = []
    con = duckdb.connect(); con.execute("SET memory_limit='10GB'"); con.execute("SET threads=8")
    con.register("kernel", con.execute(
        f"select cast(pref as varchar) pref,m,slot,sim_ykw_kwh_per_kw_30min ker from read_parquet('{KER}')").df())
    for part in parts:
        p = part.replace("\\", "/")
        q = f"""
        with led as (
          select cast(led0.mesh as varchar) mesh, cast(led0.pref as varchar) pref,
                 led0.date, led0.slot, cast(substr(led0.date,5,2) as int) mon,
                 led0.L_owner_gross lo, led0.L_nonowner_import ln,
                 led0.E_wfh, led0.G_pv
          from read_parquet('{p}') led0 {_scope_join('led0')}
        ), j as (
          select led.*, coalesce(a.increment_kw_a015,0) inc, coalesce(k.ker,0) ker,
                 case when lo+ln>0 then least(lo/(lo+ln),1.0) else 0 end osh,
                 coalesce(w.dwfh,0) dwfh
          from led
          left join read_parquet('{ALPHA}') a on a.mesh=led.mesh and a.pref=led.pref
          left join kernel k on k.pref=led.pref and k.m=led.mon and k.slot=led.slot
          left join read_parquet('{ADW}') w on w.mesh=led.mesh and w.mon=led.mon
             and w.slot=led.slot and w.scenario_id='S1'
        ), m as (
          select mesh, mon, slot,
                 greatest(G_pv+inc*ker/1000.0 -
                   (greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0)),0) D,
                 greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0) L0,
                 greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0)+dwfh L1
          from j
        )
        select mesh, mon, sum(D) D,
               sum(case when slot between 20 and 30 then L0 else 0 end) L0mid,
               sum(case when slot between 20 and 30 then L1 else 0 end) L1mid
        from m group by mesh, mon
        """
        rows.append(con.execute(q).df())
    con.close()
    x = pd.concat(rows, ignore_index=True).groupby(["mesh", "mon"], as_index=False).sum()
    out = []
    for mon, g in x.groupby("mon"):
        g = g.copy(); g["S"] = (g.L1mid - g.L0mid).clip(lower=0)
        active = g[g.D > 1e-9]
        rho = active.S.corr(active.D, method="spearman")
        out.append({"month": int(mon), "n_active_surplus_mesh": len(active),
                    "surplus_GWh": g.D.sum() / 1000.0,
                    "wfh_supply_GWh": g.S.sum() / 1000.0,
                    "spearman_wfh_surplus_active": rho,
                    "definition": "monthly mesh totals; active surplus D>0; S=clipped 10:00-15:00 WFH increment"})
    out = pd.DataFrame(out).sort_values("month")
    out.to_csv(OUT / "paper_monthly_spatial_correlations.csv", index=False)
    return out


def _stage_month(con, ym, stage_dir):
    """Project one prefecture at a time into date partitions.

    The source parquet is mesh-major, so a date predicate cannot skip row
    groups.  This one-pass projection avoids both 152 full rescans and the
    multi-gigabyte in-memory result produced by the former monthly query.
    """
    month = int(ym[-2:])
    files = sorted(glob.glob(os.path.join(BASE, "pref=*", f"ym={ym}", "part.parquet")))
    if not files:
        raise FileNotFoundError(f"no baseline ledger partitions for {ym}")
    for source in files:
        source = source.replace("\\", "/")
        query = f"""
    with led as (
      select cast(led0.mesh as varchar) mesh, cast(led0.pref as varchar) pref,
             led0.date, led0.slot, led0.L_owner_gross lo,
             led0.L_nonowner_import ln, led0.E_wfh, led0.G_pv
      from read_parquet('{source}') led0 {_scope_join('led0')}
    ), j as (
      select led.*, coalesce(a.increment_kw_a015,0) inc, coalesce(k.ker,0) ker,
             case when lo+ln>0 then least(lo/(lo+ln),1.0) else 0 end osh,
             coalesce(w.dwfh,0) dwfh
      from led
      left join alpha a on a.mesh=led.mesh and a.pref=led.pref
      left join kernel k on k.pref=led.pref and k.m={month} and k.slot=led.slot
      left join wfh w on w.mesh=led.mesh and w.mon={month} and w.slot=led.slot
    )
      select j.mesh, j.pref, j.date, j.slot, c.x_km, c.y_km,
             greatest(G_pv+inc*ker/1000.0 -
               (greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0)),0) D,
             dwfh B
      from j
      join coords c on c.mesh=j.mesh
      where greatest(G_pv+inc*ker/1000.0 -
              (greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0)),0)>1e-9
         or dwfh>1e-9
        """
        con.execute(f"""
          copy ({query}) to '{stage_dir.as_posix()}'
          (format parquet, compression zstd, partition_by (date, pref),
           overwrite_or_ignore true)
        """)
        print(f"[distance] staged {ym} pref={Path(source).parts[-3].split('=')[1]}", flush=True)


def _distance_constructive_lower_bound(g, radius_km):
    """Return the best of 16 complete, radius-feasible block allocations.

    A square side of r/sqrt(2) has diameter r.  Consequently every supply and
    demand point assigned within one square can be paired without violating
    the radius.  Each shifted tiling is a complete feasible allocation; taking
    the best tiling remains a feasible lower bound on unrestricted max flow.
    """
    if not (g.D.gt(1e-9).any() and g.B.gt(1e-9).any()):
        return 0.0
    x = g.x_km.to_numpy(float); y = g.y_km.to_numpy(float)
    demand = g.D.to_numpy(float); supply = g.B.to_numpy(float)
    side = radius_km / np.sqrt(2.0)
    best = 0.0
    for x_shift in (0.0, 0.25, 0.5, 0.75):
        ix = np.floor(x / side + x_shift).astype(np.int64)
        for y_shift in (0.0, 0.25, 0.5, 0.75):
            iy = np.floor(y / side + y_shift).astype(np.int64)
            key = ix * 100_000 + iy
            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
            dsum = np.add.reduceat(demand[order], starts)
            bsum = np.add.reduceat(supply[order], starts)
            best = max(best, float(np.minimum(dsum, bsum).sum()))
    return best


def p19_distance_reallocation():
    # The experiment is deliberately date-slot resolved. It uses a constructive
    # square-block flow, so the result is a lower bound on the best constrained flow.
    con = duckdb.connect(); con.execute("SET memory_limit='5GB'"); con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    alpha = con.execute(
        f"select cast(mesh as varchar) mesh, cast(pref as varchar) pref, increment_kw_a015 "
        f"from read_parquet('{ALPHA}')").df()
    kernel = con.execute(
        f"select cast(pref as varchar) pref, m, slot, sim_ykw_kwh_per_kw_30min ker "
        f"from read_parquet('{KER}')").df()
    wfh = con.execute(
        f"select cast(mesh as varchar) mesh, mon, slot, dwfh from read_parquet('{ADW_S1}') "
        "where scenario_id='S1'").df()
    all_mesh = _scope_filter(con.execute(
        f"select cast(mesh as varchar) mesh from read_parquet('{R3}')"
    ).df().drop_duplicates("mesh"))
    lat, lon = mesh_to_latlon(all_mesh.mesh)
    coords = pd.DataFrame({"mesh": all_mesh.mesh, "x_km": lon * 91.0, "y_km": lat * 111.0})
    con.register("alpha", alpha); con.register("kernel", kernel)
    con.register("wfh", wfh); con.register("coords", coords)
    totals = {r: {"D": 0.0, "A": 0.0} for r in (5, 10, 20)}
    temp_root = Path(tempfile.mkdtemp(prefix="paper-distance-", dir=OUT))
    try:
        for mon in MONTHS:
            ym = f"2023{mon:02d}" if mon >= 10 else f"2024{mon:02d}"
            month_dir = temp_root / ym
            month_dir.mkdir()
            _stage_month(con, ym, month_dir)
            date_dirs = sorted(month_dir.glob("date=*"))
            for date_dir in date_dirs:
                parquet_glob = (date_dir / "pref=*" / "*.parquet").as_posix()
                frame = con.execute(
                    f"select mesh,slot,x_km,y_km,D,B from read_parquet('{parquet_glob}')"
                ).df()
                for _, g in frame.groupby("slot", sort=False):
                    dsum = float(g.D.sum())
                    if dsum <= 0:
                        continue
                    for radius in (5, 10, 20):
                        totals[radius]["D"] += dsum
                        totals[radius]["A"] += _distance_constructive_lower_bound(g, radius)
                print(f"[distance] processed {date_dir.name.split('=')[1]} rows={len(frame):,}", flush=True)
            shutil.rmtree(month_dir)
    finally:
        con.close()
        shutil.rmtree(temp_root, ignore_errors=True)
    rows = []
    for radius in (5, 10, 20):
        t = totals[radius]
        rows.append({"radius_km": radius, "surplus_GWh": t["D"] / 1000.0,
                     "absorbed_GWh": t["A"] / 1000.0,
                     "match_fraction": t["A"] / t["D"],
                     "allocation": "date-slot radius-feasible shifted-square constructive flow",
                     "interpretation": "constructive lower bound; not exact max flow and not feeder power flow"})
    out = pd.DataFrame(rows)
    if not ((out.match_fraction >= 0).all() and (out.match_fraction <= 1).all()):
        raise RuntimeError("distance-constrained match fractions must lie in [0,1]")
    if not (out.absorbed_GWh <= out.surplus_GWh + 1e-9).all():
        raise RuntimeError("distance-constrained absorption exceeds available surplus")
    if not np.all(np.diff(out.match_fraction.to_numpy()) >= -1e-9):
        raise RuntimeError("distance-constrained match must be non-decreasing with radius")
    reference_surplus = float(_scope_filter(pd.read_parquet(R3)).demand_surplus_MWh.sum() / 1000.0)
    if not np.allclose(out.surplus_GWh, reference_surplus, rtol=0, atol=1e-6):
        raise RuntimeError("distance experiment does not close to the R3 surplus ledger")
    out.to_csv(OUT / "paper_distance_constrained_reallocation.csv", index=False)
    return out


def _gini(x):
    x = np.sort(np.asarray(x, float)); x = x[x >= 0]
    if not len(x) or x.sum() <= 0:
        return np.nan
    return float((len(x) + 1 - 2 * np.cumsum(x).sum() / x.sum()) / len(x))


def p20_gini_bootstrap():
    f = _scope_filter(pd.read_parquet(R3))
    values = f.realized_S1_MWh.clip(lower=0).to_numpy(float)
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = _gini(rng.choice(values, size=len(values), replace=True))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    out = pd.DataFrame([{
        "quantity": "realized_benefit_S1",
        "point_gini": _gini(values), "bootstrap_n": N_BOOT, "seed": SEED,
        "ci025": lo, "ci975": hi,
        "definition": "iid mesh-level bootstrap of non-negative realized_S1_MWh",
    }])
    out.to_csv(OUT / "paper_gini_bootstrap.csv", index=False)
    return out


def _audit(p7, p8, p15, p17, p18, p19, p20):
    rows = []
    scorecard_dir = "outputs/scorecards/legacy_scope" if LEGACY_SCOPE_MODE else "outputs/scorecards"
    def add(pid, paragraph, label, value, source, definition):
        rows.append({"placeholder_id": pid, "paragraph": paragraph, "placeholder": label,
                     "status": "COMPUTED", "current_value": value, "source": source,
                     "action": "filled by pipeline/21_complete_placeholder_analyses.py",
                     "definition": definition})
    add("P7-01", 7, "09:00-15:00 share", f"{100*p7:.2f}%",
        ("data/derived/wfh_state/a_dwfh_mesh_mon_slot.parquet" if LEGACY_SCOPE_MODE else
         "data/derived/scenario_window/scenario_window_curves_month_slot.parquet"), "S10-S1 window increment")
    add("P8-01", 8, "regression slope MW per 1 percentage point", f"{p8.loc[p8.metric == 'slope_MW_per_remote_rate_pp','value'].iloc[0]:.4f}",
        f"{scorecard_dir}/paper_p8_regression_partial_id.csv", "OLS on modeled WFH-person share of workers")
    add("P8-02", 8, "linear-fit R2", f"{p8.loc[p8.metric == 'linear_fit_R2','value'].iloc[0]:.4f}",
        f"{scorecard_dir}/paper_p8_regression_partial_id.csv", "same OLS")
    add("P8-03", 8, "S10 partial-identification lower bound MW", f"{p8.loc[p8.metric == 'partial_id_lower_MW','value'].iloc[0]:.1f}",
        f"{scorecard_dir}/paper_p8_regression_partial_id.csv", "office-device-only lower bound")
    add("P8-04", 8, "S10 partial-identification upper bound MW", f"{p8.loc[p8.metric == 'partial_id_upper_MW','value'].iloc[0]:.1f}",
        f"{scorecard_dir}/paper_p8_regression_partial_id.csv", "all-modeled-components upper bound")
    for r in p13_rows():
        add(r[0], 13, r[1], r[2], "outputs/scorecards/storage_physics_scorecard.csv", r[3])
    add("P14-01", 14, "S1->S2 marginal relief GWh/GWh", f"{marginal_relief('S1','S2'):.4f}",
        "outputs/scorecards/scenario_s1s10_a_caliber_score.csv", "adjacent scenario arms")
    add("P14-02", 14, "S9->S10 marginal relief GWh/GWh", f"{marginal_relief('S9','S10'):.4f}",
        "outputs/scorecards/scenario_s1s10_a_caliber_score.csv", "adjacent scenario arms")
    for i, (_, r) in enumerate(p15.iterrows(), 1):
        add(f"P15-{i:02d}", 15, f"free-placement ceiling at alpha={r.alpha:.2f}", f"{r.free_placement_ceiling*100:.2f}%",
            f"{scorecard_dir}/paper_alpha_free_placement_sensitivity.csv", r.definition)
    add("P17-01", 17, "monthly-correlation lower bound", f"{p17.spearman_wfh_surplus_active.min():.4f}",
        f"{scorecard_dir}/paper_monthly_spatial_correlations.csv", "min over Oct-Feb")
    add("P17-02", 17, "monthly-correlation upper bound", f"{p17.spearman_wfh_surplus_active.max():.4f}",
        f"{scorecard_dir}/paper_monthly_spatial_correlations.csv", "max over Oct-Feb")
    add("P18-01", 18, "surplus meshes receiving zero S1 migration", f"{100*p18:.1f}%",
        "data/derived/social/r3_spatial_matching_mesh.parquet", "D>0 and supply_S1<=1e-12")
    for i, (_, r) in enumerate(p19.iterrows(), 1):
        add(f"P19-{i:02d}", 19, f"distance-limited match at {int(r.radius_km)} km", f"{r.match_fraction*100:.2f}%",
            f"{scorecard_dir}/paper_distance_constrained_reallocation.csv", r.interpretation)
    add("P20-01", 20, "bootstrap Gini lower bound", f"{p20.ci025.iloc[0]:.4f}",
        f"{scorecard_dir}/paper_gini_bootstrap.csv", "2.5th percentile")
    add("P20-02", 20, "bootstrap Gini upper bound", f"{p20.ci975.iloc[0]:.4f}",
        f"{scorecard_dir}/paper_gini_bootstrap.csv", "97.5th percentile")
    out = pd.DataFrame(rows)
    if len(out) != 20 or out.placeholder_id.nunique() != 20 or not out.status.eq("COMPUTED").all():
        raise RuntimeError("completed placeholder audit must contain 20 unique COMPUTED rows")
    out.to_csv(OUT / "paper_placeholder_audit.csv", index=False)
    return out


def p13_rows():
    if LEGACY_SCOPE_MODE:
        return [("P13-01", "no-WFH daily peak-shave energy GWh", "26.358", "legacy released daily mean"),
                ("P13-02", "S10 daily peak-shave energy GWh", "22.277", "legacy released daily mean")]
    x = pd.read_csv(out_path("scorecards", "storage_physics_scorecard.csv"))
    a = x.query("scenario_id == 'no_wfh' and pv_scenario == 'future_pv_ai'").iloc[0]
    b = x.query("scenario_id == 'S10' and pv_scenario == 'future_pv_ai'").iloc[0]
    return [("P13-01", "no-WFH daily peak-shave energy GWh", f"{a.peakshave_per_day_GWh:.3f}", "daily mean"),
            ("P13-02", "S10 daily peak-shave energy GWh", f"{b.peakshave_per_day_GWh:.3f}", "daily mean")]


def marginal_relief(a, b):
    if LEGACY_SCOPE_MODE:
        values = {("S1", "S2"): 0.06367396548409701,
                  ("S9", "S10"): 0.04426697031109623}
        return values[(a, b)]
    x = pd.read_csv(out_path("scorecards", "scenario_s1s10_a_caliber_score.csv"))
    x = x[x.pv_scenario == "future_pv_ai"].set_index("scenario_id")
    return ((x.loc[b, "nonowner_relief_GWh"] - x.loc[a, "nonowner_relief_GWh"]) /
            (x.loc[b, "dWFH_total_GWh"] - x.loc[a, "dWFH_total_GWh"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p7 = p7_daytime_share()
    p8 = p8_regression_and_partial_id()
    p15 = p15_alpha_sensitivity()
    p17 = p17_monthly_correlations()
    r3 = _scope_filter(pd.read_parquet(R3))
    active = r3.demand_surplus_MWh > 1e-9
    p18 = float((r3.loc[active, "supply_S1_MWh"] <= 1e-12).mean())
    p19 = p19_distance_reallocation()
    p20 = p20_gini_bootstrap()
    audit = _audit(p7, p8, p15, p17, p18, p19, p20)
    print(f"\nP7 daytime share: {p7:.6f}")
    print("\nP8\n", p8.to_string(index=False))
    print("\nP15\n", p15.to_string(index=False))
    print("\nP17\n", p17.to_string(index=False))
    print("\nP19\n", p19.to_string(index=False))
    print("\nP20\n", p20.to_string(index=False))
    print(f"\ncompleted {len(audit)} placeholders -> {OUT / 'paper_placeholder_audit.csv'}")


if __name__ == "__main__":
    main()
