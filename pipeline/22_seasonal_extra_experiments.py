#!/usr/bin/env python3
"""Seasonal robustness experiments on the fixed legacy_34307 mesh universe.

The delivered Standard Statistics payload contains corrected mesh load only for
October 2023 through February 2024, although its inventory lists a full year.
Consequently DJF is evaluated directly from baseline_p0.  Missing months use a
transparent hybrid load reconstruction:

  five-month mesh x day-type x slot no-WFH load reference
    x observed prefecture/date/slot owner-gross-load scaling.

PV generation and owner load remain metered for every month.  The output keeps
the evidence tier on every seasonal result and never presents extrapolated
spring/summer load as directly observed Standard Statistics load.
"""
from __future__ import annotations

import calendar
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("WFH_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("WFH_SEASONAL_OUT", ROOT / "outputs" / "figures" / "extra" / "legacy_34307_seasonal"))
TMP = Path(os.environ.get("WFH_SEASONAL_TMP", ROOT / "outputs" / "_tmp" / "seasonal_extra"))
BASE = Path(os.environ.get("WFH_SEASONAL_BASE", ROOT / "data" / "derived" / "ledger" / "baseline_p0"))
OWNER = Path(os.environ.get("WFH_SEASONAL_OWNER", ROOT / "data" / "derived" / "btm" / "owner_ledger_30min"))
CAPACITY = Path(os.environ.get("WFH_SEASONAL_CAPACITY", ROOT / "data" / "derived" / "btm" / "btm_capacity_mesh_month.parquet"))
ALPHA = Path(os.environ.get("WFH_SEASONAL_ALPHA", ROOT / "data" / "derived" / "pv" / "alpha_rooftop_capacity_by_mesh.parquet"))
SCOPE = Path(os.environ.get("WFH_SEASONAL_SCOPE", ROOT / "data" / "derived" / "legacy_scope" / "study_mesh_universe.parquet"))
WFH = Path(os.environ.get("WFH_SEASONAL_WFH", ROOT / "data" / "derived" / "wfh_state" / "a_dwfh_mesh_mon_slot_v2_fullyear" / "scenario=S1" / "part.parquet"))
WFH_REGISTERED = Path(os.environ.get("WFH_SEASONAL_WFH_REGISTERED", ROOT / "data" / "derived" / "wfh_state" / "a_dwfh_mesh_mon_slot.parquet"))
LEGACY = Path(os.environ.get("WFH_SEASONAL_REPRO_SOURCE", ""))
SCOPE_LABEL = os.environ.get("WFH_SEASONAL_SCOPE_LABEL", "fixed positive-aggregate-load mesh universe")
DIRECT_MONTHS = set(os.environ.get("WFH_SEASONAL_DIRECT_MONTHS", "202310,202311,202312,202401,202402").split(","))
CALCULATION_VERSION = "seasonal_hybrid_v4_fixed_scope_completion"

MONTHS = ["202304", "202305", "202306", "202307", "202308", "202309",
          "202310", "202311", "202312", "202401", "202402", "202403"]
SEASON_BY_MONTH = {3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer",
                   9: "Autumn", 10: "Autumn", 11: "Autumn",
                   12: "Winter", 1: "Winter", 2: "Winter"}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 6.3753]
SEED = 20260804
N_NULL = 2000

# The model's calendar is shared with state, scheduling and policy calculations.
import sys
sys.path.insert(0,str(ROOT/'src'))
from wlpr.io.activity_calendar import calendar_frame, EXCLUDED_DATES
HOLIDAYS = EXCLUDED_DATES



def qpath(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def dates_frame(months: list[str] = MONTHS) -> pd.DataFrame:
    return calendar_frame(months)[['date','ym','mon','daytype']]


def connection() -> duckdb.DuckDBPyConnection:
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{qpath(TMP)}'")
    return con


def parquet_valid(con: duckdb.DuckDBPyConnection, path: Path,
                  required: set[str], min_rows: int = 1) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        columns = {row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{qpath(path)}')"
        ).fetchall()}
        if not required.issubset(columns):
            return False
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{qpath(path)}')"
        ).fetchone()[0]
        return rows >= min_rows
    except (duckdb.Error, OSError):
        return False


def copy_atomic(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp.parquet")
    tmp.unlink(missing_ok=True)
    con.execute(
        f"COPY ({query}) TO '{qpath(tmp)}' (FORMAT parquet, COMPRESSION zstd)"
    )
    tmp.replace(path)


def dataframe_to_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp.parquet")
    tmp.unlink(missing_ok=True)
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def build_reference(con: duckdb.DuckDBPyConnection, dates: pd.DataFrame) -> tuple[Path, Path]:
    ref_path = OUT / "load_reference_mesh_daytype_slot.parquet"
    pref_ref_path = OUT / "owner_load_reference_pref_daytype_slot.parquet"
    con.register("calendar_days", dates)
    scope = qpath(SCOPE)
    base = qpath(BASE / "pref=*" / "ym=*" / "*.parquet")
    if parquet_valid(con, ref_path,
                     {"mesh", "pref", "daytype", "slot", "reference_L0_MWh"},
                     min_rows=100_000):
        print("[seasonal] reusing valid mesh load reference", flush=True)
    else:
        ref_path.unlink(missing_ok=True)
        copy_atomic(con, f"""
          WITH x AS (
            SELECT cast(b.mesh AS varchar) mesh, cast(b.pref AS varchar) pref,
                   c.daytype, b.slot,
                   greatest(b.L_owner_gross - b.E_wfh *
                     CASE WHEN b.L_owner_gross+b.L_nonowner_import>0
                          THEN least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0) ELSE 0 END, 0)
                   + greatest(b.L_nonowner_import - b.E_wfh *
                     CASE WHEN b.L_owner_gross+b.L_nonowner_import>0
                          THEN 1-least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0) ELSE 1 END, 0) L0
            FROM read_parquet('{base}') b
            JOIN read_parquet('{scope}') s
              ON cast(s.mesh AS varchar)=cast(b.mesh AS varchar)
             AND cast(s.pref AS varchar)=cast(b.pref AS varchar)
             AND s.load_status='positive_aggregate_load'
            JOIN calendar_days c ON c.date=cast(b.date AS varchar)
          )
          SELECT mesh,pref,daytype,slot,avg(L0) reference_L0_MWh
          FROM x GROUP BY 1,2,3,4
        """, ref_path)

    owner_actual = qpath(OWNER / "pref=*" / "ym=*" / "*.parquet")
    actual_list = ",".join(f"'{m}'" for m in sorted(DIRECT_MONTHS))
    if parquet_valid(con, pref_ref_path,
                     {"pref", "daytype", "slot", "reference_owner_gross_MWh"},
                     min_rows=100):
        print("[seasonal] reusing valid owner-load reference", flush=True)
    else:
        pref_ref_path.unlink(missing_ok=True)
        copy_atomic(con, f"""
          WITH daily AS (
            SELECT cast(o.pref AS varchar) pref, cast(o.ymd AS varchar) date, o.slot,
                   sum(coalesce(o.gross_btm_load_kwh,0))/1000.0 owner_gross_MWh
            FROM read_parquet('{owner_actual}') o
            JOIN read_parquet('{scope}') s
              ON cast(s.mesh AS varchar)=cast(o.mesh AS varchar)
             AND cast(s.pref AS varchar)=cast(o.pref AS varchar)
             AND s.load_status='positive_aggregate_load'
            WHERE substr(cast(o.ymd AS varchar),1,6) IN ({actual_list})
            GROUP BY 1,2,3
          )
          SELECT d.pref,c.daytype,d.slot,avg(d.owner_gross_MWh) reference_owner_gross_MWh
          FROM daily d JOIN calendar_days c USING(date)
          GROUP BY 1,2,3
        """, pref_ref_path)
    return ref_path, pref_ref_path


def build_kernel(con: duckdb.DuckDBPyConnection) -> Path:
    path = OUT / "observed_stock_yield_kernel_full_year.parquet"
    if parquet_valid(con, path,
                     {"pref", "ym", "mon", "slot", "yield_kWh_per_kW_30min"},
                     min_rows=1_000):
        print("[seasonal] reusing valid observed PV kernel", flush=True)
        return path
    path.unlink(missing_ok=True)
    owner = qpath(OWNER / "pref=*" / "ym=*" / "*.parquet")
    capacity = qpath(CAPACITY)
    month_list = ",".join(f"'{m}'" for m in MONTHS)
    copy_atomic(con, f"""
        WITH generation AS (
          SELECT cast(pref AS varchar) pref, substr(cast(ymd AS varchar),1,6) ym, slot,
                 sum(coalesce(pv_gen_kwh,0)) gen_kwh,
                 count(distinct cast(ymd AS varchar)) observed_days
          FROM read_parquet('{owner}')
          WHERE substr(cast(ymd AS varchar),1,6) IN ({month_list})
          GROUP BY 1,2,3
        ), cap AS (
          SELECT cast(pref AS varchar) pref, cast(ym AS varchar) ym,
                 sum(coalesce(pv_capacity_kw,0)) cap_kw
          FROM read_parquet('{capacity}')
          WHERE cast(ym AS varchar) IN ({month_list}) GROUP BY 1,2
        )
        SELECT g.pref,g.ym,cast(substr(g.ym,5,2) AS integer) mon,g.slot,
               g.gen_kwh/nullif(c.cap_kw*g.observed_days,0) yield_kWh_per_kW_30min,
               g.observed_days,c.cap_kw,
               'metered_owner_selfcons_plus_export' kernel_source
        FROM generation g JOIN cap c USING(pref,ym)
    """, path)
    return path


def scale_columns(alias_d: str = "D", alias_b: str = "B") -> str:
    return ",\n".join(
        f"least({scale}*{alias_b},{alias_d}) A_{str(scale).replace('.', '_')}"
        for scale in SCALES
    )


def month_source_sql(con: duckdb.DuckDBPyConnection, ym: str, dates: pd.DataFrame,
                     ref_path: Path, pref_ref_path: Path, kernel_path: Path) -> tuple[str, str]:
    month_dates = dates.loc[dates.ym.eq(ym), ["date", "daytype", "mon"]].copy()
    con.register("month_days", month_dates)
    scope, alpha, kernel, wfh = map(qpath, [SCOPE, ALPHA, kernel_path, WFH])
    wfh_registered = qpath(WFH_REGISTERED)
    month = int(ym[4:])
    cols = scale_columns()

    if ym in DIRECT_MONTHS:
        base = qpath(BASE / "pref=*" / f"ym={ym}" / "*.parquet")
        source_sql = f"""
          WITH x AS (
            SELECT cast(b.mesh AS varchar) mesh, cast(b.pref AS varchar) pref,
                   cast(b.date AS varchar) date,b.slot,
                   greatest(b.L_owner_gross - b.E_wfh *
                     CASE WHEN b.L_owner_gross+b.L_nonowner_import>0
                          THEN least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0) ELSE 0 END,0)
                   + greatest(b.L_nonowner_import - b.E_wfh *
                     CASE WHEN b.L_owner_gross+b.L_nonowner_import>0
                          THEN 1-least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0) ELSE 1 END,0) L0,
                   b.G_pv + coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000.0 G,
                   coalesce(w.dwfh,0) B
            FROM read_parquet('{base}') b
            JOIN read_parquet('{scope}') s
              ON cast(s.mesh AS varchar)=cast(b.mesh AS varchar)
             AND cast(s.pref AS varchar)=cast(b.pref AS varchar)
             AND s.load_status='positive_aggregate_load'
            LEFT JOIN read_parquet('{alpha}') a
              ON cast(a.mesh AS varchar)=cast(b.mesh AS varchar) AND cast(a.pref AS varchar)=cast(b.pref AS varchar)
            LEFT JOIN read_parquet('{kernel}') k
              ON cast(k.pref AS varchar)=cast(b.pref AS varchar) AND k.ym='{ym}' AND k.slot=b.slot
            LEFT JOIN read_parquet('{wfh_registered}') w
              ON cast(w.mesh AS varchar)=cast(b.mesh AS varchar)
             AND w.mon={month} AND w.slot=b.slot AND w.scenario_id='S1'
          ), z0 AS (
            SELECT *,greatest(G-L0,0) D FROM x
          ), z AS (
            SELECT *,least(B,D) actual_absorbed,{cols} FROM z0
          )
        """
        evidence = "direct_corrected_low_voltage_ledger"
    else:
        owner = qpath(OWNER / "pref=*" / f"ym={ym}" / "*.parquet")
        ref, pref_ref = qpath(ref_path), qpath(pref_ref_path)
        source_sql = f"""
          WITH owner_mesh AS (
            SELECT cast(o.mesh AS varchar) mesh,cast(o.pref AS varchar) pref,cast(o.ymd AS varchar) date,o.slot,
                   sum(coalesce(o.gross_btm_load_kwh,0))/1000.0 owner_gross_MWh,
                   sum(coalesce(o.pv_gen_kwh,0))/1000.0 current_G_MWh
            FROM read_parquet('{owner}') o
            JOIN read_parquet('{scope}') s
              ON cast(s.mesh AS varchar)=cast(o.mesh AS varchar)
             AND cast(s.pref AS varchar)=cast(o.pref AS varchar)
             AND s.load_status='positive_aggregate_load'
            GROUP BY 1,2,3,4
          ), pref_daily AS (
            SELECT pref,date,slot,sum(owner_gross_MWh) owner_gross_MWh
            FROM owner_mesh GROUP BY 1,2,3
          ), factors AS (
            SELECT p.pref,p.date,p.slot,
                   p.owner_gross_MWh/nullif(r.reference_owner_gross_MWh,0) load_scale
            FROM pref_daily p JOIN month_days d USING(date)
            JOIN read_parquet('{pref_ref}') r
              ON r.pref=p.pref AND r.daytype=d.daytype AND r.slot=p.slot
          ), x AS (
            SELECT r.mesh,r.pref,d.date,r.slot,
                   r.reference_L0_MWh*coalesce(f.load_scale,1.0) L0,
                   coalesce(o.current_G_MWh,0)
                     + coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000.0 G,
                   coalesce(w.dwfh,0) B
            FROM read_parquet('{ref}') r
            JOIN month_days d ON d.daytype=r.daytype
            LEFT JOIN factors f ON f.pref=r.pref AND f.date=d.date AND f.slot=r.slot
            LEFT JOIN owner_mesh o
              ON o.mesh=r.mesh AND o.pref=r.pref AND o.date=d.date AND o.slot=r.slot
            LEFT JOIN read_parquet('{alpha}') a ON cast(a.mesh AS varchar)=r.mesh AND cast(a.pref AS varchar)=r.pref
            LEFT JOIN read_parquet('{kernel}') k ON k.pref=r.pref AND k.ym='{ym}' AND k.slot=r.slot
            LEFT JOIN read_parquet('{wfh}') w
              ON cast(w.mesh AS varchar)=r.mesh AND cast(w.pref AS varchar)=r.pref
             AND w.mon={month} AND w.slot=r.slot
          ), z0 AS (
            SELECT *,greatest(G-L0,0) D FROM x
          ), z AS (
            SELECT *,least(B,D) actual_absorbed,{cols} FROM z0
          )
        """
        evidence = "hybrid_load_extrapolated"

    return source_sql, evidence


def aggregate_month(con: duckdb.DuckDBPyConnection, ym: str, dates: pd.DataFrame,
                    ref_path: Path, pref_ref_path: Path, kernel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_sql, evidence = month_source_sql(con, ym, dates, ref_path, pref_ref_path, kernel_path)
    scope = qpath(SCOPE)
    sum_scales = ",".join(f"sum(A_{str(s).replace('.', '_')}) A_{str(s).replace('.', '_')}" for s in SCALES)
    grouped = con.execute(source_sql + f"""
      SELECT CASE WHEN grouping(mesh)=1 THEN 'system' ELSE 'mesh' END aggregation,
             date,slot,mesh,pref,
             sum(D) D_MWh,sum(B) B_MWh,sum(actual_absorbed) actual_absorbed_MWh,
             sum(G) future_generation_MWh,sum(L0) no_wfh_load_MWh,
             sum(D-actual_absorbed) residual_surplus_MWh,
             sum(CASE WHEN slot BETWEEN 20 AND 30 THEN L0 ELSE 0 END) midday_load_MWh,
             {sum_scales}
      FROM z
      GROUP BY GROUPING SETS ((date,slot),(mesh,pref))
    """).df()
    system = (grouped.loc[grouped.aggregation.eq("system"),
              ["date", "slot", "D_MWh", "B_MWh", "actual_absorbed_MWh",
               "future_generation_MWh", "no_wfh_load_MWh", *[f"A_{str(s).replace('.', '_')}" for s in SCALES]]]
              .sort_values(["date", "slot"]).reset_index(drop=True))
    mesh = (grouped.loc[grouped.aggregation.eq("mesh"),
            ["mesh", "pref", "D_MWh", "B_MWh", "actual_absorbed_MWh",
             "residual_surplus_MWh", "midday_load_MWh", *[f"A_{str(s).replace('.', '_')}" for s in SCALES]]]
            .rename(columns={"D_MWh": "surplus_MWh", "B_MWh": "behavior_supply_MWh",
                             "actual_absorbed_MWh": "behavior_absorbed_MWh"})
            .sort_values(["pref", "mesh"]).reset_index(drop=True))
    mesh["source_mesh_present"] = 1
    fixed_scope = con.execute(f"""
        SELECT cast(mesh AS varchar) mesh, cast(pref AS varchar) pref
        FROM read_parquet('{scope}')
        WHERE load_status='positive_aggregate_load'
    """).df()
    mesh = fixed_scope.merge(mesh, on=["mesh", "pref"], how="left", validate="one_to_one")
    numeric = [column for column in mesh.columns if column not in {"mesh", "pref"}]
    mesh[numeric] = mesh[numeric].fillna(0)
    system["ym"], system["evidence_tier"] = ym, evidence
    mesh["ym"], mesh["evidence_tier"] = ym, evidence
    system["calculation_version"] = CALCULATION_VERSION
    mesh["calculation_version"] = CALCULATION_VERSION
    return system, mesh


def seasonal_outputs(system: pd.DataFrame, mesh_month: pd.DataFrame) -> dict[str, pd.DataFrame]:
    system = system.copy()
    system["month"] = system.ym.str[4:].astype(int)
    system["season"] = system.month.map(SEASON_BY_MONTH)
    system["datetime"] = pd.to_datetime(system.date) + pd.to_timedelta(system.slot * 30, unit="min")
    mesh_month = mesh_month.copy()
    mesh_month["month"] = mesh_month.ym.str[4:].astype(int)
    mesh_month["season"] = mesh_month.month.map(SEASON_BY_MONTH)

    rows = []
    for season in SEASON_ORDER:
        x = system[system.season.eq(season)]
        D, B, actual = x.D_MWh.sum(), x.B_MWh.sum(), x.actual_absorbed_MWh.sum()
        slot = np.minimum(x.D_MWh.to_numpy(float), x.B_MWh.to_numpy(float)).sum()
        energy = min(D, B)
        rows.append({
            "season": season, "months": ",".join(map(str, sorted(x.month.unique()))),
            "days": x.date.nunique(), "surplus_GWh": D/1000, "behavior_budget_GWh": B/1000,
            "actual_absorption_GWh": actual/1000, "actual_match": actual/D,
            "same_slot_ceiling": slot/D, "spatial_wall_pp": (slot-actual)/D*100,
            "temporal_wall_pp": (energy-slot)/D*100, "magnitude_wall_pp": (D-energy)/D*100,
            "magnitude_and_temporal_wall": 1-slot/D,
            "evidence_tier": "direct_corrected_low_voltage_ledger" if season == "Winter"
                             else "hybrid_load_extrapolated",
        })
    constraints = pd.DataFrame(rows)
    constraints["closure_percent"] = (constraints.actual_match*100 + constraints.spatial_wall_pp +
                                       constraints.temporal_wall_pp + constraints.magnitude_wall_pp)

    monthly_rows = []
    for ym, x in system.groupby("ym", sort=True):
        D, B, actual = x.D_MWh.sum(), x.B_MWh.sum(), x.actual_absorbed_MWh.sum()
        slot = np.minimum(x.D_MWh, x.B_MWh).sum()
        monthly_rows.append({"ym":ym,"season":x.season.iloc[0],"surplus_GWh":D/1000,
                             "behavior_budget_GWh":B/1000,"actual_match_percent":actual/D*100,
                             "same_slot_ceiling_percent":slot/D*100,
                             "spatial_wall_pp":(slot-actual)/D*100,
                             "evidence_tier":x.evidence_tier.iloc[0]})
    monthly = pd.DataFrame(monthly_rows)

    profile = (system.groupby(["season","slot"],as_index=False)
               .agg(surplus_MWh=("D_MWh","sum"),behavior_available_MWh=("B_MWh","sum"),
                    actual_absorbed_MWh=("actual_absorbed_MWh","sum"),days=("date","nunique")))
    profile["same_slot_ceiling_MWh"] = np.minimum(profile.surplus_MWh, profile.behavior_available_MWh)
    for col in ["surplus_MWh","behavior_available_MWh","actual_absorbed_MWh","same_slot_ceiling_MWh"]:
        profile[col] = profile[col]/profile.days
    profile["hour"] = profile.slot/2

    scale_rows = []
    for season in SEASON_ORDER:
        x = system[system.season.eq(season)]
        total = x.D_MWh.sum()
        for scale in SCALES:
            col = f"A_{str(scale).replace('.', '_')}"
            actual = x[col].sum()
            free = np.minimum(scale*x.B_MWh, x.D_MWh).sum()
            scale_rows.append({"season":season,"scale":scale,"surplus_GWh":total/1000,
                               "actual_absorption_GWh":actual/1000,"actual_match":actual/total,
                               "same_slot_ceiling":free/total,"spatial_wall_pp":(free-actual)/total*100})
    scales = pd.DataFrame(scale_rows)

    mesh_season = (mesh_month.groupby(["season","mesh","pref"],as_index=False)
                   .sum(numeric_only=True))
    return {"system":system,"mesh_month":mesh_month,"mesh_season":mesh_season,
            "constraints":constraints,"monthly":monthly,"profile":profile,"scales":scales}


def null_models(mesh: pd.DataFrame, constraints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summaries, draws = [], []
    for season in SEASON_ORDER:
        x = mesh[mesh.season.eq(season)].copy()
        D = x.surplus_MWh.clip(lower=0).to_numpy(float)
        B = x.behavior_supply_MWh.clip(lower=0).to_numpy(float)
        weights = x.midday_load_MWh.clip(lower=0).to_numpy(float)
        weights = weights/weights.sum() if weights.sum() else np.ones(len(x))/len(x)
        actual = float(constraints.loc[constraints.season.eq(season),"actual_match"].iloc[0])
        for model in ["uniform_permutation","load_weighted"]:
            values = np.empty(N_NULL)
            for i in range(N_NULL):
                if model == "uniform_permutation":
                    placed = rng.permutation(B)
                else:
                    destinations = rng.choice(len(B),size=len(B),p=weights)
                    placed = np.bincount(destinations,weights=B,minlength=len(B))
                values[i] = np.minimum(D,placed).sum()/D.sum()
            draws.extend({"season":season,"model":model,"draw":i,"matching_rate":v}
                         for i,v in enumerate(values))
            q = np.quantile(values,[.025,.5,.975])
            summaries.append({"season":season,"model":model,"n":N_NULL,"actual_match":actual,
                              "null_mean":values.mean(),"null_sd":values.std(ddof=1),
                              "null_p025":q[0],"null_median":q[1],"null_p975":q[2],
                              "actual_percentile":np.mean(values<=actual)})
    return pd.DataFrame(summaries),pd.DataFrame(draws)


def spatial_diagnostics(mesh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deciles, curves = [], []
    for season in SEASON_ORDER:
        x = mesh[mesh.season.eq(season)].copy()
        positive = x[x.surplus_MWh>0].copy()
        positive["decile"] = pd.qcut(positive.surplus_MWh.rank(method="first"),10,labels=False)+1
        totals = x[["surplus_MWh","behavior_absorbed_MWh","residual_surplus_MWh"]].sum()
        for decile,g in positive.groupby("decile"):
            d,a,r = g.surplus_MWh.sum(),g.behavior_absorbed_MWh.sum(),g.residual_surplus_MWh.sum()
            deciles.append({"season":season,"decile":int(decile),"mesh_count":len(g),
                            "surplus_share_percent":d/totals.surplus_MWh*100,
                            "behavior_share_percent":a/totals.behavior_absorbed_MWh*100,
                            "residual_share_percent":r/totals.residual_surplus_MWh*100,
                            "local_capture_percent":a/d*100,"residual_percent":r/d*100})
        for column,label in [("surplus_MWh","PV surplus"),("behavior_absorbed_MWh","Behavior absorbed"),
                             ("residual_surplus_MWh","Residual surplus")]:
            values=np.sort(x[column].clip(lower=0).to_numpy(float))[::-1]
            cumulative=np.cumsum(values)/values.sum()
            curves.extend({"season":season,"metric":label,"mesh_fraction":(i+1)/len(values),
                           "cumulative_energy_share":v} for i,v in enumerate(cumulative))
    return pd.DataFrame(deciles),pd.DataFrame(curves)


def write_audit(constraints: pd.DataFrame, mesh_month: pd.DataFrame) -> None:
    rows = [
        {"input":"Standard Statistics mesh LV load","available_period":"2023-10-01 to 2024-02-29",
         "seasonal_use":"direct for Oct-Feb only","evidence":"corrected delivered payload; 152 daily archives"},
        {"input":"Standard Statistics inventory workbook","available_period":"lists 2023-04 to 2024-03",
         "seasonal_use":"inventory only","evidence":"listed files are absent from replaced payload outside Oct-Feb"},
        {"input":"Self-Consumption (Low-Voltage)","available_period":"2023-04 to 2025-03",
         "seasonal_use":"metered owner load and PV for all seasons","evidence":"24 monthly owner-ledger partitions"},
        {"input":"S1 WFH full-year field","available_period":"12 calendar months",
         "seasonal_use":"fixed residential geography; monthly mechanism response","evidence":"validated Oct-Feb; mechanism_extrapolated/cdd_prior otherwise"},
        {"input":"Missing total LV load reconstruction","available_period":"Mar-Sep",
         "seasonal_use":"winter mesh/daytype/slot reference scaled by metered prefecture owner load",
         "evidence":"hybrid_load_extrapolated; not direct Standard Statistics observation"},
    ]
    pd.DataFrame(rows).to_csv(OUT/"seasonal_input_quality_audit.csv",index=False)
    coverage = (mesh_month.groupby(["ym", "evidence_tier"], as_index=False)
                .agg(fixed_scope_meshes=("mesh", "size"),
                     source_meshes_present=("source_mesh_present", "sum")))
    coverage["source_meshes_absent"] = coverage.fixed_scope_meshes - coverage.source_meshes_present
    coverage.to_csv(OUT/"seasonal_mesh_coverage_audit.csv", index=False)
    manifest = {
        "scope": SCOPE_LABEL,
        "period":"2023-04-01 to 2024-03-31",
        "direct_months":sorted(DIRECT_MONTHS),
        "modeled_months":[m for m in MONTHS if m not in DIRECT_MONTHS],
        "seasons":{"Spring":[3,4,5],"Summer":[6,7,8],"Autumn":[9,10,11],"Winter":[12,1,2]},
        "seed":SEED,"null_draws_per_model_season":N_NULL,
        "warning":"Spring, summer and part of autumn use hybrid total-load extrapolation.",
    }
    (OUT/"experiment_scope_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    dates=dates_frame()
    con=connection()
    print("[seasonal] building five-month load reference",flush=True)
    ref,pref_ref=build_reference(con,dates)
    print("[seasonal] building 12-month observed PV kernel",flush=True)
    kernel=build_kernel(con)
    system_parts,mesh_parts=[],[]
    for i,ym in enumerate(MONTHS,1):
        system_path, mesh_path = OUT/f"date_slot_{ym}.parquet", OUT/f"mesh_{ym}.parquet"
        system_ok = parquet_valid(con, system_path,
                                  {"date", "slot", "D_MWh", "B_MWh", "actual_absorbed_MWh", "ym",
                                   "calculation_version"},
                                  min_rows=1_300)
        mesh_ok = parquet_valid(con, mesh_path,
                                {"mesh", "pref", "surplus_MWh", "behavior_supply_MWh",
                                 "behavior_absorbed_MWh", "source_mesh_present", "ym",
                                 "calculation_version"}, min_rows=34_307)
        if system_ok:
            system_ok = con.execute(
                f"SELECT count(*)=0 FROM read_parquet('{qpath(system_path)}') "
                f"WHERE calculation_version<>'{CALCULATION_VERSION}'"
            ).fetchone()[0]
        if mesh_ok:
            mesh_ok = con.execute(
                f"SELECT count(*)=0 FROM read_parquet('{qpath(mesh_path)}') "
                f"WHERE calculation_version<>'{CALCULATION_VERSION}'"
            ).fetchone()[0]
        if system_ok and mesh_ok:
            print(f"[seasonal] {i:02d}/12 {ym} reusing cached aggregates", flush=True)
            system, mesh = pd.read_parquet(system_path), pd.read_parquet(mesh_path)
        else:
            system_path.unlink(missing_ok=True)
            mesh_path.unlink(missing_ok=True)
            print(f"[seasonal] {i:02d}/12 {ym} ({'direct' if ym in DIRECT_MONTHS else 'hybrid'})",flush=True)
            system,mesh=aggregate_month(con,ym,dates,ref,pref_ref,kernel)
            dataframe_to_parquet_atomic(system, system_path)
            dataframe_to_parquet_atomic(mesh, mesh_path)
        system_parts.append(system); mesh_parts.append(mesh)
    con.close()
    data=seasonal_outputs(pd.concat(system_parts,ignore_index=True),pd.concat(mesh_parts,ignore_index=True))
    data["system"].to_parquet(OUT/"seasonal_date_slot_fields.parquet",index=False)
    data["mesh_month"].to_parquet(OUT/"seasonal_mesh_month_fields.parquet",index=False)
    data["mesh_season"].to_parquet(OUT/"seasonal_mesh_fields.parquet",index=False)
    data["constraints"].to_csv(OUT/"behavior_seasonal_constraints.csv",index=False)
    data["monthly"].to_csv(OUT/"behavior_seasonal_monthly_diagnostics.csv",index=False)
    data["profile"].to_csv(OUT/"behavior_seasonal_diurnal_profiles.csv",index=False)
    data["scales"].to_csv(OUT/"behavior_seasonal_scale_sensitivity.csv",index=False)
    null_summary,null_draws=null_models(data["mesh_season"],data["constraints"])
    null_summary.to_csv(OUT/"behavior_seasonal_null_models.csv",index=False)
    null_draws.to_parquet(OUT/"behavior_seasonal_null_draws.parquet",index=False)
    deciles,curves=spatial_diagnostics(data["mesh_season"])
    deciles.to_csv(OUT/"behavior_seasonal_spatial_deciles.csv",index=False)
    curves.to_parquet(OUT/"behavior_seasonal_concentration.parquet",index=False)
    write_audit(data["constraints"], data["mesh_month"])

    direct=data["system"][data["system"].ym.isin(DIRECT_MONTHS)]
    checks={
        "direct_surplus_GWh":direct.D_MWh.sum()/1000,
        "direct_absorption_GWh":direct.actual_absorbed_MWh.sum()/1000,
        "max_constraint_closure_error_pp":float((data["constraints"].closure_percent-100).abs().max()),
    }
    if LEGACY and (LEGACY / "behavior_date_slot_fields.parquet").exists():
        legacy_fields = pd.read_parquet(LEGACY / "behavior_date_slot_fields.parquet")
        checks["registered_surplus_GWh"] = legacy_fields.D.sum()/1000
        checks["registered_absorption_GWh"] = legacy_fields.realized_S1_MWh.sum()/1000
        checks["surplus_reproduction_abs_GWh"] = abs(checks["direct_surplus_GWh"]-checks["registered_surplus_GWh"])
        checks["absorption_reproduction_abs_GWh"] = abs(checks["direct_absorption_GWh"]-checks["registered_absorption_GWh"])
    else:
        checks["registered_surplus_GWh"] = float("nan")
        checks["registered_absorption_GWh"] = float("nan")
        checks["surplus_reproduction_abs_GWh"] = float("nan")
        checks["absorption_reproduction_abs_GWh"] = float("nan")
    pd.DataFrame([checks]).to_csv(OUT/"seasonal_reproduction_checks.csv",index=False)
    if (np.isfinite(checks["surplus_reproduction_abs_GWh"]) and
            (checks["surplus_reproduction_abs_GWh"]>1e-6 or checks["absorption_reproduction_abs_GWh"]>1e-6)):
        raise RuntimeError(f"registered-window reproduction failed: {checks}")
    if checks["max_constraint_closure_error_pp"]>1e-8:
        raise RuntimeError(f"seasonal constraint closure failed: {checks}")
    print(data["constraints"].to_string(index=False))
    print(f"[seasonal] outputs -> {OUT}")


if __name__ == "__main__":
    main()
