#!/usr/bin/env python3
"""Run the spatial and temporal questions specified in the paper draft.

The experiment scope is the 34,307 positive-aggregate-load mesh universe.  The
script materializes authoritative result tables for Figures 4-6.  It does not
modify the recovered ledgers or joint behavior-storage results.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import glob
import shutil
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import linregress, spearmanr

from wlpr.wfh_state.scenario_downscaling import scenario_multipliers, scenario_state
from wlpr.wfh_state.state_layer import mesh_state


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / _asset_location('asset_0087') / _asset_location('asset_0059') / _asset_location('asset_0060') / _asset_location('asset_0099')
DATA = V4 / _asset_location('asset_0002')
CHECKPOINTS = V4 / _asset_location('asset_0093')
LEGACY = ROOT / _asset_location('asset_0087') / _asset_location('asset_0059') / _asset_location('asset_0060') / _asset_location('asset_0094')
SEASONAL = ROOT / _asset_location('asset_0087') / _asset_location('asset_0059') / _asset_location('asset_0060') / _asset_location('asset_0082')
DERIVED = ROOT / _asset_location('asset_0002') / _asset_location('asset_0088')

R3 = LEGACY / "r3_spatial_matching_mesh.parquet"
BASE = DERIVED / _asset_location('asset_0052') / _asset_location('asset_0053')
ALPHA = DERIVED / _asset_location('asset_0054') / "alpha_rooftop_capacity_by_mesh.parquet"
KERNEL = DERIVED / _asset_location('asset_0054') / "obs_stock_yield_kernel.parquet"
ADW = DERIVED / _asset_location('asset_0055') / "a_dwfh_mesh_mon_slot.parquet"
ADW_S1 = DERIVED / _asset_location('asset_0055') / _asset_location('asset_0071') / _asset_location('asset_0072') / "part.parquet"
CURVES = DERIVED / _asset_location('asset_0080') / "scenario_window_curves_month_slot.parquet"
SCENARIO_INDEX = DERIVED / _asset_location('asset_0057') / "scenario_indices_s1s10.csv"
SCENARIO_AGG = DERIVED / _asset_location('asset_0057') / "scenario_aggregate_s1s10_a_caliber.parquet"
NETLOAD = DERIVED / _asset_location('asset_0063') / "netload_system_date_slot.parquet"
WEATHER = DERIVED / _asset_location('asset_0055') / "a_dwfh_factors_pref_date_weather.parquet"
MOBILITY = DERIVED / _asset_location('asset_0100') / "mesh_diurnal_presence_l.parquet"

EXPECTED_MESHES = 34_307
EXPECTED_DATES = 152
SEED = 20260814
RNG = np.random.default_rng(SEED)
ALPHAS = {0.05: "D005", 0.15: "D015", 0.30: "D030", 0.50: "D050"}
BEHAVIOR_SCALES = [0.0, 1.0, 3.0, 6.3753]
DISTANCE_RADII = [5.0, 10.0, 20.0]


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(DATA / name, index=False)


def mesh_to_latlon(mesh: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    s = mesh.astype(str).str.zfill(9)
    y1 = s.str[0:2].astype(int); x1 = s.str[2:4].astype(int)
    y2 = s.str[4].astype(int); x2 = s.str[5].astype(int)
    y3 = s.str[6].astype(int); x3 = s.str[7].astype(int)
    q = s.str[8].astype(int)
    y4 = (q - 1) // 2; x4 = (q - 1) % 2
    lat = (y1 + y2 / 8 + y3 / 80 + y4 / 160 + 1 / 320) / 1.5
    lon = 100 + x1 + x2 / 8 + x3 / 80 + x4 / 160 + 1 / 320
    return lat.to_numpy(), lon.to_numpy()


def gini(values: np.ndarray) -> float:
    x = np.sort(np.clip(np.asarray(values, float), 0, None))
    if not len(x) or x.sum() <= 0:
        return np.nan
    return float((len(x) + 1 - 2 * np.cumsum(x).sum() / x.sum()) / len(x))


def theil(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    x = np.clip(np.asarray(values, float), 0, None)
    mu = x.mean()
    if mu <= 0:
        return np.nan, np.nan, np.nan
    ratio = x / mu
    terms = np.zeros_like(ratio)
    positive = ratio > 0
    terms[positive] = ratio[positive] * np.log(ratio[positive])
    total = float(np.mean(terms))
    between = 0.0
    for group in np.unique(groups):
        z = x[groups == group]
        mug = z.mean()
        if mug > 0:
            between += len(z) / len(x) * (mug / mu) * np.log(mug / mu)
    return total, float(between), float(between / total if total > 0 else np.nan)


def block_ids(frame: pd.DataFrame, km: float = 10.0) -> np.ndarray:
    return (np.floor(frame.x_km.to_numpy() / km).astype(np.int64) * 10_000
            + np.floor(frame.y_km.to_numpy() / km).astype(np.int64))


def block_bootstrap_stat(
    frame: pd.DataFrame,
    stat,
    draws: int = 400,
) -> tuple[float, float]:
    blocks = block_ids(frame)
    unique = np.unique(blocks)
    indices = {b: np.flatnonzero(blocks == b) for b in unique}
    values = np.empty(draws)
    rng = np.random.default_rng(SEED)
    for i in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indices[b] for b in selected])
        values[i] = stat(frame.iloc[idx])
    return tuple(np.quantile(values, [0.025, 0.975]))


def question_1(scope: pd.DataFrame) -> dict[str, float]:
    """New question 1: is the modeled response a daytime relocation signal?"""
    curves = pd.read_parquet(CURVES)
    curves["hour"] = curves.slot / 2.0
    save(curves, "fig4_scenario_diurnal_profiles.csv")

    wide = curves.pivot_table(index=["mon", "slot"], columns="scenario_id", values="load_MW")
    delta = (wide["S10"] - wide["S1"]).rename("delta_MW").reset_index()
    delta["window"] = np.select(
        [delta.slot.between(0, 16), delta.slot.between(18, 29), delta.slot.between(36, 44)],
        ["00:00-08:00", "09:00-15:00", "18:00-22:00"], default="Other"
    )
    monthly = []
    for mon, group in delta.groupby("mon"):
        positive = group.delta_MW.clip(lower=0)
        total = float(positive.sum())
        for window, g in group.assign(positive=positive).groupby("window"):
            value = float(g.positive.sum())
            monthly.append({"month": int(mon), "window": window,
                            "delta_MW_slot_sum": value,
                            "share_percent": 100 * value / total if total else np.nan})
    monthly = pd.DataFrame(monthly)
    save(monthly, "fig4_daytime_concentration.csv")

    state = mesh_state()
    state = state[state.mesh.astype(str).isin(set(scope.mesh))].copy()
    cfg = scenario_multipliers()
    multipliers = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))
    total_workers = float(state.workers.sum())
    rates = []
    for sid in cfg.scenario_id:
        arm = scenario_state(state, sid, multipliers)
        rates.append({"scenario_id": sid,
                      "remote_rate_percent": 100 * float(arm.N_persons.sum()) / total_workers,
                      "wfh_energy_multiplier": float(multipliers[sid])})
    midday = (curves[curves.scenario_id.str.match(r"S\d+") & curves.slot.between(18, 29)]
              .groupby("scenario_id", as_index=False).load_MW.mean()
              .rename(columns={"load_MW": "midday_load_MW"}))
    dose = pd.DataFrame(rates).merge(midday, on="scenario_id")
    dose["midday_fill_vs_S1_MW"] = dose.midday_load_MW - float(
        dose.loc[dose.scenario_id.eq("S1"), "midday_load_MW"].iloc[0])
    fit = linregress(dose.remote_rate_percent, dose.midday_load_MW)
    dose["ols_fitted_MW"] = fit.intercept + fit.slope * dose.remote_rate_percent
    dose["ols_slope_MW_per_pp"] = fit.slope
    dose["ols_R2"] = fit.rvalue ** 2
    save(dose.sort_values("remote_rate_percent"), "fig4_dose_response.csv")
    loso = []
    for sid in dose.scenario_id:
        z = dose[~dose.scenario_id.eq(sid)]
        f = linregress(z.remote_rate_percent, z.midday_load_MW)
        loso.append({"left_out_scenario": sid, "slope_MW_per_pp": f.slope,
                     "R2": f.rvalue ** 2})
    save(pd.DataFrame(loso), "fig4_dose_response_loso.csv")

    net = pd.read_parquet(NETLOAD)
    net = net[net.pv_scenario.eq("future_pv_ai") & net.scenario_id.isin(["S1", "S10"])]
    nw = net.pivot_table(index=["date", "slot"], columns="scenario_id", values="net_load_MW").reset_index()
    nw["delta_MW"] = nw.S10 - nw.S1
    nw["date_ts"] = pd.to_datetime(nw.date)
    nw["daytype"] = np.where(nw.date_ts.dt.weekday < 5, "Weekday", "Weekend")
    nw["window"] = np.select(
        [nw.slot.between(0, 16), nw.slot.between(18, 29), nw.slot.between(36, 44)],
        ["00:00-08:00", "09:00-15:00", "18:00-22:00"], default="Other"
    )
    placebo = (nw.groupby(["date", "daytype", "window"], as_index=False)
               .agg(mean_delta_MW=("delta_MW", "mean"),
                    added_energy_MWh=("delta_MW", lambda x: x.sum() * 0.5)))
    save(placebo, "fig4_placebo_windows.csv")

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'"); con.execute("SET threads=8")
    con.register("scope", scope[["mesh"]])
    components = con.execute(f"""
        select a.scenario_id, a.mon, a.slot,
               sum(a.c_office_device) office_device_MWh,
               sum(a.c_lighting) lighting_MWh,
               sum(a.c_lunch) lunch_MWh,
               sum(a.c_heating) heating_MWh,
               sum(a.c_cooling) cooling_MWh,
               sum(a.c_other) other_MWh,
               sum(a.dwfh) total_MWh
        from read_parquet('{ADW.as_posix()}') a
        join scope s on cast(a.mesh as varchar)=s.mesh
        where a.scenario_id in ('S1','S10')
        group by 1,2,3
    """).df()
    con.close()
    save(components, "fig4_component_profiles.csv")

    weather = pd.read_parquet(WEATHER).groupby("date", as_index=False).tmean.mean()
    temp = nw.merge(weather, on="date", how="inner")
    day_temp = temp[["date", "tmean"]].drop_duplicates()
    day_temp["temperature_group"] = pd.qcut(
        day_temp.tmean.rank(method="first"), 3, labels=["Cool", "Mild", "Warm"]
    ).astype(str)
    temp = temp.merge(day_temp[["date", "temperature_group"]], on="date")
    temp_profile = (temp[temp.daytype.eq("Weekday")]
                    .groupby(["temperature_group", "slot"], as_index=False)
                    .agg(mean_delta_MW=("delta_MW", "mean"),
                         p10_delta_MW=("delta_MW", lambda x: x.quantile(.1)),
                         p90_delta_MW=("delta_MW", lambda x: x.quantile(.9)),
                         n_dates=("date", "nunique")))
    save(temp_profile, "fig4_temperature_stratified_profiles.csv")

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'"); con.execute("SET threads=8")
    con.register("scope", scope[["mesh"]])
    mobility = con.execute(f"""
        select cast(m.mesh as varchar) mesh, m.daytype, m.slot, m.presence
        from read_parquet('{MOBILITY.as_posix()}') m
        join scope s on cast(m.mesh as varchar)=s.mesh
    """).df()
    con.close()
    temporal = mobility.groupby(["daytype", "slot"], as_index=False).presence.sum()
    temporal = temporal.pivot(index="slot", columns="daytype", values="presence").reset_index()
    temporal["weekend_minus_weekday_presence"] = temporal.get("we", 0) - temporal.get("wd", 0)
    s1comp = components[components.scenario_id.eq("S1")].groupby("slot", as_index=False).total_MWh.sum()
    temporal = temporal.merge(s1comp.rename(columns={"total_MWh": "modeled_S1_MWh"}), on="slot", how="left")
    temporal_rho = spearmanr(temporal.weekend_minus_weekday_presence,
                             temporal.modeled_S1_MWh, nan_policy="omit").statistic
    temporal["spearman_temporal"] = temporal_rho
    save(temporal, "fig4_mobility_temporal_validation.csv")

    mid = mobility[mobility.slot.between(18, 29)].groupby(
        ["mesh", "daytype"], as_index=False).presence.mean()
    mid = mid.pivot(index="mesh", columns="daytype", values="presence").reset_index()
    mid["weekend_minus_weekday_midday"] = mid.get("we", np.nan) - mid.get("wd", np.nan)
    spatial = scope.merge(mid, on="mesh", how="left")
    spatial_rho = spearmanr(spatial.weekend_minus_weekday_midday,
                            spatial.supply_S1_MWh, nan_policy="omit").statistic
    spatial["spearman_spatial"] = spatial_rho
    save(spatial, "fig4_mobility_spatial_validation.csv")

    summary = {
        "daytime_share_percent": float(monthly.loc[monthly.window.eq("09:00-15:00"), "delta_MW_slot_sum"].sum()
                                         / monthly.delta_MW_slot_sum.sum() * 100),
        "dose_slope_MW_per_pp": float(fit.slope),
        "dose_R2": float(fit.rvalue ** 2),
        "weekend_max_abs_delta_MW": float(nw.loc[nw.daytype.eq("Weekend"), "delta_MW"].abs().max()),
        "mobility_temporal_rho": float(temporal_rho),
        "mobility_spatial_rho": float(spatial_rho),
    }
    save(pd.DataFrame([summary]), "fig4_summary.csv")
    return summary


def bootstrap_adjacent_ratio(daily: pd.DataFrame, draws: int = 500) -> tuple[float, float]:
    arr = daily[["delta_relief_GWh", "delta_behavior_GWh"]].to_numpy(float)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(draws):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        den = sample[:, 1].sum()
        values.append(sample[:, 0].sum() / den if den > 0 else np.nan)
    return tuple(np.nanquantile(values, [0.025, 0.975]))


def question_2() -> dict[str, float]:
    """New question 2: system benefits and diminishing marginal returns."""
    profiles = pd.read_parquet(SCENARIO_AGG)
    save(profiles, "fig5_netload_profiles.csv")
    net = pd.read_parquet(NETLOAD)
    future = net[net.pv_scenario.eq("future_pv_ai")].copy()

    events = []
    ramps = []
    for (sid, date), group in future.groupby(["scenario_id", "date"]):
        group = group.sort_values("slot")
        negative = np.clip(-group.net_load_MW.to_numpy(float), 0, None)
        values = group.net_load_MW.to_numpy(float)
        events.append({"scenario_id": sid, "date": date,
                       "negative_duration_h": float((negative > 0).sum() * .5),
                       "negative_energy_GWh": float(negative.sum() * .5 / 1000),
                       "minimum_net_load_MW": float(values.min())})
        slot = dict(zip(group.slot.astype(int), values))
        ramps.append({"scenario_id": sid, "date": date,
                      "ramp_15_18_MW": float(slot.get(36, np.nan) - slot.get(30, np.nan)),
                      "max_30min_ramp_MW": float(np.nanmax(np.abs(np.diff(values))))})
    events = pd.DataFrame(events)
    ramps = pd.DataFrame(ramps)
    save(events, "fig5_negative_netload_events.csv")
    save(ramps, "fig5_daily_ramps.csv")

    index = pd.read_csv(SCENARIO_INDEX)
    index = index[index.pv_scenario.eq("future_pv_ai")].sort_values("dWFH_total_GWh")
    save(index, "fig5_scenario_system_metrics.csv")
    storage = pd.read_csv(ROOT / _asset_location('asset_0087') / _asset_location('asset_0058') / "storage_physics_scorecard.csv")
    storage = storage[storage.pv_scenario.eq("future_pv_ai")]
    save(storage, "fig5_peakshave_tasks.csv")

    # Daily reverse-flow relief and behavior energy let the marginal ratios
    # carry an empirical day-resampling interval rather than a cosmetic band.
    byday = (future.groupby(["scenario_id", "date"], as_index=False)
             .agg(local_surplus_GWh=("local_surplus_MWh", lambda x: x.sum() / 1000),
                  net_energy_GWh=("net_load_MW", lambda x: x.sum() * .5 / 1000)))
    order = index.scenario_id.tolist()
    marginal_rows = []
    for prev, curr in zip(order[:-1], order[1:]):
        a = byday[byday.scenario_id.eq(prev)].set_index("date")
        b = byday[byday.scenario_id.eq(curr)].set_index("date")
        daily = pd.DataFrame({
            "delta_relief_GWh": a.local_surplus_GWh - b.local_surplus_GWh,
            "delta_behavior_GWh": b.net_energy_GWh - a.net_energy_GWh,
        }).dropna()
        ratio = daily.delta_relief_GWh.sum() / daily.delta_behavior_GWh.sum()
        lo, hi = bootstrap_adjacent_ratio(daily)
        marginal_rows.append({"from_scenario": prev, "to_scenario": curr,
                              "behavior_energy_from_GWh": float(index.loc[index.scenario_id.eq(prev), "dWFH_total_GWh"].iloc[0]),
                              "behavior_energy_to_GWh": float(index.loc[index.scenario_id.eq(curr), "dWFH_total_GWh"].iloc[0]),
                              "marginal_relief_GWh_per_GWh": ratio,
                              "bootstrap_lower": lo, "bootstrap_upper": hi,
                              "bootstrap_draws": 500})
    marginal = pd.DataFrame(marginal_rows)
    save(marginal, "fig5_marginal_relief.csv")
    return {
        "negative_days_no_wfh": int((events.query("scenario_id == 'no_wfh'").negative_duration_h > 0).sum()),
        "negative_days_S1": int((events.query("scenario_id == 'S1'").negative_duration_h > 0).sum()),
        "negative_days_S10": int((events.query("scenario_id == 'S10'").negative_duration_h > 0).sum()),
        "S1_relief_GWh": float(index.loc[index.scenario_id.eq("S1"), "nonowner_relief_GWh"].iloc[0]),
        "S10_relief_GWh": float(index.loc[index.scenario_id.eq("S10"), "nonowner_relief_GWh"].iloc[0]),
    }


def shifted_block_absorption(group: pd.DataFrame, radius_km: float) -> float:
    if not (group.D015.gt(1e-9).any() and group.B.gt(1e-9).any()):
        return 0.0
    x = group.x_km.to_numpy(float); y = group.y_km.to_numpy(float)
    demand = group.D015.to_numpy(float); supply = group.B.to_numpy(float)
    side = radius_km / np.sqrt(2.0)
    best = 0.0
    for xs in (0.0, .25, .5, .75):
        ix = np.floor(x / side + xs).astype(np.int64)
        for ys in (0.0, .25, .5, .75):
            iy = np.floor(y / side + ys).astype(np.int64)
            key = ix * 100_000 + iy
            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
            dsum = np.add.reduceat(demand[order], starts)
            bsum = np.add.reduceat(supply[order], starts)
            best = max(best, float(np.minimum(dsum, bsum).sum()))
    return best


def stage_month(con: duckdb.DuckDBPyConnection, ym: str, out: Path) -> None:
    month = int(ym[-2:])
    files = sorted(BASE.glob(f"{_asset_location('asset_0098')}{ym}{_asset_location('asset_0022')}"))
    for source in files:
        query = f"""
        with led as (
          select cast(l.mesh as varchar) mesh, cast(l.pref as varchar) pref,
                 l.date, l.slot, l.L_owner_gross lo, l.L_nonowner_import ln,
                 l.E_wfh, l.G_pv
        from read_parquet('{source.as_posix()}') l
          join scope s on cast(l.mesh as varchar)=s.mesh
                      and cast(l.pref as varchar)=s.pref
        ), j as (
          select led.*, coalesce(a.increment_kw_a005,0) i005,
                 coalesce(a.increment_kw_a015,0) i015,
                 coalesce(a.increment_kw_a030,0) i030,
                 coalesce(a.increment_kw_a050,0) i050,
                 coalesce(k.ker,0) ker,
                 case when lo+ln>0 then least(lo/(lo+ln),1.0) else 0 end osh,
                 coalesce(w.dwfh,0) B
          from led
          left join alpha a on a.mesh=led.mesh and a.pref=led.pref
          left join kernel k on k.pref=led.pref and k.m={month} and k.slot=led.slot
          left join wfh w on w.mesh=led.mesh and w.mon={month} and w.slot=led.slot
        ), m as (
          select *, greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0) L0
          from j
        )
        select m.mesh,m.pref,m.date,m.slot,c.x_km,c.y_km,m.B,
               greatest(G_pv+i005*ker/1000.0-L0,0) D005,
               greatest(G_pv+i015*ker/1000.0-L0,0) D015,
               greatest(G_pv+i030*ker/1000.0-L0,0) D030,
               greatest(G_pv+i050*ker/1000.0-L0,0) D050
        from m join coords c on c.mesh=m.mesh
        where B>1e-9 or greatest(G_pv+i050*ker/1000.0-L0,0)>1e-9
        """
        con.execute(f"""copy ({query}) to '{out.as_posix()}'
          (format parquet, compression zstd, partition_by (date,pref), overwrite_or_ignore true)""")
        print(f"[v4] staged {ym} {source.parts[-3]}", flush=True)


def heavy_spatial_experiments(scope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cached_surface = DATA / "fig5_pv_behavior_surface.csv"
    cached_distance = DATA / "fig6_distance_reallocation.csv"
    if cached_surface.exists() and cached_distance.exists():
        surface = pd.read_csv(cached_surface)
        distance = pd.read_csv(cached_distance)
        reference = float(scope.demand_surplus_MWh.sum() / 1000)
        cached_reference = float(surface.loc[
            np.isclose(surface.alpha, 0.15), "surplus_GWh"
        ].iloc[0])
        if (surface.mesh_count.eq(EXPECTED_MESHES).all()
                and distance.mesh_count.eq(EXPECTED_MESHES).all()
                and np.isclose(cached_reference, reference, atol=1e-6)):
            print("[v4] reuse heavy spatial experiment outputs", flush=True)
            return surface, distance

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'"); con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    alpha = con.execute(f"""select cast(mesh as varchar) mesh,cast(pref as varchar) pref,
      increment_kw_a005,increment_kw_a015,increment_kw_a030,increment_kw_a050
      from read_parquet('{ALPHA.as_posix()}')""").df()
    kernel = con.execute(f"""select cast(pref as varchar) pref,m,slot,
      sim_ykw_kwh_per_kw_30min ker from read_parquet('{KERNEL.as_posix()}')""").df()
    wfh = con.execute(f"""select cast(mesh as varchar) mesh,mon,slot,dwfh
      from read_parquet('{ADW_S1.as_posix()}') where scenario_id='S1'""").df()
    coords = scope[["mesh", "x_km", "y_km"]].copy()
    con.register("scope", scope[["mesh", "pref"]]); con.register("alpha", alpha)
    con.register("kernel", kernel); con.register("wfh", wfh); con.register("coords", coords)

    surface_totals = {(a, s): {"D": 0.0, "A": 0.0, "F": 0.0, "B": 0.0}
                      for a in ALPHAS for s in BEHAVIOR_SCALES}
    distance_totals = {r: 0.0 for r in DISTANCE_RADII}
    d15_total = 0.0
    temp_root = Path(tempfile.mkdtemp(prefix="v4-spatial-", dir=CHECKPOINTS))
    try:
        for ym in ["202310", "202311", "202312", "202401", "202402"]:
            month_dir = temp_root / ym
            month_dir.mkdir()
            stage_month(con, ym, month_dir)
            for date_dir in sorted(month_dir.glob(_asset_location('asset_0081'))):
                pattern = (date_dir / _asset_location('asset_0018') / _asset_location('asset_0062')).as_posix()
                frame = con.execute(f"select * from read_parquet('{pattern}')").df()
                for _, group in frame.groupby("slot", sort=False):
                    b = group.B.to_numpy(float)
                    for alpha, column in ALPHAS.items():
                        d = group[column].to_numpy(float)
                        dsum = float(d.sum()); bsum = float(b.sum())
                        for scale in BEHAVIOR_SCALES:
                            t = surface_totals[(alpha, scale)]
                            t["D"] += dsum
                            t["B"] += scale * bsum
                            t["A"] += float(np.minimum(d, scale * b).sum())
                            t["F"] += min(dsum, scale * bsum)
                    d15_total += float(group.D015.sum())
                    for radius in DISTANCE_RADII:
                        distance_totals[radius] += shifted_block_absorption(group, radius)
                print(f"[v4] spatial {date_dir.name.split('=')[1]} rows={len(frame):,}", flush=True)
            shutil.rmtree(month_dir)
    finally:
        con.close()
        shutil.rmtree(temp_root, ignore_errors=True)

    rows = []
    for (alpha, scale), t in surface_totals.items():
        rows.append({"alpha": alpha, "behavior_scale": scale,
                     "surplus_GWh": t["D"] / 1000,
                     "behavior_budget_GWh": t["B"] / 1000,
                     "anchored_absorption_GWh": t["A"] / 1000,
                     "free_placement_absorption_GWh": t["F"] / 1000,
                     "anchored_match_percent": 100 * t["A"] / t["D"],
                     "free_placement_match_percent": 100 * t["F"] / t["D"],
                     "mesh_count": EXPECTED_MESHES, "date_count": EXPECTED_DATES})
    surface = pd.DataFrame(rows)
    save(surface, "fig5_pv_behavior_surface.csv")

    actual = float(scope.realized_S1_MWh.sum())
    free = float(surface.query("alpha == 0.15 and behavior_scale == 1.0").free_placement_absorption_GWh.iloc[0] * 1000)
    dist_rows = [{"radius_km": 0.0, "absorbed_GWh": actual / 1000,
                  "surplus_GWh": d15_total / 1000,
                  "match_percent": 100 * actual / d15_total,
                  "allocation": "residential anchored"}]
    for radius in DISTANCE_RADII:
        absorbed = distance_totals[radius]
        dist_rows.append({"radius_km": radius, "absorbed_GWh": absorbed / 1000,
                          "surplus_GWh": d15_total / 1000,
                          "match_percent": 100 * absorbed / d15_total,
                          "allocation": "date-slot shifted-square feasible lower bound"})
    dist_rows.append({"radius_km": np.inf, "absorbed_GWh": free / 1000,
                      "surplus_GWh": d15_total / 1000,
                      "match_percent": 100 * free / d15_total,
                      "allocation": "same-date-slot metro free placement"})
    distance = pd.DataFrame(dist_rows)
    distance["mesh_count"] = EXPECTED_MESHES; distance["date_count"] = EXPECTED_DATES
    if not np.allclose(distance.surplus_GWh, scope.demand_surplus_MWh.sum() / 1000, atol=1e-6):
        raise RuntimeError("V4 spatial experiment does not close to the 34,307-mesh R3 ledger")
    finite = distance[np.isfinite(distance.radius_km)].match_percent.to_numpy()
    if np.any(np.diff(finite) < -1e-9):
        raise RuntimeError("distance-feasible absorption must be non-decreasing")
    save(distance, "fig6_distance_reallocation.csv")
    return surface, distance


def moran_permutation(frame: pd.DataFrame, column: str, permutations: int = 199) -> tuple[float, float]:
    coords = frame[["x_km", "y_km"]].to_numpy(float)
    pairs = np.array(list(cKDTree(coords).query_pairs(r=.85)), dtype=np.int64)
    if not len(pairs):
        return np.nan, np.nan
    x = frame[column].to_numpy(float)
    z = x - x.mean()
    denom = float(np.dot(z, z))
    observed = len(x) / (2 * len(pairs)) * 2 * float(np.sum(z[pairs[:, 0]] * z[pairs[:, 1]])) / denom
    rng = np.random.default_rng(SEED)
    null = np.empty(permutations)
    for i in range(permutations):
        zp = rng.permutation(z)
        null[i] = len(x) / (2 * len(pairs)) * 2 * float(np.sum(zp[pairs[:, 0]] * zp[pairs[:, 1]])) / denom
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1)
    return float(observed), float(p)


def question_3(scope: pd.DataFrame, distance: pd.DataFrame) -> dict[str, float]:
    """New question 3: anchoring scale and distributional consequences."""
    active = scope[scope.demand_surplus_MWh > 1e-9].copy()
    scope[["mesh", "pref", "lat", "lon", "demand_surplus_MWh",
           "supply_S1_MWh", "realized_S1_MWh", "residual_S1_MWh"]].to_parquet(
        DATA / "fig6_spatial_mesh.parquet", index=False
    )
    correlation_rows = []
    for label, sample, x, y in [
        ("Behavior vs PV generation", scope, "supply_S1_MWh", "generation_MWh"),
        ("Behavior vs active PV surplus", active, "supply_S1_MWh", "demand_surplus_MWh"),
    ]:
        rho = spearmanr(sample[x], sample[y]).statistic
        lo, hi = block_bootstrap_stat(sample, lambda z: spearmanr(z[x], z[y]).statistic)
        correlation_rows.append({"comparison": label, "n_mesh": len(sample),
                                 "spearman_rho": rho, "block_ci_lower": lo,
                                 "block_ci_upper": hi, "block_km": 10,
                                 "bootstrap_draws": 400})
    save(pd.DataFrame(correlation_rows), "fig6_correlation_contrast.csv")

    monthly = pd.read_parquet(SEASONAL / "seasonal_mesh_month_fields.parquet")
    monthly = monthly[monthly.mesh.astype(str).isin(set(scope.mesh))]
    month_rows = []
    for (ym, season, tier), group in monthly.groupby(["ym", "season", "evidence_tier"]):
        z = group[group.surplus_MWh > 1e-9]
        month_rows.append({"ym": ym, "season": season, "evidence_tier": tier,
                           "n_active_mesh": len(z),
                           "spearman_behavior_surplus": spearmanr(
                               z.behavior_supply_MWh, z.surplus_MWh).statistic,
                           "surplus_GWh": group.surplus_MWh.sum() / 1000,
                           "behavior_GWh": group.behavior_supply_MWh.sum() / 1000})
    save(pd.DataFrame(month_rows).sort_values("ym"), "fig6_monthly_correlations.csv")

    scope["load_decile"] = pd.qcut(scope.midday_load_no_wfh_MWh.rank(method="first"),
                                    10, labels=range(1, 11)).astype(int)
    gradient = scope.groupby("load_decile", as_index=False).agg(
        mesh_count=("mesh", "size"), load_MWh=("midday_load_no_wfh_MWh", "sum"),
        generation_MWh=("generation_MWh", "sum"), surplus_MWh=("demand_surplus_MWh", "sum"),
        behavior_MWh=("supply_S1_MWh", "sum"), absorbed_MWh=("realized_S1_MWh", "sum"))
    gradient["match_percent"] = 100 * gradient.absorbed_MWh / gradient.surplus_MWh.replace(0, np.nan)
    save(gradient, "fig6_load_gradient.csv")

    active["surplus_decile"] = pd.qcut(active.demand_surplus_MWh.rank(method="first"),
                                        10, labels=range(1, 11)).astype(int)
    zero_pref = active.groupby("pref", as_index=False).agg(
        active_meshes=("mesh", "size"), zero_behavior_meshes=("supply_S1_MWh", lambda x: int((x <= 1e-12).sum())),
        unserved_surplus_MWh=("residual_S1_MWh", "sum"))
    zero_pref["zero_behavior_percent"] = 100 * zero_pref.zero_behavior_meshes / zero_pref.active_meshes
    zero_decile = active.groupby("surplus_decile", as_index=False).agg(
        active_meshes=("mesh", "size"), zero_behavior_meshes=("supply_S1_MWh", lambda x: int((x <= 1e-12).sum())),
        unserved_surplus_MWh=("residual_S1_MWh", "sum"))
    zero_decile["zero_behavior_percent"] = 100 * zero_decile.zero_behavior_meshes / zero_decile.active_meshes
    zero_pref["stratum"] = "Prefecture"; zero_pref["level"] = zero_pref.pref
    zero_decile["stratum"] = "Surplus decile"; zero_decile["level"] = zero_decile.surplus_decile
    save(pd.concat([zero_pref, zero_decile], ignore_index=True, sort=False), "fig6_zero_behavior_strata.csv")

    lorenz_rows = []
    inequality_rows = []
    for label, column in [("Behavior budget", "supply_S1_MWh"),
                          ("PV surplus", "demand_surplus_MWh"),
                          ("Locally absorbed", "realized_S1_MWh")]:
        values = np.sort(np.clip(scope[column].to_numpy(float), 0, None))
        cumulative = np.r_[0, np.cumsum(values) / values.sum()]
        population = np.linspace(0, 1, len(cumulative))
        select = np.unique(np.linspace(0, len(values), 501).astype(int))
        lorenz_rows.extend({"quantity": label, "population_share": population[i],
                            "value_share": cumulative[i]} for i in select)
        point = gini(values)
        lo, hi = block_bootstrap_stat(scope, lambda z, c=column: gini(z[c].to_numpy()))
        sorted_desc = np.sort(values)[::-1]
        k = max(1, len(values) // 10)
        inequality_rows.append({"quantity": label, "gini": point,
                                "block_ci_lower": lo, "block_ci_upper": hi,
                                "top_decile_share": sorted_desc[:k].sum() / values.sum(),
                                "bottom_half_share": np.sort(values)[:len(values)//2].sum() / values.sum(),
                                "block_km": 10, "bootstrap_draws": 400})
    save(pd.DataFrame(lorenz_rows), "fig6_lorenz_curves.csv")
    save(pd.DataFrame(inequality_rows), "fig6_inequality.csv")

    spatial_rows = []
    for label, column in [("Behavior budget", "supply_S1_MWh"),
                          ("Locally absorbed", "realized_S1_MWh")]:
        moran, p = moran_permutation(scope, column)
        total, between, share = theil(scope[column].to_numpy(), scope.pref.to_numpy())
        spatial_rows.append({"quantity": label, "morans_I": moran,
                             "moran_permutation_p": p, "moran_permutations": 199,
                             "theil_total": total, "theil_between_pref": between,
                             "theil_between_pref_share": share})
    save(pd.DataFrame(spatial_rows), "fig6_spatial_decomposition.csv")
    save(distance, "fig6_distance_reallocation.csv")

    corr = pd.DataFrame(correlation_rows).set_index("comparison")
    ineq = pd.DataFrame(inequality_rows).set_index("quantity")
    return {
        "behavior_generation_rho": float(corr.loc["Behavior vs PV generation", "spearman_rho"]),
        "behavior_surplus_rho": float(corr.loc["Behavior vs active PV surplus", "spearman_rho"]),
        "zero_behavior_active_percent": float((active.supply_S1_MWh <= 1e-12).mean() * 100),
        "absorbed_gini": float(ineq.loc["Locally absorbed", "gini"]),
        "distance_20km_match_percent": float(distance.loc[distance.radius_km.eq(20), "match_percent"].iloc[0]),
    }


def write_summary(q1: dict, q2: dict, q3: dict, surface: pd.DataFrame) -> None:
    rows = []
    for question, values in [("New question 1", q1), ("New question 2", q2), ("New question 3", q3)]:
        for metric, value in values.items():
            rows.append({"question": question, "metric": metric, "value": value})
    save(pd.DataFrame(rows), "v4_new_questions_summary.csv")
    audit = pd.DataFrame([
        ("scope", len(pd.read_parquet(R3)) == EXPECTED_MESHES, EXPECTED_MESHES),
        ("winter dates", pd.read_parquet(NETLOAD).date.nunique() == EXPECTED_DATES, EXPECTED_DATES),
        ("surface rows", len(surface) == len(ALPHAS) * len(BEHAVIOR_SCALES), len(surface)),
        ("figure 4 tables", len(list(DATA.glob("fig4_*.csv"))) >= 10, len(list(DATA.glob("fig4_*.csv")))),
        ("figure 5 tables", len(list(DATA.glob("fig5_*.csv"))) >= 7, len(list(DATA.glob("fig5_*.csv")))),
        ("figure 6 tables", len(list(DATA.glob("fig6_*.csv"))) >= 9, len(list(DATA.glob("fig6_*.csv")))),
    ], columns=["check", "passed", "value"])
    save(audit, "v4_new_questions_audit.csv")
    if not audit.passed.all():
        raise RuntimeError("V4 audit failed:\n" + audit.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    scope = pd.read_parquet(R3).copy()
    scope["mesh"] = scope.mesh.astype(str)
    if len(scope) != EXPECTED_MESHES or scope.mesh.nunique() != EXPECTED_MESHES:
        raise RuntimeError("V4 requires exactly 34,307 positive-aggregate-load meshes")
    lat, lon = mesh_to_latlon(scope.mesh)
    scope["lat"] = lat; scope["lon"] = lon
    scope["x_km"] = lon * 91.0; scope["y_km"] = lat * 111.0
    q1 = question_1(scope)
    q2 = question_2()
    surface, distance = heavy_spatial_experiments(scope)
    q3 = question_3(scope, distance)
    save(pd.DataFrame([q2]), "fig5_summary.csv")
    save(pd.DataFrame([q3]), "fig6_summary.csv")
    write_summary(q1, q2, q3, surface)


if __name__ == "__main__":
    main()
