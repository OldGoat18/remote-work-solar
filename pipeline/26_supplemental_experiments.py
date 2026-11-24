#!/usr/bin/env python3
"""Supplemental analyses for the study figures.

This script only uses the registered seasonal ledgers.  It does not create
direct spring--autumn low-voltage observations or a storage SOC result when
the required post-solar deficit ledger is absent.  Such limits are written to
the audit table instead of being filled with proxy values.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / _asset_location('asset_0101')
V46 = ROOT / _asset_location('asset_0102')
SEASONAL = ROOT / _asset_location('asset_0003')
V42 = ROOT / _asset_location('asset_0103')
OUT = ROOT / _asset_location('asset_0104')
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 6.3753]
ALPHAS = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.3753]
PV_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
RATES = [0.0, 0.05, 0.10, 0.20]
WINDOWS = {
    "00:00-08:00": lambda slot: slot.between(0, 16),
    "09:00-15:00": lambda slot: slot.between(18, 30),
    "15:00-18:00": lambda slot: slot.between(31, 36),
    "18:00-22:00": lambda slot: slot.between(37, 44),
    "Other": lambda slot: ~slot.between(0, 44),
}
SEED = 20260825


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False)


def seasonal_system() -> pd.DataFrame:
    path = SEASONAL / "seasonal_date_slot_fields.parquet"
    d = pd.read_parquet(path)
    d["date"] = d["date"].astype(str)
    d["season"] = d["season"].astype(str)
    return d


def halfhour_alpha_pv_temporal_scan(d: pd.DataFrame) -> pd.DataFrame:
    """System-level half-hour scan for temporal and magnitude ceilings.

    Spatial residential anchoring is deliberately not claimed here because
    the available full-year table is system aggregated.  The monthly spatial
    E2 scan remains the linked source for spatial placement comparisons.
    """
    rows = []
    for season in SEASONS:
        x = d[d.season.eq(season)]
        for alpha in ALPHAS:
            behavior = alpha * x.B_MWh.to_numpy(float)
            for pv in PV_MULTIPLIERS:
                surplus = np.maximum(pv * x.future_generation_MWh.to_numpy(float)
                                      - x.no_wfh_load_MWh.to_numpy(float), 0.0)
                same_slot = np.minimum(behavior, surplus)
                rows.append({
                    "season": season, "alpha": alpha, "pv_multiplier": pv,
                    "surplus_GWh": surplus.sum() / 1000.0,
                    "behavior_budget_GWh": behavior.sum() / 1000.0,
                    "temporal_magnitude_ceiling_GWh": same_slot.sum() / 1000.0,
                    "temporal_magnitude_match_percent": 100 * same_slot.sum() / surplus.sum()
                    if surplus.sum() else np.nan,
                    "date_count": x.date.nunique(), "slot_count": len(x),
                    "evidence_tier": ";".join(sorted(x.evidence_tier.unique())),
                    "metric_definition": "system half-hour temporal/magnitude ceiling; no spatial placement claim",
                })
    result = pd.DataFrame(rows)
    save(result, "v47_halfhour_alpha_pv_temporal_scan.csv")
    return result


def rebound_timing_scan(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        x = d[d.season.eq(season)].copy()
        for window, selector in WINDOWS.items():
            mask = selector(x.slot)
            budget = float(x.loc[mask, "B_MWh"].sum())
            for rate in RATES:
                added = rate * budget
                rows.append({
                    "season": season, "rebound_rate": rate, "window": window,
                    "base_behavior_budget_GWh": x.B_MWh.sum() / 1000.0,
                    "additional_non_pv_window_demand_GWh": added / 1000.0,
                    "credited_additional_absorption_GWh": 0.0,
                    "identified_pv_absorption_fraction": x.actual_absorbed_MWh.sum() / x.D_MWh.sum(),
                    "window_share_of_behavior_budget_percent": 100 * budget / x.B_MWh.sum()
                    if x.B_MWh.sum() else np.nan,
                    "bias_direction": "optimistic_for_behavior; rebound is not credited to PV absorption",
                    "evidence_tier": ";".join(sorted(x.evidence_tier.unique())),
                })
    result = pd.DataFrame(rows)
    save(result, "v47_rebound_timing_scan.csv")
    return result


def seasonal_surplus_scale_scan(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, ym), x in d.groupby(["season", "ym"], sort=True):
        surplus = float(x.D_MWh.sum())
        for scale in SCALES:
            col = f"A_{str(scale).replace('.', '_')}"
            actual = float(x[col].sum())
            same_slot = float(np.minimum(scale * x.B_MWh.to_numpy(float),
                                         x.D_MWh.to_numpy(float)).sum())
            rows.append({
                "season": season, "ym": str(ym), "scale": scale,
                "surplus_GWh": surplus / 1000.0,
                "behavior_budget_GWh": scale * x.B_MWh.sum() / 1000.0,
                "anchored_absorption_GWh": actual / 1000.0,
                "same_slot_ceiling_GWh": same_slot / 1000.0,
                "anchored_match_percent": 100 * actual / surplus if surplus else np.nan,
                "same_slot_match_percent": 100 * same_slot / surplus if surplus else np.nan,
                "spatial_gap_pp": 100 * (same_slot - actual) / surplus if surplus else np.nan,
                "evidence_tier": x.evidence_tier.iloc[0],
            })
    result = pd.DataFrame(rows)
    save(result, "v47_seasonal_surplus_scale_scan.csv")
    return result


def multiscale_concentration() -> tuple[pd.DataFrame, pd.DataFrame]:
    mesh = pd.read_parquet(V4 / "fig6_spatial_mesh.parquet")
    mesh["x_km"] = (mesh.lon - mesh.lon.min()) * 91.0
    mesh["y_km"] = (mesh.lat - mesh.lat.min()) * 111.0
    metrics = {
        "PV surplus": "demand_surplus_MWh",
        "Behavior budget": "supply_S1_MWh",
        "Locally absorbed": "realized_S1_MWh",
        "Residual surplus": "residual_S1_MWh",
    }

    def gini(values: np.ndarray) -> float:
        x = np.sort(np.clip(np.asarray(values, float), 0, None))
        if len(x) == 0 or x.sum() <= 0:
            return np.nan
        return float((len(x) + 1 - 2 * np.cumsum(x).sum() / x.sum()) / len(x))

    def shares(values: np.ndarray) -> tuple[float, float]:
        x = np.sort(np.clip(np.asarray(values, float), 0, None))
        if x.sum() <= 0:
            return np.nan, np.nan
        k = max(1, len(x) // 10)
        return float(x[-k:].sum() / x.sum()), float(x[:len(x) // 2].sum() / x.sum())

    rows = []
    lorenz = []
    spatial_scales = [("mesh", None), ("500m", 0.5), ("1km", 1.0), ("2km", 2.0),
                      ("5km", 5.0), ("10km", 10.0), ("20km", 20.0)]
    for level, scale in spatial_scales:
        if scale is None:
            group = mesh.assign(unit=mesh.mesh.astype(str))
        else:
            group = mesh.assign(unit=(np.floor(mesh.x_km / scale).astype(int).astype(str)
                                      + "_" + np.floor(mesh.y_km / scale).astype(int).astype(str)))
        for label, column in metrics.items():
            agg = group.groupby("unit", as_index=False)[column].sum()
            values = agg[column].to_numpy(float)
            top, bottom = shares(values)
            rows.append({"aggregation": level, "quantity": label,
                         "n_units": len(values), "gini": gini(values),
                         "top_decile_share": top, "bottom_half_share": bottom,
                         "source": "34,307 mesh residual geography; spatial block aggregation"})
            order = np.sort(np.clip(values, 0, None))
            total = order.sum()
            if total > 0:
                cumulative = np.r_[0.0, np.cumsum(order) / total]
                population = np.linspace(0, 1, len(cumulative))
                take = np.unique(np.linspace(0, len(cumulative) - 1, 101).astype(int))
                lorenz.extend({"aggregation": level, "quantity": label,
                               "population_share": population[i],
                               "value_share": cumulative[i]} for i in take)

    mun = pd.read_csv(V46 / "E6_municipality_conversion_cloud.csv")
    for label, column in {"PV surplus": "surplus_MWh", "Behavior budget": "behavior_MWh",
                          "Locally absorbed": "absorbed_MWh", "Residual surplus": "residual_MWh"}.items():
        values = mun.groupby("munic", as_index=False)[column].sum()[column].to_numpy(float)
        top, bottom = shares(values)
        rows.append({"aggregation": "municipality", "quantity": label,
                     "n_units": len(values), "gini": gini(values),
                     "top_decile_share": top, "bottom_half_share": bottom,
                     "source": "203-municipality conversion cloud"})
    pref = mun.groupby("pref", as_index=False)[["surplus_MWh", "behavior_MWh", "absorbed_MWh", "residual_MWh"]].sum()
    for label, column in {"PV surplus": "surplus_MWh", "Behavior budget": "behavior_MWh",
                          "Locally absorbed": "absorbed_MWh", "Residual surplus": "residual_MWh"}.items():
        values = pref[column].to_numpy(float)
        top, bottom = shares(values)
        rows.append({"aggregation": "prefecture", "quantity": label,
                     "n_units": len(values), "gini": gini(values),
                     "top_decile_share": top, "bottom_half_share": bottom,
                     "source": "four-prefecture conversion aggregate"})
    result = pd.DataFrame(rows)
    curve = pd.DataFrame(lorenz)
    save(result, "v47_multiscale_concentration.csv")
    save(curve, "v47_multiscale_lorenz.csv")
    return result, curve


def date_block_bootstrap(d: pd.DataFrame, draws: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for season in SEASONS:
        x = d[d.season.eq(season)].copy()
        dates = np.array(sorted(x.date.unique()))
        by_date = {date: z for date, z in x.groupby("date")}
        for scale in [1.0, 6.3753]:
            col = f"A_{str(scale).replace('.', '_')}"
            for draw in range(draws):
                sample_dates = rng.choice(dates, size=len(dates), replace=True)
                sample = pd.concat([by_date[date] for date in sample_dates], ignore_index=True)
                surplus = sample.D_MWh.sum()
                actual = sample[col].sum()
                same = np.minimum(scale * sample.B_MWh.to_numpy(float), sample.D_MWh.to_numpy(float)).sum()
                rows.append({"season": season, "scale": scale, "draw": draw,
                             "anchored_match_percent": 100 * actual / surplus if surplus else np.nan,
                             "same_slot_match_percent": 100 * same / surplus if surplus else np.nan,
                             "spatial_gap_pp": 100 * (same - actual) / surplus if surplus else np.nan})
    result = pd.DataFrame(rows)
    save(result, "v47_seasonal_date_block_bootstrap.csv")
    return result


def gap_audit() -> pd.DataFrame:
    storage_mesh = pd.read_parquet(SEASONAL / "seasonal_date_slot_fields.parquet")
    required = {"postsolar_deficit_MWh", "evening_deficit_MWh", "charge_slots", "discharge_slots"}
    rows = [
        {"experiment": "halfhour_alpha_pv_temporal_scan", "status": "complete",
         "output": "v47_halfhour_alpha_pv_temporal_scan.csv",
         "limitation": "system temporal/magnitude ceiling; no spatial placement arm"},
        {"experiment": "rebound_rate_by_timing_window", "status": "complete",
         "output": "v47_rebound_timing_scan.csv",
         "limitation": "rebound is bookkeeping and never credited to PV absorption"},
        {"experiment": "seasonal_surplus_scale_scan", "status": "complete",
         "output": "v47_seasonal_surplus_scale_scan.csv",
         "limitation": "Spring/Summer/Autumn use hybrid load evidence"},
        {"experiment": "multiscale_concentration", "status": "complete",
         "output": "v47_multiscale_concentration.csv; v47_multiscale_lorenz.csv",
         "limitation": "spatial block aggregation is not feeder power flow"},
        {"experiment": "seasonal_date_block_bootstrap", "status": "complete",
         "output": "v47_seasonal_date_block_bootstrap.csv",
         "limitation": "date-block uncertainty for behavioral seasonal ledgers"},
        {"experiment": "seasonal_storage_soc", "status": "blocked_missing_input",
         "output": "",
         "limitation": f"required columns absent: {sorted(required - set(storage_mesh.columns))}; no proxy result generated"},
        {"experiment": "direct_spring_autumn_low_voltage_validation", "status": "blocked_missing_input",
         "output": "",
         "limitation": "direct corrected low-voltage mesh load is only available for 2023-10 through 2024-02"},
        {"experiment": "full_year_system_netload_events", "status": "blocked_missing_input",
         "output": "",
         "limitation": "seasonal ledger lacks full WFH scenario net-load and post-solar deficit fields"},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "v47_experiment_gap_audit.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = seasonal_system()
    temporal = halfhour_alpha_pv_temporal_scan(d)
    rebound = rebound_timing_scan(d)
    seasonal = seasonal_surplus_scale_scan(d)
    concentration, lorenz = multiscale_concentration()
    bootstrap = date_block_bootstrap(d)
    audit = gap_audit()
    manifest = {
        "script": str(Path(__file__).resolve()),
        "scope": "34,307 mesh seasonal ledger; 12 months; 30-minute system aggregation where stated",
        "seed": SEED,
        "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file()),
        "completed": audit.loc[audit.status.eq("complete"), "experiment"].tolist(),
        "blocked": audit.loc[audit.status.ne("complete"), "experiment"].tolist(),
        "truth_constraint": "blocked experiments have no fabricated proxy result",
    }
    (OUT / "v47_supplemental_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[v4.7] outputs -> {OUT}")
    print(f"[v4.7] temporal={len(temporal):,} rebound={len(rebound):,} seasonal={len(seasonal):,}")
    print(f"[v4.7] concentration={len(concentration):,} lorenz={len(lorenz):,} bootstrap={len(bootstrap):,}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
