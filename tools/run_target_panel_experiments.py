#!/usr/bin/env python3
"""Materialize the additional experiments used by the target panels.

The script only reads the registered seasonal ledgers.  It writes new
tables, never edits an input table.  In particular, seasonal timing results
are labelled as behavioral-availability evidence; they are not presented as
the unavailable full-year S10-minus-S1 system-net-load placebo.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTRA = ROOT / _asset_location('asset_0087') / _asset_location('asset_0059') / _asset_location('asset_0060')
OUT = EXTRA / _asset_location('asset_0201') / _asset_location('asset_0002')
SEASONAL = EXTRA / _asset_location('asset_0082')
LEGACY = EXTRA / _asset_location('asset_0094')
V4DATA = EXTRA / _asset_location('asset_0099') / _asset_location('asset_0002')
V42 = EXTRA / _asset_location('asset_0200')
BRIDGE = ROOT / _asset_location('asset_0002') / _asset_location('asset_0121') / _asset_location('asset_0054') / "repos_mesh_munic_bpg.parquet"
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 6.3753]
WINDOWS = {
    "00:00-08:00": (0, 16),
    "09:00-15:00": (18, 30),
    "18:00-22:00": (36, 44),
    "Other": None,
}
SEED = 20260903
N_NULL = 2000


def out_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False)


def seasonal_null_stratified() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare global, within-prefecture and within-exposure nulls by season."""
    mesh = pd.read_parquet(SEASONAL / "seasonal_mesh_fields.parquet")
    rows = []
    summaries = []
    rng = np.random.default_rng(SEED)
    models = ["global_uniform", "global_load_weighted",
              "within_prefecture", "within_exposure_decile"]
    for season in SEASONS:
        x = mesh[mesh.season.eq(season)].copy().reset_index(drop=True)
        x["D"] = x.surplus_MWh.clip(lower=0)
        x["B"] = x.behavior_supply_MWh.clip(lower=0)
        x["exposure_decile"] = pd.qcut(
            x.D.rank(method="first"), 10, labels=False
        ).astype(int) + 1
        D = x.D.to_numpy(float)
        B = x.B.to_numpy(float)
        weights = x.midday_load_MWh.clip(lower=0).to_numpy(float)
        weights = weights / weights.sum() if weights.sum() else np.ones(len(x)) / len(x)
        actual = float(x.behavior_absorbed_MWh.sum() / max(x.surplus_MWh.sum(), 1e-12))
        groups_pref = [g.to_numpy() for _, g in x.groupby("pref", sort=False).groups.items()]
        groups_dec = [g.to_numpy() for _, g in x.groupby("exposure_decile", sort=False).groups.items()]
        for model in models:
            values = np.empty(N_NULL)
            for draw in range(N_NULL):
                if model == "global_uniform":
                    placed = rng.permutation(B)
                elif model == "global_load_weighted":
                    destination = rng.choice(len(B), size=len(B), p=weights)
                    placed = np.bincount(destination, weights=B, minlength=len(B))
                else:
                    placed = np.zeros(len(B))
                    groups = groups_pref if model == "within_prefecture" else groups_dec
                    for idx in groups:
                        placed[idx] = rng.permutation(B[idx])
                values[draw] = np.minimum(D, placed).sum() / max(D.sum(), 1e-12) * 100
            for draw, value in enumerate(values):
                rows.append({"season": season, "model": model, "draw": draw,
                             "matching_rate_percent": value, "observed_match_percent": actual * 100,
                             "observed_minus_draw_pp": actual * 100 - value,
                             "n_mesh": len(x), "n_groups": len(groups_pref) if "prefecture" in model else len(groups_dec) if "decile" in model else 1})
            q025, q50, q975 = np.quantile(values, [.025, .5, .975])
            summaries.append({"season": season, "model": model, "n": N_NULL,
                              "observed_match_percent": actual * 100,
                              "null_mean_percent": values.mean(), "null_sd_percent": values.std(ddof=1),
                              "null_p025_percent": q025, "null_median_percent": q50,
                              "null_p975_percent": q975,
                              "observed_percentile": np.mean(values <= actual * 100) * 100,
                              "definition": "same seasonal surplus and behavior marginal; placement restricted by model"})
    draws = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    draws.to_parquet(OUT / "v47_seasonal_stratified_null_draws.parquet", index=False)
    out_csv(summary, "v47_seasonal_stratified_null_summary.csv")
    return draws, summary


def common_capacity_storage() -> pd.DataFrame:
    """Run the existing three dispatch policies at the stored common capacity."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import run_spatial_storage_sensitivity as v42

    dates, slots, surplus, load = v42._storage_series()
    capacity_gwh = 106.027066707611 / 1000.0
    policies = {
        "Perfect foresight": lambda h, e: v42._perfect_dispatch(capacity_gwh, h, e, surplus, load),
        "Daily MPC (24 h)": lambda h, e: v42._daily_mpc_dispatch(capacity_gwh, h, e, dates, slots, surplus, load),
        "Naive rule": lambda h, e: v42._naive_dispatch(capacity_gwh, h, e, slots, surplus, load),
    }
    rows = []
    for policy, solve in policies.items():
        for duration in [2, 4, 6, 8, 12]:
            for efficiency in [0.80, 0.85, 0.90, 0.95]:
                result = solve(duration, efficiency)
                captured = float(result["captured_GWh"])
                discharged = float(result["discharged_GWh"])
                rows.append({
                    "policy": policy, "common_capacity_MWh": capacity_gwh * 1000,
                    "duration_h": duration, "efficiency": efficiency,
                    "power_MW": capacity_gwh * 1000 / duration,
                    "captured_GWh": captured, "discharged_GWh": discharged,
                    "conversion_loss_GWh": max(0.0, captured - discharged),
                    "max_soc_percent": result["max_soc_percent"],
                    "end_soc_error_MWh": result["end_soc_error_MWh"],
                    "effective_input_cycles": captured / max(capacity_gwh, 1e-12),
                    "source_basis": "recomputed_30min_dispatch at stored common capacity",
                })
    result = pd.DataFrame(rows)
    out_csv(result, "v47_common_capacity_policy_scan.csv")
    return result


def seasonal_spatial_scale_scan() -> pd.DataFrame:
    """Compute season-specific local ceilings at 0.5--5 km aggregation scales."""
    seasonal = pd.read_parquet(SEASONAL / "seasonal_mesh_fields.parquet")
    coords = pd.read_parquet(V4DATA / "fig6_spatial_mesh.parquet")[["mesh", "lat", "lon"]]
    seasonal = seasonal.merge(coords, on="mesh", how="left", validate="many_to_one")
    rows = []
    for season in SEASONS:
        base = seasonal[seasonal.season.eq(season)].copy()
        for scale_km in [0.5, 1.0, 2.0, 5.0]:
            # Longitude is converted locally to km using the fixed project
            # approximation used by the multiscale concentration experiment.
            base["gx"] = np.floor((base.lon - base.lon.min()) * 91.0 / scale_km).astype(int)
            base["gy"] = np.floor((base.lat - base.lat.min()) * 111.0 / scale_km).astype(int)
            g = base.groupby(["gx", "gy"], as_index=False).agg(
                surplus_MWh=("surplus_MWh", "sum"), behavior_MWh=("behavior_supply_MWh", "sum"),
                absorbed_MWh=("behavior_absorbed_MWh", "sum"), residual_MWh=("residual_surplus_MWh", "sum"),
                n_mesh=("mesh", "nunique"))
            total = g.surplus_MWh.sum()
            local = np.minimum(g.surplus_MWh, g.behavior_MWh).sum()
            rows.append({"season": season, "aggregation_scale_km": scale_km,
                         "n_groups": len(g), "n_mesh": base.mesh.nunique(),
                         "surplus_GWh": total / 1000, "behavior_GWh": g.behavior_MWh.sum() / 1000,
                         "absorbed_GWh": g.absorbed_MWh.sum() / 1000,
                         "local_ceiling_GWh": local / 1000,
                         "anchored_match_percent": g.absorbed_MWh.sum() / max(total, 1e-12) * 100,
                         "local_ceiling_match_percent": local / max(total, 1e-12) * 100,
                         "spatial_gap_pp": (local - g.absorbed_MWh.sum()) / max(total, 1e-12) * 100,
                         "evidence_tier": "direct_corrected_low_voltage_ledger" if season == "Winter" else "hybrid_load_extrapolated",
                         "definition": "seasonal mesh spatial aggregation; local ceiling is a co-location proxy, not power flow"})
    result = pd.DataFrame(rows)
    out_csv(result, "v47_seasonal_spatial_scale_scan.csv")
    return result


def seasonal_behavior_timing_scan() -> pd.DataFrame:
    """Create date x window distributions from the full seasonal B/A ledger."""
    d = pd.read_parquet(SEASONAL / "seasonal_date_slot_fields.parquet").copy()
    rows = []
    for (season, date), x in d.groupby(["season", "date"], sort=True):
        for window, bounds in WINDOWS.items():
            if bounds is None:
                mask = ~x.slot.between(0, 16) & ~x.slot.between(18, 30) & ~x.slot.between(36, 44)
            else:
                mask = x.slot.between(*bounds)
            y = x.loc[mask]
            rows.append({"season": season, "date": str(date), "window": window,
                         "behavior_available_MWh": y.B_MWh.sum(),
                         "actual_absorbed_MWh": y.actual_absorbed_MWh.sum(),
                         "surplus_MWh": y.D_MWh.sum(),
                         "evidence_tier": x.evidence_tier.iloc[0],
                         "definition": "seasonal behavior-availability timing; not S10-minus-S1 system placebo"})
    result = pd.DataFrame(rows)
    out_csv(result, "v47_seasonal_behavior_timing_scan.csv")
    return result


def mobility_spatial_strata() -> pd.DataFrame:
    """Report spatial validation by prefecture and PV-supply exposure decile."""
    d = pd.read_csv(V4DATA / "fig4_mobility_spatial_validation.csv")
    d = d.dropna(subset=["weekend_minus_weekday_midday", "supply_S1_MWh"]).copy()
    d["exposure_decile"] = pd.qcut(d.supply_S1_MWh.rank(method="first"), 10, labels=False) + 1
    rows = []
    for level, groups in [("prefecture", d.groupby("pref")), ("supply_exposure_decile", d.groupby("exposure_decile"))]:
        for group_id, x in groups:
            if len(x) < 3:
                rho = np.nan
            else:
                rho = x["weekend_minus_weekday_midday"].corr(x["supply_S1_MWh"], method="spearman")
            rows.append({"stratum_level": level, "stratum": str(group_id),
                         "n_mesh": len(x), "spearman_rho": rho,
                         "mean_mobility_difference": x.weekend_minus_weekday_midday.mean(),
                         "mean_behavior_budget_MWh": x.supply_S1_MWh.mean(),
                         "positive_mobility_difference_percent": (x.weekend_minus_weekday_midday > 0).mean() * 100,
                         "definition": "spatial consistency diagnostic; not a causal estimate"})
    result = pd.DataFrame(rows)
    out_csv(result, "v47_mobility_spatial_strata.csv")
    return result


def municipality_scale_relief() -> pd.DataFrame:
    """Aggregate all stored seasonal behavior scales to municipality level."""
    mesh = pd.read_parquet(SEASONAL / "seasonal_mesh_fields.parquet")
    bridge = pd.read_parquet(BRIDGE)[["mesh", "munic", "B_PG"]].copy()
    bridge["mesh"] = bridge.mesh.astype(str)
    bridge["munic"] = bridge.munic.astype(str)
    bridge = bridge.sort_values("B_PG", ascending=False).drop_duplicates("mesh")[["mesh", "munic"]]
    mesh["mesh"] = mesh.mesh.astype(str)
    z = mesh.merge(bridge, on="mesh", how="inner", validate="many_to_one")
    z["pref"] = z.munic.str[:2].astype(int)
    rows = []
    for scale in SCALES:
        col = f"A_{str(scale).replace('.', '_')}"
        g = z.groupby(["munic", "pref"], as_index=False).agg(
            n_mesh=("mesh", "nunique"), surplus_MWh=("surplus_MWh", "sum"),
            behavior_base_MWh=("behavior_supply_MWh", "sum"), absorbed_MWh=(col, "sum"))
        g["scale"] = scale
        g["behavior_budget_MWh"] = scale * g.behavior_base_MWh
        g["residual_MWh"] = (g.surplus_MWh - g.absorbed_MWh).clip(lower=0)
        g["match_percent"] = g.absorbed_MWh / g.surplus_MWh.replace(0, np.nan) * 100
        g["behavior_utilization_percent"] = g.absorbed_MWh / g.behavior_budget_MWh.replace(0, np.nan) * 100
        g["relief_vs_zero_MWh"] = g.absorbed_MWh
        g["evidence_tier"] = "direct_corrected_low_voltage_ledger" if scale == 1.0 else "hybrid_load_extrapolated_or_upper_bound"
        rows.append(g)
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["munic", "scale"])
    result["marginal_relief_per_added_behavior_MWh"] = result.groupby("munic").relief_vs_zero_MWh.diff() / result.groupby("munic").behavior_budget_MWh.diff()
    out_csv(result, "v47_municipality_scale_relief.csv")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draws, null_summary = seasonal_null_stratified()
    storage = common_capacity_storage()
    spatial = seasonal_spatial_scale_scan()
    timing = seasonal_behavior_timing_scan()
    mobility = mobility_spatial_strata()
    municipality = municipality_scale_relief()
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "seed": SEED,
        "null_draws_per_season_model": N_NULL,
        "outputs": {
            "v47_seasonal_stratified_null_draws.parquet": len(draws),
            "v47_seasonal_stratified_null_summary.csv": len(null_summary),
            "v47_common_capacity_policy_scan.csv": len(storage),
            "v47_seasonal_spatial_scale_scan.csv": len(spatial),
            "v47_seasonal_behavior_timing_scan.csv": len(timing),
            "v47_mobility_spatial_strata.csv": len(mobility),
            "v47_municipality_scale_relief.csv": len(municipality),
        },
        "truth_boundaries": [
            "Seasonal timing scan is behavior availability/absorption, not a full-year S10-minus-S1 system placebo.",
            "Seasonal spatial scan is mesh aggregation/local co-location proxy, not feeder power flow.",
            "Mobility result is spatial consistency diagnostic, not causal validation.",
            "Municipality scales are stored A_scale columns; no unobserved intermediate scenario is invented.",
        ],
    }
    (OUT / "v47_target_panel_experiments_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
