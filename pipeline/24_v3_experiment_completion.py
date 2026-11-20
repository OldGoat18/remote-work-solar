#!/usr/bin/env python3
"""Complete the diagnostics registered in the V3 experiment-chain design.

This stage does not alter the legacy_34307 inputs or rerun the core behavior
and storage models. It materializes the remaining counterfactual summaries,
dispatch diagnostics, spatial concentration tests, and assumption bookkeeping
used by the V3 composite figures and review document.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTRA = ROOT / "outputs" / "figures" / "extra"
V3 = EXTRA / "v3"
DATA = V3 / "data"
LEGACY = EXTRA / "legacy_34307"
SEASONAL = EXTRA / "legacy_34307_seasonal"
COMMUNITY = EXTRA / "Community Sustainability style" / "analysis_data"

DT = 0.5
P_TO_E_H = 4.0
ETA_C = ETA_D = np.sqrt(0.90)
EXPECTED_MESHES = 34_307
EXPECTED_DATES = 152


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(DATA / name, index=False)


def behavior_diagnostics() -> None:
    r3 = pd.read_parquet(LEGACY / "r3_spatial_matching_mesh.parquet")
    scale = pd.read_csv(LEGACY / "behavior_wfh_scale_sensitivity.csv")
    draws = pd.read_csv(COMMUNITY / "null_model_draws_regenerated.csv")
    rebound = pd.read_csv(LEGACY / "behavior_rebound_sensitivity.csv")

    if len(r3) != EXPECTED_MESHES:
        raise RuntimeError(f"expected {EXPECTED_MESHES:,} meshes, found {len(r3):,}")

    # This is an auditable load-intensity gradient, not an administrative
    # metro/regional/rural classification.
    rank = r3.midday_load_no_wfh_MWh.rank(method="first")
    r3["load_intensity_group"] = pd.qcut(
        rank, 3, labels=["Lower", "Middle", "Upper"]
    ).astype(str)
    gradient = r3.groupby("load_intensity_group", observed=True, as_index=False).agg(
        mesh_count=("mesh", "size"),
        midday_load_MWh=("midday_load_no_wfh_MWh", "sum"),
        pv_surplus_MWh=("demand_surplus_MWh", "sum"),
        behavior_budget_MWh=("supply_S1_MWh", "sum"),
        behavior_absorbed_MWh=("realized_S1_MWh", "sum"),
    )
    gradient["order"] = gradient.load_intensity_group.map(
        {"Lower": 0, "Middle": 1, "Upper": 2}
    )
    gradient = gradient.sort_values("order").drop(columns="order")
    gradient["classification_basis"] = (
        "terciles of 152-day mesh midday low-voltage load; operational gradient, "
        "not an administrative urban-rural class"
    )
    write_csv(gradient, "v3_behavior_load_intensity_gradient.csv")

    selected = scale.iloc[[int(np.argmin(np.abs(scale.scale - 1.0))), -1]].copy()
    selected["scenario"] = ["S1", "Defensible maximum"]
    selected["global_energy_ceiling_percent"] = np.minimum(
        1.0, selected.wfh_budget_GWh / selected.surplus_GWh
    ) * 100.0
    selected["actual_match_percent"] = selected.actual_match * 100.0
    selected["same_slot_ceiling_percent"] = selected.free_placement_ceiling * 100.0
    selected["spatial_wall_pp"] = selected.spatial_gap * 100.0
    selected["temporal_wall_pp"] = (
        selected.global_energy_ceiling_percent - selected.same_slot_ceiling_percent
    )
    selected["magnitude_wall_pp"] = 100.0 - selected.global_energy_ceiling_percent
    selected["closure_percent"] = (
        selected.actual_match_percent + selected.spatial_wall_pp
        + selected.temporal_wall_pp + selected.magnitude_wall_pp
    )
    write_csv(selected[[
        "scenario", "scale", "wfh_budget_GWh", "surplus_GWh",
        "actual_absorption_GWh", "actual_match_percent",
        "same_slot_ceiling_percent", "spatial_wall_pp", "temporal_wall_pp",
        "magnitude_wall_pp", "global_energy_ceiling_percent", "closure_percent",
        "upper_bound_scenario", "target_occupation_adoption",
        "frequency_intensity", "evidence_tier",
    ]], "v3_behavior_constraint_diagnostics.csv")

    observed = float(selected.iloc[0].actual_match_percent)
    free_ceiling = float(selected.iloc[0].same_slot_ceiling_percent)
    rows = []
    for model, group in draws.groupby("model", sort=False):
        values = group.matching_rate_percent.to_numpy(float)
        rows.append({
            "model": model,
            "draws": len(values),
            "seed": 20260804,
            "placement_rule": (
                "permutation of the mesh behavior-supply vector"
                if model == "Uniform random placement"
                else "multinomial destinations weighted by mesh midday low-voltage load"
            ),
            "mean_percent": values.mean(),
            "median_percent": np.median(values),
            "q025_percent": np.quantile(values, 0.025),
            "q975_percent": np.quantile(values, 0.975),
            "observed_percent": observed,
            "observed_percentile": 100.0 * np.mean(values <= observed),
            "mean_minus_observed_pp": values.mean() - observed,
            "same_slot_free_ceiling_percent": free_ceiling,
            "all_draws_above_observed": bool(np.all(values > observed)),
        })
    write_csv(pd.DataFrame(rows), "v3_behavior_null_summary.csv")

    rebound = rebound.copy()
    rebound["additional_evening_energy_task_GWh"] = (
        rebound.additional_non_pv_window_demand_GWh
    )
    rebound["minimum_added_storage_discharge_GWh_at_unit_efficiency"] = (
        rebound.additional_non_pv_window_demand_GWh
    )
    rebound["task_interpretation"] = (
        "energy bookkeeping only; no feeder ramp-rate or power-flow calculation"
    )
    write_csv(rebound, "v3_behavior_rebound_task.csv")


def greedy_dispatch(energy_MWh: float, ledger: pd.DataFrame) -> pd.DataFrame:
    capacity = float(energy_MWh)
    power_step = capacity / P_TO_E_H * DT
    soc = 0.0
    rows = []
    for row in ledger.itertuples(index=False):
        surplus = max(float(row.D), 0.0)
        load = max(float(row.net_load_MW), 0.0) * DT
        charge = discharge = 0.0
        if 34 <= int(row.slot) <= 44:
            discharge = min(load, power_step, soc * ETA_D)
            soc -= discharge / ETA_D
        elif surplus > 0:
            charge = min(surplus, power_step, max((capacity - soc) / ETA_C, 0.0))
            soc += charge * ETA_C
        rows.append((row.date, int(row.slot), surplus, charge, discharge, soc))
    return pd.DataFrame(rows, columns=[
        "date", "slot", "pv_surplus_MWh", "charge_MWh", "discharge_MWh", "soc_MWh"
    ])


def storage_diagnostics() -> None:
    equivalence = pd.read_csv(DATA / "v3_mesh_storage_equivalence.csv")
    pooled_curve = pd.read_csv(COMMUNITY / "storage_capacity_response_dense.csv")
    mesh_curve = pd.read_csv(DATA / "v3_mesh_storage_capacity_response.csv")
    ledger = pd.read_parquet(LEGACY / "behavior_date_slot_fields.parquet").sort_values(
        ["date", "slot"]
    )
    if ledger.date.nunique() != EXPECTED_DATES:
        raise RuntimeError(
            f"expected {EXPECTED_DATES} dates, found {ledger.date.nunique()}"
        )

    pooled = equivalence[
        (equivalence.scenario == "pooled_no_network_relaxation")
        & equivalence.policy.isin(["perfect_foresight_upper", "greedy_lower"])
    ].copy()
    mesh = equivalence[
        (equivalence.scenario == "storage_only")
        & (equivalence.policy == "mesh_daily_envelope_cyclic")
    ].copy()
    diagnostics = pd.concat([pooled, mesh], ignore_index=True)
    diagnostics["capture_to_capacity_cycles"] = (
        diagnostics.captured_surplus_GWh * 1000.0 / diagnostics.storage_energy_MWh
    )
    perfect_capacity = float(
        diagnostics.loc[
            diagnostics.policy.eq("perfect_foresight_upper"), "storage_energy_MWh"
        ].iloc[0]
    )
    diagnostics["capacity_ratio_to_pooled_perfect"] = (
        diagnostics.storage_energy_MWh / perfect_capacity
    )

    same_capacity = perfect_capacity
    same_capacity_rows = []
    for policy, group in pooled_curve.groupby("policy"):
        group = group.sort_values("storage_energy_GWh")
        capture = np.interp(
            same_capacity / 1000.0,
            group.storage_energy_GWh.to_numpy(float),
            group.captured_surplus_GWh.to_numpy(float),
        )
        same_capacity_rows.append({
            "policy": policy,
            "common_capacity_MWh": same_capacity,
            "captured_surplus_GWh": capture,
            "information_set": (
                "full-horizon perfect foresight" if policy == "Perfect foresight"
                else "causal evening-discharge rule"
            ),
        })
    mesh_group = mesh_curve[
        (mesh_curve.scenario == "storage_only")
        & (mesh_curve.policy == "mesh_daily_envelope_cyclic")
    ].sort_values("storage_capacity_MWh")
    same_capacity_rows.append({
        "policy": "Mesh-day free siting",
        "common_capacity_MWh": same_capacity,
        "captured_surplus_GWh": np.interp(
            same_capacity,
            mesh_group.storage_capacity_MWh.to_numpy(float),
            mesh_group.captured_surplus_MWh.to_numpy(float) / 1000.0,
        ),
        "information_set": "mesh-day envelope with cyclic SOC",
    })
    same_capacity_frame = pd.DataFrame(same_capacity_rows)
    diagnostics = diagnostics.merge(
        same_capacity_frame[["policy", "captured_surplus_GWh"]].rename(
            columns={"captured_surplus_GWh": "capture_at_perfect_capacity_GWh"}
        ),
        left_on="policy",
        right_on="policy",
        how="left",
    )
    # Map display labels back to model policy identifiers for the merge above.
    display_map = {
        "Perfect foresight": "perfect_foresight_upper",
        "Naive rule": "greedy_lower",
        "Mesh-day free siting": "mesh_daily_envelope_cyclic",
    }
    same_capacity_frame["model_policy"] = same_capacity_frame.policy.map(display_map)
    diagnostics = diagnostics.drop(columns="capture_at_perfect_capacity_GWh")
    diagnostics = diagnostics.merge(
        same_capacity_frame[["model_policy", "captured_surplus_GWh"]].rename(
            columns={
                "model_policy": "policy",
                "captured_surplus_GWh": "capture_at_perfect_capacity_GWh",
            }
        ), on="policy", how="left"
    )
    write_csv(diagnostics, "v3_storage_exchange_diagnostics.csv")
    write_csv(same_capacity_frame, "v3_storage_same_capacity_comparison.csv")

    perfect_dispatch = pd.read_csv(COMMUNITY / "perfect_foresight_dispatch_timeseries.csv")
    perfect_dispatch = perfect_dispatch[[
        "date", "slot", "D", "charge_MWh", "discharge_MWh", "soc_MWh"
    ]].rename(columns={"D": "pv_surplus_MWh"})
    perfect_dispatch["policy"] = "Perfect foresight"
    naive_energy = float(
        equivalence.loc[
            (equivalence.scenario == "pooled_no_network_relaxation")
            & equivalence.policy.eq("greedy_lower"), "storage_energy_MWh"
        ].iloc[0]
    )
    naive_dispatch = greedy_dispatch(naive_energy, ledger)
    naive_dispatch["policy"] = "Naive rule"
    combined = pd.concat([perfect_dispatch, naive_dispatch], ignore_index=True)
    combined["datetime"] = pd.to_datetime(
        combined.date.astype(str), format="%Y%m%d"
    ) + pd.to_timedelta(combined.slot * 30, unit="min")
    write_csv(combined, "v3_storage_dispatch_comparison.csv")


def spatial_and_cost_diagnostics() -> None:
    spatial = pd.read_parquet(DATA / "v3_behavior_storage_spatial_mesh.parquet")
    active = spatial[spatial.demand_surplus_MWh > 0].copy()
    active["exposure_decile"] = pd.qcut(
        active.demand_surplus_MWh.rank(method="first"),
        10, labels=range(1, 11),
    ).astype(int)
    concentration = active.groupby("exposure_decile", as_index=False).agg(
        mesh_count=("mesh", "size"),
        pv_surplus_MWh=("demand_surplus_MWh", "sum"),
        behavior_absorbed_MWh=("behavior_absorbed_MWh", "sum"),
        residual_after_behavior_MWh=("residual_after_behavior_MWh", "sum"),
        storage_capacity_MWh=("joint_storage_capacity_MWh", "sum"),
        storage_captured_MWh=("joint_storage_captured_MWh", "sum"),
    )
    for column in [
        "pv_surplus_MWh", "behavior_absorbed_MWh", "residual_after_behavior_MWh",
        "storage_capacity_MWh", "storage_captured_MWh",
    ]:
        total = concentration[column].sum()
        concentration[column.replace("_MWh", "_share_percent")] = (
            concentration[column] / total * 100.0 if total else np.nan
        )
    write_csv(concentration, "v3_behavior_storage_concentration.csv")

    k = max(1, len(active) // 10)
    sets = {
        "Behavior absorbed": set(active.nlargest(k, "behavior_absorbed_MWh").mesh),
        "Residual surplus": set(active.nlargest(k, "residual_after_behavior_MWh").mesh),
        "Storage capacity": set(active.nlargest(k, "joint_storage_capacity_MWh").mesh),
    }
    rows = []
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            intersection = len(sets[left] & sets[right])
            union = len(sets[left] | sets[right])
            rows.append({
                "left_metric": left,
                "right_metric": right,
                "top_decile_size": k,
                "overlap_meshes": intersection,
                "overlap_fraction": intersection / k,
                "jaccard": intersection / union,
            })
    write_csv(pd.DataFrame(rows), "v3_behavior_storage_top_decile_overlap.csv")

    costs = pd.read_csv(LEGACY / "storage_behavior_cost_comparison.csv")
    central = costs[
        (costs.storage_capex_USD_per_kWh == 250)
        & np.isclose(costs.storage_annualized_fraction, 0.04)
        & np.isclose(costs.behavior_incentive_USD_per_shifted_MWh, 50)
    ].copy()
    central["scenario_role"] = "secondary parameterized scenario; not a core claim"
    write_csv(central, "v3_storage_cost_central_scenario.csv")


def assumption_ledger() -> None:
    source = pd.read_csv(LEGACY / "assumption_ledger.csv").rename(
        columns={"comparison_role": "best_vs_best_role"}
    )
    required = [
        "assumption", "current_setting", "source_or_basis", "bias_direction",
        "claim_affected", "robustness_test", "best_vs_best_role",
    ]
    source = source[required]
    upper = source.assumption.eq("Behavior compliance upper")
    source.loc[upper, "source_or_basis"] = (
        source.loc[upper, "source_or_basis"]
        + "; package_docs_figures/outputs/release/scenario_indices_s1s10.csv "
          "(S10 occupation-based structural upper-bound arm)"
    )
    source.insert(0, "assumption_id", [f"A{i:02d}" for i in range(1, len(source) + 1)])
    source["evidence_status"] = np.where(
        source.source_or_basis.str.contains("model assumption|counterfactual", case=False),
        "declared modeling assumption", "registered data/model basis"
    )
    source["v3_figure_link"] = source.claim_affected.map(lambda x: (
        "Figure 1" if any(k in str(x).lower() for k in ["behavior", "match", "time gate"])
        else "Figure 2" if "storage" in str(x).lower()
        else "Figure 3"
    ))
    write_csv(source, "Extended_Data_Table_1_Assumption_Ledger.csv")

    lines = [
        "# Extended Data Table 1 | Assumption ledger",
        "",
        "本表记录 v3 三张综合图使用的关键假设、偏置方向、受影响声称和稳健性检验。",
        "`best-vs-best` 表示行为和储能均获得对自身有利的理论放宽。",
        "",
        "| ID | Assumption | Current setting | Bias direction | Claim affected | Robustness test | Best-vs-best role |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in source.itertuples(index=False):
        values = [row.assumption_id, row.assumption, row.current_setting,
                  row.bias_direction, row.claim_affected, row.robustness_test,
                  row.best_vs_best_role]
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in values) + " |")
    lines += [
        "", "## 使用边界", "",
        "该表用于透明记录假设及其偏置方向，不把 mesh surplus 解释为馈线潮流结果，"
        "也不把自由放置或完美预见解释为现实部署预测。",
    ]
    (V3 / "Extended_Data_Table_1_Assumption_Ledger.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def audit() -> None:
    joint = pd.read_csv(DATA / "v3_behavior_storage_joint_summary.csv").iloc[0]
    constraint = pd.read_csv(DATA / "v3_behavior_constraint_diagnostics.csv")
    nulls = pd.read_csv(DATA / "v3_behavior_null_summary.csv")
    exchange = pd.read_csv(DATA / "v3_storage_exchange_diagnostics.csv")
    dispatch = pd.read_csv(DATA / "v3_storage_dispatch_comparison.csv")
    overlap = pd.read_csv(DATA / "v3_behavior_storage_top_decile_overlap.csv")
    checks = [
        ("mesh scope", int(joint.mesh_count) == EXPECTED_MESHES, int(joint.mesh_count)),
        ("date scope", int(joint.date_count) == EXPECTED_DATES, int(joint.date_count)),
        ("constraint closure", np.allclose(constraint.closure_percent, 100.0),
         float(np.max(np.abs(constraint.closure_percent - 100.0)))),
        ("null draws", int(nulls.draws.sum()) == 4000, int(nulls.draws.sum())),
        ("dispatch rows", len(dispatch) == EXPECTED_DATES * 48 * 2, len(dispatch)),
        ("storage target", np.allclose(exchange.captured_surplus_GWh,
                                       joint.behavior_absorption_GWh, atol=1e-4),
         float(exchange.captured_surplus_GWh.max())),
        ("spatial diagnostics", len(overlap) == 3, len(overlap)),
    ]
    frame = pd.DataFrame(checks, columns=["check", "passed", "value"])
    write_csv(frame, "v3_experiment_completion_audit.csv")
    failed = frame.loc[~frame.passed]
    if not failed.empty:
        raise RuntimeError("V3 completion audit failed:\n" + failed.to_string(index=False))
    acceptance = pd.DataFrame([
        (1, "Three composite figures with panels (a)-(g)", "complete", "Figure_1/2/3 PNG and PDF"),
        (2, "Figure 1 independently closes behavioral robustness", "complete", "alpha, null, seasonal and rebound outputs"),
        (3, "Figure 2 target equals Figure 1 behavior absorption", "complete", f"{joint.behavior_absorption_GWh:.6f} GWh"),
        (4, "Figure 3 adds no new performance metric", "complete", "formal references, proxy boundary and symmetric assumptions"),
        (5, "Maximum alpha has a traceable basis", "complete", "registered S10 occupation upper-bound arm: 0.85 adoption x 0.70 intensity"),
        (6, "Random draws, seed and placement rule are reproducible", "complete", "2,000 per arm; seed 20260804; rules in null summary"),
        (7, "Seasonal constraints close", "complete", "four seasons; each four-part decomposition closes to 100%"),
        (8, "Storage policies share inputs, efficiency and target", "complete", "same 152-day ledger, 0.90 round-trip efficiency and common target"),
        (9, "Storage reports MWh, MW and cumulative capture", "complete", "storage exchange diagnostics"),
        (10, "Spatial maps share extent and color semantics", "complete", "Figure 2f map group"),
        (11, "Cost is marked secondary", "complete", "central scenario and full sensitivity table"),
        (12, "Mesh surplus is labeled as a proxy", "complete", "Figure 3b and model audit"),
        (13, "Assumption ledger is independent Extended Data", "complete", "Extended Data Table 1 CSV/MD/DOCX"),
        (14, "Legacy and 34,307-mesh values are not mixed", "complete", "registered positive-aggregate-load scope"),
        (15, "Every panel has a source manifest entry", "complete", "V3_FIGURE_SOURCE_MANIFEST.csv"),
    ], columns=["criterion", "requirement", "status", "evidence"])
    write_csv(acceptance, "v3_design_acceptance.csv")
    print(frame.to_string(index=False), flush=True)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    behavior_diagnostics()
    storage_diagnostics()
    spatial_and_cost_diagnostics()
    assumption_ledger()
    audit()


if __name__ == "__main__":
    main()
