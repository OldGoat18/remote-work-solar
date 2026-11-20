#!/usr/bin/env python3
"""V3 mesh-level storage siting and behavior-storage complementarity experiment.

The experiment is fixed to the 34,307 positive-aggregate-load mesh universe.
It derives exact mesh-day energy envelopes from the corrected 152-day ledger,
simulates cross-day SOC at every mesh, and freely allocates capacity across
meshes. The storage model is network-free and uses daily envelopes; it is not
an AC power-flow or feeder-hosting-capacity calculation.
"""
from __future__ import annotations

import glob
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "extra" / "v3"
DATA = OUT / "data"
CHECKPOINT = OUT / "checkpoints" / "mesh_day_storage"
CURVES = OUT / "checkpoints" / "mesh_storage_curves"
BASE = ROOT / "data" / "derived" / "ledger" / "baseline_p0"
SCOPE = ROOT / "data" / "derived" / "legacy_scope" / "study_mesh_universe.parquet"
ALPHA = ROOT / "data" / "derived" / "pv" / "alpha_rooftop_capacity_by_mesh.parquet"
KERNEL = ROOT / "data" / "derived" / "pv" / "obs_stock_yield_kernel.parquet"
WFH = ROOT / "data" / "derived" / "wfh_state" / "a_dwfh_mesh_mon_slot.parquet"
R3 = ROOT / "outputs" / "figures" / "extra" / "legacy_34307" / "r3_spatial_matching_mesh.parquet"
POOLED_EQUIVALENCE = (ROOT / "outputs" / "figures" / "extra" / "legacy_34307" /
                      "storage_equivalence.csv")

ETA_C = float(np.sqrt(0.90))
ETA_D = float(np.sqrt(0.90))
P_TO_E_H = 4.0
DT = 0.5
N_BUCKETS = 32
PROTOCOL = "v3_mesh_day_soc_34k_v2"
FACTORS = np.array([
    0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03,
    0.06, 0.10, 0.16, 0.25, 0.40, 0.65, 1.0, 1.5,
], dtype=float)


def qpath(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=6")
    con.execute("SET preserve_insertion_order=false")
    temp = OUT / "checkpoints" / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{qpath(temp)}'")
    return con


def _valid_parquet(con: duckdb.DuckDBPyConnection, path: Path,
                   required: set[str], minimum_rows: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        columns = {r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{qpath(path)}')").fetchall()}
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{qpath(path)}')").fetchone()[0]
        return required.issubset(columns) and rows >= minimum_rows
    except (duckdb.Error, OSError):
        return False


def _copy_atomic(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp.parquet")
    tmp.unlink(missing_ok=True)
    con.execute(f"COPY ({query}) TO '{qpath(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(path)


def build_s1_wfh(con: duckdb.DuckDBPyConnection) -> Path:
    path = OUT / "checkpoints" / "s1_wfh_scope.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    required = {"mesh", "pref", "mon", "slot", "dwfh"}
    if _valid_parquet(con, path, required, 1_000_000):
        print("[v3] reuse scoped S1 WFH checkpoint", flush=True)
        return path
    scope = qpath(SCOPE)
    wfh = qpath(WFH)
    _copy_atomic(con, f"""
        SELECT cast(w.mesh AS varchar) mesh, cast(s.pref AS varchar) pref,
               w.mon, w.slot, w.dwfh
        FROM read_parquet('{wfh}') w
        JOIN read_parquet('{scope}') s
          ON cast(s.mesh AS varchar)=cast(w.mesh AS varchar)
         AND s.load_status='positive_aggregate_load'
        WHERE w.scenario_id='S1' AND w.mon IN (10,11,12,1,2)
    """, path)
    print("[v3] built scoped S1 WFH checkpoint", flush=True)
    return path


def build_daily_primitives(con: duckdb.DuckDBPyConnection, wfh_path: Path) -> list[Path]:
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    parts = sorted(Path(p) for p in glob.glob(str(BASE / "pref=*" / "ym=*" / "*.parquet")))
    if len(parts) != 20:
        raise RuntimeError(f"expected 20 corrected baseline partitions, found {len(parts)}")
    required = {
        "mesh", "pref", "date", "bucket", "surplus_no_behavior_MWh",
        "behavior_absorbed_MWh", "surplus_after_behavior_MWh",
        "postsolar_deficit_no_behavior_MWh", "postsolar_deficit_after_behavior_MWh",
        "midday_surplus_no_behavior_MWh", "midday_surplus_after_behavior_MWh",
        "evening_deficit_no_behavior_MWh", "evening_deficit_after_behavior_MWh",
    }
    outputs: list[Path] = []
    for index, part in enumerate(parts, start=1):
        pref = part.parent.parent.name
        ym = part.parent.name
        output = CHECKPOINT / f"{pref}_{ym}.parquet"
        outputs.append(output)
        if _valid_parquet(con, output, required, 10_000):
            print(f"[v3] reuse daily primitive {index:02d}/20 {output.name}", flush=True)
            continue
        p = qpath(part)
        scope, alpha, kernel, wfh = map(qpath, [SCOPE, ALPHA, KERNEL, wfh_path])
        _copy_atomic(con, f"""
          WITH x AS (
            SELECT cast(b.mesh AS varchar) mesh, cast(b.pref AS varchar) pref,
                   cast(b.date AS varchar) date, b.slot,
                   greatest(b.L_owner_gross - b.E_wfh *
                     CASE WHEN b.L_owner_gross+b.L_nonowner_import>0
                          THEN least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0)
                          ELSE 0 END,0)
                   + greatest(b.L_nonowner_import - b.E_wfh *
                     CASE WHEN b.L_owner_gross+b.L_nonowner_import>0
                          THEN 1-least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0)
                          ELSE 1 END,0) L0,
                   b.G_pv + coalesce(a.increment_kw_a015,0)
                     * coalesce(k.sim_ykw_kwh_per_kw_30min,0)/1000.0 G,
                   coalesce(w.dwfh,0) B
            FROM read_parquet('{p}') b
            JOIN read_parquet('{scope}') s
              ON cast(s.mesh AS varchar)=cast(b.mesh AS varchar)
             AND cast(s.pref AS varchar)=cast(b.pref AS varchar)
             AND s.load_status='positive_aggregate_load'
            LEFT JOIN read_parquet('{alpha}') a
              ON cast(a.mesh AS varchar)=cast(b.mesh AS varchar)
             AND cast(a.pref AS varchar)=cast(b.pref AS varchar)
            LEFT JOIN read_parquet('{kernel}') k
              ON cast(k.pref AS varchar)=cast(b.pref AS varchar)
             AND k.m=cast(substr(cast(b.date AS varchar),5,2) AS integer)
             AND k.slot=b.slot
            LEFT JOIN read_parquet('{wfh}') w
              ON cast(w.mesh AS varchar)=cast(b.mesh AS varchar)
             AND cast(w.pref AS varchar)=cast(b.pref AS varchar)
             AND w.mon=cast(substr(cast(b.date AS varchar),5,2) AS integer)
             AND w.slot=b.slot
          ), z AS (
            SELECT *, greatest(G-L0,0) D0, greatest(G-L0-B,0) RB,
                   greatest(L0-G,0) DEF0, greatest(L0+B-G,0) DEFB
            FROM x
          )
          SELECT mesh,pref,date,cast(hash(mesh)%{N_BUCKETS} AS integer) bucket,
                 sum(D0) surplus_no_behavior_MWh,
                 sum(D0-RB) behavior_absorbed_MWh,
                 sum(RB) surplus_after_behavior_MWh,
                 sum(CASE WHEN slot>=30 THEN DEF0 ELSE 0 END) postsolar_deficit_no_behavior_MWh,
                 sum(CASE WHEN slot>=30 THEN DEFB ELSE 0 END) postsolar_deficit_after_behavior_MWh,
                 sum(CASE WHEN slot BETWEEN 20 AND 30 THEN D0 ELSE 0 END) midday_surplus_no_behavior_MWh,
                 sum(CASE WHEN slot BETWEEN 20 AND 30 THEN RB ELSE 0 END) midday_surplus_after_behavior_MWh,
                 sum(CASE WHEN slot BETWEEN 34 AND 44 THEN DEF0 ELSE 0 END) evening_deficit_no_behavior_MWh,
                 sum(CASE WHEN slot BETWEEN 34 AND 44 THEN DEFB ELSE 0 END) evening_deficit_after_behavior_MWh,
                 count(CASE WHEN D0>1e-12 THEN 1 END) charge_slots_no_behavior,
                 count(CASE WHEN RB>1e-12 THEN 1 END) charge_slots_after_behavior,
                 count(CASE WHEN slot>=30 AND DEF0>1e-12 THEN 1 END) discharge_slots_no_behavior,
                 count(CASE WHEN slot>=30 AND DEFB>1e-12 THEN 1 END) discharge_slots_after_behavior,
                 count(CASE WHEN slot BETWEEN 20 AND 30 AND D0>1e-12 THEN 1 END) midday_charge_slots_no_behavior,
                 count(CASE WHEN slot BETWEEN 20 AND 30 AND RB>1e-12 THEN 1 END) midday_charge_slots_after_behavior,
                 count(CASE WHEN slot BETWEEN 34 AND 44 AND DEF0>1e-12 THEN 1 END) evening_discharge_slots_no_behavior,
                 count(CASE WHEN slot BETWEEN 34 AND 44 AND DEFB>1e-12 THEN 1 END) evening_discharge_slots_after_behavior
          FROM z GROUP BY mesh,pref,date
        """, output)
        print(f"[v3] built daily primitive {index:02d}/20 {output.name}", flush=True)
    return outputs


def _date_axis(con: duckdb.DuckDBPyConnection, pattern: str) -> list[str]:
    return [r[0] for r in con.execute(
        f"SELECT DISTINCT date FROM read_parquet('{pattern}') ORDER BY date").fetchall()]


def _matrix(frame: pd.DataFrame, meshes: pd.Index, dates: list[str], column: str) -> np.ndarray:
    matrix = frame.pivot(index="mesh", columns="date", values=column)
    matrix = matrix.reindex(index=meshes, columns=dates, fill_value=0.0).fillna(0.0)
    return matrix.to_numpy(dtype=np.float64)


def simulate_daily(surplus: np.ndarray, deficit: np.ndarray,
                   charge_slots: np.ndarray, discharge_slots: np.ndarray,
                   capacities: np.ndarray, cyclic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate local daily-envelope SOC for every mesh and capacity level."""
    energy = capacities
    power_step = energy / P_TO_E_H * DT
    soc = np.zeros_like(energy)
    max_passes = 20 if cyclic else 1
    final_capture = final_discharge = None
    gap = np.zeros_like(energy)
    for pass_id in range(max_passes):
        start = soc.copy()
        captured = np.zeros_like(energy)
        discharged = np.zeros_like(energy)
        for day in range(surplus.shape[1]):
            q = np.minimum(surplus[:, day, None], power_step * charge_slots[:, day, None])
            q = np.minimum(q, np.maximum((energy - soc) / ETA_C, 0.0))
            soc += q * ETA_C
            captured += q
            d = np.minimum(deficit[:, day, None], power_step * discharge_slots[:, day, None])
            d = np.minimum(d, np.maximum(soc * ETA_D, 0.0))
            soc -= d / ETA_D
            discharged += d
        gap = soc - start
        final_capture, final_discharge = captured, discharged
        if not cyclic or np.nanmax(np.abs(gap)) < 1e-8:
            break
    if cyclic:
        # Remove any terminal inventory increase that did not complete a cycle.
        final_capture = np.maximum(final_capture - np.maximum(gap, 0.0) / ETA_C, 0.0)
    return final_capture, final_discharge, gap


def build_curves(con: duckdb.DuckDBPyConnection, daily_files: list[Path]) -> list[Path]:
    CURVES.mkdir(parents=True, exist_ok=True)
    pattern = qpath(CHECKPOINT / "*.parquet")
    dates = _date_axis(con, pattern)
    if len(dates) != 152:
        raise RuntimeError(f"expected 152 dates, found {len(dates)}")
    outputs = []
    required = {"mesh", "pref", "scenario", "policy", "capacity_MWh", "captured_MWh", "protocol"}
    for bucket in range(N_BUCKETS):
        output = CURVES / f"bucket={bucket:02d}.parquet"
        outputs.append(output)
        if _valid_parquet(con, output, required, 1000):
            print(f"[v3] reuse storage curve bucket {bucket+1:02d}/{N_BUCKETS}", flush=True)
            continue
        frame = con.execute(f"""
            SELECT * EXCLUDE(bucket) FROM read_parquet('{pattern}')
            WHERE bucket={bucket} ORDER BY mesh,date
        """).df()
        meshes = pd.Index(sorted(frame.mesh.unique()), name="mesh")
        pref = frame.groupby("mesh").pref.first().reindex(meshes)
        rows = []
        for scenario, suffix in (("storage_only", "no_behavior"),
                                 ("behavior_plus_storage", "after_behavior")):
            surplus_all = _matrix(frame, meshes, dates, f"surplus_{suffix}_MWh")
            deficit_all = _matrix(frame, meshes, dates, f"postsolar_deficit_{suffix}_MWh")
            charge_all = _matrix(frame, meshes, dates, f"charge_slots_{suffix}")
            discharge_all = _matrix(frame, meshes, dates, f"discharge_slots_{suffix}")
            surplus_naive = _matrix(frame, meshes, dates, f"midday_surplus_{suffix}_MWh")
            deficit_naive = _matrix(frame, meshes, dates, f"evening_deficit_{suffix}_MWh")
            charge_naive = _matrix(frame, meshes, dates, f"midday_charge_slots_{suffix}")
            discharge_naive = _matrix(frame, meshes, dates, f"evening_discharge_slots_{suffix}")
            reference = np.maximum(surplus_all.max(axis=1), 0.0)
            capacities = reference[:, None] * FACTORS[None, :]
            perfect_capture, perfect_discharge, perfect_gap = simulate_daily(
                surplus_all, deficit_all, charge_all, discharge_all, capacities, cyclic=True)
            naive_capture, naive_discharge, naive_gap = simulate_daily(
                surplus_naive, deficit_naive, charge_naive, discharge_naive, capacities, cyclic=False)
            for policy, capture, discharge, gap in (
                ("mesh_daily_envelope_cyclic", perfect_capture, perfect_discharge, perfect_gap),
                ("mesh_naive_midday_evening", naive_capture, naive_discharge, naive_gap),
            ):
                block = pd.DataFrame({
                    "mesh": np.repeat(meshes.to_numpy(), len(FACTORS)),
                    "pref": np.repeat(pref.to_numpy(), len(FACTORS)),
                    "scenario": scenario,
                    "policy": policy,
                    "level": np.tile(np.arange(len(FACTORS)), len(meshes)),
                    "capacity_factor": np.tile(FACTORS, len(meshes)),
                    "capacity_MWh": capacities.ravel(),
                    "captured_MWh": capture.ravel(),
                    "discharged_MWh": discharge.ravel(),
                    "cyclic_gap_MWh": gap.ravel(),
                    "model_grain": "mesh-day chronological SOC envelope",
                    "protocol": PROTOCOL,
                })
                rows.append(block)
        result = pd.concat(rows, ignore_index=True)
        result.to_parquet(output, index=False)
        print(f"[v3] built storage curve bucket {bucket+1:02d}/{N_BUCKETS}", flush=True)
    return outputs


def _segments(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mesh, pref), group in curves.groupby(["mesh", "pref"], sort=False):
        group = group.sort_values("capacity_MWh")
        x = group.capacity_MWh.to_numpy(float)
        y = np.maximum.accumulate(group.captured_MWh.to_numpy(float))
        keep = np.r_[True, np.diff(x) > 1e-12]
        x, y = x[keep], y[keep]
        if len(x) < 2:
            continue
        dx, dy = np.diff(x), np.diff(y)
        slope = np.divide(dy, dx, out=np.zeros_like(dy), where=dx > 0)
        # Conservative concavity correction preserves segment precedence.
        slope = np.minimum.accumulate(slope)
        dy = slope * dx
        y0 = np.r_[0.0, np.cumsum(dy)[:-1]]
        for i in range(len(dx)):
            if slope[i] <= 0 or dy[i] <= 0:
                continue
            rows.append((mesh, pref, x[i], x[i+1], y0[i], y0[i]+dy[i],
                         dx[i], dy[i], slope[i], i))
    return pd.DataFrame(rows, columns=[
        "mesh", "pref", "capacity_start_MWh", "capacity_end_MWh",
        "capture_start_MWh", "capture_end_MWh", "delta_capacity_MWh",
        "delta_capture_MWh", "marginal_capture_per_MWh", "segment",
    ])


def _allocate(segments: pd.DataFrame, *, target_capture_MWh: float | None = None,
              capacity_budget_MWh: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (target_capture_MWh is None) == (capacity_budget_MWh is None):
        raise ValueError("provide exactly one allocation stopping rule")
    ordered = segments.sort_values(
        ["marginal_capture_per_MWh", "mesh", "segment"], ascending=[False, True, True]).copy()
    allocations = []
    used_capacity = used_capture = 0.0
    response = [(0.0, 0.0)]
    for row in ordered.itertuples(index=False):
        fraction = 1.0
        if target_capture_MWh is not None and used_capture + row.delta_capture_MWh > target_capture_MWh:
            fraction = (target_capture_MWh - used_capture) / row.delta_capture_MWh
        if capacity_budget_MWh is not None and used_capacity + row.delta_capacity_MWh > capacity_budget_MWh:
            fraction = (capacity_budget_MWh - used_capacity) / row.delta_capacity_MWh
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if fraction <= 0:
            break
        dc = row.delta_capacity_MWh * fraction
        dy = row.delta_capture_MWh * fraction
        allocations.append((row.mesh, row.pref, dc, dy))
        used_capacity += dc
        used_capture += dy
        response.append((used_capacity, used_capture))
        if target_capture_MWh is not None and used_capture >= target_capture_MWh - 1e-8:
            break
        if capacity_budget_MWh is not None and used_capacity >= capacity_budget_MWh - 1e-8:
            break
    allocation = pd.DataFrame(allocations, columns=[
        "mesh", "pref", "allocated_capacity_MWh", "captured_surplus_MWh"])
    if not allocation.empty:
        allocation = allocation.groupby(["mesh", "pref"], as_index=False).sum()
    response_frame = pd.DataFrame(response, columns=["storage_capacity_MWh", "captured_surplus_MWh"])
    return allocation, response_frame


def optimize_siting(con: duckdb.DuckDBPyConnection) -> None:
    curve_pattern = qpath(CURVES / "*.parquet")
    daily_pattern = qpath(CHECKPOINT / "*.parquet")
    benchmark = con.execute(f"""
        SELECT sum(surplus_no_behavior_MWh) D,
               sum(behavior_absorbed_MWh) A,
               sum(surplus_after_behavior_MWh) R,
               count(distinct mesh) meshes, count(distinct date) dates
        FROM read_parquet('{daily_pattern}')
    """).fetchone()
    total_surplus, behavior_target, residual, meshes, dates = map(float, benchmark)
    if int(meshes) != 34307 or int(dates) != 152:
        raise RuntimeError(f"scope mismatch: meshes={meshes}, dates={dates}")
    r3 = pd.read_parquet(R3)
    r3_target = float(r3.realized_S1_MWh.sum())
    if not np.isclose(behavior_target, r3_target, atol=1e-5, rtol=0):
        raise RuntimeError(f"behavior target mismatch: daily={behavior_target}, R3={r3_target}")

    equivalence_rows = []
    response_rows = []
    allocation_frames: dict[tuple[str, str], pd.DataFrame] = {}
    segment_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for scenario in ("storage_only", "behavior_plus_storage"):
        for policy in ("mesh_daily_envelope_cyclic", "mesh_naive_midday_evening"):
            curves = con.execute(f"""
                SELECT mesh,pref,capacity_MWh,captured_MWh
                FROM read_parquet('{curve_pattern}')
                WHERE scenario='{scenario}' AND policy='{policy}'
            """).df()
            segments = _segments(curves)
            segment_cache[(scenario, policy)] = segments
            allocation, response = _allocate(segments, target_capture_MWh=behavior_target)
            if response.captured_surplus_MWh.iloc[-1] < behavior_target - 1e-5:
                raise RuntimeError(f"target not reached for {scenario}/{policy}")
            capacity = float(response.storage_capacity_MWh.iloc[-1])
            allocation["scenario"] = scenario
            allocation["policy"] = policy
            allocation_frames[(scenario, policy)] = allocation
            equivalence_rows.append({
                "scenario": scenario,
                "policy": policy,
                "target_behavior_absorption_GWh": behavior_target / 1000.0,
                "storage_energy_MWh": capacity,
                "storage_power_MW": capacity / P_TO_E_H,
                "captured_surplus_GWh": response.captured_surplus_MWh.iloc[-1] / 1000.0,
                "round_trip_efficiency": ETA_C * ETA_D,
                "cyclic_soc": policy == "mesh_daily_envelope_cyclic",
                "cross_day_dispatch": True,
                "free_mesh_siting": True,
                "network_constraints": False,
                "model_grain": "mesh-day chronological SOC envelope",
            })
            sampled = response.iloc[np.unique(np.linspace(
                0, len(response)-1, min(240, len(response)), dtype=int))].copy()
            sampled["scenario"] = scenario
            sampled["policy"] = policy
            response_rows.append(sampled)

    equivalence = pd.DataFrame(equivalence_rows)
    pooled = pd.read_csv(POOLED_EQUIVALENCE)
    pooled_rows = []
    for row in pooled.itertuples(index=False):
        pooled_rows.append({
            "scenario": "pooled_no_network_relaxation",
            "policy": row.policy,
            "target_behavior_absorption_GWh": row.target_behavior_relief_GWh,
            "storage_energy_MWh": row.storage_energy_GWh * 1000.0,
            "storage_power_MW": row.storage_power_GW * 1000.0,
            "captured_surplus_GWh": row.captured_surplus_GWh,
            "round_trip_efficiency": row.round_trip_efficiency,
            "cyclic_soc": bool(row.cyclic_soc),
            "cross_day_dispatch": bool(row.cross_day_dispatch),
            "free_mesh_siting": bool(row.free_mesh_siting),
            "network_constraints": bool(row.power_flow_constraints),
            "model_grain": row.model_grain,
        })
    equivalence = pd.concat([equivalence, pd.DataFrame(pooled_rows)], ignore_index=True)
    equivalence.to_csv(DATA / "v3_mesh_storage_equivalence.csv", index=False)
    pd.concat(response_rows, ignore_index=True).to_csv(
        DATA / "v3_mesh_storage_capacity_response.csv", index=False)
    all_allocations = pd.concat(allocation_frames.values(), ignore_index=True)
    all_allocations.to_parquet(DATA / "v3_mesh_storage_target_allocations.parquet", index=False)

    perfect_key = ("storage_only", "mesh_daily_envelope_cyclic")
    fixed_capacity = float(equivalence.loc[
        (equivalence.scenario == perfect_key[0]) & (equivalence.policy == perfect_key[1]),
        "storage_energy_MWh"].iloc[0])
    joint_alloc, joint_response = _allocate(
        segment_cache[("behavior_plus_storage", "mesh_daily_envelope_cyclic")],
        capacity_budget_MWh=fixed_capacity)
    storage_after_behavior = float(joint_response.captured_surplus_MWh.iloc[-1])
    storage_only_capture = behavior_target
    joint_total = behavior_target + storage_after_behavior
    overlap = behavior_target + storage_only_capture - joint_total
    if abs(overlap) < 1e-6:
        overlap = 0.0
    arms = pd.DataFrame([
        {"arm": "baseline", "behavior_capture_GWh": 0.0, "storage_capture_GWh": 0.0,
         "total_capture_GWh": 0.0, "residual_surplus_GWh": total_surplus / 1000.0},
        {"arm": "behavior_only", "behavior_capture_GWh": behavior_target / 1000.0,
         "storage_capture_GWh": 0.0, "total_capture_GWh": behavior_target / 1000.0,
         "residual_surplus_GWh": residual / 1000.0},
        {"arm": "storage_only", "behavior_capture_GWh": 0.0,
         "storage_capture_GWh": storage_only_capture / 1000.0,
         "total_capture_GWh": storage_only_capture / 1000.0,
         "residual_surplus_GWh": (total_surplus-storage_only_capture) / 1000.0},
        {"arm": "behavior_plus_storage", "behavior_capture_GWh": behavior_target / 1000.0,
         "storage_capture_GWh": storage_after_behavior / 1000.0,
         "total_capture_GWh": joint_total / 1000.0,
         "residual_surplus_GWh": (total_surplus-joint_total) / 1000.0},
    ])
    arms["storage_capacity_MWh"] = [0.0, 0.0, fixed_capacity, fixed_capacity]
    arms.to_csv(DATA / "v3_behavior_storage_joint_arms.csv", index=False)

    joint_summary = pd.DataFrame([{
        "mesh_count": int(meshes), "date_count": int(dates),
        "pv_surplus_GWh": total_surplus / 1000.0,
        "behavior_absorption_GWh": behavior_target / 1000.0,
        "behavior_match_percent": behavior_target / total_surplus * 100.0,
        "fixed_storage_capacity_MWh": fixed_capacity,
        "storage_only_capture_GWh": storage_only_capture / 1000.0,
        "storage_after_behavior_capture_GWh": storage_after_behavior / 1000.0,
        "joint_total_capture_GWh": joint_total / 1000.0,
        "behavior_storage_overlap_GWh": overlap / 1000.0,
        "storage_increment_after_behavior_positive": storage_after_behavior > 0,
        "behavior_increment_after_storage_positive": joint_total > storage_only_capture,
        "strict_superadditivity_supported": joint_total > behavior_target + storage_only_capture + 1e-6,
        "interpretation": "both services retain positive marginal capture; complementarity is sequential residual service, not strict superadditivity",
    }])
    joint_summary.to_csv(DATA / "v3_behavior_storage_joint_summary.csv", index=False)

    joint_alloc = joint_alloc.rename(columns={
        "allocated_capacity_MWh": "joint_storage_capacity_MWh",
        "captured_surplus_MWh": "joint_storage_captured_MWh",
    })
    spatial = r3.merge(joint_alloc, on=["mesh", "pref"], how="left")
    spatial[["joint_storage_capacity_MWh", "joint_storage_captured_MWh"]] = spatial[[
        "joint_storage_capacity_MWh", "joint_storage_captured_MWh"]].fillna(0.0)
    spatial["behavior_absorbed_MWh"] = spatial.realized_S1_MWh.clip(lower=0)
    spatial["residual_after_behavior_MWh"] = spatial.residual_S1_MWh.clip(lower=0)
    spatial["residual_after_joint_MWh"] = np.maximum(
        spatial.residual_after_behavior_MWh - spatial.joint_storage_captured_MWh, 0.0)
    spatial.to_parquet(DATA / "v3_behavior_storage_spatial_mesh.parquet", index=False)

    active = spatial.loc[spatial.demand_surplus_MWh > 0].copy()
    active["exposure_decile"] = pd.qcut(
        active.demand_surplus_MWh.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    deciles = active.groupby("exposure_decile", as_index=False).agg(
        mesh_count=("mesh", "size"),
        pv_surplus_MWh=("demand_surplus_MWh", "sum"),
        behavior_absorbed_MWh=("behavior_absorbed_MWh", "sum"),
        storage_capacity_MWh=("joint_storage_capacity_MWh", "sum"),
        storage_captured_MWh=("joint_storage_captured_MWh", "sum"),
        residual_after_joint_MWh=("residual_after_joint_MWh", "sum"),
    )
    deciles["behavior_capture_percent"] = (
        deciles.behavior_absorbed_MWh / deciles.pv_surplus_MWh * 100.0)
    deciles["joint_capture_percent"] = (
        (deciles.behavior_absorbed_MWh + deciles.storage_captured_MWh)
        / deciles.pv_surplus_MWh * 100.0)
    deciles.to_csv(DATA / "v3_behavior_storage_exposure_deciles.csv", index=False)

    pd.DataFrame([{
        "scope": "positive_aggregate_load", "mesh_count": int(meshes),
        "date_count": int(dates), "time_step": "mesh-day envelope from 30-min ledger",
        "power_to_energy_hours": P_TO_E_H, "round_trip_efficiency": ETA_C*ETA_D,
        "pooled_upper_policy": "30-min pooled perfect-foresight LP; cyclic SOC; no network constraints",
        "mesh_cyclic_policy": "daily-envelope residual surplus; post-solar deficits; cyclic fixed-point SOC",
        "lower_policy": "10:00-15:00 charge; 17:00-22:00 discharge; empty initial SOC",
        "siting": "free capacity allocation across meshes by marginal lifetime capture",
        "network_constraints": False,
        "limitation": "mesh daily envelope preserves exact daily energy totals but not within-window 30-min order; pooled LP remains the theoretical upper bound",
    }]).to_csv(DATA / "v3_mesh_storage_model_audit.csv", index=False)
    print(equivalence.to_string(index=False), flush=True)
    print(joint_summary.to_string(index=False), flush=True)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    con = connection()
    try:
        wfh = build_s1_wfh(con)
        daily = build_daily_primitives(con, wfh)
        build_curves(con, daily)
        optimize_siting(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
