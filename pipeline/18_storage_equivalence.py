"""SOC storage equivalence experiment on the public-data reconstruction.

The model uses date-resolved system load and mesh-grain PV surplus, enforces
cyclic SOC, permits cross-day dispatch, and compares perfect foresight with a
causal greedy policy. Free mesh siting is an explicit upper-bound assumption.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from wlpr.io.paths import derived_path, out_path

SYS = derived_path("netload_mesh_date", "netload_system_date_slot.parquet")
R3 = derived_path("social", "r3_spatial_matching_mesh.parquet")
SCOPE = os.environ.get("WLPR_SCOPE")
SCOPE_STATUS = os.environ.get("WLPR_SCOPE_STATUS", "positive")
_CUSTOM_OUT = os.environ.get("WLPR_EXTRA_OUTDIR")
OUTDIR = Path(_CUSTOM_OUT or out_path("scorecards"))
FIGDIR = Path(os.environ.get("WLPR_EXTRA_FIGDIR") or (_CUSTOM_OUT or out_path("figures", "extra")))
DT = 0.5
ETA_C = ETA_D = np.sqrt(0.90)
P_TO_E_H = 4.0


def _scoped_r3():
    scoped_output = OUTDIR / "r3_spatial_matching_mesh.parquet"
    if SCOPE and scoped_output.exists():
        return pd.read_parquet(scoped_output)
    frame = pd.read_parquet(R3)
    if not SCOPE:
        return frame
    scope = pd.read_parquet(SCOPE)
    if SCOPE_STATUS == "positive" and "load_status" in scope:
        scope = scope[scope.load_status == "positive_aggregate_load"]
    elif SCOPE_STATUS != "all":
        raise ValueError(f"unsupported WLPR_SCOPE_STATUS={SCOPE_STATUS!r}")
    keep = set(scope.mesh.astype(str))
    frame = frame[frame.mesh.astype(str).isin(keep)].copy()
    if len(frame) != len(keep):
        raise RuntimeError(f"scope/R3 mismatch: expected {len(keep):,}, found {len(frame):,}")
    return frame


def _mesh_centers(mesh):
    s = pd.Series(mesh).astype(str).str.zfill(9)
    y1 = s.str[0:2].astype(int); x1 = s.str[2:4].astype(int)
    y2 = s.str[4].astype(int); x2 = s.str[5].astype(int)
    y3 = s.str[6].astype(int); x3 = s.str[7].astype(int)
    q = s.str[8].astype(int)
    y4 = (q - 1) // 2; x4 = (q - 1) % 2
    lat = (y1 + y2 / 8 + y3 / 80 + y4 / 160 + 1 / 320) / 1.5
    lon = 100 + x1 + x2 / 8 + x3 / 80 + x4 / 160 + 1 / 320
    return lat.to_numpy(), lon.to_numpy()


def _series():
    if SCOPE:
        no = pd.read_parquet(OUTDIR / "behavior_date_slot_fields.parquet").sort_values(["date", "slot"])
        if no.empty:
            raise FileNotFoundError("scoped behavior date-slot fields are missing")
        surplus = no.D.to_numpy(float)
        load = np.maximum(no.net_load_MW.to_numpy(float), 0.0) * DT
        return no.date.to_numpy(), no.slot.to_numpy(int), surplus, load
    x = pd.read_parquet(SYS)
    no = x[(x.scenario_id == "no_wfh") & (x.pv_scenario == "future_pv_ai")].sort_values(["date", "slot"])
    if no.empty:
        raise FileNotFoundError("no-WFH future-PV date-resolved surface is missing")
    # The date-resolved net curve supplies discharge demand. The mesh-grain
    # surplus is stored as a date-slot system aggregate by pipeline/08.
    surplus = no.local_surplus_MWh.to_numpy(float)
    load = np.maximum(no.net_load_MW.to_numpy(float), 0.0) * DT
    return no.date.to_numpy(), no.slot.to_numpy(int), surplus, load


def _lp(E_gwh: float, surplus: np.ndarray, load: np.ndarray):
    T = len(surplus); E = E_gwh * 1000.0; P = E / P_TO_E_H; pdt = P * DT
    n = 3 * T
    c = np.zeros(n); c[:T] = -1.0
    Aeq = lil_matrix((T, n)); beq = np.zeros(T)
    for t in range(T):
        nxt = (t + 1) % T
        Aeq[t, 2*T + nxt] = 1.0
        Aeq[t, 2*T + t] = -1.0
        Aeq[t, t] = -ETA_C
        Aeq[t, T + t] = 1.0 / ETA_D
    bounds = [(0.0, float(min(surplus[t], pdt))) for t in range(T)]
    bounds += [(0.0, float(min(load[t], pdt))) for t in range(T)]
    bounds += [(0.0, E)] * T
    res = linprog(c, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"storage LP failed at E={E_gwh}: {res.message}")
    return dict(E_GWh=E_gwh, P_GW=P/1000.0, captured_GWh=res.x[:T].sum()/1000.0,
                discharge_GWh=res.x[T:2*T].sum()/1000.0, charge=res.x[:T],
                discharge=res.x[T:2*T], soc=res.x[2*T:])


def _greedy(E_gwh: float, surplus: np.ndarray, load: np.ndarray, slots):
    E = E_gwh * 1000.0; P = E / P_TO_E_H; pdt = P * DT
    # Start empty: the lower benchmark cannot spend energy supplied before the
    # study window. It is intentionally not forced to end at the initial SOC.
    soc = 0.0; charge = np.zeros(len(surplus)); discharge = np.zeros(len(surplus))
    for i, (s, l, slot) in enumerate(zip(surplus, load, slots)):
        if 34 <= slot <= 44:
            d = min(l, pdt, soc * ETA_D); discharge[i] = d; soc -= d / ETA_D
        elif s > 0:
            q = min(s, pdt, (E - soc) / ETA_C); charge[i] = q; soc += q * ETA_C
    return dict(E_GWh=E_gwh, P_GW=P/1000.0, captured_GWh=charge.sum()/1000.0,
                discharge_GWh=discharge.sum()/1000.0, charge=charge, discharge=discharge,
                end_soc_MWh=soc)


def _capacity_for(policy, surplus, load, slots, target):
    """Find the smallest four-hour capacity reaching target captured GWh."""
    lo, hi = 0.0, 0.25
    fn = _lp if policy == "perfect_foresight_upper" else _greedy
    def run(E):
        return fn(E, surplus, load) if policy == "perfect_foresight_upper" else fn(E, surplus, load, slots)
    while True:
        r = run(hi)
        if r["captured_GWh"] >= target: break
        hi *= 2.0
        if hi > 5000: raise RuntimeError("storage capacity bracket exceeded 5000 GWh")
    for _ in range(22):
        mid = (lo + hi) / 2
        r = run(mid)
        if r["captured_GWh"] >= target: hi = mid
        else: lo = mid
    return run(hi)


def _residual_geography(E_gwh):
    f = _scoped_r3()
    f["behavior_absorbed_S1_MWh"] = f.realized_S1_MWh.clip(lower=0)
    f["residual_surplus_S1_MWh"] = f.residual_S1_MWh.clip(lower=0)
    total = f.residual_surplus_S1_MWh.sum()
    f["storage_equivalent_allocation_MWh"] = E_gwh * 1000.0 * f.residual_surplus_S1_MWh / total if total else 0.0
    behavior_total = f.behavior_absorbed_S1_MWh.sum()
    f["behavior_absorption_share"] = f.behavior_absorbed_S1_MWh / behavior_total if behavior_total else 0.0
    f["residual_surplus_share"] = f.residual_surplus_S1_MWh / total if total else 0.0
    f["share_difference_residual_minus_behavior"] = f.residual_surplus_share - f.behavior_absorption_share
    if not np.isclose(f.storage_equivalent_allocation_MWh.sum(), E_gwh * 1000.0, rtol=0, atol=1e-8):
        raise RuntimeError("mesh storage allocation does not close to the pooled equivalent capacity")
    mesh_output = (OUTDIR / "storage_residual_geography_mesh.parquet" if SCOPE
                   else Path(derived_path("storage", "storage_residual_geography_mesh.parquet")))
    f.to_parquet(mesh_output, index=False)
    corr = f.behavior_absorbed_S1_MWh.corr(f.residual_surplus_S1_MWh, method="spearman")
    k = max(1, len(f)//10)
    top = f.nlargest(k, "behavior_absorbed_S1_MWh")
    top_residual = f.nlargest(k, "residual_surplus_S1_MWh")
    bset, rset = set(top.mesh), set(top_residual.mesh)
    overlap = len(bset & rset)
    zero_behavior_residual_share = (f.loc[f.behavior_absorbed_S1_MWh <= 1e-12, "residual_surplus_S1_MWh"].sum() / total
                                    if total else np.nan)
    residual_before = float(f.demand_surplus_MWh.clip(lower=0).sum())
    pd.DataFrame([
        {"metric":"behavior_vs_residual_surplus_spearman", "value":corr},
        {"metric":"residual_surplus_total_GWh", "value":total/1000},
        {"metric":"residual_share_after_behavior", "value":total/residual_before if residual_before else np.nan},
        {"metric":"storage_equivalent_allocation_GWh", "value":E_gwh},
        {"metric":"behavior_top_decile_storage_allocation_share", "value":top.storage_equivalent_allocation_MWh.sum()/f.storage_equivalent_allocation_MWh.sum()},
        {"metric":"top_decile_mesh_overlap_fraction", "value":overlap/k},
        {"metric":"top_decile_jaccard", "value":overlap/len(bset | rset)},
        {"metric":"zero_behavior_mesh_residual_share", "value":zero_behavior_residual_share},
        {"metric":"mean_absolute_standardized_share_difference", "value":float(f.share_difference_residual_minus_behavior.abs().mean())},
        {"metric":"mesh_siting_unique", "value":"no"},
        {"metric":"strict_negative_spatial_complementarity_supported", "value":"no"},
    ]).to_csv(OUTDIR / "storage_residual_geography.csv", index=False)
    pd.DataFrame([{
        "mesh_count":len(f),
        "behavior_vs_residual_spearman":corr,
        "behavior_residual_top_decile_overlap_fraction":overlap/k,
        "behavior_residual_top_decile_jaccard":overlap/len(bset | rset),
        "zero_behavior_mesh_residual_share":zero_behavior_residual_share,
        "strict_negative_spatial_complementarity_supported":False,
        "interpretation":"behavior removes only part of local surplus; residual storage remains concentrated in high-exposure meshes",
    }]).to_csv(OUTDIR / "storage_spatial_diagnostics.csv", index=False)
    siting = f[["mesh", "pref", "behavior_absorbed_S1_MWh", "residual_surplus_S1_MWh",
                "storage_equivalent_allocation_MWh"]].copy()
    siting["allocation_rule"] = "residual-proportional geography diagnostic; not a mesh-level SOC optimum"
    siting["optimization_status"] = "pooled SOC relaxation solved; stationary mesh capacity allocation remains non-identified"
    siting.to_csv(OUTDIR / "storage_mesh_siting_allocation.csv", index=False)

    lat, lon = _mesh_centers(f.mesh)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    panels = [(f.behavior_absorbed_S1_MWh.to_numpy(float), "S1 behavior absorption", "viridis"),
              (f.storage_equivalent_allocation_MWh.to_numpy(float), "Residual-proportional storage allocation", "magma")]
    for ax, (raw_values, title, cmap) in zip(axes, panels):
        values = np.log1p(raw_values)
        positive = values > 0
        vmax = float(np.quantile(values[positive], .99)) if positive.any() else 1.0
        ax.scatter(lon[~positive], lat[~positive], c="#d9d9d9", s=1.4, alpha=.45, linewidths=0)
        sc = ax.scatter(lon[positive], lat[positive], c=values[positive], s=2.0, alpha=.78,
                        cmap=cmap, vmin=0, vmax=vmax, linewidths=0)
        ax.set_title(title); ax.set_xlabel("longitude"); ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=.045, pad=.02, label="log(1 + MWh)")
    axes[0].set_ylabel("latitude")
    fig.tight_layout(); fig.savefig(FIGDIR / "storage_residual_geography.png", dpi=180); fig.savefig(FIGDIR / "storage_residual_geography.pdf"); plt.close(fig)


def _siting_equivalence(perfect, greedy):
    rows = []
    for policy, result in (("perfect_foresight_upper", perfect), ("greedy_lower", greedy)):
        rows.append({
            "policy":policy,
            "storage_energy_GWh":result["E_GWh"],
            "storage_power_GW":result["P_GW"],
            "network_constraints":False,
            "pooled_dispatch":True,
            "mesh_capacity_allocation":"proportional to residual surplus",
            "mesh_allocation_unique":False,
            "equivalence_condition":"free siting, common efficiency and power-to-energy ratio, and no feeder constraints",
            "interpretation":"pooled result is the free-siting/no-network relaxation; the map is one feasible capacity allocation, not a feeder SOC solution",
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "storage_siting_equivalence.csv", index=False)


def _anchor_shadow_capacity(surplus, load, slots):
    scale = pd.read_csv(OUTDIR / "behavior_wfh_scale_sensitivity.csv")
    s1 = scale.loc[(scale.scale - 1.0).abs().idxmin()]
    anchored = float(s1.actual_absorption_GWh)
    free = float(s1.free_placement_ceiling * s1.surplus_GWh)
    gap = free - anchored
    rows = []
    for policy in ("perfect_foresight_upper", "greedy_lower"):
        result = _capacity_for(policy, surplus, load, slots, gap)
        rows.append({
            "policy":policy,
            "anchored_absorption_GWh":anchored,
            "same_slot_free_absorption_GWh":free,
            "spatial_anchoring_gap_GWh":gap,
            "storage_energy_GWh":result["E_GWh"],
            "storage_energy_MWh":result["E_GWh"] * 1000.0,
            "storage_power_MW":result["P_GW"] * 1000.0,
            "captured_surplus_GWh":result["captured_GWh"],
            "interpretation":"free-siting storage capacity needed to recover the S1 same-slot free-placement minus anchored absorption gap",
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "storage_anchor_shadow_capacity.csv", index=False)


def solve():
    dates, slots, surplus, load = _series()
    target_gwh = float(_scoped_r3().realized_S1_MWh.sum() / 1000.0)
    # Include sub-0.1 GWh capacities so the policy-bracket figure shows the
    # behavior-equivalent intersection instead of starting above the target.
    sweep = np.geomspace(0.01, 500.0, 24)
    perfect_sweep = [_lp(float(E), surplus, load) for E in sweep]
    first = next((float(E) for E, r in zip(sweep, perfect_sweep) if r["captured_GWh"] >= target_gwh), float(sweep[-1]))
    lo, hi = 0.0, first
    for _ in range(18):
        mid = (lo + hi) / 2
        if _lp(mid, surplus, load)["captured_GWh"] >= target_gwh: hi = mid
        else: lo = mid
    perfect = _capacity_for("perfect_foresight_upper", surplus, load, slots, target_gwh)
    greedy = _capacity_for("greedy_lower", surplus, load, slots, target_gwh)
    rows = []
    for policy, r in [("perfect_foresight_upper", perfect), ("greedy_lower", greedy)]:
        rows.append({"policy":policy, "target_behavior_relief_GWh":target_gwh,
                     "storage_energy_GWh":r["E_GWh"], "storage_power_GW":r["P_GW"],
                     "captured_surplus_GWh":r["captured_GWh"],
                     "capture_to_capacity":r["captured_GWh"]/r["E_GWh"],
                     "capacity_per_behavior_relief":r["E_GWh"]/target_gwh,
                     "round_trip_efficiency":ETA_C*ETA_D,
                     "cyclic_soc":policy == "perfect_foresight_upper", "cross_day_dispatch":True,
                     "free_mesh_siting":True, "model_grain":"pooled date-slot SOC dispatch; residual-proportional mesh diagnostic",
                     "siting_equivalence":"pooled dispatch is the unconstrained spatial upper bound; stationary mesh SOC siting is not uniquely identified",
                     "mesh_siting_unique":False,
                     "power_flow_constraints":False})
    out = pd.DataFrame(rows); out.to_csv(OUTDIR / "storage_equivalence.csv", index=False)
    _siting_equivalence(perfect, greedy)
    _anchor_shadow_capacity(surplus, load, slots)
    bracket = []
    for E in sweep:
        p = _lp(float(E), surplus, load); g = _greedy(float(E), surplus, load, slots)
        for policy, r in [("perfect_foresight_upper",p),("greedy_lower",g)]:
            bracket.append({"policy":policy, "storage_energy_GWh":E, "storage_power_GW":r["P_GW"], "captured_surplus_GWh":r["captured_GWh"], "target_GWh":target_gwh})
    pd.DataFrame(bracket).to_csv(OUTDIR / "storage_policy_bracket.csv", index=False)
    _residual_geography(perfect["E_GWh"])
    q = pd.DataFrame(bracket)
    plt.figure(figsize=(6.2,3.8))
    for policy, g in q.groupby("policy"): plt.plot(g.storage_energy_GWh, g.captured_surplus_GWh, marker="o", ms=3, label=policy)
    plt.axhline(target_gwh, color="#c23b22", ls="--", label="S1 behavior target")
    plt.xscale("log"); plt.xlabel("storage energy capacity (GWh)"); plt.ylabel("captured surplus (GWh)"); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(FIGDIR / "storage_policy_bracket.png", dpi=180); plt.savefig(FIGDIR / "storage_policy_bracket.pdf"); plt.close()
    return out


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)
    print(solve().to_string(index=False))
