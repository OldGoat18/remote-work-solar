"""Behavior-side robustness experiments for the public-data reconstruction.

Runs on the existing mesh x date x slot ledger and keeps all WFH increments
prefecture-conserving.  It reports (1) WFH-scale sensitivity, (2) randomized
placement nulls, (3) seasonal data availability and the winter benchmark, and
(4) the explicitly omitted rebound direction.  Spring/summer are never filled
from winter; they are emitted as unavailable until those source months exist.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb

from wlpr.io.paths import derived_path, out_path

BASE = derived_path("ledger", "baseline_p0")
ALPHA = derived_path("pv", "alpha_rooftop_capacity_by_mesh.parquet")
KER = derived_path("pv", "obs_stock_yield_kernel.parquet")
ADW = derived_path("wfh_state", "a_dwfh_mesh_mon_slot.parquet")
R3 = derived_path("social", "r3_spatial_matching_mesh.parquet")
IDX = derived_path("scenario_s1s10", "scenario_indices_s1s10.csv")
CFG = derived_path("wfh_state", "scenario_config_v2.csv")
SCOPE = os.environ.get("WLPR_SCOPE")
SCOPE_STATUS = os.environ.get("WLPR_SCOPE_STATUS", "positive")
_CUSTOM_OUT = os.environ.get("WLPR_EXTRA_OUTDIR")
OUTDIR = _CUSTOM_OUT or out_path("scorecards")
FIGDIR = os.environ.get("WLPR_EXTRA_FIGDIR") or (_CUSTOM_OUT or out_path("figures", "extra"))
PV_CALIBRATION_SCALE = float(os.environ.get("WLPR_PV_SCALE", "1.0"))
LOAD_CALIBRATION_SCALE = float(os.environ.get("WLPR_LOAD_SCALE", "1.0"))
WFH_CALIBRATION_SCALE = float(os.environ.get("WLPR_WFH_SCALE", "1.0"))
SEED = 20260804
N_NULL = 2000


def _scope_frame():
    scope = pd.read_parquet(SCOPE)
    if SCOPE_STATUS == "positive" and "load_status" in scope:
        scope = scope[scope.load_status == "positive_aggregate_load"]
    elif SCOPE_STATUS != "all":
        raise ValueError(f"unsupported WLPR_SCOPE_STATUS={SCOPE_STATUS!r}")
    return scope


def _scoped_r3():
    scoped_output = Path(OUTDIR) / "r3_spatial_matching_mesh.parquet"
    if SCOPE and scoped_output.exists():
        return pd.read_parquet(scoped_output)
    frame = pd.read_parquet(R3)
    if not SCOPE:
        return frame
    scope = _scope_frame()
    keep = set(scope.mesh.astype(str))
    frame = frame[frame.mesh.astype(str).isin(keep)].copy()
    if len(frame) != len(keep):
        raise RuntimeError(f"scope/R3 mismatch: expected {len(keep):,}, found {len(frame):,}")
    return frame


def _scope_join(alias="b"):
    if not SCOPE:
        return ""
    scope = str(Path(SCOPE).resolve()).replace("\\", "/")
    status_clause = " and sc.load_status='positive_aggregate_load'" if SCOPE_STATUS == "positive" else ""
    return (f"inner join read_parquet('{scope}') sc "
            f"on cast(sc.mesh as varchar)=cast({alias}.mesh as varchar) "
            f"and cast(sc.pref as varchar)=cast({alias}.pref as varchar) "
            f"{status_clause}")


def _scales():
    cfg = pd.read_csv(CFG)
    s10 = cfg.loc[cfg.scenario_id == "S10"].iloc[0]
    if s10.evidence_tier != "upper-bound":
        raise RuntimeError("S10 must remain the registered upper-bound scenario")
    maximum = float(s10.wfh_energy_multiplier)
    base = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    return base + [maximum], s10


def _winter_scale():
    if SCOPE:
        scale = pd.read_csv(Path(OUTDIR) / "behavior_wfh_scale_sensitivity.csv")
        s1 = scale.loc[np.isclose(scale.scale, 1.0)].iloc[0]
        return float(s1.wfh_budget_GWh), float(s1.surplus_GWh), float(s1.actual_match), float(s1.free_placement_ceiling)
    idx = pd.read_csv(IDX)
    s1 = idx[(idx.scenario_id == "S1") & (idx.pv_scenario == "future_pv_ai")].iloc[0]
    r3 = pd.read_csv(out_path("scorecards", "r3_spatial_reallocation_scorecard.csv"))
    r = r3[r3.scope == "all_slots"].iloc[0]
    return float(s1.dWFH_total_GWh), float(r.surplus_D_GWh), float(r.match_actual), float(r.match_spatial_opt)


def _exact_scale_fields():
    """Rebuild all-slot date-slot fields with one consistent definition."""
    parts = sorted(glob.glob(os.path.join(BASE, "pref=*", "ym=*", "*.parquet")))
    con = duckdb.connect(); con.execute("SET memory_limit='10GB'"); con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    rows, mesh_rows = [], []
    scales, _ = _scales()
    for part in parts:
        p = part.replace("\\", "/")
        q = f"""
        with x as (
          select cast(b.mesh as varchar) mesh, cast(b.pref as varchar) pref, b.date, b.slot,
                 b.L_owner_gross lo, b.L_nonowner_import ln, b.E_wfh, b.G_pv gp,
                 coalesce(a.increment_kw_a015,0) inc, coalesce(k.sim_ykw_kwh_per_kw_30min,0) ker,
                 coalesce(w.dwfh,0) dwfh, coalesce(w10.dwfh,0) dwfh10,
                 case when lo+ln>0 then least(lo/(lo+ln),1.0) else 0 end osh
          from read_parquet('{p}') b
          {_scope_join('b')}
          left join read_parquet('{ALPHA}') a on cast(a.mesh as varchar)=cast(b.mesh as varchar)
             and cast(a.pref as varchar)=cast(b.pref as varchar)
          left join read_parquet('{KER}') k on cast(k.pref as varchar)=cast(b.pref as varchar)
             and k.m=cast(substr(b.date,5,2) as int) and k.slot=b.slot
          left join read_parquet('{ADW}') w on cast(w.mesh as varchar)=cast(b.mesh as varchar)
             and w.mon=cast(substr(b.date,5,2) as int) and w.slot=b.slot and w.scenario_id='S1'
          left join read_parquet('{ADW}') w10 on cast(w10.mesh as varchar)=cast(b.mesh as varchar)
             and w10.mon=cast(substr(b.date,5,2) as int) and w10.slot=b.slot and w10.scenario_id='S10'
        ), y as (
          select mesh,pref,date,slot,
             dwfh*{WFH_CALIBRATION_SCALE} B, dwfh10*{WFH_CALIBRATION_SCALE} B10,
             (greatest(lo-E_wfh*osh,0)+greatest(ln-E_wfh*(1-osh),0))*{LOAD_CALIBRATION_SCALE} L0,
             (gp+inc*ker/1000.0)*{PV_CALIBRATION_SCALE} G
          from x
        ), z as (
          select mesh,pref,date,slot,B,B10,L0,G,L0-G net_energy,greatest(G-L0,0) D from y
        )
        select date, slot, {"mesh, pref," if SCOPE else ""}
               sum(D) D, sum(B) B, sum(B10) B10, sum(net_energy)*2.0 net_load_MW,
               sum(G) generation_MWh,
               sum(case when slot between 20 and 30 then L0 else 0 end) midday_load_no_wfh_MWh,
               sum(greatest(G-(L0+B),0)) residual_S1_MWh,
               sum(D-greatest(G-(L0+B),0)) realized_S1_MWh,
               sum(case when slot between 20 and 30 then B else 0 end) supply_S1_MWh,
               sum(greatest(G-(L0+B10),0)) residual_S10_MWh,
               sum(D-greatest(G-(L0+B10),0)) realized_S10_MWh,
               sum(case when slot between 20 and 30 then B10 else 0 end) supply_S10_MWh,
               {", ".join(f"sum(least({scale}*B,D)) A_{str(scale).replace('.', '_')}" for scale in scales)}
        from z {"group by grouping sets ((date,slot),(mesh,pref))" if SCOPE else "group by date, slot"}
        """
        result = con.execute(q).df()
        if SCOPE:
            rows.append(result[result.mesh.isna()].drop(columns=["mesh", "pref"]))
            mesh_rows.append(result[result.mesh.notna()].drop(columns=["date", "slot"]))
        else:
            rows.append(result)
    con.close()
    system = pd.concat(rows, ignore_index=True).groupby(["date", "slot"], as_index=False).sum()
    if not SCOPE:
        return system, None
    mesh = (pd.concat(mesh_rows, ignore_index=True)
            .groupby(["mesh", "pref"], as_index=False).sum(numeric_only=True))
    keep = ["mesh", "pref", "D", "generation_MWh", "midday_load_no_wfh_MWh",
            "residual_S1_MWh", "realized_S1_MWh", "supply_S1_MWh",
            "residual_S10_MWh", "realized_S10_MWh", "supply_S10_MWh"]
    mesh = mesh[keep].rename(columns={"D":"demand_surplus_MWh"})
    expected = len(_scope_frame())
    if len(mesh) != expected:
        raise RuntimeError(f"scoped R3 expected {expected:,} meshes, got {len(mesh):,}")
    return system, mesh


def scale_sensitivity():
    """Scale the observed S1 WFH field against the fixed surplus field."""
    f, scoped_r3 = _exact_scale_fields()
    f.to_parquet(Path(OUTDIR) / "behavior_date_slot_fields.parquet", index=False)
    if scoped_r3 is not None:
        scoped_r3.to_parquet(Path(OUTDIR) / "r3_spatial_matching_mesh.parquet", index=False)
    D = f.D.to_numpy(float)
    S = f.B.to_numpy(float)
    scales, s10 = _scales()
    rows = []
    for scale in scales:
        # WFH supply is scaled; local absorption is capped by local surplus.
        # The base surplus is fixed because the sensitivity asks how much
        # additional movable demand is needed to absorb it.
        A = f[f"A_{str(scale).replace('.', '_')}"].sum()
        B = (scale * S).sum()
        # Free placement is constrained within each date-slot, preserving the
        # behavior's temporal budget and the surplus's temporal availability.
        A_spatial = np.minimum(scale * S, D).sum()
        rows.append(dict(experiment="wfh_scale", scale=scale,
                         wfh_budget_GWh=B / 1000, surplus_GWh=D.sum() / 1000,
                         actual_absorption_GWh=A / 1000,
                         actual_match=A / D.sum() if D.sum() else np.nan,
                         free_placement_ceiling=A_spatial / D.sum() if D.sum() else np.nan,
                         spatial_gap=(A_spatial - A) / D.sum() if D.sum() else np.nan,
                         magnitude_wall=1 - A_spatial / D.sum() if D.sum() else np.nan,
                         upper_bound_scenario="S10" if np.isclose(scale, float(s10.wfh_energy_multiplier)) else "",
                         target_occupation_adoption=0.85 if np.isclose(scale, float(s10.wfh_energy_multiplier)) else np.nan,
                         frequency_intensity=0.70 if np.isclose(scale, float(s10.wfh_energy_multiplier)) else np.nan,
                         evidence_tier=str(s10.evidence_tier) if np.isclose(scale, float(s10.wfh_energy_multiplier)) else "scale_sensitivity",
                         evidence="date-slot WFH supply scaled; date-slot free-placement upper bound"))
    out = pd.DataFrame(rows)
    out.to_csv(Path(OUTDIR) / "behavior_wfh_scale_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(out.scale, out.actual_match * 100, marker="o", lw=2, label="anchored placement")
    ax.plot(out.scale, out.free_placement_ceiling * 100, marker="s", lw=2, label="same-slot free placement")
    ax.fill_between(out.scale, out.actual_match * 100, out.free_placement_ceiling * 100,
                    color="#e6a23c", alpha=.18, label="spatial gap")
    ax.set_xlabel("WFH scale relative to S1")
    ax.set_ylabel("surplus absorption (%)")
    ax.set_title("Behavior-side robustness: WFH scale")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(FIGDIR) / "behavior_wfh_scale_sensitivity.png", dpi=180)
    fig.savefig(Path(FIGDIR) / "behavior_wfh_scale_sensitivity.pdf")
    plt.close(fig)
    r3 = _scoped_r3()
    s10_budget = float(f.B10.sum() / 1000.0)
    max_row = out.iloc[-1]
    summary = pd.DataFrame([
        {"case":"S1_observed_anchor", "behavior_budget_GWh":float(out.loc[np.isclose(out.scale, 1.0), "wfh_budget_GWh"].iloc[0]),
         "absorbed_GWh":float(out.loc[np.isclose(out.scale, 1.0), "actual_absorption_GWh"].iloc[0]),
         "match_fraction":float(out.loc[np.isclose(out.scale, 1.0), "actual_match"].iloc[0]),
         "spatial_setting":"observed S1 residential anchor", "interpretation":"headline current behavior"},
        {"case":"S10_structural_upper_exact_geography", "behavior_budget_GWh":s10_budget,
         "absorbed_GWh":float(r3.realized_S10_MWh.sum()/1000),
         "match_fraction":float(r3.realized_S10_MWh.sum()/r3.demand_surplus_MWh.sum()),
         "spatial_setting":"S10 occupation-based residential geography",
         "interpretation":"85% target-occupation adoption and 70% frequency intensity"},
        {"case":"S10_energy_on_S1_geography_adversarial", "behavior_budget_GWh":float(max_row.wfh_budget_GWh),
         "absorbed_GWh":float(max_row.actual_absorption_GWh), "match_fraction":float(max_row.actual_match),
         "spatial_setting":"S1 geography scaled by S10 energy multiplier",
         "interpretation":"more favorable anchored result than exact S10 geography"},
        {"case":"S10_same_slot_free_placement", "behavior_budget_GWh":float(max_row.wfh_budget_GWh),
         "absorbed_GWh":float(max_row.free_placement_ceiling*max_row.surplus_GWh),
         "match_fraction":float(max_row.free_placement_ceiling),
         "spatial_setting":"free placement within every date-slot",
         "interpretation":"behavior-side best case; theoretical spatial upper bound"},
    ])
    summary.to_csv(Path(OUTDIR) / "behavior_adversarial_upper_bound.csv", index=False)
    target_mwh = float(out.loc[np.isclose(out.scale, 1.0), "actual_absorption_GWh"].iloc[0] * 1000.0)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if np.minimum(mid * S, D).sum() >= target_mwh:
            hi = mid
        else:
            lo = mid
    free_scale = hi
    anchored_budget = float(S.sum() / 1000.0)
    free_budget = free_scale * anchored_budget
    pd.DataFrame([{
        "target_absorption_GWh": target_mwh / 1000.0,
        "anchored_scale": 1.0,
        "same_slot_free_scale": free_scale,
        "anchored_wfh_budget_GWh": anchored_budget,
        "same_slot_free_wfh_budget_GWh": free_budget,
        "spatial_anchor_extra_wfh_budget_GWh": anchored_budget - free_budget,
        "spatial_anchor_budget_multiplier": anchored_budget / free_budget,
        "definition": "extra relocated-energy budget needed by S1 geography to match same-slot free placement at the S1 captured-energy target",
    }]).to_csv(Path(OUTDIR) / "behavior_spatial_anchor_shadow.csv", index=False)
    return out


def null_models():
    """Random placement nulls preserving total supply and surplus marginals."""
    f = _scoped_r3()
    D = np.maximum(f.demand_surplus_MWh.to_numpy(float), 0)
    S = np.maximum(f.supply_S1_MWh.to_numpy(float), 0)
    rng = np.random.default_rng(SEED)
    # Two nulls: uniform permutation and population/load-weighted placement.
    # The latter uses validated midday baseline load as the residence proxy.
    weights = np.maximum(f.midday_load_no_wfh_MWh.fillna(0).to_numpy(float), 0)
    weights = weights / weights.sum() if weights.sum() else np.ones(len(f)) / len(f)
    actual = float(f.realized_S1_MWh.sum() / f.demand_surplus_MWh.sum())
    rows = []
    for model in ["uniform_permutation", "load_weighted"]:
        vals = []
        for i in range(N_NULL):
            if model == "uniform_permutation":
                placed = rng.permutation(S)
            else:
                # Random multinomial placement preserves total energy while
                # allowing multiple supply draws in a load-weighted location.
                destinations = rng.choice(len(S), size=len(S), p=weights)
                placed = np.bincount(destinations, weights=S, minlength=len(S))
            vals.append(float(np.minimum(D, placed).sum() / D.sum()))
        q = np.quantile(vals, [0.025, 0.5, 0.975])
        rows.append(dict(model=model, n=N_NULL, actual_match=actual,
                         null_mean=float(np.mean(vals)), null_sd=float(np.std(vals, ddof=1)),
                         null_p025=float(q[0]), null_median=float(q[1]), null_p975=float(q[2]),
                         actual_percentile=float(np.mean(np.asarray(vals) <= actual)),
                         null_experiment="mesh-window placement null; same D and same midday WFH supply marginal"))
    out = pd.DataFrame(rows)
    out.to_csv(Path(OUTDIR) / "behavior_null_models.csv", index=False)
    # Distribution plot for the uniform null; recompute compactly from seed.
    vals = []
    for _ in range(N_NULL):
        vals.append(float(np.minimum(D, rng.permutation(S)).sum() / D.sum()))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(vals, bins=35, color="#7aa6c2", alpha=.8, label="uniform permutation null")
    ax.axvline(actual, color="#c23b22", lw=2, label=f"observed S1 = {actual:.3f}")
    ax.set_xlabel("local absorption fraction"); ax.set_ylabel("count"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(Path(FIGDIR) / "behavior_null_distribution.png", dpi=180); fig.savefig(Path(FIGDIR) / "behavior_null_distribution.pdf"); plt.close(fig)
    return out


def seasonal_status():
    """Emit a truthful seasonal status table; only available source months are computed."""
    requested = {"spring": {3, 4, 5}, "summer": {6, 7, 8}, "winter": {10, 11, 12, 1, 2}}
    present = {10, 11, 12, 1, 2}
    rows = []
    # Winter values use the current public rebuild. Spring/summer are blocked
    # by missing source months and are not imputed.
    scale = pd.read_csv(Path(OUTDIR) / "behavior_wfh_scale_sensitivity.csv")
    r = scale.loc[np.isclose(scale.scale, 1.0)].iloc[0]
    for season, months in requested.items():
        missing = sorted(months - present)
        if missing:
            rows.append(dict(season=season, status="MISSING_SOURCE_MONTHS", missing_months=",".join(map(str, missing)),
                             actual_match=np.nan, spatial_wall_pp=np.nan, magnitude_wall=np.nan,
                             missing_load_months=",".join(map(str, missing)),
                             missing_weather_months=",".join(map(str, missing)),
                             missing_pv_kernel_months=",".join(map(str, missing)),
                             required_rebuild="baseline_p0 + weather factors + PV yield kernel + date-slot R3",
                             note="Standard Statistics source staged here contains Oct-Feb only; no seasonal imputation"))
        else:
            rows.append(dict(season=season, status="COMPUTED", missing_months="",
                             actual_match=float(r.actual_match), spatial_wall_pp=float(r.spatial_gap*100),
                             magnitude_wall=float(r.magnitude_wall), missing_load_months="",
                             missing_weather_months="", missing_pv_kernel_months="", required_rebuild="",
                             note="public reconstruction"))
    out = pd.DataFrame(rows)
    out.to_csv(Path(OUTDIR) / "behavior_seasonal_wall.csv", index=False)
    available = out[out.status == "COMPUTED"]
    if not available.empty:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.bar(available.season, available.spatial_wall_pp, color="#d98c3f")
        ax2 = ax.twinx()
        ax2.plot(available.season, available.magnitude_wall * 100, "o-", color="#345995")
        ax.set_ylabel("spatial gap (percentage points)")
        ax2.set_ylabel("magnitude wall (%)")
        ax.set_title("Seasonal wall decomposition: available seasons")
        fig.tight_layout()
        fig.savefig(Path(FIGDIR) / "behavior_seasonal_wall.png", dpi=180)
        fig.savefig(Path(FIGDIR) / "behavior_seasonal_wall.pdf")
        plt.close(fig)
    return out


def rebound_statement():
    out = pd.DataFrame([dict(assumption="rebound_load", value=0.0,
                             status="OMITTED", bias_direction="optimistic_for_behavior",
                             claim_effect="cannot raise identified local PV absorption; can increase evening ramp and storage need",
                             rationale="additional non-PV-window demand cannot increase the identified relocation field;"
                                      "evening rebound worsens ramp/storage metrics")])
    out.to_csv(Path(OUTDIR) / "behavior_rebound_assumption.csv", index=False); return out


def rebound_sensitivity():
    """Bookkeep rebound energy without crediting it as relocated PV-window load."""
    s1_budget, _, actual, _ = _winter_scale()
    rows = []
    for rate in (0.0, 0.05, 0.10, 0.20):
        rows.append({
            "rebound_rate": rate,
            "s1_relocated_budget_GWh": s1_budget,
            "additional_non_pv_window_demand_GWh": rate * s1_budget,
            "identified_pv_absorption_fraction": actual,
            "credited_additional_absorption_GWh": 0.0,
            "bias_direction":"weakly worse for behavior; evening rebound increases ramp and storage duty",
        })
    out = pd.DataFrame(rows)
    out.to_csv(Path(OUTDIR) / "behavior_rebound_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(out.rebound_rate * 100, out.additional_non_pv_window_demand_GWh,
           width=3.2, color="#c9824a", label="additional non-PV-window demand")
    ax.set_xlabel("rebound rate relative to S1 budget (%)")
    ax.set_ylabel("additional demand (GWh)")
    ax2 = ax.twinx()
    ax2.plot(out.rebound_rate * 100, out.identified_pv_absorption_fraction * 100,
             "o-", color="#2f6f8f", label="identified local PV absorption")
    ax2.set_ylabel("identified absorption (%)")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(Path(FIGDIR) / "behavior_rebound_sensitivity.png", dpi=180)
    fig.savefig(Path(FIGDIR) / "behavior_rebound_sensitivity.pdf"); plt.close(fig)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
    pd.DataFrame([{
        "scope":f"legacy_scope_{SCOPE_STATUS}" if SCOPE else "public_reconstruction_all_meshes",
        "mesh_count":len(_scoped_r3()),
        "scope_file":str(Path(SCOPE).resolve()) if SCOPE else "",
        "seed":SEED,
        "null_draws":N_NULL,
        "pv_calibration_scale":PV_CALIBRATION_SCALE,
        "load_calibration_scale":LOAD_CALIBRATION_SCALE,
        "wfh_calibration_scale":WFH_CALIBRATION_SCALE,
    }]).to_csv(Path(OUTDIR) / "experiment_scope_manifest.csv", index=False)
    a, n, s, r, rs = scale_sensitivity(), null_models(), seasonal_status(), rebound_statement(), rebound_sensitivity()
    print(a.to_string(index=False)); print(n.to_string(index=False)); print(s.to_string(index=False)); print(r.to_string(index=False)); print(rs.to_string(index=False))
    print("17_behavior_robustness: COMPLETED (seasonal gaps reported, not imputed)")

if __name__ == "__main__":
    main()
