"""Step 05b — future-PV alpha-adoption rooftop capacity scenario (ported from
configured asset).

Future PV is defined transparently as an ADOPTION FRACTION (alpha) of the official MOE REPOS R3
building-rooftop CAPACITY potential (B_IC, kW), with observed current PV preserved by a per-mesh floor:

    future_cap(mesh) = max( current_cap(mesh),  alpha * REPOS_B_IC(mesh) )   # per-mesh floor => no double-count
    increment(mesh)  = future_cap - current_cap = max(alpha*B_IC - current, 0)

alpha grid: 5 / 15 / 30 / 50 / 100 %. Central = 15 %. 100 % = technical rooftop ceiling (NOT a forecast).

Calibers (never conflate): current metered ~2.83 GW = 1x ; REPOS roof potential ~76.85 GW = ceiling.
Units: REPOS B_IC = 設備容量 (potential CAPACITY, kW); B_PG = 年間発電 (kWh/yr).

PORT NOTE: the source also produced a Mydaiz reference-mapping scorecard that reads
future_pv_ai_capacity_mesh.parquet (the EXCLUDED future-PV user-state / Mydaiz subsystem). That block is
NOT ported here; only the alpha-rooftop core (central=15%) and the capacity scorecard are kept.

Reads frozen mesh_denominator_panel + derived btm capacity panel + derived REPOS roof potential.
Writes configured asset (+ configured asset).
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path, input_path, out_path

ALPHAS = [0.05, 0.15, 0.30, 0.50, 1.00]
CENTRAL = 0.15
OCT_FEB = ["202310", "202311", "202312", "202401", "202402"]

CAP_SRC = derived_path(_asset_location('asset_0086'), "btm_capacity_mesh_month.parquet")
REPOS_SRC = derived_path(_asset_location('asset_0054'), "repos_mesh500_potential.parquet")
OUT = derived_path(_asset_location('asset_0054'), "alpha_rooftop_capacity_by_mesh.parquet")
SC = out_path(_asset_location('asset_0058'), "alpha_capacity_scorecard.csv")


def _ms(df, col="mesh"):
    df[col] = df[col].astype(str)
    return df


def build_alpha_rooftop() -> dict:
    # --- project mesh universe (the analysis scope; headline numbers restrict to this) ---
    den = _ms(pd.read_parquet(input_path("mesh_denominator_panel"), columns=["mesh", "pref"]))
    proj = den.drop_duplicates("mesh")[["mesh", "pref"]].copy()

    # --- current PV capacity per mesh (発電設備容量, Oct-Feb mean) ---
    cap = _ms(pd.read_parquet(CAP_SRC))
    cap["ym"] = cap["ym"].astype(str)
    cur = (cap[cap.ym.isin(OCT_FEB)].groupby("mesh", as_index=False)
           .agg(current_cap_kw=("pv_capacity_kw", "mean")))
    cur = _ms(cur)

    # --- REPOS official building-rooftop CAPACITY potential per mesh (B_IC, kW) ---
    rep = _ms(pd.read_parquet(REPOS_SRC, columns=["mesh", "B_IC", "B_PG"]))
    rep = rep.groupby("mesh", as_index=False).agg(pot_cap_kw_B_IC=("B_IC", "sum"), pot_gen_kwh_B_PG=("B_PG", "sum"))

    full_pot_GW = rep.pot_cap_kw_B_IC.sum() / 1e6          # full REPOS 4-pref roof capacity potential

    # --- build the per-mesh layer on the PROJECT universe ---
    m = proj.merge(cur, on="mesh", how="left").merge(rep, on="mesh", how="left")
    m[["current_cap_kw", "pot_cap_kw_B_IC", "pot_gen_kwh_B_PG"]] = m[["current_cap_kw", "pot_cap_kw_B_IC", "pot_gen_kwh_B_PG"]].fillna(0.0)
    inproj_pot_GW = m.pot_cap_kw_B_IC.sum() / 1e6
    cur_GW = m.current_cap_kw.sum() / 1e6
    cov_mesh = float((m.pot_cap_kw_B_IC > 0).mean())       # share of project meshes with REPOS roof potential

    for a in ALPHAS:
        tag = f"a{int(a*100):03d}"
        fut = np.maximum(m.current_cap_kw, a * m.pot_cap_kw_B_IC)
        m[f"future_cap_kw_{tag}"] = fut
        m[f"increment_kw_{tag}"] = fut - m.current_cap_kw
    m.to_parquet(OUT, index=False)

    # --- capacity scorecard (the 4 required scope numbers + per-alpha in-project totals) ---
    rows = [
        dict(item="full_4pref_REPOS_roof_potential_GW", value=round(full_pot_GW, 3),
             note="all REPOS meshes pref 11-14, B_IC capacity (official)"),
        dict(item="in_project_REPOS_roof_potential_GW", value=round(inproj_pot_GW, 3),
             note="REPOS B_IC restricted to the 26,039 project meshes (HEADLINE scope)"),
        dict(item="project_coverage_pct_of_full_potential", value=round(100 * inproj_pot_GW / full_pot_GW, 1),
             note="in-project potential / full 4-pref potential"),
        dict(item="project_mesh_with_repos_roof_pct", value=round(100 * cov_mesh, 1),
             note="share of project meshes carrying REPOS roof potential"),
        dict(item="current_metered_capacity_GW_inproject", value=round(cur_GW, 3),
             note="発電設備容量 Oct-Feb mean, project meshes (~2.83 target)"),
    ]
    for a in ALPHAS:
        tag = f"a{int(a*100):03d}"
        fut_GW = m[f"future_cap_kw_{tag}"].sum() / 1e6
        inc_GW = m[f"increment_kw_{tag}"].sum() / 1e6
        n_inc = int((m[f"increment_kw_{tag}"] > 0).sum())
        lab = "CENTRAL" if abs(a - CENTRAL) < 1e-9 else ("CEILING" if a >= 1.0 else "scenario")
        rows.append(dict(item=f"future_capacity_GW_alpha_{int(a*100)}pct", value=round(fut_GW, 3),
                         note=f"[{lab}] x{fut_GW/cur_GW:.2f} current | increment {inc_GW:.2f} GW | {n_inc:,} meshes grow"))
    sc = pd.DataFrame(rows)
    sc.to_csv(SC, index=False)

    # --- console summary ---
    print("=== alpha rooftop future-PV capacity (project scope) ===")
    print(f"full 4-pref REPOS roof potential : {full_pot_GW:8.2f} GW")
    print(f"in-project REPOS roof potential  : {inproj_pot_GW:8.2f} GW  (coverage {100*inproj_pot_GW/full_pot_GW:.1f}% of full, {100*cov_mesh:.1f}% of meshes)")
    print(f"current metered (in-project)     : {cur_GW:8.2f} GW")
    for a in ALPHAS:
        tag = f"a{int(a*100):03d}"
        fut_GW = m[f"future_cap_kw_{tag}"].sum() / 1e6
        flag = "  <- CENTRAL" if abs(a - CENTRAL) < 1e-9 else ("  <- technical ceiling" if a >= 1 else "")
        print(f"  alpha={int(a*100):3d}%  future {fut_GW:7.2f} GW  ({fut_GW/cur_GW:5.2f}x current){flag}")
    print("[saved] alpha_rooftop_capacity_by_mesh.parquet, alpha_capacity_scorecard.csv")

    return {"central_alpha": CENTRAL, "cur_GW": cur_GW,
            "central_future_GW": float(m[f"future_cap_kw_a{int(CENTRAL*100):03d}"].sum() / 1e6),
            "n_mesh": len(m)}


def run_gates() -> dict:
    out = build_alpha_rooftop()
    assert abs(out["central_alpha"] - 0.15) < 1e-9, "central alpha must be 15%"
    df = pd.read_parquet(OUT)
    # future >= current per mesh at every alpha (floor identity)
    for a in ALPHAS:
        tag = f"a{int(a*100):03d}"
        assert (df[f"future_cap_kw_{tag}"] >= df["current_cap_kw"] - 1e-9).all(), f"floor broken at {tag}"
        assert (df[f"increment_kw_{tag}"] >= -1e-9).all(), f"negative increment at {tag}"
    return out


def main() -> None:
    out = run_gates()
    print(f"[05b] alpha_rooftop: central alpha={out['central_alpha']:.0%} | current {out['cur_GW']:.2f} GW "
          f"-> central future {out['central_future_GW']:.2f} GW | {out['n_mesh']:,} meshes")
    print("05b_alpha_rooftop: ALL GATES PASS")


if __name__ == "__main__":
    main()
