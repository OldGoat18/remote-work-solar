"""Household-state CELL layer (plan §14.3 / §13.1) — CLEAN official-source rebuild, deterministic IPF.

Ported verbatim (numerics unchanged) from the source
`configured asset`.

Replaces the RETIRED v60 kWh-derived household probe (rejected by configured asset
as load-reverse-engineered / circular — its weights were a baseline-load proxy, forbidden by non-identifiability).
This layer reads ONLY the H0 whitelist:
  - T001108 census worker / non-worker HOUSEHOLD counts (via mesh_state: H_worker, H_nonworker)
  - housing_mesh_panel detached/apartment dwelling share + heat-loss (census-derived housing, whitelisted)
  - mesh_state's non-circular WFH propensity (occupation T001193 × survey telework rate × intensity)
NO kWh, NO load-reverse-engineered weights, NO retired version artifacts.

It synthesises the joint household-state distribution per mesh by DETERMINISTIC IPF over the official margins
(worker-status × dwelling), then reconciles the cell aggregate to mesh_state EXACTLY (S1 bit-identity preserved —
the cells are a faithful decomposition of the mesh totals). The cells expose the heterogeneity the mesh-average
hides: DWELLING-stratified heat-loss (detached ≫ apartment, conserved to the mesh mean) and WORKER-STATUS
home-presence (non-worker households are home, worker households are the WFH-marginal). Promotion to drive the
component HEADLINE (which would move S1) is a separate decision; here it is the clean, red-line-safe substrate.

    PYTHONPATH=src python -m wlpr.wfh_state.household_cells
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location
import os

import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path
from wlpr.wfh_state import state_layer as SL

OUT = derived_path(_asset_location('asset_0055'))
HOUSE = derived_path(_asset_location('asset_0107'), "housing_mesh_panel.parquet")
DET_APT_HL_RATIO = 1.30     # building-physics prior: detached heat-loss / apartment heat-loss (conserved to mesh mean)
Q_HOME = 0.85               # daytime home-presence allowance for a non-working member (H0 caliber, same as state_layer)


def ipf_2x2(row_margin, col_margin, n_iter=50, tol=1e-12):
    """Deterministic IPF of a 2×2 joint from its row & col margins (seed = independence). Vectorised over meshes:
    row_margin (M,2), col_margin (M,2) with equal row/col totals -> joint (M,2,2)."""
    rt = np.asarray(row_margin, float); ct = np.asarray(col_margin, float)
    X = np.einsum("mi,mj->mij", rt, ct) / np.clip(rt.sum(1), 1e-12, None)[:, None, None]   # independence seed
    for _ in range(n_iter):
        rs = X.sum(2); X *= np.divide(rt, rs, out=np.ones_like(rs), where=rs > 0)[:, :, None]
        cs = X.sum(1); X *= np.divide(ct, cs, out=np.ones_like(cs), where=cs > 0)[:, None, :]
        if np.abs(X.sum(2) - rt).max() < tol and np.abs(X.sum(1) - ct).max() < tol:
            break
    return X


def build_cells():
    """mesh × {worker-status × dwelling} household-state cells from clean official margins, reconciled to mesh_state."""
    ms = SL.mesh_state()
    hp = pd.read_parquet(HOUSE)[["mesh", "det_sh"]].copy()
    hp["mesh"] = hp.mesh.astype("int64").astype(str)
    m = ms.merge(hp, on="mesh", how="left")
    det = np.clip(pd.to_numeric(m.det_sh, errors="coerce").fillna(m.det_sh.median()).values, 0.0, 1.0)
    apt = 1.0 - det
    Hw = m.H_worker.values; Hn = m.H_nonworker.values; Ht = np.clip(Hw + Hn, 1e-9, None)

    # IPF the worker-status × dwelling joint per mesh (margins: [Hw,Hn] and [det,apt]×Ht)
    row = np.column_stack([Hw, Hn])
    col = np.column_stack([det * Ht, apt * Ht])
    J = ipf_2x2(row, col)                         # (M,2,2): [worker/nonworker] × [detached/apartment]

    # dwelling-stratified heat-loss, CONSERVED to the mesh mean (det = ratio × apt; det·share + apt·share = mesh)
    hl = m.heatloss.values
    apt_hl = hl / np.clip(det * DET_APT_HL_RATIO + apt, 1e-9, None)
    det_hl = DET_APT_HL_RATIO * apt_hl

    # per-mesh WFH split: only WORKER households carry WFH; split across dwelling by the worker-row of J
    Np = m.N_persons.values; Hwfh = m.H_wfh.values; pi = m.pi_worker0.values
    vm = m.vacancy_margin.values                        # mesh sharing margin (carried to every cell -> S1 bit-identity)
    f_nw = np.where(Hw > 0, np.clip((m.p_home0.values * Ht - Hn * Q_HOME) / np.clip(Hw * Q_HOME, 1e-9, None), 0, 1), 0.0)
    rows = []
    DW = ["detached", "apartment"]; WS = ["worker", "nonworker"]
    worker_hh_share = np.divide(J[:, 0, :], np.clip(J[:, 0, :].sum(1, keepdims=True), 1e-9, None))   # dwelling split of worker hh
    for di, dw in enumerate(DW):
        whh = J[:, 0, di]; nhh = J[:, 1, di]            # worker / nonworker households in this dwelling
        wshare = np.divide(whh, np.clip(Hw, 1e-9, None))
        rows.append(pd.DataFrame({
            "mesh": m.mesh.values, "pref": m.pref.values, "worker_status": "worker", "dwelling_type": dw,
            "households": whh, "wfh_persons": Np * wshare, "wfh_households": Hwfh * wshare,
            "heatloss": (det_hl if dw == "detached" else apt_hl),
            "vacancy_margin": vm,                       # headline driver = mesh sharing margin (bit-identity)
            "p_home0": f_nw * Q_HOME,                   # cell-native worker home-presence (diagnostic / fidelity sidecar)
        }))
        rows.append(pd.DataFrame({
            "mesh": m.mesh.values, "pref": m.pref.values, "worker_status": "nonworker", "dwelling_type": dw,
            "households": nhh, "wfh_persons": np.zeros(len(m)), "wfh_households": np.zeros(len(m)),
            "heatloss": (det_hl if dw == "detached" else apt_hl),
            "vacancy_margin": vm,
            "p_home0": np.full(len(m), Q_HOME),
        }))
    cells = pd.concat(rows, ignore_index=True)
    return cells, ms, J, row, col


def main():
    print("[household_cells] CLEAN official-source household-state cells (IPF over census worker-status × dwelling).")
    print("  sources: T001108 worker/non-worker households · housing_mesh_panel dwelling+heatloss · T001193 WFH propensity")
    print("  NO kWh / NO load-reverse-engineered weights / NO v60 retired probe (H0 whitelist only).\n")
    cells, ms, J, row, col = build_cells()

    # (1) IPF margins conserved
    rresid = float(np.abs(J.sum(2) - row).max()); cresid = float(np.abs(J.sum(1) - col).max())
    print(f"  IPF margins: worker-status max|Δ|={rresid:.2e}  dwelling max|Δ|={cresid:.2e}  (<1e-6 -> reconciled)")

    # (2) reconciliation to mesh_state (S1-safe): cell aggregate reproduces N_persons, H_wfh, and heatloss mean
    agg = cells.groupby("mesh").agg(N=("wfh_persons", "sum"), H=("wfh_households", "sum")).reset_index()
    chk = agg.merge(ms[["mesh", "N_persons", "H_wfh", "heatloss"]], on="mesh")
    rN = float((chk.N - chk.N_persons).abs().max()); rH = float((chk.H - chk.H_wfh).abs().max())
    # heat-loss conservation: household-weighted dwelling heat-loss == mesh heatloss
    hlw = cells.groupby("mesh").apply(lambda g: np.average(g.heatloss, weights=g.households) if g.households.sum() > 0 else np.nan)
    hl_resid = float(np.nanmax(np.abs(hlw.reindex(chk.mesh).values - chk.heatloss.values)))
    print(f"  reconcile to mesh_state: N_persons max|Δ|={rN:.2e}  H_wfh max|Δ|={rH:.2e}  heatloss-mean max|Δ|={hl_resid:.2e}")
    s1_safe = max(rN, rH, hl_resid) < 1e-6

    # (3) heterogeneity exposed
    dw = cells.groupby("dwelling_type").apply(lambda g: pd.Series({
        "wfh_households": g.wfh_households.sum(),
        "heatloss_w": np.average(g.heatloss, weights=g.households) if g.households.sum() > 0 else 0.0}))
    print("\n  [housing] dwelling-stratified heat-loss (detached ≫ apartment; conserved to mesh mean):")
    print(dw.round(3).to_string())
    p0 = cells.groupby("worker_status").apply(lambda g: np.average(g.p_home0, weights=np.clip(g.households, 1e-9, None)))
    print(f"\n  [home-presence] worker-status p_home0: worker={p0['worker']:.3f}  nonworker={p0['nonworker']:.3f} "
          f"(worker households are the WFH-marginal; non-worker are already home)")

    out = cells.groupby(["mesh", "pref", "worker_status", "dwelling_type"]).agg(
        households=("households", "sum"), wfh_persons=("wfh_persons", "sum"),
        wfh_households=("wfh_households", "sum"), heatloss=("heatloss", "mean"),
        vacancy_margin=("vacancy_margin", "mean"), p_home0=("p_home0", "mean")).reset_index()
    out.to_parquet(os.path.join(OUT, "household_state_cells_v2.parquet"), index=False)
    print(f"\n[saved] {os.path.join(OUT, 'household_state_cells_v2.parquet')} ({len(out):,} mesh×worker-status×dwelling cells)")

    det_hl = float(dw.loc["detached", "heatloss_w"]); apt_hl = float(dw.loc["apartment", "heatloss_w"])
    ok = rresid < 1e-6 and cresid < 1e-6 and s1_safe and det_hl > apt_hl
    print(f"\n=== HOUSEHOLD-CELLS VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(clean official-source IPF; reconciles to mesh_state -> S1-safe; dwelling/worker heterogeneity exposed) ===")
    if not ok:
        raise SystemExit("household_cells reconciliation/heterogeneity failed")


if __name__ == "__main__":
    main()
