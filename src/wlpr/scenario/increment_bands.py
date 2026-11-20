"""Step 10c — the partial-identification INCREMENT band (Layer B Monte-Carlo) for the WFH scenario relief.

Ported (numerics verbatim) from the LAYER B half of the source
`_maintainer/builders/build_uncertainty_full_diag.py`, producing ONLY `scenario_increment_bands.csv`.

DROPPED on port: Layer A (validation predictive bands by grain), the V1/V2/B1/B2/B3 figures, and the 13-assertion
verify checklist (figure/diagnostic surface, not a product). Also DROPPED with the BTM removal: the owner_BTM and
future_BTM channels — the BTM self-consumption split belongs to the excluded eta/least PV/BTM user-state twin.
The surviving band channel is `nonowner_relief_GWh` (the grid-level WFH PV-relief increment).

THE ONE INVARIANT (RED LINE): TWO LAYERS, never merged.
  LAYER A  VALIDATION band : split-conformal on real residuals (lives in the validation cluster; here it enters
                             ONLY as a single labelled outer FLOOR factor q_val).
  LAYER B  INCREMENT band  : Monte-Carlo over DEFINITIONAL assumptions -> the S1-S10 relief increment RANGE.
                             The increment is NON-IDENTIFIABLE from aggregate LV; its range is dominated by
                             DEFINITIONAL assumption width no data can shrink.

q_val_saitama / q_val_pooled / sigma_post are read from the validation cluster's conformal scorecards
(out_path("scorecards", w1_conformal_coverage.csv / w1_fold_coefficient_stability.csv)).

Output: derived_path("scenario_s1s10", "scenario_increment_bands.csv").
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path, out_path

IDX = derived_path("scenario_s1s10", "scenario_indices_s1s10.csv")
COV = out_path("scorecards", "w1_conformal_coverage.csv")
FOLD = out_path("scorecards", "w1_fold_coefficient_stability.csv")
OUT = derived_path("scenario_s1s10", "scenario_increment_bands.csv")
ORDER = [f"S{i}" for i in range(1, 11)]
# BTM channels are excluded (eta/least engine). The surviving partial-ID band channel is grid-level relief.
CHANNELS = ["nonowner_relief_GWh"]
PV_OF = {"nonowner_relief_GWh": "future_pv_ai"}


def _ladders(idx):
    """per (channel, pv) sorted (mult, value) ladders for piecewise-linear interpolation through all 10 nodes."""
    out = {}
    base = idx[idx.scenario_id.isin(ORDER)].copy()
    for pv in ["current", "future_pv_ai"]:
        sub = base[base.pv_scenario == pv].sort_values("wfh_energy_multiplier")
        for ch in CHANNELS:
            if ch in sub.columns:
                out[(ch, pv)] = (sub.wfh_energy_multiplier.values.astype(float), sub[ch].values.astype(float))
    return out


def _tri(rng, lo, mode, hi, n):
    lo, hi = float(lo), float(hi); mode = float(np.clip(mode, lo, hi))
    if hi - lo < 1e-9:
        return np.full(n, mode)
    return rng.triangular(lo, mode, hi, n)


def layer_b(q_val_saitama, q_val_pooled, sigma_post):
    from wlpr.wfh_state import scenario_downscaling as SD  # deferred: belongs to the wfh_state cluster
    idx = pd.read_csv(IDX)
    lad = _ladders(idx)
    cache = {t: SD.base_rates(t) for t in SD.RATE}
    N = 20000; rng = np.random.default_rng(20260619)
    chans = CHANNELS
    pv_of = PV_OF
    recs = []
    HW_FLOOR = 0.005   # degeneracy guard ONLY (keeps the triangular non-degenerate if target==current); NOT a prior edge

    def mc_mult(table, freq, raises, tier):
        """N independent draws of the multiplier under perturbed definitional assumptions."""
        # frequency intensity prior
        if freq == "observed":
            Ifr = np.full(N, SD.I_FREQ[freq])
        elif freq == "high":
            Ifr = _tri(rng, 0.60, 0.70, 0.70, N)                  # ceiling, one-sided
        elif freq == "2-3 days":
            Ifr = _tri(rng, 0.40, 0.50, 0.60, N)
        else:  # 3-4 days
            Ifr = _tri(rng, 0.60, 0.70, 0.80, N)
        if table is None:
            pr = np.ones(N)
        else:
            rates, workers = cache[table]; codes = list(rates)
            w = np.array([workers[c] for c in codes]); base = np.array([rates[c] for c in codes])
            denom = float((w * base).sum()); idxc = {c: i for i, c in enumerate(codes)}
            Rdraw = np.tile(base, (N, 1))
            for c, target in raises.items():
                cur = rates.get(c, 0.0); hw = max(0.5 * (target - cur), HW_FLOOR)
                if tier == "upper-bound":
                    samp = _tri(rng, target - hw, target, target, N)       # downward-only ceiling
                else:
                    samp = _tri(rng, target - hw, target, min(target + hw, 1.0), N)
                Rdraw[:, idxc[c]] = np.maximum(base[idxc[c]], samp)         # engine max(rate,raise) clamp
            pr = (Rdraw @ w) / denom
        return pr * (Ifr / SD.I_BASE), pr

    def corner_mult(table, freq, raises, tier):
        lof = {"observed": SD.I_FREQ["observed"], "high": 0.60, "2-3 days": 0.40, "3-4 days": 0.60}[freq]
        hif = {"observed": SD.I_FREQ["observed"], "high": 0.70, "2-3 days": 0.60, "3-4 days": 0.80}[freq]

        def pr_at(edge):
            if table is None:
                return 1.0
            rates, workers = cache[table]; num = 0.0; den = 0.0
            for c in rates:
                w = workers[c]; base = rates[c]; den += w * base
                if c in raises:
                    target = raises[c]; cur = rates.get(c, 0.0); hw = max(0.5*(target-cur), HW_FLOOR)
                    if tier == "upper-bound":
                        val = target if edge == "hi" else target - hw
                    else:
                        val = (min(target + hw, 1.0)) if edge == "hi" else (target - hw)
                    num += w * max(base, val)
                else:
                    num += w * base
            return num / den
        return pr_at("lo") * (lof / SD.I_BASE), pr_at("hi") * (hif / SD.I_BASE)

    for (sid, name, grp, lever, freq, tier, table, raises) in SD.SCEN:
        mult_pub = float(idx[idx.scenario_id == sid].wfh_energy_multiplier.iloc[0])
        if sid == "S1":
            mult_mc = np.full(N, mult_pub); pr = np.ones(N); cml, cmh = mult_pub, mult_pub
        else:
            mult_mc, pr = mc_mult(table, freq, raises, tier)
            cml, cmh = corner_mult(table, freq, raises, tier)
        n_raised = 0 if table is None else len(raises)
        for ch in chans:
            pv = pv_of[ch]
            if (ch, pv) not in lad:
                continue
            xm, ym = lad[(ch, pv)]
            central = float(idx[(idx.scenario_id == sid) & (idx.pv_scenario == pv)][ch].iloc[0])
            if central == 0:
                continue
            btm_mc = np.interp(mult_mc, xm, ym)
            if sid == "S1":
                d_lo = d_hi = 1.0; def_p10 = def_p90 = central
            else:
                def_p10 = float(np.percentile(btm_mc, 10)); def_p90 = float(np.percentile(btm_mc, 90))
                d_lo = def_p10 / central; d_hi = def_p90 / central
                if tier == "upper-bound":
                    d_hi = 1.0; def_p90 = central                              # ceiling: no upside
            corner_L = float(np.interp(cml, xm, ym)); corner_U = float(np.interp(cmh, xm, ym))
            if tier == "upper-bound":
                corner_U = central
            q = q_val_saitama
            # quadrature (log) headline + direct-multiplication (safer outer)
            L_quad = central * np.exp(-np.sqrt(np.log(d_lo)**2 + np.log(1 - q)**2))
            U_quad = central * np.exp(+np.sqrt(np.log(max(d_hi, 1e-9))**2 + np.log(1 + q)**2))
            L_dir = central * d_lo * (1 - q); U_dir = central * d_hi * (1 + q)
            if tier == "upper-bound":      # CEILING: validation band adds DOWNSIDE only, never upside
                U_quad = central; U_dir = central
            shape = ("symmetric_validation_only" if tier == "observed" else
                     "ceiling_downward_only" if tier == "upper-bound" else
                     "policy_counterfactual_nonforecast" if tier == "policy-counterfactual" else "two_sided")
            mc_w = def_p90 - def_p10; cn_w = corner_U - corner_L
            recs.append(dict(scenario_id=sid, pv_scenario=pv, channel=ch, central=round(central, 3),
                             def_p10=round(def_p10, 3), def_p90=round(def_p90, 3),
                             d_lo=round(d_lo, 4), d_hi=round(d_hi, 4), q_val=round(q, 4),
                             q_val_source="G2_forward_Saitama_worst",
                             L_quadrature=round(float(L_quad), 3), U_quadrature=round(float(U_quad), 3),
                             L_direct_mult=round(float(L_dir), 3), U_direct_mult=round(float(U_dir), 3),
                             corner_L=round(corner_L, 3), corner_U=round(corner_U, 3),
                             evidence_tier=tier, band_shape=shape, n_raised_categories=n_raised,
                             mc_vs_corner_ratio=(round(mc_w / cn_w, 3) if cn_w > 1e-9 else None),
                             caveat_flag=("NON_FORECAST" if tier == "policy-counterfactual" else
                                          "CEILING" if tier == "upper-bound" else "")))
    B = pd.DataFrame(recs)
    B.to_csv(OUT, index=False)
    print(f"[saved] {OUT}  ({len(B)} rows; channel={CHANNELS})")
    return B


def run_gates(B):
    fails = []

    def ck(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            fails.append(name)

    ck("1 layers separated (q_val is a single labelled column, not merged into the increment)",
       set(["q_val", "q_val_source"]).issubset(B.columns) and B.q_val_source.eq("G2_forward_Saitama_worst").all())
    ck("2 both envelopes present (quadrature + direct)",
       set(["L_quadrature", "U_quadrature", "L_direct_mult", "U_direct_mult"]).issubset(B.columns))
    s1 = B[B.scenario_id == "S1"]
    ck("3 S1 immovable anchor (def band = 0)",
       bool((s1.def_p10 == s1.central).all() and (s1.def_p90 == s1.central).all()) if len(s1) else False)
    ub = B[B.evidence_tier == "upper-bound"]
    ck("4 upper-bound ceilings add no upside (U_quad <= central)",
       bool((ub.U_quadrature <= ub.central + 1e-6).all()) if len(ub) else True)
    ck("5 BTM-free band (no owner_BTM / future_BTM channel)",
       not B.channel.str.contains("_BTM").any(), f"channels={sorted(B.channel.unique())}")
    if fails:
        raise SystemExit("increment_bands: GATE FAILS -> " + ", ".join(fails))


def main():
    cov = pd.read_csv(COV)
    q_val_saitama = float(cov[(cov.heldout == "ALL") & (cov.pref == "Saitama")].q80_band.iloc[0])
    q_val_pooled = float(cov[cov.heldout == "ALL"].q80_band.mean())
    fold = pd.read_csv(FOLD); cvm = fold.groupby("coef").cv_abs.mean()
    sigma_post = float(np.sqrt(cvm.get("a_base", 0)**2 + cvm.get("b_hdd", 0)**2))   # c_cdd EXCLUDED
    print(f"[floor] q_val Saitama-worst={q_val_saitama:.4f} (pooled {q_val_pooled:.4f}); "
          f"sigma_post(a_base,b_hdd only)={sigma_post:.4f}")
    B = layer_b(q_val_saitama, q_val_pooled, sigma_post)
    print("=== GATES ===")
    run_gates(B)
    print("10c_increment_bands: ALL GATES PASS (partial-ID relief increment band, two-layer separated)")


if __name__ == "__main__":
    main()
