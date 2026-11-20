"""R3 benefit-distribution indicators — the inequality / targeting / clustering suite for WFH's spatial
benefit (ported from build_benefit_distribution.py; built from the R3 per-mesh parquet, no rep-day, no system
pre-aggregate).

R3 is the BENEFIT-DISTRIBUTION result: WFH's system benefit = the future-PV midday surplus it absorbs locally
(realized reverse-pressure relief). This builder characterises that benefit's spatial distribution:

  (1) HOW CONCENTRATED is the benefit?   Gini ; Atkinson A(epsilon) ; top-1% / top-decile share ;
                                         bottom-half share ; Lorenz (curve in the figure).
  (2) Does it go WHERE IT IS NEEDED?     Concentration index CI ranked by NEED (= PV surplus) and dissimilarity D
                                         vs the surplus field. Read on the WFH SUPPLY (behaviour), NOT the
                                         realized benefit (which is mechanically confined to surplus meshes).
                                         Honest targeting is CI(WFH vs need) < 0 with dissimilarity -> 1.
  (3) WHERE does the inequality come from?  Theil index decomposed BETWEEN vs WITHIN municipality.
  (SI) Is it spatially CLUSTERED?        Moran's I on the regular 500m grid (rook contiguity from the mesh codes).

Output: outputs/scorecards/r3_benefit_distribution_scorecard.csv
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path, input_path, out_path

MESHP = derived_path("social", "r3_spatial_matching_mesh.parquet")
OUTS = out_path("scorecards", "r3_benefit_distribution_scorecard.csv")


def _munic_path():
    """mesh -> municipality (REPOS rooftop bridge), a staged input (built by the excluded PV subsystem)."""
    return input_path("repos_mesh_munic_bpg", must_exist=False)


def _gini(x):
    x = np.sort(np.asarray(x, float)); x = x[x >= 0]
    if len(x) == 0 or x.sum() <= 0:
        return float("nan")
    n = len(x); cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _atkinson(x, eps=0.5):
    """A(eps) = 1 - EDE/mean, EDE = [ (1/n) sum x^(1-eps) ]^(1/(1-eps)). eps>0 = inequality aversion."""
    x = np.asarray(x, float); x = x[x >= 0]; mu = x.mean()
    if len(x) == 0 or mu <= 0:
        return float("nan")
    if abs(eps - 1.0) < 1e-9:
        if (x <= 0).any():
            return 1.0
        ede = np.exp(np.log(x).mean())
    else:
        ede = (np.mean(x ** (1 - eps))) ** (1 / (1 - eps))
    return float(1 - ede / mu)


def _top_share(x, q):
    x = np.sort(np.asarray(x, float))[::-1]
    if x.sum() <= 0:
        return float("nan")
    k = max(1, int(round(q * len(x))))
    return float(x[:k].sum() / x.sum())


def _bottom_share(x, q):
    x = np.sort(np.asarray(x, float))
    if x.sum() <= 0:
        return float("nan")
    k = max(1, int(round(q * len(x))))
    return float(x[:k].sum() / x.sum())


def _concentration_index(b, need):
    """CI of benefit b when meshes are ranked by NEED ascending. CI = 2*cov(b, fractional-need-rank)/mean(b).
    >0: b concentrates in HIGH-need meshes (well targeted); <0: in LOW-need meshes (regressive)."""
    b = np.asarray(b, float); mu = b.mean()
    if mu <= 0:
        return float("nan")
    fr = pd.Series(np.asarray(need, float)).rank(method="average").values
    fr = (fr - 0.5) / len(fr)
    return float(2 * np.cov(b, fr, bias=True)[0, 1] / mu)


def _dissimilarity(a, b):
    """Index of dissimilarity between two spatial distributions: 0 = identical, 1 = fully disjoint."""
    a = np.asarray(a, float).clip(min=0); b = np.asarray(b, float).clip(min=0)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def _theil(x):
    x = np.asarray(x, float); N = len(x); X = x.sum()
    if X <= 0:
        return float("nan")
    p = x[x > 0] / X
    return float(np.sum(p * np.log(p * N)))


def _theil_between_share(x, groups):
    x = np.asarray(x, float); N = len(x); X = x.sum()
    if X <= 0:
        return float("nan")
    df = pd.DataFrame({"x": x, "g": groups})
    g = df.groupby("g").agg(Xg=("x", "sum"), ng=("x", "size"))
    sg = (g.Xg / X).values; fg = (g.ng / N).values
    m = sg > 0
    t_between = float(np.sum(sg[m] * np.log(sg[m] / fg[m])))
    t_total = _theil(x)
    return t_between / t_total if t_total and t_total > 0 else float("nan")


def _grid_index(mesh):
    s = str(mesh).zfill(9)
    y1 = int(s[0:2]); x1 = int(s[2:4]); y2 = int(s[4]); x2 = int(s[5]); y3 = int(s[6]); x3 = int(s[7]); q = int(s[8])
    y4 = (q - 1) // 2; x4 = (q - 1) % 2
    return ((y1 * 8 + y2) * 10 + y3) * 2 + y4, ((x1 * 8 + x2) * 10 + x3) * 2 + x4   # exact 500m grid indices


def _morans_i(vals, meshes):
    """Global Moran's I with rook contiguity on the regular 500m grid (adjacency from the JIS mesh codes)."""
    vals = np.asarray(vals, float)
    coords = [_grid_index(m) for m in meshes]
    idx = {c: k for k, c in enumerate(coords)}
    z = vals - vals.mean()
    num = 0.0; W = 0
    for k, (iy, ix) in enumerate(coords):
        for d in ((iy + 1, ix), (iy - 1, ix), (iy, ix + 1), (iy, ix - 1)):
            j = idx.get(d)
            if j is not None:
                num += z[k] * z[j]; W += 1
    denom = float(np.sum(z * z))
    return float((len(vals) / W) * (num / denom)) if W > 0 and denom > 0 else float("nan")


def build():
    f = pd.read_parquet(MESHP)
    R = f.realized_S1_MWh.values        # realized benefit (the R3 subject)
    S = f.supply_S1_MWh.values          # WFH supply / behaviour
    D = f.demand_surplus_MWh.values     # PV surplus = NEED
    meshes = f.mesh.values

    munic_p = _munic_path()
    mm = pd.read_parquet(munic_p)
    dom = mm.sort_values("B_PG", ascending=False).drop_duplicates("mesh")[["mesh", "munic"]]
    munic = f[["mesh"]].merge(dom, on="mesh", how="left").munic.fillna("NA").values

    rows = []
    for name, x in [("realized_benefit", R), ("wfh_supply", S)]:
        rows.append(dict(
            quantity=name,
            gini=round(_gini(x), 4),
            atkinson_eps0p5=round(_atkinson(x, 0.5), 4),
            top_1pct_share=round(_top_share(x, 0.01), 4),
            top_decile_share=round(_top_share(x, 0.10), 4),
            bottom_half_share=round(_bottom_share(x, 0.50), 4),
            CI_vs_surplus_need=round(_concentration_index(x, D), 4),
            dissimilarity_vs_surplus=round(_dissimilarity(x, D), 4),
            theil_total=round(_theil(x), 4),
            theil_between_munic_share=round(_theil_between_share(x, munic), 4),
            morans_I=round(_morans_i(x, meshes), 4),
        ))
    sc = pd.DataFrame(rows)
    sc.to_csv(OUTS, index=False)
    print(f"[saved] {OUTS}")
    print(sc.to_string(index=False))

    rb = sc[sc.quantity == "realized_benefit"].iloc[0]
    ws = sc[sc.quantity == "wfh_supply"].iloc[0]
    print("\n=== READING ===")
    print(f"  concentration : benefit Gini {rb.gini}, Atkinson(0.5) {rb.atkinson_eps0p5}, top-decile {rb.top_decile_share:.0%}, "
          f"bottom-half {rb.bottom_half_share:.2%} (half the grids get ~nothing)")
    print(f"  targeting     : CI(WFH vs need) {ws.CI_vs_surplus_need:+.3f} (<0 = WFH concentrates in LOW-surplus meshes); "
          f"dissimilarity(WFH,surplus) {ws.dissimilarity_vs_surplus:.2f}")
    print(f"                  [benefit CI {rb.CI_vs_surplus_need:+.2f} is MECHANICAL: relief requires surplus (R<=D); do not read as targeting]")
    print(f"  decomposition : Theil between-municipality share -- WFH {ws.theil_between_munic_share:.0%} (a between-city phenomenon), "
          f"benefit {rb.theil_between_munic_share:.0%} (mostly WITHIN city = hyperlocal co-location)")
    print(f"  clustering    : Moran's I -- WFH {ws.morans_I} (strongly clustered), benefit {rb.morans_I}")
    return sc


def run_gates(sc):
    rb = sc[sc.quantity == "realized_benefit"].iloc[0]
    ws = sc[sc.quantity == "wfh_supply"].iloc[0]
    print("=== GATES ===")
    fails = []

    def ck(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
        if not ok:
            fails.append(n)

    ck("1 benefit Gini in (0,1) and matches the R3 scorecard (~0.78)", 0.70 < rb.gini < 0.85, f"{rb.gini}")
    ck("2 Atkinson in (0,1), top-decile share > 0.5 (benefit highly concentrated)",
       0 < rb.atkinson_eps0p5 < 1 and rb.top_decile_share > 0.5, f"A {rb.atkinson_eps0p5} / top10 {rb.top_decile_share}")
    ck("3 WFH behaviour is REGRESSIVE vs need: CI(WFH vs surplus) < 0", ws.CI_vs_surplus_need < 0,
       f"{ws.CI_vs_surplus_need}")
    ck("4 benefit is spatially DISJOINT from need: dissimilarity(benefit,surplus) > 0.3",
       rb.dissimilarity_vs_surplus > 0.3, f"{rb.dissimilarity_vs_surplus}")
    ck("5 Theil between-municipality share in (0,1)", 0 < rb.theil_between_munic_share < 1,
       f"{rb.theil_between_munic_share}")
    ck("6 benefit is spatially CLUSTERED: Moran's I > 0", rb.morans_I > 0, f"{rb.morans_I}")
    if fails:
        raise SystemExit("benefit_distribution: GATE FAILS -> " + ", ".join(fails))
    print("benefit_distribution: ALL GATES PASS")


def main():
    sc = build()
    run_gates(sc)


if __name__ == "__main__":
    main()
