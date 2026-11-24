"""Fig 3 — R3 spatial-matching figure (the paper's CLIMAX visual).

Ported from the source spatial-matching figure implementation. Built STRICTLY from the R3 mesh parquet +
scorecards; no rep-day, no system pre-aggregate. Nature-style: panel letters a-f, no titles, all-English,
despined.

THE FINDING (mesh x date x slot, S1 = current census-driven WFH geography):
  WFH co-locates with PV GENERATION (rho 0.85) but ANTI-locates with the PV SURPLUS that actually needs
  absorbing (rho -0.21 over surplus-bearing meshes). Surplus = generation - load; WFH IS load, so the top WFH
  decile holds 47% of WFH and 23% of generation but only ~1.4% of surplus. Only ~3.8% of surplus is absorbed by
  same-mesh WFH; the 96% remainder defines where storage/aggregation value falls. The realized relief concentrates
  MORE than the WFH behaviour itself (Gini 0.70 -> 0.78). The anti-location is LOAD-mediated (mobility-free).

Geography is real: JIS 500m mesh codes -> lat/lon centroid (inverse of repos_mesh_munic.latlon_to_mesh500).

Panels:
  a  WFH midday-increment field over Tokyo (where the behaviour is)
  b  PV surplus field over Tokyo  (where the absorbable energy is -- visibly disjoint from a)
  c  by WFH decile: share of WFH / PV generation / PV surplus (the quantified flip)
  d  Lorenz curves: realized benefit vs WFH supply -- the benefit is extremely concentrated
  e  DECISIVE 3-way split of the future-PV midday surplus -- absorbed now / reachable if co-located / storage wall
  f  co-location vs mismatch map

Output: configured asset,pdf}.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from wlpr.io.paths import derived_path, out_path

MESHP = derived_path(_asset_location('asset_0056'), "r3_spatial_matching_mesh.parquet")
SCORE = out_path(_asset_location('asset_0058'), "r3_spatial_matching_scorecard.csv")
REALLOC = out_path(_asset_location('asset_0058'), "r3_spatial_reallocation_scorecard.csv")   # spatial-vs-magnitude decomposition
BDIST = out_path(_asset_location('asset_0058'), "r3_benefit_distribution_scorecard.csv")      # inequality / concentration indicators
OUTDIR = os.path.dirname(out_path(_asset_location('asset_0059'), _asset_location('asset_0056'), "fig_spatial_matching_r3.png"))

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333", "axes.unicode_minus": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 300,
})
C_WFH, C_SURP, C_GEN = "#3B6EA5", "#55A868", "#C44E52"
TOKYO_LAT = 35.7


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _panel(ax, letter, dx=-0.085):
    ax.text(dx, 1.035, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")


def mesh_to_latlon(mesh):
    """Inverse of repos_mesh_munic.latlon_to_mesh500 -> centroid (lat, lon) of each JIS 500m mesh."""
    s = pd.Series(mesh).astype(str).str.zfill(9)
    y1 = s.str[0:2].astype(int); x1 = s.str[2:4].astype(int)
    y2 = s.str[4].astype(int); x2 = s.str[5].astype(int)
    y3 = s.str[6].astype(int); x3 = s.str[7].astype(int)
    q = s.str[8].astype(int); y4 = (q - 1) // 2; x4 = (q - 1) % 2
    lat = (y1 + y2 / 8 + y3 / 80 + y4 / 160 + 1 / 320) / 1.5
    lon = 100 + x1 + x2 / 8 + x3 / 80 + x4 / 160 + 1 / 320
    return lat.values, lon.values


def _map(ax, lat, lon, val, cmap, label):
    """Hotspot map: light-grey basemap (full Tokyo outline) + value>0 meshes coloured by gamma-emphasised
    within-field percentile rank, so only true hotspots saturate and the spatial pattern (not just 'full of
    colour') is what reads. Meshes with no value stay grey -- e.g. the surplus-free urban core in panel b."""
    ax.scatter(lon, lat, s=2.2, marker="s", color="#ececec", linewidths=0, rasterized=True)   # basemap = outline
    pos = val > 0
    pr = pd.Series(val[pos]).rank(pct=True).values ** 2.4                                      # gamma -> hotspots pop
    order = np.argsort(pr)                                                                     # draw hotspots last
    sc = ax.scatter(lon[pos][order], lat[pos][order], c=pr[order], s=2.8, marker="s", cmap=cmap,
                    vmin=0, vmax=1, linewidths=0, rasterized=True)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(TOKYO_LAT)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.040, pad=0.02, ticks=[0, 1.0])
    cb.ax.set_yticklabels(["low", "high"], fontsize=7)
    cb.outline.set_visible(False)
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=8.8, va="top", ha="left", fontweight="bold")


def build():
    os.makedirs(OUTDIR, exist_ok=True)
    f = pd.read_parquet(MESHP)
    sc = pd.read_csv(SCORE)
    s1 = sc[sc.arm == "S1"].iloc[0]
    rc = pd.read_csv(REALLOC)
    rc = rc[rc.scope == "all_slots"].iloc[0]
    bd = pd.read_csv(BDIST).set_index("quantity")
    bd_b, bd_s = bd.loc["realized_benefit"], bd.loc["wfh_supply"]

    lat, lon = mesh_to_latlon(f.mesh.values)
    S = f.supply_S1_MWh.values            # WFH midday increment (behaviour / absorption capacity)
    D = f.demand_surplus_MWh.values       # PV surplus (the energy to absorb)
    G = f.generation_MWh.values           # PV generation
    R = f.realized_S1_MWh.values          # realized benefit (relief)

    fig = plt.figure(figsize=(13.8, 8.7))
    gs = fig.add_gridspec(2, 3, left=0.045, right=0.985, top=0.95, bottom=0.085, wspace=0.22, hspace=0.26)

    # ---- a: WFH midday-increment field ----
    ax = fig.add_subplot(gs[0, 0]); _map(ax, lat, lon, S, "Blues", "WFH midday increment"); _panel(ax, "a")
    # ---- b: PV surplus field (disjoint from a) ----
    ax = fig.add_subplot(gs[0, 1]); _map(ax, lat, lon, D, "Greens", "PV midday surplus"); _panel(ax, "b")

    # ---- c: by-WFH-decile share of WFH / generation / surplus ----
    ax = fig.add_subplot(gs[0, 2])
    dfd = pd.DataFrame({"S": S, "G": G, "D": D})
    dfd["dec"] = pd.qcut(dfd.S.rank(method="first"), 10, labels=False) + 1
    g = dfd.groupby("dec").agg(S=("S", "sum"), G=("G", "sum"), D=("D", "sum"))
    x = g.index.values
    ax.plot(x, 100 * g.S / g.S.sum(), "-o", color=C_WFH, lw=1.8, ms=3.5, label="WFH increment")
    ax.plot(x, 100 * g.G / g.G.sum(), "-s", color=C_GEN, lw=1.8, ms=3.2, label="PV generation")
    ax.plot(x, 100 * g.D / g.D.sum(), "-^", color=C_SURP, lw=1.8, ms=3.5, label="PV surplus")
    ax.annotate("top WFH decile holds\n%.0f%% WFH, %.0f%% generation\nbut only %.1f%% surplus"
                % (100 * g.S.iloc[-1] / g.S.sum(), 100 * g.G.iloc[-1] / g.G.sum(),
                   100 * s1.surplus_share_top_wfh_decile),
                xy=(10, 100 * g.D.iloc[-1] / g.D.sum()), xytext=(3.7, 33), fontsize=7.6, color="#333333",
                ha="left", arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    ax.set_xticks(range(1, 11)); ax.set_xlabel("Mesh decile by WFH increment (1 = lowest)")
    ax.set_ylabel("Share of city total (%)"); ax.legend(fontsize=7.6, loc="upper left")
    ax.grid(True, axis="y", alpha=0.18, lw=0.5); ax.set_axisbelow(True); _despine(ax); _panel(ax, "c")

    # ---- d: Lorenz -- the benefit is extremely concentrated (and more than the behaviour) ----
    ax = fig.add_subplot(gs[1, 0])

    def _lorenz(x):
        x = np.sort(np.asarray(x, float)); x = x[x >= 0]
        c = np.insert(np.cumsum(x), 0, 0.0); c = c / c[-1]
        return np.linspace(0, 1, len(c)), c

    sx, sy = _lorenz(S); bx, by = _lorenz(R)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.9, ls="--")
    ax.plot(sx, sy, color=C_WFH, lw=1.8, label="WFH supply  (Gini %.2f)" % bd_s.gini)
    ax.plot(bx, by, color=C_SURP, lw=2.0, label="Realized benefit  (Gini %.2f)" % bd_b.gini)
    ax.fill_between(bx, by, np.interp(bx, sx, sy), color=C_SURP, alpha=0.08, lw=0)
    ax.annotate("bottom 50%% of grids:\n%.1f%% of the benefit" % (100 * bd_b.bottom_half_share),
                xy=(0.5, np.interp(0.5, bx, by)), xytext=(0.05, 0.55), fontsize=7.6, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    ax.text(0.97, 0.05, "Atkinson(0.5) %.2f\ntop 10%% → %.0f%%" % (bd_b.atkinson_eps0p5, 100 * bd_b.top_decile_share),
            transform=ax.transAxes, fontsize=7.2, va="bottom", ha="right", color="#555555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative share of meshes (low → high)"); ax.set_ylabel("Cumulative share of quantity")
    ax.legend(fontsize=7.4, loc="upper left"); _despine(ax); _panel(ax, "d")

    # ---- e: DECISIVE -- is the 3.8% a spatial or a magnitude limit? the 3-way decomposition of the surplus ----
    ax = fig.add_subplot(gs[1, 1])
    a_now, a_gain, a_wall = 100 * rc.match_actual, 100 * rc.spatial_gain, 100 * rc.magnitude_wall
    ceiling = a_now + a_gain
    segs = [(a_now, C_SURP, "absorbed\nnow"),
            (a_gain, C_WFH, "reachable if WFH\nco-located with surplus"),
            (a_wall, "#b3b3b3", "beyond any WFH\n→ storage")]
    left = 0
    for w, c, _ in segs:
        ax.barh(0, w, left=left, height=0.42, color=c, edgecolor="white", lw=0.6)
        left += w
    # internal labels for the two wide segments; the thin "absorbed now" gets an external arrow
    ax.text(a_now + a_gain / 2, 0, "%s\n%.0f%%" % (segs[1][2], a_gain), ha="center", va="center",
            fontsize=7.2, color="white")
    ax.text(a_now + a_gain + a_wall / 2, 0, "%s\n%.0f%%" % (segs[2][2], a_wall), ha="center", va="center",
            fontsize=7.2, color="#333333")
    ax.annotate("absorbed now\n%.1f%%" % a_now, xy=(a_now, 0.21), xytext=(7, 0.66),
                fontsize=7.4, color="#2f6b3f", ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    ax.plot([ceiling, ceiling], [-0.34, 0.34], color="#333333", lw=1.0, ls=(0, (3, 2)))
    ax.text(ceiling, -0.52, "ceiling if optimally\nplaced  %.0f%%  (%.0f×)" % (ceiling, ceiling / a_now),
            ha="center", va="top", fontsize=7.2, color="#333333")
    ax.set_xlim(0, 100); ax.set_ylim(-0.95, 0.95); ax.set_yticks([])
    ax.set_xlabel("Share of future-PV midday surplus (%)")
    _despine(ax); ax.spines["left"].set_visible(False); _panel(ax, "e")

    # ---- f: co-location vs mismatch map ----
    ax = fig.add_subplot(gs[1, 2])
    sp = S > 1e-9; dp = D > 1e-9
    hiW = S >= np.median(S[sp]) if sp.any() else np.zeros_like(S, bool)
    hiS = D >= np.median(D[dp]) if dp.any() else np.zeros_like(D, bool)
    colocated = hiW & hiS & dp                       # benefit realized in place (rare)
    mismatch = (~hiW) & hiS & dp                     # surplus, little local WFH -> storage task
    reshape = hiW & (~hiS)                            # WFH reshapes local load, little surplus
    rest = ~(colocated | mismatch | reshape)
    ax.scatter(lon[rest], lat[rest], s=2.0, marker="s", color="#e3e3e3", linewidths=0, rasterized=True)
    ax.scatter(lon[reshape], lat[reshape], s=2.4, marker="s", color=C_WFH, alpha=0.55, linewidths=0, rasterized=True)
    ax.scatter(lon[mismatch], lat[mismatch], s=2.6, marker="s", color=C_SURP, linewidths=0, rasterized=True)
    ax.scatter(lon[colocated], lat[colocated], s=3.4, marker="s", color="#000000", linewidths=0, rasterized=True)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(TOKYO_LAT)))
    ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_visible(False)
    leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor="#000000", ms=6, label="co-located (benefit)"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor=C_SURP, ms=6, label="surplus, low WFH (storage)"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor=C_WFH, ms=6, label="WFH, low surplus")]
    ax.legend(handles=leg, fontsize=6.8, loc="upper left", handletextpad=0.3, borderpad=0.4,
              frameon=True, facecolor="white", framealpha=0.78, edgecolor="none")
    ax.text(0.98, 0.02, "only %.1f%% of surplus\nabsorbed locally" % (100 * s1.local_match_fraction),
            transform=ax.transAxes, fontsize=7.8, va="bottom", ha="right", color="#7a2a2a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
    _panel(ax, "f")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig_spatial_matching_r3.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return f, sc


def run_gates(f, sc):
    s1 = sc[sc.arm == "S1"].iloc[0]
    fails = []

    def ck(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
        if not ok:
            fails.append(n)

    ck("1 mesh parquet carries the geometry-decodable key + fields",
       {"mesh", "supply_S1_MWh", "demand_surplus_MWh", "generation_MWh", "realized_S1_MWh"} <= set(f.columns))
    la, lo = mesh_to_latlon(f.mesh.values)
    ck("2 decoded centroids land in the Tokyo metro box (lat 34.9-36.3, lon 138.7-140.9)",
       float(np.nanmin(la)) > 34.9 and float(np.nanmax(la)) < 36.3 and
       float(np.nanmin(lo)) > 138.7 and float(np.nanmax(lo)) < 140.9,
       f"lat[{la.min():.2f},{la.max():.2f}] lon[{lo.min():.2f},{lo.max():.2f}]")
    ck("3 figure encodes the mismatch finding (gen co-location > 0.5, surplus anti-location < 0)",
       s1.colocation_wfh_generation > 0.5 and s1.colocation_wfh_surplus_active < 0,
       f"gen {s1.colocation_wfh_generation} / surplus {s1.colocation_wfh_surplus_active}")
    rc_all = pd.read_csv(REALLOC); rc_all = rc_all[rc_all.scope == "all_slots"].iloc[0]
    ck("3b decomposition panel sourced: spatial ceiling >> actual (siting limit, not magnitude)",
       rc_all.match_spatial_opt > 3 * rc_all.match_actual and rc_all.spatial_gain > 0.10,
       f"actual {rc_all.match_actual} -> ceiling {rc_all.match_spatial_opt}")
    bdf = pd.read_csv(BDIST).set_index("quantity")
    ck("3c Lorenz panel sourced: benefit highly concentrated (Gini>0.7, bottom-half near zero)",
       bdf.loc["realized_benefit", "gini"] > 0.7 and bdf.loc["realized_benefit", "bottom_half_share"] < 0.05,
       f"Gini {bdf.loc['realized_benefit','gini']} / bottom-half {bdf.loc['realized_benefit','bottom_half_share']}")
    for ext in ("png", "pdf"):
        p = os.path.join(OUTDIR, f"fig_spatial_matching_r3.{ext}")
        ck(f"4 out fig_spatial_matching_r3.{ext} nonempty", os.path.exists(p) and os.path.getsize(p) > 20000)
    return fails


def main():
    f, sc = build()
    png = os.path.join(OUTDIR, "fig_spatial_matching_r3.png")
    print(f"[saved] {png} (+ pdf)")
    fails = run_gates(f, sc)
    if fails:
        raise SystemExit("spatial_matching: GATE FAILS -> " + ", ".join(fails))
    print("spatial_matching: ALL GATES PASS")


if __name__ == "__main__":
    main()
