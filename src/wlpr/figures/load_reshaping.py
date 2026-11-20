"""Fig 1 — WFH re-times residential demand toward midday (the load-reshaping result).

Ported from _maintainer/builders/build_load_reshaping_figure.py. Built STRICTLY from the validated-window
scenario LOAD curves (scenario_window_curves: mesh x date x slot -> month x slot, anchored S1 == observed
current load). No rep-day input, no PV here -- this is the demand re-timing BEFORE any PV/storage framing.
Nature-style: panel letters a-d, no titles, all-English, despined.

ANCHOR (permanent): S1 == the OBSERVED current load (NMAE-validated). no_wfh (the synthetic without-WFH
counterfactual) is meaningless and NEVER appears in the figure. The full S1->S10 family is the comparison
axis; every increment/fill is relative to S1.

  a  residential LV load over the day (window mean), S1->S10 gradient: more WFH progressively lifts the midday
     valley and leaves the evening peak untouched -- demand flattens toward solar hours.
  b  twin-vs-observed diurnal validation curve (twin +/-RMSE band = the model's OWN per-slot error).
  c  twin-vs-observed daily-total validation scatter.
  d  the afternoon->evening ramp: one floating range bar per scenario; valley fills S1->S10, peak unchanged.

The F=2.0 unit fix (MWh per 30-min slot -> average power MW) and the per-slot RMSE band are PART OF THE
SOURCE -- kept verbatim. Output: outputs/figures/scenario_window/fig_load_reshaping.{png,pdf}.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize

from wlpr.io.paths import input_path, derived_path, out_path

CURVES = derived_path("scenario_window", "scenario_window_curves_month_slot.parquet")
SIDX = derived_path("scenario_s1s10", "scenario_indices_s1s10.csv")
TWIN = lambda: input_path("lv_twin_pred_pref_v4")   # obs_mwh / pred_mwh, pref x date x slot
BANDS = derived_path("scenario_s1s10", "scenario_increment_bands.csv")   # partial-ID WFH-increment band
OUTDIR = os.path.dirname(out_path("figures", "scenario_window", "fig_load_reshaping.png"))
WORK = slice(18, 35)        # 9-17h work window (slots)
WINDOW = [10, 11, 12, 1, 2]
MONTHNAME = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb"}

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333", "axes.unicode_minus": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 300,
})
CMAP = LinearSegmentedColormap.from_list("s1s10", ["#3B6EA5", "#9A6BA1", "#C44E52"])   # S1 blue -> S10 red
C_S1, C_S10, C_ZERO = "#3B6EA5", "#C44E52", "#bbbbbb"


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _panel(ax, letter, dx=-0.085):
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")


def _ladder():
    m = pd.read_csv(SIDX).drop_duplicates("scenario_id").set_index("scenario_id").wfh_energy_multiplier.to_dict()
    return sorted([f"S{i}" for i in range(1, 11)], key=lambda s: m.get(s, 0.0))


def _scen_cbar(fig, ax):
    sm = cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["S1", "S10"], fontsize=7.5)
    cb.outline.set_visible(False); cb.set_label("WFH intensity", fontsize=7.5)


def build():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_parquet(CURVES)
    w = df.groupby(["scenario_id", "slot"]).load_MW.mean().unstack(0)      # window-mean (MW): for the ladder + gates
    hours = np.arange(48) / 2.0
    order = _ladder()
    cof = {sid: CMAP(i / (len(order) - 1)) for i, sid in enumerate(order)}
    tr = np.array([w[s].values[WORK].min() for s in order]) / 1000.0
    try:
        bd = pd.read_csv(BANDS); b10 = bd[bd.scenario_id == "S10"]
        dlo, dhi = float(b10.d_lo.mean()), float(b10.d_hi.mean())          # partial-ID WFH band (S10 = widest)
    except Exception:
        dlo, dhi = 0.88, 1.12

    fig = plt.figure(figsize=(13.6, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.16, 1.0], left=0.055, right=0.915,
                          top=0.895, bottom=0.08, hspace=0.50)
    gtop = gs[0].subgridspec(1, 5, wspace=0.10)
    gbot = gs[1].subgridspec(1, 3, wspace=0.34, width_ratios=[1.25, 0.85, 1.25])

    # ---- a: WFH load reshaping BY MONTH (small multiples) + S10 partial-ID band ----
    mon_w = {m: df[df.mon == m].groupby(["scenario_id", "slot"]).load_MW.mean().unstack(0) / 1000.0 for m in WINDOW}
    allv = np.concatenate([mon_w[m][order].values.ravel() for m in WINDOW])
    ylo, yhi = allv.min() * 0.98, allv.max() * 1.05
    ax0 = None
    for i, m in enumerate(WINDOW):
        ax = fig.add_subplot(gtop[0, i], sharey=ax0)
        if ax0 is None:
            ax0 = ax
        ax.axvspan(9, 15, color="#F2C14E", alpha=0.15, lw=0)
        mw = mon_w[m]; s1 = mw["S1"].values; inc10 = mw["S10"].values - s1
        ax.fill_between(hours, s1 + inc10 * dlo, s1 + inc10 * dhi, color=C_S10, alpha=0.18, lw=0)
        for sid in order:
            ax.plot(hours, mw[sid].values, color=cof[sid], lw=1.2)
        ax.set_xlim(0, 24); ax.set_xticks([0, 12, 24]); ax.set_ylim(ylo, yhi)
        ax.set_title(MONTHNAME[m], fontsize=10, pad=4)
        ax.set_xlabel("Hour", fontsize=8)
        ax.text(0.5, 0.03, "+%.1f GW" % (mw["S10"].values[WORK].min() - s1[WORK].min()),
                transform=ax.transAxes, fontsize=7.0, ha="center", va="bottom", color="#777777")
        if i == 0:
            ax.set_ylabel("Residential LV load (GW)")
        else:
            ax.tick_params(labelleft=False)
        _despine(ax)
    fig.text(0.011, 0.875, "a", fontsize=13, fontweight="bold")
    fig.text(0.485, 0.925, "WFH lifts the midday trough into the solar window — each validated month, S1→S10 "
             "(shaded = scenario partial-ID band on S10)", ha="center", fontsize=8.4, color="#444444", style="italic")
    sm = cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)); sm.set_array([])
    cax = fig.add_axes([0.925, 0.565, 0.012, 0.27])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 1]); cb.ax.set_yticklabels(["S1", "S10"], fontsize=7.5)
    cb.outline.set_visible(False); cb.set_label("WFH intensity", fontsize=7.5)

    # ---- b (validation CURVE): twin reproduces the diurnal shape; band = the model's OWN per-slot error (RMSE),
    #      not the day-to-day load spread. dark = observed, orange = twin; legend/caption in the empty band above
    #      midday so nothing occludes the curve. ----
    ax = fig.add_subplot(gbot[0, 0]); tw = None
    C_OBS, C_TW = "#2b2b2b", "#E0801A"
    try:
        tw = pd.read_parquet(TWIN())
        sysd = tw.groupby(["d", "slot"]).agg(obs=("obs_mwh", "sum"), pred=("pred_mwh", "sum")).reset_index()
        g = sysd.groupby("slot"); hh = np.arange(48) / 2.0
        F = 2.0   # obs/pred are MWh per 30-min slot (energy); x2 -> average power (MW) == CURVES load_MW scale
        om = (g.obs.mean() * F / 1000.0).values; pm = (g.pred.mean() * F / 1000.0).values
        rmse = (sysd.assign(r=(sysd.pred - sysd.obs) * F / 1000.0).groupby("slot").r
                .apply(lambda s: float(np.sqrt((s ** 2).mean()))).values)
        ax.fill_between(hh, pm - rmse, pm + rmse, color=C_TW, alpha=0.16, lw=0, label="twin ±RMSE")
        ax.plot(hh, om, color=C_OBS, lw=2.1, label="observed", zorder=3, solid_capstyle="round")
        ax.plot(hh, pm, color=C_TW, lw=1.6, ls=(0, (3.5, 2)), label="grid twin", zorder=4)
        nm = float(np.abs(sysd.pred - sysd.obs).sum() / sysd.obs.sum())
        lo = float(min(om.min(), (pm - rmse).min())); hi = float(max(om.max(), (pm + rmse).max()))
        ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(lo - 0.4, hi + (hi - lo) * 0.36)        # top headroom -> legend sits clear of the curve
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Residential LV load (GW)")
        ax.text(0.0, 1.03, "Oct–Feb mean day · NMAE %.3f · ±RMSE band (±%.0f%%)"
                % (nm, 100.0 * float(np.mean(rmse / pm))), transform=ax.transAxes,
                fontsize=6.6, va="bottom", ha="left", color="#666666")
        ax.legend(fontsize=6.6, loc="upper left", handlelength=1.8, borderaxespad=0.5, labelspacing=0.32)
    except Exception:
        ax.text(0.5, 0.5, "validation\nunavailable", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    _despine(ax); _panel(ax, "b", dx=-0.14)

    # ---- c (validation SCATTER, kept): twin vs observed daily total ----
    ax = fig.add_subplot(gbot[0, 1])
    try:
        dd = tw.groupby("d").agg(obs=("obs_mwh", "sum"), pred=("pred_mwh", "sum")).reset_index()
        o = dd.obs.values / 1000.0; p = dd.pred.values / 1000.0
        nm = float(np.abs(p - o).sum() / o.sum())
        lim = [min(o.min(), p.min()) * 0.97, max(o.max(), p.max()) * 1.03]
        ax.plot(lim, lim, color="#bbbbbb", lw=0.9, ls=(0, (4, 2)))
        ax.scatter(o, p, s=10, color=C_S1, alpha=0.5, edgecolors="none")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed (GWh/day)"); ax.set_ylabel("Twin (GWh/day)")
        ax.text(0.05, 0.95, "daily total\n%d days · NMAE %.3f" % (len(o), nm), transform=ax.transAxes,
                fontsize=7.0, va="top", color="#333333")
    except Exception:
        pass
    ax.grid(True, alpha=0.15, lw=0.5); ax.set_axisbelow(True); _despine(ax); _panel(ax, "c", dx=-0.20)

    # ---- d: the afternoon->evening ramp the grid must climb. One floating range bar per scenario, from the midday
    #      valley (bottom) to the evening peak (top). Tops stay level (peak unchanged); bottoms climb S1->S10 (the valley
    #      fills); the bar IS the peak-to-valley ramp and it shrinks. The old two lines were mirror images of this. ----
    ax = fig.add_subplot(gbot[0, 2])
    peak = np.array([w[s].values.max() for s in order]) / 1000.0
    gap = peak - tr
    x = np.arange(len(order)); pk = float(np.mean(peak))
    for i, s in enumerate(order):
        ax.bar(i, gap[i], bottom=tr[i], width=0.70, color=cof[s], alpha=0.92,
               edgecolor="white", lw=0.6, zorder=2)
    ax.plot([-0.65, len(order) - 0.35], [pk, pk], color="#9a9a9a", lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(len(order) - 0.4, pk + 0.06, "evening peak · unchanged", fontsize=6.6, ha="right", va="bottom", color="#777777")
    ax.annotate("", xy=(len(order) - 1, tr[-1] - 0.7), xytext=(0, tr[0] - 0.7),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.1, connectionstyle="arc3,rad=0.12"))
    ax.text((len(order) - 1) / 2.0, tr.min() - 1.05, "valley fills → ramp narrows %.1f→%.1f GW" % (gap[0], gap[-1]),
            ha="center", va="top", fontsize=7.0, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.2)
    ax.set_xlabel("WFH scenario (S1→S10 by intensity)"); ax.set_ylabel("Residential LV load (GW)")
    ax.set_ylim(tr.min() - 1.5, pk + 0.95)
    ax.grid(True, axis="y", alpha=0.15, lw=0.5); ax.set_axisbelow(True)
    _despine(ax); _panel(ax, "d", dx=-0.085)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig_load_reshaping.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return w, order


def run_gates(w, order):
    fails = []

    def ck(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
        if not ok:
            fails.append(n)

    s1 = w["S1"].values[WORK].min(); s10 = w["S10"].values[WORK].min()
    ck("1 full S1->S10 family present (10 scenarios)", all(f"S{i}" in w.columns for i in range(1, 11)),
       f"{sum(f'S{i}' in w.columns for i in range(1,11))}/10")
    ck("2 more WFH lifts the midday valley (S10 trough > S1 trough)", s10 > s1,
       f"S1 trough {s1:.0f} -> S10 {s10:.0f} MW")
    ck("3 evening peak ~unchanged across the family (a daytime phenomenon)",
       abs(w["S10"].values.max() - w["S1"].values.max()) < 0.02 * w["S1"].values.max(),
       f"peak S1 {w['S1'].values.max():.0f} vs S10 {w['S10'].values.max():.0f} MW")
    for ext in ("png", "pdf"):
        p = os.path.join(OUTDIR, f"fig_load_reshaping.{ext}")
        ck(f"4 out fig_load_reshaping.{ext} nonempty", os.path.exists(p) and os.path.getsize(p) > 15000)
    return fails


def main():
    w, order = build()
    png = os.path.join(OUTDIR, "fig_load_reshaping.png")
    print(f"[saved] {png} (+ pdf)")
    fails = run_gates(w, order)
    if fails:
        raise SystemExit("load_reshaping: GATE FAILS -> " + ", ".join(fails))
    print("load_reshaping: ALL GATES PASS")


if __name__ == "__main__":
    main()
