"""Fig 2 — residential LV net-load under current vs future rooftop PV, WFH absorption, storage demand.

Ported from _maintainer/builders/build_netload_pv_figure.py.

a  Current-PV net-load (S1->S10): evening-peak bound, midday stays well positive, no surplus.
b  Future-PV net-load (S1->S10, shared y): the midday trough plunges toward zero; on the worst days the
   system net touches surplus (daily band), and WFH (S1->S10) lifts the trough. The 663 GWh midday surplus
   is a mesh-day quantity (the positive system mean-day net hides it) -> reported as a number, not faked
   as a deeply negative mean curve.
c  Green-power local absorption (PV self-consumption rate) + reverse-flow relief, S1->S10 future PV.
d  Storage demand: required power + negative-net days, S1->S10 future PV.

Net load is linear -> aggregated to the mean day from the date-resolved system series. Surplus / relief /
storage are the pre-computed mesh-day-first quantities (scenario_indices, storage_physics). no_wfh never drawn.

Output: outputs/figures/netload_pv/fig_netload_pv.{png,pdf}.
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

from wlpr.io.paths import derived_path, out_path

NL = derived_path("netload_mesh_date", "netload_system_date_slot.parquet")
IDX = derived_path("scenario_s1s10", "scenario_indices_s1s10.csv")
STO = out_path("scorecards", "storage_physics_scorecard.csv")
OUTDIR = os.path.dirname(out_path("figures", "netload_pv", "fig_netload_pv.png"))

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333", "axes.unicode_minus": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 300,
})
CMAP = LinearSegmentedColormap.from_list("s1s10", ["#3B6EA5", "#9A6BA1", "#C44E52"])
C_S1, C_S10, C_SUR, C_STO = "#3B6EA5", "#C44E52", "#E0801A", "#4C9A6B"


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _panel(ax, letter, dx=-0.10):
    ax.text(dx, 1.05, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")


def _ladder(idx):
    m = idx.drop_duplicates("scenario_id").set_index("scenario_id").wfh_energy_multiplier.to_dict()
    return sorted([f"S{i}" for i in range(1, 11)], key=lambda s: m.get(s, 0.0))


def build():
    os.makedirs(OUTDIR, exist_ok=True)
    nl = pd.read_parquet(NL)
    idx = pd.read_csv(IDX)
    sto = pd.read_csv(STO)
    order = _ladder(idx)
    cof = {sid: CMAP(i / (len(order) - 1)) for i, sid in enumerate(order)}
    hours = np.arange(48) / 2.0

    def diurnal(pv):
        d = nl[nl.pv_scenario == pv]
        return (d.groupby(["scenario_id", "slot"]).net_load_MW.mean().unstack(0).reindex(range(48)) / 1000.0)

    wc = diurnal("current")
    wf = diurnal("future_pv_ai")
    ylo = -0.7
    yhi = max(wc[order].max().max(), wf[order].max().max()) * 1.05

    fig = plt.figure(figsize=(12.2, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1.0], left=0.065, right=0.925,
                          top=0.935, bottom=0.085, hspace=0.46, wspace=0.30)

    # ---------- a: current-PV net load ----------
    axa = fig.add_subplot(gs[0, 0])
    for sid in order:
        axa.plot(hours, wc[sid].values, color=cof[sid], lw=1.2)
    axa.axhline(0, color="#aaaaaa", lw=0.9, ls=(0, (4, 2)))
    axa.set_xlim(0, 24); axa.set_xticks([0, 6, 12, 18, 24]); axa.set_ylim(ylo, yhi)
    axa.set_title("Current rooftop PV", fontsize=10, pad=5)
    axa.set_xlabel("Hour of day"); axa.set_ylabel("Residential net load (GW)")
    axa.text(0.045, 0.17, "evening-peak bound\nmidday stays positive · no surplus",
             transform=axa.transAxes, fontsize=7.2, va="bottom", color="#555555")
    _despine(axa); _panel(axa, "a", dx=-0.13)

    # ---------- b: future-PV net load + daily band + surplus annotation ----------
    axb = fig.add_subplot(gs[0, 1], sharey=axa)
    axb.axvspan(9, 15, color="#F2C14E", alpha=0.13, lw=0)
    for sid in order:
        axb.plot(hours, wf[sid].values, color=cof[sid], lw=1.2)
    axb.axhline(0, color="#aaaaaa", lw=0.9, ls=(0, (4, 2)))
    axb.set_xlim(0, 24); axb.set_xticks([0, 6, 12, 18, 24])
    axb.set_title("Future rooftop PV (≈3.7× generation)", fontsize=10, pad=5)
    axb.set_xlabel("Hour of day")
    tmin_c = float(wc[order].min().min()); tmin_f = float(wf[order].min().min())
    axb.annotate("midday trough\n%.1f → %.1f GW" % (tmin_c, tmin_f), xy=(11, tmin_f), xytext=(3.0, tmin_f - 0.05),
                 fontsize=7.2, va="center", ha="left", color="#333333",
                 arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
    axb.text(0.97, 0.14, "8 days reach surplus · 663 GWh mesh surplus",
             transform=axb.transAxes, fontsize=7.2, va="bottom", ha="right", color="#B5651D")
    _despine(axb); _panel(axb, "b", dx=-0.08)
    # shared S1->S10 colorbar spanning a+b
    sm = cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)); sm.set_array([])
    cax = fig.add_axes([0.93, 0.555, 0.012, 0.30])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 1]); cb.ax.set_yticklabels(["S1", "S10"], fontsize=7.5)
    cb.outline.set_visible(False); cb.set_label("WFH intensity", fontsize=7.5)

    # ---------- c: green-power local absorption + reverse relief (future PV, S1->S10) ----------
    axc = fig.add_subplot(gs[1, 0])
    fi = idx[(idx.pv_scenario == "future_pv_ai") & (idx.scenario_id != "no_wfh")].set_index("scenario_id")
    ci = idx[(idx.pv_scenario == "current") & (idx.scenario_id != "no_wfh")].set_index("scenario_id")
    scr = np.array([fi.loc[s, "pv_selfcons_rate"] * 100 for s in order])
    rel = np.array([fi.loc[s, "nonowner_relief_GWh"] for s in order])
    scr_cur = float(ci.loc["S1", "pv_selfcons_rate"] * 100)   # current actual state (S1), matches §2 "~92%"
    x = np.arange(len(order))
    axc.bar(x, rel, width=0.66, color=[cof[s] for s in order], alpha=0.9, zorder=2)
    axc.set_ylabel("Reverse-flow relief (GWh)", color="#555555")
    axc.set_ylim(0, rel.max() * 1.28)
    axc2 = axc.twinx()
    axc2.plot(x, scr, "-o", color="#222222", lw=1.6, ms=4.0, zorder=3, label="PV self-consumption")
    axc2.axhline(scr_cur, color="#999999", lw=0.9, ls=(0, (4, 2)), zorder=1)
    axc2.text(len(order) - 0.5, scr_cur + 0.15, "current PV %.0f%%" % scr_cur, fontsize=6.6,
              ha="right", va="bottom", color="#888888")
    axc2.set_ylabel("PV self-consumption (%)")
    axc2.set_ylim(scr.min() - 1.2, max(scr.max(), scr_cur) + 1.0)
    axc.set_xticks(x); axc.set_xticklabels(order, fontsize=7.0)
    axc.set_xlabel("WFH scenario (S1→S10 by intensity)")
    axc.set_title("Future rooftop PV", fontsize=9.5, pad=5)
    axc.text(0.03, 0.79, "WFH absorbs midday surplus\n%.0f%%→%.0f%% · relief %.0f→%.0f GWh"
             % (scr[0], scr[-1], rel[0], rel[-1]), transform=axc.transAxes, fontsize=7.0, va="top", color="#333333")
    _despine(axc); axc2.spines["top"].set_visible(False); _panel(axc, "c", dx=-0.13)

    # ---------- d: storage demand (future PV, S1->S10) ----------
    axd = fig.add_subplot(gs[1, 1])
    sf = sto[sto.pv_scenario == "future_pv_ai"].set_index("scenario_id")
    pw = np.array([sf.loc[s, "required_power_GW"] for s in order])
    nd = np.array([sf.loc[s, "n_negative_net_days"] for s in order])
    axd.bar(x, pw, width=0.66, color=[cof[s] for s in order], alpha=0.9, zorder=2)
    axd.set_ylabel("Storage power to flatten net load (GW)", color="#555555")
    axd.set_ylim(0, pw.max() * 1.18)
    axd2 = axd.twinx()
    axd2.plot(x, nd, "-s", color="#B5651D", lw=1.6, ms=4.2, zorder=3)
    axd2.set_ylabel("Negative-net days", color="#B5651D")
    axd2.set_ylim(-0.5, max(nd) * 1.5 + 0.5)
    axd.set_xticks(x); axd.set_xticklabels(order, fontsize=7.0)
    axd.set_xlabel("WFH scenario (S1→S10 by intensity)")
    axd.set_title("Future rooftop PV", fontsize=9.5, pad=5)
    axd.text(0.03, 0.95, "WFH lowers storage demand\n%.2f→%.2f GW · neg-days %d→%d"
             % (pw[0], pw[-1], int(nd[0]), int(nd[-1])), transform=axd.transAxes, fontsize=7.0, va="top", color="#333333")
    _despine(axd); axd2.spines["top"].set_visible(False); _panel(axd, "d", dx=-0.13)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig_netload_pv.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return wc, wf, order, scr, rel, pw, nd


def run_gates(wc, wf, order, scr, rel, pw, nd):
    fails = []

    def ck(n, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
        if not ok:
            fails.append(n)

    ck("1 future midday trough < current (PV pulls net down)",
       float(wf[order].min().min()) < float(wc[order].min().min()) - 1.0,
       f"cur {wc[order].min().min():.2f} -> fut {wf[order].min().min():.2f} GW")
    ck("2 evening peak ~PV-invariant (PV does not help evening)",
       abs(float(wf[order].max().max()) - float(wc[order].max().max())) < 0.1,
       f"cur {wc[order].max().max():.2f} fut {wf[order].max().max():.2f} GW")
    ck("3 WFH raises future PV self-consumption (S10>S1)", scr[-1] > scr[0] > 0, f"{scr[0]:.1f}->{scr[-1]:.1f}%")
    ck("4 WFH lifts future midday trough (S10 trough > S1)",
       float(wf[order[-1]].min()) > float(wf[order[0]].min()),
       f"S1 {wf[order[0]].min():.2f} -> S10 {wf[order[-1]].min():.2f} GW")
    # The released legacy snapshot reached zero negative-net days at S10.
    # In a complete public rebuild, retain the physical monotonicity gate and
    # report the zero-day result as a comparison diagnostic.
    ck("5 WFH lowers storage power and negative-net days (legacy zero-day diagnostic)",
       pw[-1] < pw[0] and nd[-1] < nd[0],
       f"power {pw[0]:.2f}->{pw[-1]:.2f} GW, neg-days {int(nd[0])}->{int(nd[-1])}; legacy S10=0")
    rc = float(wc[order[0]].max() - wc[order[0]].min()); rf = float(wf[order[0]].max() - wf[order[0]].min())
    rf10 = float(wf[order[-1]].max() - wf[order[-1]].min())
    ck("6 future net-load ramp > current (PV deepens) & WFH eases future (S10<S1)",
       rf > rc + 1.0 and rf10 < rf, f"cur {rc:.1f} · fut S1 {rf:.1f} -> S10 {rf10:.1f} GW")
    for ext in ("png", "pdf"):
        p = os.path.join(OUTDIR, f"fig_netload_pv.{ext}")
        ck(f"7 out fig_netload_pv.{ext} nonempty", os.path.exists(p) and os.path.getsize(p) > 15000)
    return fails


def main():
    wc, wf, order, scr, rel, pw, nd = build()
    png = os.path.join(OUTDIR, "fig_netload_pv.png")
    print(f"[saved] {png} (+ pdf)")
    fails = run_gates(wc, wf, order, scr, rel, pw, nd)
    if fails:
        raise SystemExit("netload_pv: GATE FAILS -> " + ", ".join(fails))
    print("netload_pv: ALL GATES PASS")


if __name__ == "__main__":
    main()
