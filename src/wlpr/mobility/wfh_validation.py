"""Step 14b — independent mobility validation of the WFH digital twin (SI-tier). Ports
`configured asset`.

The production WFH weight is residence × occupation (use_mobility=False) -- this NEVER feeds the model, it only
validates it. Reads the frozen 30-min diurnal presence curves (extract_presence.py) and asks what independent
mobile presence can confirm about the census × occupation WFH placement. FOUR findings, descending strength:

  (1) POPULATION skeleton   corr(data_l residence-pop, census pop)            ~ +0.94   [validates foundation]
  (2) WFH TIMING (x-modal)  residential weekday commute-gap SHAPE vs WFH load increment SHAPE (daytime) ~ +0.86
  (3) WFH PLACEMENT         corr(WFH energy, curve-shape residentiality)      ~ +0.40   [WFH lands residential]
  (4) WFH MAGNITUDE         DATA WALL -- NOT identifiable (no pre-pandemic baseline; data_l residence-anchored
                            can't see commute movement, data_m activity-inflated isn't headcount). Reported,
                            never claimed.

CALIBER (permanent): ex_volume = presence proxy. data_m = current-location, ACTIVITY-weighted (regional total
swells midday -> night-undercount bias); data_l = population-scale, residence-anchored (CV 0.145, total ~=
census pop). All conclusions use NORMALIZED shape + weekday-MINUS-weekend differencing + RANK correlation ->
immune to the bias.

Outputs: out_path("figures","mobility_validation",{fig_diurnal_presence,fig_crossmodal}.{png,pdf})
         out_path("scorecards","mobility_wfh_validation_scorecard.csv")
GATES (exit 1 on fail): population corr >= 0.85; timing daytime corr >= 0.70; placement corr >= 0.20;
         magnitude bound flagged NON-IDENTIFIED (documented, not claimed); figures non-empty.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wlpr.io.paths import derived_path, out_path

MIDDAY = list(range(20, 31))          # 10:00-15:30
NIGHT = [0, 1, 2, 3, 4, 5, 46, 47]    # ~00:00-03:00 + 23:00-23:30
DAY = list(range(12, 42))             # 06:00-21:00 (work-relevant window for the timing test)

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#333333", "legend.frameon": False, "savefig.dpi": 300})


def _mob_m():
    return derived_path(_asset_location('asset_0100'), "mesh_diurnal_presence_m.parquet")


def _mob_l():
    return derived_path(_asset_location('asset_0100'), "mesh_diurnal_presence_l.parquet")


def _r3():
    return derived_path(_asset_location('asset_0056'), "r3_spatial_matching_mesh.parquet")


def _sw():
    return derived_path(_asset_location('asset_0054'), "wfh_spatial_weight.parquet")


def _wfhcurve():
    return derived_path(_asset_location('asset_0080'), "scenario_window_curves_month_slot.parquet")


def _figdir():
    return os.path.dirname(out_path(_asset_location('asset_0059'), _asset_location('asset_0136'), _asset_location('asset_0137')))


def _score():
    return out_path(_asset_location('asset_0058'), "mobility_wfh_validation_scorecard.csv")


def _curves(path):
    c = pd.read_parquet(path)
    piv = c.pivot_table(index="mesh", columns=["daytype", "slot"], values="presence")
    wd = piv["wd"].reindex(columns=range(48))
    we = piv["we"].reindex(columns=range(48))
    keep = wd.notna().sum(axis=1).ge(40) & we.notna().sum(axis=1).ge(40) & (wd.mean(axis=1) > 5)
    return wd[keep], we[keep]


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def build():
    figdir = _figdir()
    os.makedirs(figdir, exist_ok=True)
    score = _score()
    sw = pd.read_parquet(_sw())[["mesh", "pop", "working_pop"]]
    r3 = pd.read_parquet(_r3())[["mesh", "supply_S1_MWh"]]
    for d in (sw, r3):
        d["mesh"] = d.mesh.astype("int64").astype(str)
    res = {}

    # ---------------- data_m world: residentiality, placement, timing, figure ----------------
    wd, we = _curves(_mob_m())
    night_wd, mid_wd = wd[NIGHT].mean(axis=1), wd[MIDDAY].mean(axis=1)
    Fm = pd.DataFrame(index=wd.index)
    Fm["residentiality"] = (night_wd - mid_wd) / (night_wd + mid_wd)
    Fm = Fm.reset_index().merge(sw, on="mesh").merge(r3, on="mesh")
    Fm = Fm[Fm.working_pop > 0]
    res["placement_corr_m"] = float(spearmanr(Fm.supply_S1_MWh, Fm.residentiality).statistic)

    # timing: residential (top-tertile) weekday commute-gap shape vs WFH load increment shape
    rrank = Fm.set_index("mesh").residentiality
    rmesh = rrank[rrank >= rrank.quantile(2 / 3)].index
    wdn = wd.loc[wd.index.intersection(rmesh)].div(wd.loc[wd.index.intersection(rmesh)].mean(axis=1), axis=0).mean()
    wen = we.loc[we.index.intersection(rmesh)].div(we.loc[we.index.intersection(rmesh)].mean(axis=1), axis=0).mean()
    gap = (wen - wdn).reindex(range(48))                         # >0 where weekday emptier = commute-out shape
    w = pd.read_parquet(_wfhcurve()).groupby(["scenario_id", "slot"]).load_MW.mean().unstack(0)
    wfh_inc = (w["S10"] - w["S1"]).reindex(range(48))            # WFH load added (MW), midday-centred
    res["timing_corr_day"] = float(spearmanr(gap.values[DAY], wfh_inc.values[DAY]).statistic)
    res["timing_corr_full"] = float(spearmanr(gap.values, wfh_inc.values).statistic)

    # ---------------- data_l world: population skeleton + magnitude wall ----------------
    wdl, wel = _curves(_mob_l())
    Fl = pd.DataFrame(index=wdl.index)
    Fl["res_pop"] = wel.mean(axis=1)
    Fl["mid_wd"] = wdl[MIDDAY].mean(axis=1)
    Fl["mid_we"] = wel[MIDDAY].mean(axis=1)
    Fl["night_wd"] = wdl[NIGHT].mean(axis=1)
    Fl["residentiality"] = (Fl.night_wd - Fl.mid_wd) / (Fl.night_wd + Fl.mid_wd)
    Fl = Fl.reset_index().merge(sw, on="mesh").merge(r3, on="mesh")
    Fl = Fl[(Fl.working_pop > 0) & (Fl.res_pop > 0)]
    res["population_corr"] = float(spearmanr(Fl.res_pop, Fl["pop"]).statistic)
    res["population_corr_workers"] = float(spearmanr(Fl.res_pop, Fl.working_pop).statistic)
    res["placement_corr_l"] = float(spearmanr(Fl.supply_S1_MWh, Fl.residentiality).statistic)
    res["data_l_cv"] = float((wdl.std(axis=1) / wdl.mean(axis=1).clip(lower=1e-9)).median())

    # magnitude WALL: calibrate data_l persons -> census scale, commute gap, loose home-worker upper bound
    k = Fl["pop"].sum() / Fl.res_pop.sum()
    resid = Fl[Fl.residentiality > 0].copy()
    commuters = float((resid.mid_we - resid.mid_wd).clip(lower=0).sum() * k)
    workers = float(resid.working_pop.sum())
    res["magnitude_home_worker_upper_bound"] = round(1 - commuters / workers, 3)
    res["magnitude_identified"] = False     # NEVER claimed -- documented data wall

    _figures(figdir, wd, we, Fm, gap, wfh_inc)
    pd.DataFrame([res]).to_csv(score, index=False)
    print(f"[saved] {score}")
    return res


def _figures(figdir, wd, we, Fm, gap, wfh_inc):
    hours = np.arange(48) / 2.0
    resmask = Fm.set_index("mesh").residentiality > 0
    # fig 1: residential vs workplace mean diurnal presence (normalized), weekday vs weekend
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for j, (mask, title) in enumerate([(resmask[resmask].index, "Residential meshes"),
                                       (resmask[~resmask].index, "Workplace meshes")]):
        idx = wd.index.intersection(mask)
        nwd = wd.loc[idx].div(wd.loc[idx].mean(axis=1), axis=0).mean()
        nwe = we.loc[idx].div(we.loc[idx].mean(axis=1), axis=0).mean()
        ax[j].axvspan(9, 16, color="#F2C14E", alpha=0.15, lw=0)
        ax[j].plot(hours, nwd.values, color="#C44E52", lw=1.8, label="weekday")
        ax[j].plot(hours, nwe.values, color="#3B6EA5", lw=1.8, ls=(0, (4, 2)), label="weekend")
        ax[j].set_title(title, fontsize=10); ax[j].set_xlabel("Hour"); ax[j].set_xlim(0, 24); ax[j].set_xticks([0, 6, 12, 18, 24])
        _despine(ax[j])
        if j == 0:
            ax[j].set_ylabel("Presence / daily mean"); ax[j].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"fig_diurnal_presence.{ext}"))
    plt.close(fig)

    # fig 2: cross-modal -- two independent worlds, one midday signature
    fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
    ax1.axvspan(9, 16, color="#F2C14E", alpha=0.13, lw=0)
    l1, = ax1.plot(hours, gap.values, color="#3B6EA5", lw=2.0,
                   label="people missing on weekdays\n(mobile presence: weekend − weekday)")
    ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Presence gap (weekend − weekday, norm.)", color="#3B6EA5")
    ax1.tick_params(axis="y", labelcolor="#3B6EA5"); ax1.set_xlim(0, 24); ax1.set_xticks([0, 6, 12, 18, 24])
    ax1.axhline(0, color="#cccccc", lw=0.7)
    ax2 = ax1.twinx()
    l2, = ax2.plot(hours, wfh_inc.values / 1000.0, color="#C44E52", lw=2.0, ls=(0, (4, 2)),
                   label="WFH load added (model: S10 − S1)")
    ax2.set_ylabel("WFH load increment (GW)", color="#C44E52"); ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    ax1.legend(handles=[l1, l2], fontsize=7.5, loc="upper left")
    ax1.set_title("Two independent worlds, one midday signature", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"fig_crossmodal.{ext}"))
    plt.close(fig)


def run_gates(r: dict) -> None:
    """SI validation gates (exit 1 on fail). population>=0.85, timing>=0.70, placement>=0.20,
    magnitude NON-IDENTIFIED, figures non-empty."""
    figdir = _figdir()
    fails = []

    def ck(name, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {d}")
        if not ok:
            fails.append(name)

    print("\n=== GATES ===")
    ck("1 population skeleton corr >= 0.85", r["population_corr"] >= 0.85, f"{r['population_corr']:+.3f}")
    ck("2 WFH timing daytime corr >= 0.70", r["timing_corr_day"] >= 0.70, f"{r['timing_corr_day']:+.3f}")
    ck("3 WFH placement corr >= 0.20 (both series)",
       min(r["placement_corr_m"], r["placement_corr_l"]) >= 0.20,
       f"m {r['placement_corr_m']:+.3f} / l {r['placement_corr_l']:+.3f}")
    ck("4 WFH magnitude NON-IDENTIFIED (never claimed)", r["magnitude_identified"] is False)
    for ext in ("png", "pdf"):
        for fn in ("fig_diurnal_presence", "fig_crossmodal"):
            pth = os.path.join(figdir, f"{fn}.{ext}")
            ck(f"5 {fn}.{ext} non-empty", os.path.exists(pth) and os.path.getsize(pth) > 10000)
    if fails:
        raise SystemExit("build_mobility_wfh_validation: GATE FAILS -> " + ", ".join(fails))
    print("build_mobility_wfh_validation: ALL GATES PASS")


def main():
    for p in (_mob_m(), _mob_l()):
        if not os.path.exists(p):
            raise SystemExit(f"missing diurnal parquet {p} -> run extract_presence.py first")
    r = build()
    print("\n=== MOBILITY WFH VALIDATION ===")
    print(f"  (1) population skeleton corr(data_l res_pop, census pop) = {r['population_corr']:+.3f} "
          f"(workers {r['population_corr_workers']:+.3f})")
    print(f"  (2) WFH timing  corr(commute-gap, WFH load increment) day = {r['timing_corr_day']:+.3f} "
          f"(full {r['timing_corr_full']:+.3f})")
    print(f"  (3) WFH placement corr(WFH energy, residentiality) data_m {r['placement_corr_m']:+.3f} / "
          f"data_l {r['placement_corr_l']:+.3f}")
    print(f"  (4) WFH magnitude: home-worker UPPER BOUND {r['magnitude_home_worker_upper_bound']} -- "
          f"NON-IDENTIFIED (data wall), magnitude stays survey-anchored")
    print(f"      data_l diurnal CV {r['data_l_cv']:.3f} (population-stable)")
    run_gates(r)


if __name__ == "__main__":
    main()
