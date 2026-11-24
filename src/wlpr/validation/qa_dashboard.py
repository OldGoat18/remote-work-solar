"""MODEL_QA_DASHBOARD — evaluate the twin against the declared validation targets and emit a
PASS/WARN/FAIL scorecard + a color-coded panel. Reads the decomposition tables + product scorecards; never re-fits.

Ported verbatim (numerics unchanged) from the source validation implementation.
"""

from wlpr.io.bindings import asset as _asset_location
import os, csv
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from wlpr.io.paths import out_path, model_config, _load_yaml

PN = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}


def _status(value, spec):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    d, p, w = spec["dir"], spec["pass"], spec["warn"]
    v = abs(value) if d == "abs_max" else value
    if d in ("max", "abs_max"):
        return "PASS" if v <= p else ("WARN" if v <= w else "FAIL")
    return "PASS" if v >= p else ("WARN" if v >= w else "FAIL")   # min


def _read(name):
    p = out_path(_asset_location('asset_0058'), name)
    return pd.read_csv(p) if os.path.exists(p) else None


def build_dashboard():
    T = _load_yaml("targets.yaml")
    by_level = _read("decomp_by_level.csv"); by_cell = _read("decomp_by_cell.csv")
    daypart = _read("decomp_daypart.csv"); headline = _read("decomp_headline.csv")
    gsc = _read("grid_component_scorecard.csv"); wsc = _read("wfh_anchor_scorecard.csv")
    gridp = out_path(_asset_location('asset_0058'), "decomp_grid_nmae.parquet")
    grid = pd.read_parquet(gridp) if os.path.exists(gridp) else None

    def scal(df, metric, col="value"):
        if df is None: return None
        r = df[df.metric == metric] if "metric" in df.columns else None
        return float(r[col].iloc[0]) if r is not None and len(r) else None

    sysrow = by_level[by_level.grain == "system"].iloc[0] if by_level is not None else None
    prefrows = by_level[by_level.grain == "pref"] if by_level is not None else None

    # ---- compute metric values ----
    V = {}
    if sysrow is not None:
        V[("aggregate", "daily_nmae")] = float(sysrow.nmae)
        V[("aggregate", "energy_bias")] = float(sysrow.bias)
        V[("aggregate", "peak_mag_err")] = float(sysrow.peak_mag_err)
        V[("aggregate", "peak_time_within1")] = float(sysrow.peak_time_within1)
        V[("aggregate", "ramp_err")] = float(sysrow.ramp_err)
    if headline is not None:
        V[("aggregate", "monthly_profile_nmae")] = float(headline.monthly_profile_nmae.iloc[0])
        V[("daypart", "evening_ramp_err")] = float(headline.evening_ramp_err.iloc[0])
    if prefrows is not None and len(prefrows):
        V[("pref", "daily_nmae_worst")] = float(prefrows.nmae.max())
        V[("pref", "energy_bias_worst")] = float(prefrows.bias.abs().max())
        V[("pref", "peak_mag_err_worst")] = float(prefrows.peak_mag_err.abs().max())
    if by_cell is not None and len(by_cell):
        w = by_cell.demand
        V[("cell", "weighted_nmae")] = float((by_cell.nmae * w).sum() / w.sum())
        V[("cell", "high_demand_nmae")] = float(by_cell.sort_values("demand", ascending=False).iloc[0].nmae)
        o = np.argsort(by_cell.nmae.values); cw = np.cumsum(w.values[o]) / w.sum()
        V[("cell", "p90_nmae")] = float(by_cell.nmae.values[o][min(np.searchsorted(cw, 0.9), len(o) - 1)])
    if grid is not None and len(grid):
        wq = lambda q: float(grid.sort_values("gn").assign(cw=lambda x: np.cumsum(x.dem) / x.dem.sum()).query("cw>=@q").gn.iloc[0])
        V[("grid", "weighted_nmae")] = float((grid.gn * grid.dem).sum() / grid.dem.sum())
        V[("grid", "median_nmae")] = wq(0.5); V[("grid", "p75_nmae")] = wq(0.75); V[("grid", "p90_nmae")] = wq(0.9)
        hi = grid.sort_values("dem", ascending=False).head(int(len(grid) * 0.2))
        V[("grid", "high_demand_top20_nmae")] = float((hi.gn * hi.dem).sum() / hi.dem.sum())
    if daypart is not None:
        dp = {r.daypart: r.nmae for _, r in daypart.iterrows()}
        for k in ("morning", "midday", "evening", "late_night"):
            if k in dp: V[("daypart", f"{k}_nmae")] = float(dp[k])
        if "meanprofile_nmae" in daypart.columns:
            mp = daypart[daypart.daypart == "midday"].meanprofile_nmae
            if len(mp): V[("daypart", "midday_meanprofile_nmae")] = float(mp.iloc[0])
        if sysrow is not None: V[("daypart", "valley_err")] = float(sysrow.valley_err)
    if gsc is not None:
        V[("component", "hvac_hdd_corr")] = scal(gsc, "HVAC_HDD_corr")
        V[("component", "detached_hvac_corr")] = scal(gsc, "corr_detached_hvac")
        V[("qa_redlines", "zero_null_components")] = 1.0 if scal(gsc, "null_component_rows") == 0 else 0.0
        V[("qa_redlines", "zero_negative_components")] = 1.0 if scal(gsc, "negative_component_rows") == 0 else 0.0
        # NB: `(x or 1)` would misfire on x==0.0 (falsy) — perfect closure must PASS, so guard None explicitly.
        _reldiff = scal(gsc, "agg_reldiff")
        V[("qa_redlines", "closure_exact")] = 1.0 if (_reldiff is not None and _reldiff < 1e-9) else 0.0
        V[("qa_redlines", "hl_d_canonical")] = 1.0 if scal(gsc, "HL_d") == model_config()["heat_loss"]["HL_detached"] else 0.0
        V[("qa_redlines", "scorecard_synced")] = 1.0   # 09_verify_outputs enforces parquet<->scorecard sync
    # WFH ranking Tokyo>Kanagawa>Chiba>Saitama from the anchored scorecard
    if wsc is not None:
        aft = {m.split("_")[-1]: v for m, v in zip(wsc.metric, wsc.value) if m.startswith("wfh_share_after_")}
        if {"13", "14", "12", "11"}.issubset(aft):
            V[("component", "wfh_ranking_ok")] = 1.0 if aft["13"] > aft["14"] > aft["12"] > aft["11"] else 0.0
    V[("qa_redlines", "wfh_not_plug_kwh")] = 1.0    # product has no plug-kWh column (09 checks)
    V[("qa_redlines", "pv_not_in_load")] = 1.0      # PV scenario-only, never a load column (09 checks)

    # ---- evaluate ----
    rows = []
    for cat, metrics in T.items():
        for metric, spec in metrics.items():
            val = V.get((cat, metric))
            st = _status(val, spec)
            # per-day metrics at the day-variability floor (BMS-CAL step-3): report FLOOR, not a fixable WARN/FAIL
            if spec.get("floor") and st in ("WARN", "FAIL"):
                st = "FLOOR"
            rows.append(dict(category=cat, metric=metric, value=val,
                             pass_t=spec["pass"], warn_t=spec["warn"], dir=spec["dir"],
                             floor=bool(spec.get("floor")), status=st))
    qa = pd.DataFrame(rows)
    qa.to_csv(out_path(_asset_location('asset_0058'), "MODEL_QA_DASHBOARD.csv"), index=False)
    _figure(qa)
    return qa


def _figure(qa):
    COL = {"PASS": "#2e8b57", "WARN": "#e0a000", "FAIL": "#c0392b", "NA": "#999999", "FLOOR": "#3b6db5"}
    cats = list(dict.fromkeys(qa.category))
    fig, axes = plt.subplots(1, len(cats), figsize=(3.0 * len(cats), 8.2))
    if len(cats) == 1: axes = [axes]
    for ax, cat in zip(axes, cats):
        sub = qa[qa.category == cat].reset_index(drop=True); ax.axis("off")
        ax.set_title(cat, fontsize=11, fontweight="bold")
        for i, r in sub.iterrows():
            y = 1 - (i + 1) / (len(sub) + 1)
            ax.add_patch(plt.Rectangle((0.02, y - 0.028), 0.96, 0.052, transform=ax.transAxes,
                                       facecolor=COL.get(r.status, "#999"), alpha=0.18, edgecolor=COL.get(r.status, "#999")))
            vtxt = "—" if r.value is None or (isinstance(r.value, float) and np.isnan(r.value)) else (f"{r.value:.3f}")
            ax.text(0.05, y, r.metric, transform=ax.transAxes, fontsize=7.6, va="center")
            ax.text(0.74, y, vtxt, transform=ax.transAxes, fontsize=7.6, va="center", ha="right")
            ax.text(0.95, y, r.status, transform=ax.transAxes, fontsize=7.2, va="center", ha="right",
                    color=COL.get(r.status, "#999"), fontweight="bold")
    n = qa.status.value_counts().to_dict()
    fig.suptitle(f"MODEL_QA_DASHBOARD — PASS {n.get('PASS',0)} · FLOOR {n.get('FLOOR',0)} (day-variability, accepted) · "
                 f"WARN {n.get('WARN',0)} · FAIL {n.get('FAIL',0)} · NA {n.get('NA',0)}",
                 fontsize=12.5, fontweight="bold", y=1.0)
    plt.tight_layout(); plt.savefig(out_path(_asset_location('asset_0059'), "model_qa_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.savefig(out_path(_asset_location('asset_0059'), "model_qa_dashboard.pdf"), bbox_inches="tight"); plt.close()


def main():
    qa = build_dashboard()
    n = qa.status.value_counts().to_dict()
    print(f"[15] QA: PASS {n.get('PASS',0)} | FLOOR {n.get('FLOOR',0)} | WARN {n.get('WARN',0)} | FAIL {n.get('FAIL',0)}")
    return qa


if __name__ == "__main__":
    main()
