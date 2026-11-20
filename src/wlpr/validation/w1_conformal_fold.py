"""W1 — uncertainty wrapper for the FORWARD driver model (L1 family: load = a + b*HDD + c*CDD per
pref x daytype), the part of the product that generalizes to unobserved days. Ported from the source
`_maintainer/builders/build_w1_conformal_fold_stability.py` (Bayesian challenger withdrawn with supervisor).

1. LOMO-by-month (5 folds, Oct23-Feb24) refit at pref-daily level vs the FROZEN ground-truth LV
   observations (stdstats, flow=順 voltage=低圧, prefs 11-14).
2. Fold-spread coefficient stability table: 4 pref x 2 daytype x 3 coefs x 5 folds.
3. Calibrated MULTIPLICATIVE split-conformal 80% band (daily-twin method): for held-out month m,
   calibration scores = held-out relative residuals of the OTHER folds; band = pred * (1 +/- q80).
   Coverage evaluated per pref and per held-out month.
Day types from the empirical calendar registry (is_workday_defacto), NOT hand lists.

This is the producer of the two scorecards consumed by scenario/increment_bands.py
(w1_conformal_coverage.csv, w1_fold_coefficient_stability.csv).

Clean-port note: the source pinned the HDD/CDD degree-day bases by taking medians off the forward-2024
`l1_weather_daily.parquet` PV table; those medians are EXACTLY the package's centralized convention
(hdd_base 18.0 / cdd_base 24.0 in configs/model.yaml), so this port reads them from config and drops the
forward-PV L1 dependency with zero numerical change.

Claim boundary: wraps the forward DRIVER model's day-scatter risk; NOT the deployed slot twin's closure
accuracy, and it licenses no WFH plug-kWh claim.

    PYTHONPATH=src python -m wlpr.validation.w1_conformal_fold
"""
from __future__ import annotations

import os

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wlpr.io.paths import derived_path, input_path, model_config, out_path  # noqa: E402

OBS = input_path("lv_load_hourly", must_exist=False)
AME = input_path("weather_amedas", must_exist=False)  # UTC timestamps -> +9h JST before any daily cut
CAL = derived_path("calendar", "calendar_registry.parquet")
COV = out_path("scorecards", "w1_conformal_coverage.csv")
FOLD = out_path("scorecards", "w1_fold_coefficient_stability.csv")
FIG = out_path("figures", "validation", "w1_conformal_band.png")
PN = {"11": "Saitama", "12": "Chiba", "13": "Tokyo", "14": "Kanagawa"}
YMS = ["202310", "202311", "202312", "202401", "202402"]


def _fit(sub):
    X = np.column_stack([np.ones(len(sub)), sub.HDD.values, sub.CDD.values])
    beta, *_ = np.linalg.lstsq(X, sub.mwh.values, rcond=None)
    return beta


def _predict(beta, sub):
    return beta[0] + beta[1] * sub.HDD.values + beta[2] * sub.CDD.values


def build_w1_conformal_fold():
    for required in (OBS, AME):
        if not os.path.exists(required):
            raise FileNotFoundError(f"input missing: {required} (stage inputs before running validation)")
    cfg = model_config()
    hdd_base = float(cfg["weather"]["hdd_base_c"])   # 18.0 — == source l1_weather_daily median, exactly
    cdd_base = float(cfg["weather"]["cdd_base_c"])   # 24.0 — == source l1_weather_daily median, exactly

    con = duckdb.connect(); con.execute("SET memory_limit='4GB'"); con.execute("SET threads=3")
    day = con.execute(f"""
        select pref_code pref, date_str d, sum(kwh)/1000.0 mwh
        from read_parquet('{OBS}')
        where flow='順' and voltage='低圧' and pref_code in ('11','12','13','14')
          and substr(date_str,1,6) in ({",".join(f"'{y}'" for y in YMS)})
        group by 1,2""").df()

    # obs-window weather from frozen AMeDAS (UTC -> +9h JST); degree-day bases pinned to package convention
    stations = [r[0] for r in con.execute(f"select distinct station from read_parquet('{AME}')").fetchall()]
    smap = {}
    for code, cands in [("11", ["熊谷", "さいたま", "埼玉"]), ("12", ["千葉"]),
                        ("13", ["東京"]), ("14", ["横浜", "神奈川"])]:
        smap[code] = next((s for s in stations for c in cands if c in s), None)
    assert all(smap.values()), f"station map incomplete: {smap}"
    pairs = ",".join(f"('{k}','{v}')" for k, v in smap.items())
    wx = con.execute(f"""
        with mp as (select * from (values {pairs}) as t(pref, station))
        select mp.pref, strftime(a.timestamp + to_hours(9), '%Y%m%d') ymd, avg(a.temp) tmean
        from read_parquet('{AME}') a join mp on mp.station=a.station
        where strftime(a.timestamp + to_hours(9), '%Y%m') in ({",".join(f"'{y}'" for y in YMS)})
        group by 1,2""").df()
    con.close()
    wx["HDD"] = (hdd_base - wx.tmean).clip(lower=0)
    wx["CDD"] = (wx.tmean - cdd_base).clip(lower=0)
    print(f"[weather] AMeDAS JST daily rebuilt: bases HDD<{hdd_base:.1f} CDD>{cdd_base:.1f}, "
          f"stations {smap}, {len(wx)} pref-days")

    cal = pd.read_parquet(CAL)[["ymd", "is_workday_defacto", "is_yearend"]]
    df = (day.rename(columns={"d": "ymd"})
          .merge(wx[["pref", "ymd", "HDD", "CDD"]], on=["pref", "ymd"]).merge(cal, on="ymd"))
    df["ym"] = df.ymd.str[:6]
    df["dt"] = np.where(df.is_workday_defacto, "wd", "nonwd")
    assert df.groupby("pref").size().min() >= 150, "unexpected obs coverage"

    coef_rows, pred_rows = [], []
    for (pf, dt), g in df.groupby(["pref", "dt"]):
        for hold in YMS:
            tr, te = g[g.ym != hold], g[g.ym == hold]
            if len(te) == 0 or len(tr) < 20:
                continue
            beta = _fit(tr)
            coef_rows.append(dict(pref=PN[pf], daytype=dt, fold_heldout=hold,
                                  a_base=beta[0], b_hdd=beta[1], c_cdd=beta[2], n_train=len(tr)))
            yh = _predict(beta, te)
            for ymd, o, p_ in zip(te.ymd, te.mwh, yh):
                pred_rows.append(dict(pref=pf, daytype=dt, ym=hold, ymd=ymd, obs=o, pred=p_,
                                      is_yearend=bool(cal.set_index("ymd").loc[ymd, "is_yearend"])))
    coef = pd.DataFrame(coef_rows)
    pred = pd.DataFrame(pred_rows).sort_values(["pref", "ymd"])

    # ---- fold-spread stability table
    stab = (coef.melt(id_vars=["pref", "daytype", "fold_heldout"],
                      value_vars=["a_base", "b_hdd", "c_cdd"], var_name="coef")
            .groupby(["pref", "daytype", "coef"])
            .agg(mean=("value", "mean"), sd=("value", "std"),
                 vmin=("value", "min"), vmax=("value", "max"), nfold=("value", "size"))
            .reset_index())
    stab["cv_abs"] = (stab.sd / stab["mean"].abs()).round(4)
    stab[["mean", "sd", "vmin", "vmax"]] = stab[["mean", "sd", "vmin", "vmax"]].round(3)
    stab["evidence_label"] = "lomo_fold_spread_oct23_feb24"
    stab.to_csv(FOLD, index=False)

    # ---- multiplicative split-conformal 80% band
    pred["rres"] = (pred.obs - pred.pred).abs() / pred.pred
    cov_rows = []
    for pf, g in pred.groupby("pref"):
        for hold in YMS:
            calib = g[g.ym != hold].rres.values  # other folds' HELD-OUT residuals
            te = g[g.ym == hold]
            if len(te) == 0 or len(calib) < 30:
                continue
            q80 = float(np.quantile(calib, 0.80))
            inside = ((te.obs >= te.pred * (1 - q80)) & (te.obs <= te.pred * (1 + q80))).mean()
            pred.loc[te.index, "q80"] = q80
            cov_rows.append(dict(pref=PN[pf], heldout=hold, n_days=len(te), q80_band=round(q80, 4),
                                 coverage=round(float(inside), 3),
                                 nmae=round(float((te.obs - te.pred).abs().sum() / te.obs.sum()), 4)))
    cov = pd.DataFrame(cov_rows)
    overall = (cov.groupby("pref").apply(
        lambda x: pd.Series(dict(coverage=np.average(x.coverage, weights=x.n_days),
                                 q80_band=x.q80_band.mean(), nmae=np.average(x.nmae, weights=x.n_days))),
        include_groups=False).round(4).reset_index())
    overall["heldout"] = "ALL"; overall["n_days"] = cov.groupby("pref").n_days.sum().values
    cov = pd.concat([cov, overall], ignore_index=True)
    cov["evidence_label"] = "split_conformal_multiplicative_80pct_lomo"
    cov.to_csv(COV, index=False)
    print(cov[cov.heldout == "ALL"].to_string(index=False))

    _figure(pred, coef, cov)
    print(f"[saved] {FOLD} + {COV} + {FIG}")
    return cov, coef


def _figure(pred, coef, cov):
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [2.6, 1]})
    for i, pf in enumerate(["11", "12", "13", "14"]):
        g = pred[pred.pref == pf].sort_values("ymd").reset_index(drop=True)
        x = np.arange(len(g))
        ax = axes[i, 0]
        ax.fill_between(x, g.pred * (1 - g.q80), g.pred * (1 + g.q80),
                        color="#9ecae1", alpha=0.55, lw=0, label="80% conformal band")
        ax.plot(x, g.pred, color="#08519c", lw=1.1, label="LOMO held-out prediction")
        ax.scatter(x, g.obs, s=7, color="#252525", zorder=3, label="observed daily LV")
        ye = g[g.is_yearend]
        ax.scatter(x[ye.index], ye.obs, s=16, facecolors="none", edgecolors="#d62728",
                   zorder=4, label="year-end days")
        ov = cov[(cov.pref == PN[pf]) & (cov.heldout == "ALL")].iloc[0]
        ax.set_title(f"{PN[pf]} - coverage {ov.coverage:.2f} (target 0.80) | band +/-{ov.q80_band:.3f} | "
                     f"held-out daily NMAE {ov.nmae:.3f}", fontsize=9.5)
        mstart = g.groupby(g.ymd.str[:6]).apply(lambda s: s.index.min())
        ax.set_xticks(mstart.values); ax.set_xticklabels(mstart.index, fontsize=8)
        ax.set_ylabel("daily LV (MWh)")
        if i == 0:
            ax.legend(fontsize=7.5, ncol=2)
        ax.grid(alpha=0.25)
        axc = axes[i, 1]
        sub = coef[coef.pref == PN[pf]]
        labels, pos = [], 0
        for dt in ["wd", "nonwd"]:
            for cname in ["a_base", "b_hdd", "c_cdd"]:
                vals = sub[(sub.daytype == dt)][cname].values * 1e-3
                axc.plot([pos] * len(vals), vals, "o", ms=4, alpha=0.7, color="#08519c")
                axc.plot([pos - 0.18, pos + 0.18], [vals.mean()] * 2, color="#d62728", lw=1.6)
                labels.append(f"{dt}\n{cname.split('_')[1] if '_' in cname else cname}")
                pos += 1
        axc.set_xticks(range(pos)); axc.set_xticklabels(labels, fontsize=7)
        axc.set_ylabel("GWh-scale coef (5 LOMO folds)")
        axc.set_title("coefficient fold-spread", fontsize=9)
        axc.grid(alpha=0.25)
    fig.suptitle("W1 - forward driver model (a + b*HDD + c*CDD per pref x daytype): LOMO fold stability "
                 "+ calibrated multiplicative 80% conformal band | frozen LV ground truth | "
                 "no WFH plug-kWh claim", fontsize=11)
    fig.text(0.01, 0.005, "window Oct 2023 - Feb 2024 | daytype from empirical calendar registry | "
             "evidence: split_conformal_multiplicative_80pct_lomo", fontsize=7.5, color="#777777")
    fig.tight_layout(rect=[0, 0.012, 1, 0.97])
    fig.savefig(FIG, dpi=140)
    fig.savefig(FIG.replace(".png", ".pdf"))
    plt.close(fig)


def run_gates() -> bool:
    """The two scorecards exist; pooled coverage is within a sane band of the 0.80 conformal target."""
    fails = []
    for p in (COV, FOLD):
        if not os.path.exists(p):
            fails.append(f"missing {os.path.basename(p)}")
    if not fails:
        cov = pd.read_csv(COV)
        pooled = cov[cov.heldout == "ALL"]
        if not (0.70 <= float(pooled.coverage.mean()) <= 0.92):
            fails.append(f"pooled coverage {pooled.coverage.mean():.3f} outside [0.70,0.92] of 0.80 target")
        if not (pooled.q80_band > 0).all():
            fails.append("non-positive conformal band width")
    for f in fails:
        print(f"  [FAIL] {f}")
    if not fails:
        print("w1_conformal_fold: ALL GATES PASS (split-conformal 80% band + LOMO fold stability)")
    return not fails


def main():
    print("=== W1 conformal/fold uncertainty (forward driver model, LOMO + split-conformal 80%) ===")
    build_w1_conformal_fold()
    if not run_gates():
        raise SystemExit("w1_conformal_fold: GATE FAILS")


if __name__ == "__main__":
    main()
