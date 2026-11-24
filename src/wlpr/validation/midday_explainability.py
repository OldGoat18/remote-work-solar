"""Decisive test: is the per-day midday residual reducible by available, generalizable day-features?

Ported verbatim (numerics unchanged) from the source `configured asset`.

Regress the signed midday residual on (daily HDD, daily mean temp, daily temp-range, weekend, day-of-week), per pref,
and report BOTH in-sample R^2 and leave-one-month-out CV R^2. If LOMO R^2 is meaningfully > 0, a feature-conditioned
component-aware calibration can reduce the scatter (we then build it). If LOMO R^2 ~ 0, the scatter is irreducible
day-variation given the frozen inputs -> documented obs floor (the honest 'perfection' is mean-profile + systematic-bias
removal, not per-day NMAE below the floor).
"""

from wlpr.io.bindings import asset as _asset_location
import os, datetime
import numpy as np, pandas as pd, duckdb
from wlpr.io.paths import input_path, out_path

STN = {'11': '埼玉熊谷', '12': '千葉', '13': '東京', '14': '神奈川横浜'}
MIDDAY = list(range(19, 31)); ALLM = ['202310', '202311', '202312', '202401', '202402']
HOL = {'20231009', '20231103', '20231123', '20231229', '20231230', '20231231',
       '20240101', '20240102', '20240103', '20240108', '20240211', '20240212', '20240223'}


def _r2(y, yhat, w):
    ybar = (y * w).sum() / w.sum()
    ss_res = (w * (y - yhat) ** 2).sum(); ss_tot = (w * (y - ybar) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def run_explainability():
    SH = input_path("closure_surface_share").replace("\\", "/")
    AME = input_path("weather_amedas").replace("\\", "/")
    tmp = out_path(_asset_location('asset_0085'), _asset_location('asset_0149')); os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{tmp}'"); con.execute("SET preserve_insertion_order=false")
    pds = con.execute(f"select pref, d, slot, sum(obs) o, sum(pv*oshare) p from '{SH}' group by 1,2,3").df()
    # weather daily features per pref
    wf = {}
    for pref, st in STN.items():
        t = con.execute(f"""select strftime(cast(timestamp+INTERVAL 9 HOUR as DATE),'%Y%m%d') d,
            avg(temp) tmean, max(temp)-min(temp) trange from '{AME}' where station='{st}' group by 1""").df()
        for _, r in t.iterrows():
            wf[(pref, r.d)] = (max(18 - r.tmean, 0), r.tmean, r.trange)
    is_wknd = lambda d: 1.0 if (datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])).weekday() >= 5 or d in HOL) else 0.0
    dow = lambda d: datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])).weekday()

    rows = []
    for (pref, d), g in pds.groupby(['pref', 'd']):
        g = g.set_index('slot').reindex(range(48)).fillna(0); o = g.o.values; p = g.p.values
        om = o[MIDDAY].sum()
        if om <= 0 or (pref, d) not in wf:
            continue
        hdd, tmean, trange = wf[(pref, d)]
        rows.append(dict(pref=pref, d=d, ym=d[:6], res=(p[MIDDAY].sum()-om)/om, w=om,
                         hdd=hdd, tmean=tmean, trange=trange, weekend=is_wknd(d), dow=dow(d)))
    R = pd.DataFrame(rows)

    # design matrix: intercept + hdd + tmean + trange + weekend + dow one-hot (6) ; fit per pref
    res = []
    for pref, g in R.groupby('pref'):
        g = g.copy()
        feats = ['hdd', 'tmean', 'trange', 'weekend']
        X = np.column_stack([np.ones(len(g))] + [g[f].values for f in feats] +
                            [(g.dow.values == k).astype(float) for k in range(6)])
        y = g.res.values; w = g.w.values
        Wsq = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(X * Wsq, y * np.sqrt(w), rcond=None)
        r2_in = _r2(y, X @ beta, w)
        # LOMO CV
        yhat = np.full(len(g), np.nan)
        for ho in ALLM:
            tr = g.ym != ho; teidx = np.where(g.ym.values == ho)[0]
            if tr.sum() < 10 or len(teidx) == 0:
                continue
            Xtr = X[tr.values]; ytr = y[tr.values]; wtr = w[tr.values]
            b, *_ = np.linalg.lstsq(Xtr * np.sqrt(wtr)[:, None], ytr * np.sqrt(wtr), rcond=None)
            yhat[teidx] = X[teidx] @ b
        m = ~np.isnan(yhat)
        r2_cv = _r2(y[m], yhat[m], w[m])
        res.append(dict(pref=pref, n=len(g), r2_insample=r2_in, r2_LOMO_cv=r2_cv,
                        resid_std=float(np.sqrt(np.average((y-np.average(y, weights=w))**2, weights=w)))))
    out = pd.DataFrame(res)
    out.to_csv(out_path(_asset_location('asset_0058'), "midday_explainability.csv"), index=False)
    return out


def main():
    expl = run_explainability()
    print("[15] midday LOMO-CV R^2 by pref: " + " ".join(f"{int(r.pref)}:{r.r2_LOMO_cv:+.2f}" for _, r in expl.iterrows())
          + "  (all <0 -> irreducible floor)")
    return expl


if __name__ == "__main__":
    main()
