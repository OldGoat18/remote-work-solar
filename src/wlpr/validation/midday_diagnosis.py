"""Sharp diagnosis of the per-day midday residual — find its STRUCTURE (systematic & fixable) vs zero-mean noise.

Ported verbatim (numerics unchanged) from the source `configured asset`.

Key fact: product pred = pred_v4(pref) x observed-share, so the grid shape error == the pref shape error. The midday
WARN is therefore a pref x day midday-shape residual of pred_v4. We test whether that residual is systematic along:
  - weekday vs weekend (midday is higher at home on weekends),
  - daily HDD (cold-day midday HVAC),
  - month,
and split per-day midday NMAE into systematic (mean) vs scatter (noise).
"""

from wlpr.io.bindings import asset as _asset_location
import os
import numpy as np, pandas as pd, duckdb
from wlpr.io.paths import input_path, out_path

MLAB = {'202310': 'Oct', '202311': 'Nov', '202312': 'Dec', '202401': 'Jan', '202402': 'Feb'}
MIDDAY = list(range(19, 31))   # 09:30-15:00
PN = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}
STN = {'11': '埼玉熊谷', '12': '千葉', '13': '東京', '14': '神奈川横浜'}
HOL = {'20231009', '20231103', '20231123', '20231229', '20231230', '20231231',
       '20240101', '20240102', '20240103', '20240108', '20240211', '20240212', '20240223'}


def diagnose_midday():
    import datetime
    SH = input_path("closure_surface_share").replace("\\", "/")
    AME = input_path("weather_amedas").replace("\\", "/")
    tmp = out_path(_asset_location('asset_0085'), _asset_location('asset_0148')); os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{tmp}'"); con.execute("SET preserve_insertion_order=false")
    # pref x day x slot obs/pred
    pds = con.execute(f"""select pref, d, slot, sum(obs) o, sum(pv*oshare) p
        from '{SH}' group by 1,2,3""").df()
    # daily HDD per pref (own station, +9h JST, base 18)
    hdd = {}
    for pref, st in STN.items():
        t = con.execute(f"""select strftime(cast(timestamp+INTERVAL 9 HOUR as DATE),'%Y%m%d') d, avg(temp) t
            from '{AME}' where station='{st}' group by 1""").df()
        for _, r in t.iterrows():
            hdd[(pref, r.d)] = max(18 - r.t, 0)
    is_wknd = lambda d: datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])).weekday() >= 5 or d in HOL

    rows = []
    for (pref, d), g in pds.groupby(['pref', 'd']):
        g = g.set_index('slot').reindex(range(48)).fillna(0)
        o = g.o.values; p = g.p.values
        om = o[MIDDAY].sum()
        if om <= 0:
            continue
        mid_res = (p[MIDDAY].sum() - om) / om           # signed midday level residual
        mid_nmae = np.abs(p[MIDDAY] - o[MIDDAY]).sum() / om
        rows.append(dict(pref=pref, d=d, ym=d[:6], weekend=is_wknd(d), hdd=hdd.get((pref, d), np.nan),
                         midday_obs=om, mid_res=mid_res, mid_nmae=mid_nmae))
    R = pd.DataFrame(rows)

    out = {}
    # 1) systematic vs scatter (demand-weighted) overall and by weekend
    def split(df):
        w = df.midday_obs
        sysmean = float((df.mid_res * w).sum() / w.sum())          # signed systematic bias
        nmae = float((df.mid_nmae * w).sum() / w.sum())
        scatter = float(np.sqrt((((df.mid_res - sysmean)**2) * w).sum() / w.sum()))
        return pd.Series(dict(midday_nmae=nmae, systematic_bias=sysmean, scatter_std=scatter, n=len(df)))
    out['overall'] = R.groupby(lambda x: 'all').apply(split).reset_index(drop=True)
    out['by_weekend'] = R.groupby('weekend').apply(split).reset_index()
    out['by_month'] = R.groupby('ym').apply(split).reset_index().assign(month=lambda x: x.ym.map(MLAB))
    out['by_pref_weekend'] = R.groupby(['pref', 'weekend']).apply(split).reset_index().assign(pref_name=lambda x: x.pref.map(PN))
    # 2) residual vs HDD correlation (is cold-day midday systematic?)
    rr = R.dropna(subset=['hdd'])
    out['hdd_corr'] = pd.DataFrame([dict(
        corr_midres_hdd=float(np.corrcoef(rr.mid_res, rr.hdd)[0, 1]),
        corr_midres_hdd_weekday=float(np.corrcoef(rr[~rr.weekend].mid_res, rr[~rr.weekend].hdd)[0, 1]) if (~rr.weekend).sum() > 2 else np.nan,
        corr_midres_hdd_weekend=float(np.corrcoef(rr[rr.weekend].mid_res, rr[rr.weekend].hdd)[0, 1]) if (rr.weekend).sum() > 2 else np.nan,
    )])
    for k, v in out.items():
        v.to_csv(out_path(_asset_location('asset_0058'), f"midday_diag_{k}.csv"), index=False)
    R.to_csv(out_path(_asset_location('asset_0058'), "midday_diag_perday.csv"), index=False)
    out['_R'] = R
    return out


def main():
    out = diagnose_midday()
    hc = out['hdd_corr'].iloc[0]
    print(f"[15] midday residual↔HDD corr {hc.corr_midres_hdd:+.3f}")
    return out


if __name__ == "__main__":
    main()
