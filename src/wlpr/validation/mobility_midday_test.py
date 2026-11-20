"""Decisive test: does RAW daily mobility presence explain the per-day midday residual that weather/calendar could not?

Ported verbatim (numerics unchanged) from the source `validation/mobility_midday_test.py`.

Joins raw-tar-derived daily pref presence (midday + night) to the per-day midday diagnosis (Dec-Feb overlap), and asks:
  A. corr(midday presence, midday OBS load)   — does presence track residential midday load? (mechanism validation)
  B. corr(home-retention = mid/night, signed midday residual) — does presence explain the residual direction?
  C. LOMO-CV R^2 of  mid_res ~ home_retention (+hdd+weekend)  vs weather-only — does presence give a GENERALIZABLE
     signal (positive held-out R^2) where weather alone was negative? (the floor-breaking test)
Mobility is presence-support, NOT load truth — used here as an occupancy DRIVER/validation, never as kWh.
"""
import numpy as np, pandas as pd
from wlpr.io.paths import input_path, out_path

MO = ['202312', '202401', '202402']
PN = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}


def _r2(y, yhat, w):
    yb = (y*w).sum()/w.sum(); ss = (w*(y-yhat)**2).sum(); st = (w*(y-yb)**2).sum()
    return 1 - ss/st if st > 0 else np.nan


def _lomo_r2(df, feats):
    y = df.mid_res.values; w = df.midday_obs.values
    X = np.column_stack([np.ones(len(df))] + [df[f].values for f in feats])
    yhat = np.full(len(df), np.nan)
    for ho in df.ym.unique():
        tr = df.ym.values != ho; te = np.where(df.ym.values == ho)[0]
        if tr.sum() < 8 or len(te) == 0:
            continue
        b, *_ = np.linalg.lstsq(X[tr]*np.sqrt(w[tr])[:, None], y[tr]*np.sqrt(w[tr]), rcond=None)
        yhat[te] = X[te] @ b
    m = ~np.isnan(yhat)
    return _r2(y[m], yhat[m], w[m])


def run_mobility_test():
    mob = pd.read_csv(input_path("mobility_presence_panel"))   # frozen raw-derived presence panel
    mob['pref'] = mob.pref.astype(str); mob['date'] = mob.date.astype(str)
    mob = mob[(mob.midday_presence > 0) & (mob.night_presence > 0)].copy()
    mob['home_retention'] = mob.midday_presence / mob.night_presence
    # residential-load-weighted (emphasises residential occupancy; unweighted is commercial-contaminated)
    if 'midday_presence_w' in mob.columns:
        mob['home_retention_w'] = mob.midday_presence_w / mob.night_presence_w
    diag = pd.read_csv(out_path("scorecards", "midday_diag_perday.csv"))
    diag['pref'] = diag.pref.astype(str); diag['d'] = diag.d.astype(str)
    df = diag.merge(mob, left_on=['pref', 'd'], right_on=['pref', 'date'])
    df = df[df.ym.astype(str).isin(MO)].copy()
    out = {}
    # A presence vs obs midday load (pooled, demand-normalised within pref to remove scale)
    df['obs_z'] = df.groupby('pref').midday_obs.transform(lambda x: (x-x.mean())/x.std())
    df['pres_z'] = df.groupby('pref').midday_presence.transform(lambda x: (x-x.mean())/x.std())
    df['hr_z'] = df.groupby('pref').home_retention.transform(lambda x: (x-x.mean())/x.std())
    haw = 'home_retention_w' in df.columns
    rows = []
    for pref, g in df.groupby('pref'):
        rec = dict(pref=pref, pref_name=PN[pref], n=len(g),
                   corr_presence_obs=float(np.corrcoef(g.midday_presence, g.midday_obs)[0, 1]),
                   corr_homeret_resid=float(np.corrcoef(g.home_retention, g.mid_res)[0, 1]),
                   r2_weather_only=_lomo_r2(g, ['hdd', 'weekend']),
                   r2_weather_plus_presence=_lomo_r2(g, ['hdd', 'weekend', 'home_retention', 'midday_presence']))
        if haw:
            gg = g.dropna(subset=['home_retention_w', 'midday_presence_w'])
            rec['corr_presW_obs'] = float(np.corrcoef(gg.midday_presence_w, gg.midday_obs)[0, 1]) if len(gg) > 3 else np.nan
            rec['corr_homeretW_resid'] = float(np.corrcoef(gg.home_retention_w, gg.mid_res)[0, 1]) if len(gg) > 3 else np.nan
            rec['r2_weather_plus_presW'] = _lomo_r2(gg, ['hdd', 'weekend', 'home_retention_w', 'midday_presence_w']) if len(gg) > 12 else np.nan
        rows.append(rec)
    out['by_pref'] = pd.DataFrame(rows)
    # pooled within-pref-standardised
    out['pooled'] = pd.DataFrame([dict(
        n=len(df),
        corr_presZ_obsZ=float(np.corrcoef(df.pres_z, df.obs_z)[0, 1]),
        corr_hrZ_resid=float(np.corrcoef(df.hr_z, df.mid_res)[0, 1]))])
    for k, v in out.items():
        v.to_csv(out_path("scorecards", f"mobility_midday_test_{k}.csv"), index=False)
    return out


def main():
    out = run_mobility_test()["by_pref"]
    print("[15] mobility +presence LOMO-CV R^2: " + " ".join(
        f"{int(r.pref)}:{r.r2_weather_plus_presence:+.2f}" for _, r in out.iterrows()) + "  (stays <0 -> floor robust)")
    return out


if __name__ == "__main__":
    main()
