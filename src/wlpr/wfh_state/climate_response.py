"""Layer 2 — per-date climate response (the §3 MAGNITUDE driver).

Ported verbatim (numerics unchanged) from the source
`configured asset`.

Turns AMeDAS weather into a per (pref, date) HVAC daily MAGNITUDE g_hdd/g_cdd. This is the climate-depth
upgrade over H1 (which used one monthly g_hdd/g_cdd): the daily magnitude tracks weather per date, giving
genuine per-date climate skill, while the intra-day SHAPE stays occupancy-driven in component_loads (so we
do NOT re-create the v87.6 instantaneous-temp bimodal-at-commute-peaks defect).

Weather red line: AMeDAS timestamps are UTC -> +9h JST before any join. Per-pref temperature is the
own-station / Tokyo-station blend (model.yaml pop_weight_own_station), matching the validated twin.
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd

from wlpr.io.paths import input_path, model_config

SMAP = {"11": "埼玉熊谷", "12": "千葉", "13": "東京", "14": "神奈川横浜"}
HDD_FULL, CDD_FULL = 12.0, 8.0   # saturation span: g reaches 1 at 12 HDD / 8 CDD (verbatim from H1)
CDD_P = 1.5                      # cooling CONVEXITY exponent (§13.5.5: cooling 比 heating 更非线性; heating stays linear)
HUMIDEX_K = 3.0                  # humidex: up to +3°C apparent temperature at 100% RH on a hot day (observed rhum)


def month_climatology_tmean(prefs, months):
    """Multi-year month-mean temperature per (pref, month), on the EXACT H1 caliber
    (build_wfh_attribution_layer.temp_profiles): avg temp per (pref, month, hour) over ALL years, 24-hour
    interpolate, then mean over the 24 hours. Own-station. This is what the mon-grain (PV-feeding) layer uses
    so the v2 S1 reproduces the locked identity; the per-DATE weather (daily_tmean) drives the date-grain G3."""
    ame = input_path("weather_amedas")
    st2pref = {SMAP[p]: p for p in prefs}
    slist = ",".join(f"'{SMAP[p]}'" for p in prefs)
    mlist = ",".join(str(int(m)) for m in months)
    con = duckdb.connect(); con.execute("SET threads=4")
    df = con.execute(f"""
        select station, cast(strftime(timestamp + to_hours(9), '%m') as int) m,
               cast(strftime(timestamp + to_hours(9), '%H') as int) hh, avg(temp) th
        from read_parquet('{ame}')
        where station in ({slist})
          and cast(strftime(timestamp + to_hours(9), '%m') as int) in ({mlist})
        group by 1, 2, 3""").df()
    con.close()
    out = {}
    for (st, m), g in df.groupby(["station", "m"]):
        arr = g.set_index("hh").th.reindex(range(24)).interpolate(limit_direction="both").values
        out[(st2pref[st], int(m))] = float(np.nanmean(arr))
    return out


def daily_tmean(prefs, months):
    """Per (pref, date) mean temperature (JST) from each prefecture's OWN AMeDAS station (SMAP), matching the
    H1 WFH-HVAC caliber (build_wfh_attribution_layer.temp_profiles uses own-station, NOT the LV-twin pop-weight
    blend) so the v2 mon-grain reproduces the locked S1 identity. Returns {(pref,'YYYYMMDD'): Tmean}; Tokyo is
    the fallback only when a pref's own station is missing a day. AMeDAS is UTC -> +9h before the date cut."""
    ame = input_path("weather_amedas")
    stations = sorted(set(SMAP.values()) | {"東京"})
    slist = ",".join(f"'{s}'" for s in stations)
    mlist = ",".join(str(int(m)) for m in months)
    con = duckdb.connect(); con.execute("SET threads=4")
    df = con.execute(f"""
        select station,
               strftime(timestamp + to_hours(9), '%Y%m%d') ymd,
               cast(strftime(timestamp + to_hours(9), '%m') as int) mon,
               avg(temp) tmean
        from read_parquet('{ame}')
        where station in ({slist})
          and cast(strftime(timestamp + to_hours(9), '%m') as int) in ({mlist})
        group by 1, 2, 3""").df()
    con.close()
    tok = df[df.station == "東京"].set_index("ymd").tmean.to_dict()
    out = {}
    for pref in prefs:
        own = df[df.station == SMAP[pref]].set_index("ymd").tmean.to_dict()
        for d in set(own) | set(tok):
            to = own.get(d)
            out[(pref, d)] = float(to if to is not None else tok.get(d))
    return out


def g_hdd(tmean):
    return min(max(18.0 - tmean, 0.0) / HDD_FULL, 1.0)


def humidex_temp(tmean, rhum):
    """Apparent (humidex) cooling temperature: humid air feels hotter, but ONLY when already warm. The bump is
    0 below 20°C (so winter is untouched) and ramps to HUMIDEX_K·(rhum-60)/40 above 28°C. Observed rhum."""
    warm = min(max((tmean - 20.0) / 8.0, 0.0), 1.0)          # 0 below 20°C, 1 above 28°C
    return tmean + HUMIDEX_K * max(rhum - 60.0, 0.0) / 40.0 * warm


def g_cdd(tmean, rhum=None):
    """Cooling magnitude — CONVEX in CDD (exponent CDD_P>1, more nonlinear than the linear heating ramp, §13.5.5)
    and, when observed rhum is supplied, driven by the HUMIDEX apparent temperature so humid summer days raise
    the MAIN cooling magnitude (not just redistribute it). Oct-Feb: tmean<24 and the humidex bump is ~0 (cold),
    so g_cdd≡0 -> the S1 PV-feed / LV floor / closure are untouched by construction. Monotone (Gate-5 holds)."""
    t = tmean if rhum is None else humidex_temp(tmean, rhum)
    return min((max(t - 24.0, 0.0) / CDD_FULL) ** CDD_P, 1.0)


def month_climatology_rhum(prefs, months):
    """Multi-year month-mean relative humidity per (pref, month) (own-station, JST). Feeds the humidex into the
    summer (cdd_prior) cooling magnitude in the mon-grain full-year emit. None where a station lacks the month."""
    ame = input_path("weather_amedas")
    st2pref = {SMAP[p]: p for p in prefs}
    slist = ",".join(f"'{SMAP[p]}'" for p in prefs)
    mlist = ",".join(str(int(m)) for m in months)
    con = duckdb.connect(); con.execute("SET threads=4")
    df = con.execute(f"""
        select station, cast(strftime(timestamp + to_hours(9), '%m') as int) m, avg(rhum) rh
        from read_parquet('{ame}')
        where station in ({slist})
          and cast(strftime(timestamp + to_hours(9), '%m') as int) in ({mlist})
        group by 1, 2""").df()
    con.close()
    return {(st2pref[st], int(m)): (float(rh) if rh == rh else None) for st, m, rh in df.itertuples(index=False)}


# --- §15.6 secondary climate variables (humidity / wind / solar) -------------------------------------------
# These are SECOND-ORDER magnitude modifiers on the HVAC daily magnitude only — never a direct total-load term
# (§16.3). They enter the DATE-GRAIN path and are window-renormalised to the survey anchor, so the mon-grain
# rep-day that feeds PV / the S1 bit-identity is UNTOUCHED. Evidence labels are explicit:
#   humidity (rhum), wind (wspd) = OBSERVED AMeDAS  -> tier "observed_secondary"
#   solar gain                   = geometry-derived (AMeDAS tsun is empty) -> tier "climate_prior_geometry"
RH_REF, BETA_RH = 60.0, 0.80      # cooling-comfort: humid days feel hotter -> more cooling demand
WS_REF, BETA_WS = 3.5, 0.18       # envelope loss (m/s ref): windy days lose more heat AND cool -> both HVAC up
BETA_SOLAR = 0.25                 # daytime solar gain (climate_prior): more summer sun -> more cooling
SEC_CLIP = (0.6, 1.6)             # bound every modifier so a single noisy day cannot dominate the window


def daily_secondary(prefs, months):
    """Per (pref, date) OBSERVED humidity (rhum, %) and wind (wspd) from each pref's OWN AMeDAS station
    (own-station caliber, Tokyo fallback), JST (+9h). Returns {(pref,'YYYYMMDD'): (rhum, wspd)}.
    AMeDAS wspd is km/h -> converted to m/s. Used ONLY as a date-grain second-order HVAC modifier."""
    ame = input_path("weather_amedas")
    stations = sorted(set(SMAP.values()) | {"東京"})
    slist = ",".join(f"'{s}'" for s in stations)
    mlist = ",".join(str(int(m)) for m in months)
    con = duckdb.connect(); con.execute("SET threads=4")
    df = con.execute(f"""
        select station, strftime(timestamp + to_hours(9), '%Y%m%d') ymd,
               avg(rhum) rhum, avg(wspd) wspd
        from read_parquet('{ame}')
        where station in ({slist})
          and cast(strftime(timestamp + to_hours(9), '%m') as int) in ({mlist})
        group by 1, 2""").df()
    con.close()
    tok = df[df.station == "東京"].set_index("ymd")[["rhum", "wspd"]].to_dict("index")
    out = {}
    for pref in prefs:
        own = df[df.station == SMAP[pref]].set_index("ymd")[["rhum", "wspd"]].to_dict("index")
        for d in set(own) | set(tok):
            rec = own.get(d) or tok.get(d)
            rh = rec.get("rhum"); ws = rec.get("wspd")
            rh = float(rh) if rh == rh and rh is not None else RH_REF
            ws = float(ws) / 3.6 if ws == ws and ws is not None else WS_REF   # km/h -> m/s
            out[(pref, d)] = (rh, ws)
    return out


def solar_gain_doy(doy):
    """Geometry-derived daily solar-gain magnitude in ~[0,1] (climate_prior; AMeDAS tsun is empty so this is
    NOT an observation). Tracks clear-sky daily insolation via solar declination: long high-sun summer days
    -> large daytime cooling gain; short low-sun winter days -> small. Used only to modulate cooling magnitude."""
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + int(doy)) / 365.0)
    lat = np.deg2rad(35.6)
    w0 = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))            # half-day angle
    # daily extraterrestrial insolation ∝ w0·sinφ·sinδ + cosφ·cosδ·sinw0, normalised by the Kanto annual
    # MAX (summer-solstice value ≈ 1.15) so summer ≈ 1.0 and winter ≈ 0.37 (genuine seasonal contrast).
    ins = w0 * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(w0)
    return float(np.clip(ins / 1.15, 0.0, 1.0))


def m_cool_humidity(rhum):
    """Cooling-comfort multiplier from observed RH (>1 on humid days). Centered at RH_REF=60%."""
    return float(np.clip(1.0 + BETA_RH * (rhum - RH_REF) / 100.0, *SEC_CLIP))


def m_env_wind(wspd):
    """Envelope-loss multiplier from observed wind (>1 on windy days). Applies to BOTH heating and cooling."""
    return float(np.clip(1.0 + BETA_WS * (wspd - WS_REF) / WS_REF, *SEC_CLIP))


def m_solar_cool(doy):
    """Solar-gain cooling multiplier (>1 in high-sun season). Geometry-derived climate_prior."""
    return float(np.clip(1.0 + BETA_SOLAR * (solar_gain_doy(doy) - 0.5), *SEC_CLIP))


def monthly_secondary_cool(prefs, months):
    """Monthly-mean cooling-comfort magnitude multiplier (humidity × wind × solar) per (pref, month). §15.6:
    humidity/wind/solar must drive a real cooling RESPONSE, not just a within-window redistribution. This is
    used for the NON-VALIDATED summer (cdd_prior) months, where there is NO survey anchor to redistribute
    within — so the secondary climate genuinely lifts the cooling magnitude there (the validated Oct-Feb window
    stays temperature-anchored, cooling≡0, so S1 is untouched). Returns {(pref, month): cool_mult}."""
    sec = daily_secondary(prefs, months)
    acc = {}
    for (pref, d), (rh, ws) in sec.items():
        acc.setdefault((pref, int(d[4:6])), []).append((rh, ws))
    out = {}
    for (pref, m), vals in acc.items():
        rh = float(np.mean([v[0] for v in vals])); ws = float(np.mean([v[1] for v in vals]))
        mid_doy = int(pd.Timestamp(2024, m, 15).dayofyear)
        out[(pref, m)] = float(np.clip(m_cool_humidity(rh) * m_env_wind(ws) * m_solar_cool(mid_doy), *SEC_CLIP))
    return out


def secondary_factors(pref, date, sec_map):
    """Combined per-date HVAC magnitude modifiers (heat_mult, cool_mult) from the secondary climate vars.
    heating: wind envelope loss only.  cooling: humidity comfort × wind envelope × solar gain.
    Returns (heat_mult, cool_mult) both ~1.0; missing data -> (1.0, 1.0) (neutral)."""
    rec = sec_map.get((pref, date))
    doy = int(pd.Timestamp(int(date[:4]), int(date[4:6]), int(date[6:8])).dayofyear)
    if rec is None:
        return 1.0, m_solar_cool(doy)
    rh, ws = rec
    mw = m_env_wind(ws)
    heat_mult = mw
    cool_mult = float(np.clip(m_cool_humidity(rh) * mw * m_solar_cool(doy), *SEC_CLIP))
    return heat_mult, cool_mult
