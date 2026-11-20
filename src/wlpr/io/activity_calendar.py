"""One registered activity calendar for the 2023-04 to 2024-03 study year.

This specifies the modeled work arrangement; it is not an observed attendance
calendar. Ordinary residential load and generation retain every calendar date.
"""
from __future__ import annotations
import pandas as pd

START='2023-04-01'
END='2024-03-31'
VERSION='study_activity_calendar_20261113'
EXCLUDED_DATES=frozenset({
 '20230429','20230503','20230504','20230505','20230717','20230811',
 '20230918','20230923','20231009','20231103','20231123',
 '20231229','20231230','20231231','20240101','20240102','20240103',
 '20240108','20240211','20240212','20240223','20240320',
})
CLOSURE_DATES=frozenset({'20231229','20231230','20231231','20240102','20240103'})

def calendar_frame(months=None):
    dates=pd.date_range(START,END,freq='D')
    rows=[]
    for dt in dates:
        d=dt.strftime('%Y%m%d');weekend=dt.weekday()>=5
        active=not weekend and d not in EXCLUDED_DATES
        rows.append(dict(ymd=d,date=d,ym=d[:6],mon=dt.month,
          daytype='workday' if active else 'nonworkday',
          day_class='eligible' if active else ('weekend' if weekend else 'weekday_holiday'),
          is_workday_defacto=active,is_yearend=d in {'20231229','20231230','20231231'},
          is_holiday_window=d in EXCLUDED_DATES,is_project_closure=d in CLOSURE_DATES,
          source=VERSION))
    out=pd.DataFrame(rows)
    if months is not None:
        requested=[str(x) for x in months]
        missing=set(requested)-set(out.ym)
        if missing:raise ValueError('No registered activity calendar for months '+str(sorted(missing)))
        out=out[out.ym.isin(requested)].copy()
    return out.reset_index(drop=True)

def workday_factors(dates):
    lookup=calendar_frame().set_index('date').is_workday_defacto.to_dict()
    requested=[str(d) for d in dates]
    missing=set(requested)-set(lookup)
    if missing:raise ValueError('Dates outside registered study calendar: '+str(sorted(missing)[:10]))
    return {d:float(lookup[d]) for d in requested}
