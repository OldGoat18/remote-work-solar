#!/usr/bin/env python3
"""Leave-one-month-out test of owner scaling against observed aggregate LV import.

This tests a load proxy at prefecture scale, not mesh/no-WFH decomposition or
summer accuracy. No held-out total-load observation enters the reference.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/_asset_location('asset_0105')

def main():
    lv=pd.read_parquet(ROOT/_asset_location('asset_0195'))
    lv['pref']=lv.pref_code.astype(int);lv['date']=lv.date_str.astype(str);lv['slot']=lv.hour_str.astype(int)-1
    lv['time']=pd.to_datetime(lv.date,format='%Y%m%d')+pd.to_timedelta(lv.slot*30,unit='min')
    lv['month']=lv.time.dt.strftime('%Y%m');lv['weekend']=lv.time.dt.dayofweek>=5
    owner=pd.read_csv(OUT/_asset_location('asset_0194'))
    owner['time']=pd.to_datetime(owner.timestamp,format='%Y%m%d %H:%M')
    owner['owner_gross_kWh']=owner.import_kWh+owner.self_consumption_kWh
    d=lv.merge(owner[['pref','time','owner_gross_kWh']],on=['pref','time'],validate='one_to_one')
    assert len(d)==len(lv) and d.owner_gross_kWh.notna().all()
    predictions=[];metrics=[];keys=['pref','weekend','slot']
    for month in sorted(d.month.unique()):
        train=d[d.month!=month];test=d[d.month==month]
        ref=train.groupby(keys).agg(reference_LV_kWh=('kwh','mean'),reference_owner_kWh=('owner_gross_kWh','mean')).reset_index()
        x=test.merge(ref,on=keys,validate='many_to_one')
        for model in ['owner_scaled','unscaled_reference']:
            prediction=x.reference_LV_kWh*(x.owner_gross_kWh/x.reference_owner_kWh if model=='owner_scaled' else 1.)
            assert prediction.notna().all()
            z=x[['pref','time','month','kwh']].copy();z['prediction_kWh']=prediction;z['model']=model
            predictions.append(z)
            for pref,g in z.groupby('pref'):
                error=g.prediction_kWh-g.kwh
                metrics.append(dict(held_out_month=month,pref=pref,model=model,n=len(g),
                                    NMAE=error.abs().sum()/g.kwh.sum(),relative_bias=error.sum()/g.kwh.sum(),
                                    RMSE_kWh=np.sqrt(np.mean(error**2)),
                                    scope='aggregate forward LV import; no held-out import in reference; Oct-Feb only'))
    pd.DataFrame(metrics).to_csv(OUT/'aggregate_LV_month_holdout_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'aggregate_LV_month_holdout_predictions.parquet',index=False)

if __name__=='__main__':main()
