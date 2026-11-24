#!/usr/bin/env python3
"""Read actual two-year owner data; test a first-year kernel on year two.

Owner forecast error is NOT validation of total LV load or WFH causality.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/_asset_location('asset_0105')

def main():
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1GB'")
    rows=[]
    for path in sorted((ROOT/_asset_location('asset_0203')).rglob(_asset_location('asset_0062'))):
        p=str(path).replace("'","''")
        r=con.execute(f"""SELECT count(*) n_rows,count(distinct mesh) meshes,
            min(ymd) min_date,max(ymd) max_date,count(distinct ymd) dates,
            count(distinct (ymd,slot)) date_slots,min(slot) min_slot,max(slot) max_slot,
            count(*) FILTER(WHERE ymd IS NULL OR slot IS NULL OR gross_btm_load_kwh IS NULL OR pv_gen_kwh IS NULL) null_rows,
            count(*) FILTER(WHERE ymd IS NULL OR slot IS NULL) null_key_rows,
            count(*) FILTER(WHERE gross_btm_load_kwh IS NULL) null_load_rows,
            count(*) FILTER(WHERE pv_gen_kwh IS NULL) null_generation_rows,
            sum(gross_btm_load_kwh) visible_gross_load_kWh,sum(pv_gen_kwh) visible_generation_kWh,
            max(abs(gross_btm_load_kwh-selfcons_kwh-import_kwh)) load_identity_error_kWh,
            max(abs(pv_gen_kwh-selfcons_kwh-export_kwh)) pv_identity_error_kWh
            FROM read_parquet('{p}')""").df().iloc[0].to_dict()
        r.update(pref=path.parent.parent.name.split('=')[-1],month=path.parent.name.split('=')[-1],path=str(path.relative_to(ROOT)))
        rows.append(r)
        print(f"Owner partition {r['pref']}/{r['month']}: {r['dates']} actual dates",flush=True)
    pd.DataFrame(rows).to_csv(OUT/_asset_location('asset_0202'),index=False)
    d=pd.read_csv(OUT/_asset_location('asset_0194'))
    d['time']=pd.to_datetime(d.timestamp,format='%Y%m%d %H:%M')
    for c in ['self_consumption_kWh','import_kWh','export_kWh','installed_capacity_kW','contracts']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    assert d[['self_consumption_kWh','import_kWh','export_kWh','installed_capacity_kW','contracts']].notna().all().all()
    d['generation_kWh']=d.self_consumption_kWh+d.export_kWh
    d['gross_load_kWh']=d.self_consumption_kWh+d.import_kWh
    aggregate=d.groupby(['pref','month']).agg(aggregate_gross_load_kWh=('gross_load_kWh','sum'),
                                             aggregate_generation_kWh=('generation_kWh','sum')).reset_index()
    visibility=pd.DataFrame(rows)
    visibility['pref']=visibility.pref.astype(int);visibility['month']=visibility.month.astype(int)
    visibility=visibility.merge(aggregate,on=['pref','month'],validate='one_to_one')
    visibility['visible_load_fraction']=visibility.visible_gross_load_kWh/visibility.aggregate_gross_load_kWh
    visibility['visible_generation_fraction']=visibility.visible_generation_kWh/visibility.aggregate_generation_kWh
    visibility['null_row_fraction']=visibility.null_rows/visibility.n_rows
    visibility.to_csv(OUT/'owner_mesh_visibility.csv',index=False)
    d['calendar_month']=d.time.dt.month;d['slot']=d.time.dt.hour*2+d.time.dt.minute//30
    d['weekend']=d.time.dt.dayofweek>=5
    train=d[d.time<'2024-04-01'].copy();test=d[d.time>='2024-04-01'].copy()
    predictions=[];metrics=[]
    for target,denom in [('generation_kWh','installed_capacity_kW'),('gross_load_kWh','contracts')]:
        train['normalized']=train[target]/train[denom]
        for name,keys in [('month_slot',['pref','calendar_month','slot']),('month_daytype_slot',['pref','calendar_month','weekend','slot']),('annual_slot_baseline',['pref','slot'])]:
            kernel=train.groupby(keys).normalized.mean().rename('unit_prediction').reset_index()
            pred=test.merge(kernel,on=keys,how='left',validate='many_to_one')
            pred['prediction_kWh']=pred.unit_prediction*pred[denom]
            assert pred.prediction_kWh.notna().all()
            for (pref,month),g in pred.groupby(['pref','month']):
                error=g.prediction_kWh-g[target];scale=g[target].sum()
                metrics.append(dict(target=target,model=name,pref=pref,month=month,n=len(g),
                                    NMAE=error.abs().sum()/scale,relative_bias=error.sum()/scale,
                                    RMSE_kWh=np.sqrt(np.mean(error**2)),
                                    train_period='202304-202403',test_period='202404-202503',
                                    boundary='observed owner cohort only; not total LV or WFH magnitude validation'))
            pred['target']=target;pred['model']=name;pred['observed_kWh']=pred[target]
            predictions.append(pred[['pref','timestamp','target','model','prediction_kWh','observed_kWh']])
    pd.DataFrame(metrics).to_csv(OUT/'owner_year2_holdout_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'owner_year2_holdout_predictions.parquet',index=False)
    (OUT/'owner_holdout_manifest.json').write_text(json.dumps(dict(
        method='Year-one fitted unit kernel; year-two unseen electricity outcomes. Same-slot year-two capacity/contracts are exposure offsets, not forecast targets.',
        fitted_rows=len(train),test_rows=len(test),prefectures=sorted(d.pref.unique().tolist()),
        forbidden_claims=['independent WFH effect','all-household LV load','feeder feasibility']),indent=2))

if __name__=='__main__': main()
