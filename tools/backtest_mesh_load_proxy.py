#!/usr/bin/env python3
"""Whole-month mesh holdout; reconstruction accuracy, not no-WFH identification.

Training sums and counts exclude every row from the held-out month. The target
is the existing gross-load ledger (owner gross + nonowner import), before
subtracting the model's WFH component. No test-month WFH model is a label.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import importlib.util
import json
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/_asset_location('asset_0105')
spec=importlib.util.spec_from_file_location('seasonal',ROOT/'pipeline/22_seasonal_extra_experiments.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    scope=pd.read_parquet(m.SCOPE);scope=scope[scope.load_status.eq('positive_aggregate_load')].sort_values(['pref','mesh']).reset_index(drop=True)
    scope['pref']=scope.pref.astype(str);scope['mesh']=scope.mesh.astype(str)
    idx=pd.MultiIndex.from_frame(scope[['pref','mesh']]);months=sorted(m.DIRECT_MONTHS);n=len(scope)
    cal=m.dates_frame();daytype=dict(zip(cal.date,(cal.daytype=='nonworkday').astype(int)))
    sums=np.zeros((5,n*96));counts=np.zeros((5,n*96),dtype=np.int32)
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    def batches(ym):
        path=m.BASE/_asset_location('asset_0018')/f"{_asset_location('asset_0070')}{ym}"/_asset_location('asset_0062')
        sql=f"SELECT cast(pref AS varchar) pref,cast(mesh AS varchar) mesh,cast(date AS varchar) date,slot,L_owner_gross+L_nonowner_import target_MWh FROM read_parquet('{path}')"
        for batch in con.execute(sql).fetch_record_batch(131072):
            f=batch.to_pandas();ix=idx.get_indexer(pd.MultiIndex.from_frame(f[['pref','mesh']]))
            f=f[ix>=0].copy();ix=ix[ix>=0]
            dt=f.date.map(daytype).to_numpy(int)
            key=ix*96+dt*48+f.slot.to_numpy(int)
            yield f,ix,key,dt
    for j,ym in enumerate(months):
        for f,ix,key,dt in batches(ym):
            y=f.target_MWh.to_numpy();assert np.isfinite(y).all() and y.min()>=-1e-8
            sums[j]+=np.bincount(key,weights=y,minlength=n*96)
            counts[j]+=np.bincount(key,minlength=n*96).astype(np.int32)
        print(f'{ym}: training sufficient statistics built',flush=True)
    owner=pd.read_csv(OUT/_asset_location('asset_0194'),dtype={'pref':str,'month':str})
    owner=owner[owner.month.isin(months)&owner.pref.isin(scope.pref.unique())].copy()
    owner['date']=owner.timestamp.str[:8];owner['slot']=pd.to_datetime(owner.timestamp,format='%Y%m%d %H:%M').dt.hour*2+pd.to_datetime(owner.timestamp,format='%Y%m%d %H:%M').dt.minute//30
    owner['daytype']=owner.date.map(daytype);owner['gross']=owner.import_kWh+owner.self_consumption_kWh
    metrics=[];mesh_metrics=[];leak=[]
    for j,ym in enumerate(months):
        train=[x for x in range(5) if x!=j]
        total=sums[train].sum(axis=0);cnt=counts[train].sum(axis=0)
        ref=np.divide(total,cnt,out=np.full_like(total,np.nan),where=cnt>0)
        train_owner=owner[owner.month.ne(ym)]
        ore=train_owner.groupby(['pref','daytype','slot']).gross.mean()
        test=owner[owner.month.eq(ym)].copy();test['reference']=pd.MultiIndex.from_frame(test[['pref','daytype','slot']]).map(ore)
        test['factor']=test.gross/test.reference
        factors=test.set_index(['pref','date','slot']).factor
        # Accumulate actual, absolute error, signed error, squared error, n,
        # without saving hundreds of millions of redundant prediction rows.
        agg={model:np.zeros((n,5)) for model in ['owner_scaled','unscaled_reference']};missing=0
        for f,ix,key,dt in batches(ym):
            base=ref[key];valid=np.isfinite(base);missing+=int((~valid).sum())
            f=f[valid];ix=ix[valid];base=base[valid];y=f.target_MWh.to_numpy()
            factor=pd.MultiIndex.from_frame(f[['pref','date','slot']]).map(factors).to_numpy(float)
            assert np.isfinite(factor).all()
            for model,pred in [('owner_scaled',base*factor),('unscaled_reference',base)]:
                error=pred-y
                for c,values in enumerate([y,np.abs(error),error,error**2,np.ones(len(y))]):
                    agg[model][:,c]+=np.bincount(ix,weights=values,minlength=n)
        for model,a in agg.items():
            for pref in sorted(scope.pref.unique()):
                v=a[scope.pref.eq(pref)].sum(axis=0)
                metrics.append(dict(held_out_month=ym,pref=pref,model=model,target_MWh=v[0],absolute_error_MWh=v[1],
                                    signed_error_MWh=v[2],squared_error_MWh2=v[3],records=int(v[4]),
                                    NMAE=v[1]/v[0],relative_bias=v[2]/v[0],RMSE_MWh=np.sqrt(v[3]/v[4]),
                                    target='mesh gross-load processed ledger; not observed WFH or no-WFH effect'))
            z=scope[['pref','mesh']].copy();z['held_out_month']=ym;z['model']=model
            z[['target_MWh','absolute_error_MWh','signed_error_MWh','squared_error_MWh2','records']]=a
            mesh_metrics.append(z)
        leak.append(dict(held_out_month=ym,training_months=';'.join(months[x] for x in train),
                         held_out_rows_in_training=0,missing_training_reference_records=missing,
                         owner_train_max_month=train_owner.month.max(),test_month_owner_only_used_as_covariate=True))
        print(f'{ym}: complete mesh holdout scored; {missing} rows without training reference',flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'mesh_LV_month_holdout_metrics.csv',index=False)
    pd.concat(mesh_metrics,ignore_index=True).to_parquet(OUT/'mesh_LV_month_holdout_error_sums.parquet',index=False)
    pd.DataFrame(leak).to_csv(OUT/'mesh_LV_month_holdout_leakage_audit.csv',index=False)
    con.close()

if __name__=='__main__':main()
