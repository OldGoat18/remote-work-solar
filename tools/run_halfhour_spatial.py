#!/usr/bin/env python3
"""Reconstruct monthly mesh x date x slot D from original partition inputs.

B is constant across dates within each source month/slot. Sorting D across
dates is an exact sufficient statistic for sum_date min(D, B), including
after moving the complete B trajectory with one all-year mesh mapping.
This avoids storing or permuting a 600-million-row table on disk.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location
import argparse
import calendar
import gc
import importlib.util
import json
import os
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from run_spatial_order_uncertainty import spatial_inputs, make_permutations, SEED

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/_asset_location('asset_0003')
OUT=ROOT/_asset_location('asset_0105')
spec=importlib.util.spec_from_file_location('seasonal',ROOT/'pipeline/22_seasonal_extra_experiments.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
COMP=['c_office_device','c_lighting','c_lunch','c_heating','c_cooling','c_other']

@njit(parallel=True)
def score_maps(sorted_D,prefix,B,maps):
    scores=np.zeros(len(maps));nd=sorted_D.shape[2]
    for k in prange(len(maps)):
        score=0.
        for i in range(sorted_D.shape[0]):
            donor=maps[k,i]
            for t in range(sorted_D.shape[1]):
                b=B[donor,t]
                lo=0;hi=nd
                while lo<hi:
                    mid=(lo+hi)//2
                    if sorted_D[i,t,mid]<b:lo=mid+1
                    else:hi=mid
                score+=prefix[i,t,lo]+(nd-lo)*b
        scores[k]=score
    return scores

def reconstruct_month(ym,keys,con):
    nd=calendar.monthrange(int(ym[:4]),int(ym[4:]))[1]
    D=np.zeros((len(keys),20,nd),dtype=np.float64)
    seen=np.zeros(D.shape,dtype=bool)
    blo=np.full((len(keys),20),np.inf);bhi=np.zeros_like(blo)
    idx=pd.MultiIndex.from_frame(keys[['pref','mesh']].astype(str))
    audit=[]
    ref=SRC/'load_reference_mesh_daytype_slot.parquet'
    for pref in sorted(keys.pref.unique()):
        sql,evidence=m.month_source_sql(con,ym,m.dates_frame(),ref,
            SRC/'owner_load_reference_pref_daytype_slot.parquet',SRC/'observed_stock_yield_kernel_full_year.parquet')
        sql=sql.replace(_asset_location('asset_0018'),f"{_asset_location('asset_0016')}{pref}")
        if ym not in m.DIRECT_MONTHS:
            # Restrict ONLY to this pref and slots used in the statistic;
            # prefecture owner scaling remains identical at these slots.
            sql=sql.replace(f"FROM read_parquet('{m.qpath(ref)}') r",
                            f"FROM (SELECT * FROM read_parquet('{m.qpath(ref)}') WHERE cast(pref AS varchar)='{pref}') r")
            sql=sql.replace('GROUP BY 1,2,3,4','WHERE o.slot BETWEEN 17 AND 36 GROUP BY 1,2,3,4')
        query=sql+' SELECT mesh,pref,date,slot,D,B FROM z WHERE slot BETWEEN 17 AND 36'
        count=0
        for batch in con.execute(query).fetch_record_batch(131072):
            f=batch.to_pandas();ix=idx.get_indexer(pd.MultiIndex.from_frame(f[['pref','mesh']].astype(str)))
            keep=ix>=0
            f=f[keep];ix=ix[keep]
            if not len(f):continue
            t=f.slot.to_numpy(int)-17;day=f.date.astype(str).str[-2:].astype(int).to_numpy()-1
            flat=(ix*20+t)*nd+day
            assert len(np.unique(flat))==len(flat) and not seen.ravel()[flat].any(),'Duplicate mesh/date/slot'
            vals=f.D.to_numpy();b=f.B.to_numpy()
            assert np.isfinite(vals).all() and np.isfinite(b).all() and min(vals.min(),b.min())>=-1e-9
            D.ravel()[flat]=vals;seen.ravel()[flat]=True
            np.minimum.at(blo,(ix,t),b);np.maximum.at(bhi,(ix,t),b)
            count+=len(f)
        audit.append(dict(ym=ym,pref=pref,rows=count,evidence_tier=evidence))
        print(f'{ym} pref {pref}: {count} records reconstructed',flush=True)
    present=seen.any(axis=2)
    assert np.max(np.abs(bhi[present]-blo[present]))<1e-10,'B varies by date; sufficient statistic invalid'
    # Trajectory permutation needs a complete common calendar for each mesh.
    # Partial calendars are not silently filled with zero.
    counts=seen.sum(axis=2)
    assert np.isin(counts,[0,nd]).all(),'Partial mesh calendar requires explicit missing-data treatment'
    return D,bhi,seen,audit

def components(con,ym,keys,scenario):
    if ym in m.DIRECT_MONTHS and scenario=='S1':
        source=m.WFH_REGISTERED;condition="scenario_id='S1'"
    else:
        source=ROOT/f"{_asset_location('asset_0021')}{scenario}{_asset_location('asset_0022')}"
        condition='true'
    fields=['dwfh']+COMP
    pref_col='' if ym in m.DIRECT_MONTHS and scenario=='S1' else ',cast(pref AS varchar) pref'
    f=con.execute(f"SELECT cast(mesh AS varchar) mesh{pref_col},slot,{','.join(fields)} FROM read_parquet('{source}') WHERE mon={int(ym[4:])} AND {condition}").df()
    assert not f.duplicated(['mesh','slot']).any()
    assert keys.mesh.is_unique,'Component source lacks pref key'
    if pref_col:
        ix=pd.MultiIndex.from_frame(keys[['pref','mesh']].astype(str)).get_indexer(pd.MultiIndex.from_frame(f[['pref','mesh']]))
    else:ix=pd.Index(keys.mesh).get_indexer(f.mesh)
    good=ix>=0
    f=f[good];ix=ix[good];t=f.slot.to_numpy(int)-17
    assert ((t>=0)&(t<20)).all()
    arr=np.zeros((len(keys),20,len(fields)));arr[ix,t]=f[fields].to_numpy()
    assert np.max(np.abs(arr[:,:,0]-arr[:,:,1:].sum(axis=2)))<1e-8
    return arr

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--months',nargs='*');parser.add_argument('--draws',type=int,default=2000)
    args=parser.parse_args();set_num_threads(min(4,os.cpu_count() or 1))
    monthly=pd.read_parquet(SRC/'seasonal_mesh_month_fields.parquet')
    keys,_,_=spatial_inputs(monthly)
    complete_mesh=(monthly.groupby('mesh').source_mesh_present.min()==1)
    keys=keys[keys.mesh.isin(complete_mesh[complete_mesh].index)].reset_index(drop=True)
    print(f'Using {len(keys)} meshes complete in all 12 source months; excluded {34307-len(keys)} absent-month meshes',flush=True)
    annual=pd.read_parquet(SRC/'seasonal_date_slot_fields.parquet')
    permutations={};ids={}
    for model,draw,p in make_permutations(keys,args.draws):
        permutations.setdefault(model,[]).append(p)
    permutations={k:np.stack(v) for k,v in permutations.items()}
    con=duckdb.connect();con.execute("SET memory_limit='3GB'");con.execute('SET threads=2')
    tmp=OUT/_asset_location('asset_0007');tmp.mkdir(exist_ok=True);con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    months=args.months or sorted(monthly.ym.unique())
    for ym in months:
        out=OUT/f'halfhour_{ym}.json'
        if out.exists() and json.loads(out.read_text()).get('draws')==args.draws:
            print(f'{ym} reusing checked month outputs',flush=True);continue
        D,B,seen,audit=reconstruct_month(ym,keys,con);nd=D.shape[2]
        g=monthly[monthly.ym.eq(ym)&monthly.mesh.isin(keys.mesh)]
        expected=g.behavior_absorbed_MWh.sum();actual=np.minimum(D,B[:,:,None]).sum()
        budget=(B*seen.sum(axis=2)).sum()
        assert abs(actual-expected)<1e-5,f'Identity mismatch {ym}: {actual} vs {expected}'
        assert abs(budget-g.behavior_supply_MWh.sum())<1e-5,f'B scope mismatch {ym}'
        # Uniform zero source meshes, if present, cannot receive B under a
        # constrained null; unrestricted sensitivity reports this separately.
        assert seen.all(),'Unobserved target cells prevent an exact complete-trajectory null'
        sorted_D=np.sort(D,axis=2);prefix=np.concatenate([np.zeros((*D.shape[:2],1)),np.cumsum(sorted_D,axis=2)],axis=2)
        identity=score_maps(sorted_D,prefix,B,np.arange(len(keys),dtype=np.int32)[None,:])[0]
        assert abs(identity-expected)<1e-5
        rows=[]
        for model,maps in permutations.items():
            scores=score_maps(sorted_D,prefix,B,maps)
            rows.extend(dict(ym=ym,model=model,draw=i,absorption_MWh=x) for i,x in enumerate(scores))
        pd.DataFrame(rows).to_parquet(OUT/f'halfhour_null_{ym}.parquet',index=False)
        ablations=[]
        for scenario in ['S1','S8','S10']:
            c=components(con,ym,keys,scenario)
            if scenario=='S1':assert np.max(np.abs(c[:,:,0]-B))<1e-8
            for omit in ['none','office','lighting_lunch','hvac']:
                keep=c[:,:,0].copy()
                if omit=='office':keep-=c[:,:,1]
                if omit=='lighting_lunch':keep-=c[:,:,2]+c[:,:,3]
                if omit=='hvac':keep-=c[:,:,4]+c[:,:,5]
                assert keep.min()>-1e-9
                keep=np.maximum(keep,0)
                score=score_maps(sorted_D,prefix,keep,np.arange(len(keys),dtype=np.int32)[None,:])[0]
                ablations.append(dict(ym=ym,scenario=scenario,omitted_component=omit,
                                      annualized_label='monthly actual day count',behavior_MWh=keep.sum()*nd,
                                      absorption_MWh=score,baseline_surplus_MWh=g.surplus_MWh.sum()))
        pd.DataFrame(ablations).to_csv(OUT/f'halfhour_ablation_{ym}.csv',index=False)
        info=dict(ym=ym,draws=args.draws,seed=SEED,records=int(seen.sum()),audit=audit,
                  actual_MWh=expected,identity_MWh=identity,identity_error_MWh=abs(identity-expected),
                  B_budget_error_MWh=abs(budget-g.behavior_supply_MWh.sum()),baseline_surplus_MWh=g.surplus_MWh.sum(),
                  common_complete_meshes=len(keys),excluded_absent_month_meshes=34307-len(keys),
                  method='exact date-sorted surplus sufficient statistic; B proven constant by date within month/slot')
        out.write_text(json.dumps(info,indent=2));print(f'{ym} exact half-hour null and component ablations saved',flush=True)
        del D,B,seen,sorted_D,prefix;gc.collect()
    con.close()
    complete=sorted(OUT.glob('halfhour_202*.json'))
    if len(complete)==12:
        infos=[json.loads(p.read_text()) for p in complete]
        draws=pd.concat([pd.read_parquet(OUT/f"halfhour_null_{x['ym']}.parquet") for x in infos])
        ab=pd.concat([pd.read_csv(OUT/f"halfhour_ablation_{x['ym']}.csv") for x in infos])
        ab.to_csv(OUT/'s1_component_ablation.csv',index=False)
        rows=[]
        seasons=dict(zip(monthly.ym,monthly.season));draws['season']=draws.ym.map(seasons)
        for period in ['Annual','Spring','Summer','Autumn','Winter']:
            use=infos if period=='Annual' else [x for x in infos if seasons[x['ym']]==period]
            target=sum(x['actual_MWh'] for x in use);den=sum(x['baseline_surplus_MWh'] for x in use)
            for model,z in draws[draws.ym.isin([x['ym'] for x in use])].groupby('model'):
                vals=z.groupby('draw').absorption_MWh.sum().to_numpy();p=(1+(vals<=target).sum())/(len(vals)+1)
                rows.append(dict(period=period,model=model,draws=len(vals),actual_match_percent=100*target/den,
                                 null_mean_percent=100*vals.mean()/den,null_p025_percent=100*np.quantile(vals,.025)/den,
                                 null_p975_percent=100*np.quantile(vals,.975)/den,lower_tail_p=p,
                                 lower_tail_mc_se=np.sqrt(p*(1-p)/(len(vals)+1)),
                                 interpretation='whole-year trajectory permutation; exact half-hour matching; conditional placement null'))
        pd.DataFrame(rows).to_csv(OUT/'halfhour_spatial_null_summary.csv',index=False)

if __name__=='__main__':main()
