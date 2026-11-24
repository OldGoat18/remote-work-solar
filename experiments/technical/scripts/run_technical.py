"""Technical revision: common interval boundaries, calendar, end uses and policy.
Versioned outputs only. The old seasonal source is read, never overwritten.
All energies MWh/half-hour. NMAE cases are deterministic, not confidence limits.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys, importlib.util, json, calendar, gc
import numpy as np
import pandas as pd
import duckdb
from numba import njit, set_num_threads
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));OUT=BASE/_asset_location('asset_0002');TMP=BASE/_asset_location('asset_0007')
OLD=BASE.parent/_asset_location('asset_0023');SRC=ROOT/_asset_location('asset_0003')
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
from input_mesh_alignment import m,components,INPUT_VERSION
sp=importlib.util.spec_from_file_location('supp',(ROOT / 'experiments/supplemental/scripts/run_supplemental.py'));supp=importlib.util.module_from_spec(sp);sp.loader.exec_module(supp)
GRID=np.linspace(0,1,21)

@njit
def score_grid(D,B,mi,nm):
    # Exact evaluation at every grid knot; each municipality has independent x.
    out=np.zeros((len(GRID),nm))
    for i in range(D.shape[0]):
        j=mi[i]
        if j<0:continue
        for t in range(20):
            v=np.sort(D[i,t]);cs=np.zeros(len(v)+1)
            for d in range(len(v)):cs[d+1]=cs[d]+v[d]
            for k in range(len(GRID)):
                b=B[i,t]*(1+GRID[k]);a=np.searchsorted(v,b)
                out[k,j]+=cs[a]+(len(v)-a)*b
    return out

from lunch_constraints import lunch_schedule


def main():
 set_num_threads(4)
 keys=pd.read_csv(OLD/'mesh_population_scope.csv',dtype={'mesh':str,'pref':str,'munic':str})
 muni=pd.read_csv(OLD/'policy_municipality_inputs.csv',dtype={'munic':str,'pref':str})
 mi=keys.munic.map(dict(zip(muni.munic,range(len(muni))))).fillna(-1).to_numpy(np.int64)
 idx=pd.MultiIndex.from_frame(keys[['pref','mesh']]);datesall=m.dates_frame()
 metrics=pd.read_csv(ROOT/_asset_location('asset_0048'))
 la=float(metrics.loc[metrics.asset.eq('mesh LV baseline'),'mean_NMAE'].iloc[0]);pa=float(metrics.loc[metrics.asset.eq('owner_year2:month_slot'),'mean_NMAE'].iloc[0])
 scenarios=[dict(case=f'L{l:+d}_P{p:+d}',load=l*la,pv=p*pa,a=1.) for l in [-1,0,1] for p in [-1,0,1]]
 # Source-defined activity scenarios, not empirical errors in kWh/person-day.
 scenarios += [dict(case='frequency_0.50',load=0.,pv=0.,a=.50/.349),dict(case='frequency_0.70',load=0.,pv=0.,a=.70/.349)]
 (OUT/'scenario_definitions.json').write_text(json.dumps(scenarios,indent=2))
 con=duckdb.connect();con.execute('set threads=4');con.execute("set memory_limit='4GB'");con.execute(f"set temp_directory='{TMP}'")
 for ym in m.MONTHS:
  dest=OUT/f'month_{ym}.json'
  if dest.exists() and (ym in m.DIRECT_MONTHS or json.loads(dest.read_text()).get('input_version')==INPUT_VERSION):print('reuse',ym,flush=True);continue
  nd=calendar.monthrange(int(ym[:4]),int(ym[4:]))[1];shape=(len(keys),20,nd)
  cache=TMP/f'inputs_{ym}.npz'
  if cache.exists():
   z=np.load(cache);G=z['G'];L=z['L'];GI=z['GI'];B=z['B']
  else:
   G=np.zeros(shape);L=np.zeros(shape);GI=np.zeros(shape);B=np.zeros(shape[:2])
   for pref in ['11','12','13','14']:
    sql,_=m.month_source_sql(con,ym,datesall,SRC/'load_reference_mesh_daytype_slot.parquet',SRC/'owner_load_reference_pref_daytype_slot.parquet',SRC/'observed_stock_yield_kernel_full_year.parquet')
    inc='coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000.0'
    sql=sql.replace('coalesce(w.dwfh,0) B',f'coalesce(w.dwfh,0) B, {inc} GI').replace(_asset_location('asset_0018'),f"{_asset_location('asset_0016')}{pref}")
    if ym not in m.DIRECT_MONTHS:
     ref=m.qpath(SRC/'load_reference_mesh_daytype_slot.parquet');sql=sql.replace(f"FROM read_parquet('{ref}') r",f"FROM (SELECT * FROM read_parquet('{ref}') WHERE cast(pref AS varchar)='{pref}') r")
    for batch in con.execute(sql+' SELECT pref,mesh,date,slot,G,L0,GI,B FROM x WHERE slot BETWEEN 17 AND 36').fetch_record_batch(262144):
     f=batch.to_pandas();ix=idx.get_indexer(pd.MultiIndex.from_frame(f[['pref','mesh']].astype(str)));ok=ix>=0;f=f[ok];ix=ix[ok]
     t=f.slot.to_numpy(int)-17;d=f.date.astype(str).str[-2:].astype(int).to_numpy()-1
     G[ix,t,d]=f.G;L[ix,t,d]=f.L0;GI[ix,t,d]=f.GI;B[ix,t]=f.B
    print('read',ym,pref,flush=True)
   np.savez_compressed(cache,G=G,L=L,GI=GI,B=B)
  cal=datesall[datesall.ym.eq(ym)];wd=cal.daytype.eq('workday').to_numpy();ds=cal.date.to_numpy();season=dict(zip([3,4,5,6,7,8,9,10,11,12,1,2],['Spring']*3+['Summer']*3+['Autumn']*3+['Winter']*3))[int(ym[4:])]
  comp=components(con,ym,keys,'S1')
  if np.any(comp[:,:,0]!=B):
   B=comp[:,:,0].copy()
   np.savez_compressed(cache,G=G,L=L,GI=GI,B=B)
  pd.DataFrame([dict(ym=ym,component=k,eligible_MWh=float(comp[:,:,j+1].sum()*wd.sum()),all_date_template_MWh=float(comp[:,:,j+1].sum()*nd)) for j,k in enumerate(['office_device','lighting','lunch','heating','cooling','other'])]).to_csv(OUT/f'components_{ym}.csv',index=False)
  rows=[];sche=[];gridrows=[];meshrows=[];spatial=[]
  for case in scenarios:
   D=np.maximum(G+case['pv']*GI-(1+(case['load'] if ym not in m.DIRECT_MONTHS else 0))*L,0)
   b=B*case['a'];gs=(G+case['pv']*GI).sum(axis=0);ls=((1+(case['load'] if ym not in m.DIRECT_MONTHS else 0))*L).sum(axis=0);bs=b.sum(axis=0)
   local=np.minimum(D,b[:,:,None]).sum(axis=(0,1));pool=np.minimum(D.sum(axis=0),bs[:,None]).sum(axis=0);net=np.maximum(gs-ls,0);regional=np.minimum(net,bs[:,None]).sum(axis=0)
   diff=(np.minimum(gs,ls+bs[:,None])-np.minimum(gs,ls)).sum(axis=0);assert np.max(abs(diff-regional))<1e-7
   for d,date in enumerate(ds):
    dt=pd.Timestamp(date);kind='eligible' if wd[d] else ('weekend' if dt.dayofweek>=5 else 'weekday_holiday')
    rows.append(dict(case=case['case'],date=date,season=season,day_class=kind,local_template_MWh=local[d],pool_template_MWh=pool[d],regional_template_MWh=regional[d],demand_template_MWh=b.sum(),local_MWh=local[d]*wd[d],pool_MWh=pool[d]*wd[d],regional_MWh=regional[d]*wd[d],demand_MWh=b.sum()*wd[d],regional_net_daytime_MWh=net[:,d].sum()))
   dw=np.ascontiguousarray(D[:,:,wd]);v=supp.local_schedules(dw,b)
   lunch,shift=lunch_schedule(dw,b,comp[:,:,3]*case['a'])
   for d,date in enumerate(ds[wd]):
    for name,value in zip(['anchored','permutation_expectation','ideal_sorting','lunch_constrained'],list(v[d])+[lunch[d]]):sche.append(dict(case=case['case'],date=date,season=season,schedule=name,service_MWh=value))
   scores=score_grid(dw,b,mi,len(muni))
   for k,x in enumerate(GRID):
    gridrows.extend(dict(case=case['case'],ym=ym,munic=mu,x=x,service_MWh=scores[k,j]) for j,mu in enumerate(muni.munic))
   # Regional 48-slot corrections saved as daytime sums; outside A=0 always.
   pd.DataFrame(dict(date=np.repeat(ds,20),slot=np.tile(np.arange(17,37),nd),G=gs.T.ravel(),L0=ls.T.ravel(),A=(bs[:,None]*wd).T.ravel(),positive_pool=D.sum(axis=0).T.ravel(),local_service=(np.minimum(D,b[:,:,None])*wd).sum(axis=0).T.ravel())).to_parquet(OUT/f'regional_{ym}_{case["case"]}.parquet',index=False)
   if case['case']=='L+0_P+0':
    u=np.minimum(D,b[:,:,None])*wd
    z=keys[['mesh','pref','munic','lon','lat','N_persons','workers']].copy();z['ym']=ym;z['season']=season;z['service_MWh']=u.sum(axis=(1,2));z['surplus_MWh']=D.sum(axis=(1,2));z['eligible_surplus_MWh']=dw.sum(axis=(1,2));z['demand_MWh']=b.sum(axis=1)*wd.sum();z['L0_MWh']=L.sum(axis=(1,2));z.to_parquet(OUT/f'mesh_{ym}.parquet',index=False)
    for r in [.5,1.,2.,5.,10.,20.]:
     xy=np.column_stack([np.floor(keys.lon/(r/91)),np.floor(keys.lat/(r/111))]);_,ix=np.unique(xy,axis=0,return_inverse=True)
     dd=np.zeros((ix.max()+1,20,nd));bb=np.zeros((ix.max()+1,20));np.add.at(dd,ix,D);np.add.at(bb,ix,b)
     spatial.append(dict(ym=ym,season=season,cell_side_km=r,simultaneous_MWh=float((np.minimum(dd,bb[:,:,None])*wd).sum()),local_MWh=float(u.sum())))
    np.savez_compressed(TMP/f'policy_{ym}.npz',D=dw,B=B,mi=mi)
   del D,dw
   print('computed',ym,case['case'],flush=True)
  pd.DataFrame(rows).to_csv(OUT/f'metrics_{ym}.csv',index=False);pd.DataFrame(sche).to_csv(OUT/f'schedule_{ym}.csv',index=False);pd.DataFrame(gridrows).to_parquet(OUT/f'policy_grid_{ym}.parquet',index=False);pd.DataFrame(spatial).to_csv(OUT/f'spatial_{ym}.csv',index=False)
  dest.write_text(json.dumps(dict(ym=ym,eligible_dates=int(wd.sum()),dates=nd,calendar_correction='weekday representative template gated by existing holiday-adjusted workday registry',cases=len(scenarios),input_version=INPUT_VERSION)))
  del G,L,GI,B,comp;gc.collect()
 con.close()
 for name in ['metrics','schedule','components','spatial']:
  pd.concat([pd.read_csv(OUT/f'{name}_{ym}.csv') for ym in m.MONTHS]).to_csv(OUT/f'{name}_all.csv',index=False)
 pd.concat([pd.read_parquet(OUT/f'mesh_{ym}.parquet') for ym in m.MONTHS]).to_parquet(OUT/'mesh_month.parquet',index=False)
 pd.concat([pd.read_parquet(OUT/f'policy_grid_{ym}.parquet') for ym in m.MONTHS]).groupby(['case','munic','x'],as_index=False).service_MWh.sum().to_parquet(OUT/'policy_grid_annual.parquet',index=False)
 print('complete',flush=True)
if __name__=='__main__':main()
