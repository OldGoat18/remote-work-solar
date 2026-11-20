"""Calendar-corrected spatial summaries and targeted construction sensitivity."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=Path(__file__).resolve().parents[1];O=BASE/'data';SRC=ROOT/'outputs/figures/extra/legacy_34307_seasonal'
sys.path.insert(0,str(BASE/'scripts'));from run_technical import m
old=pd.read_parquet(SRC/'seasonal_mesh_month_fields.parquet');old.mesh=old.mesh.astype(str);old.ym=old.ym.astype(str)
z=pd.read_parquet(O/'mesh_month.parquet');z.mesh=z.mesh.astype(str);z.ym=z.ym.astype(str)
print(old.columns.tolist(),flush=True)
for col in ['surplus_MWh','midday_load_MWh']:
 if col in old:z=z.drop(columns=[col],errors='ignore').merge(old[['mesh','ym',col]],on=['mesh','ym'],validate='one_to_one')
keys=pd.read_csv(BASE.parent/'20261111_supplemental/data/mesh_population_scope.csv',dtype={'mesh':str,'pref':str,'munic':str})
scales=[0,.5,1,1.5,2,3,5,6.3753];rows=[];alternative=[];cells=[];components=[]
for ym in m.MONTHS:
 q=np.load(BASE/f'tmp/inputs_{ym}.npz');G=q['G'];L=q['L'];B=q['B'];D=np.maximum(G-L,0);wd=m.dates_frame().query('ym==@ym').daytype.eq('workday').to_numpy();dw=D[:,:,wd]
 for a in scales:
  u=np.minimum(dw,a*B[:,:,None]).sum(axis=(1,2));f=keys[['mesh','pref','munic']].copy();f['scale']=a;f['ym']=ym;f['service_MWh']=u;rows.append(f)
 # Equal worker-based redistribution within prefecture, fixed each pref/slot total.
 b=B.copy()
 for pref in keys.pref.unique():
  ix=keys.pref.eq(pref).to_numpy();w=keys.loc[ix,'workers'].to_numpy();b[ix]=w[:,None]/w.sum()*B[ix].sum(axis=0)
 f=keys[['mesh','pref']].copy();f['ym']=ym;f['alternative_demand_MWh']=b.sum(axis=1)*wd.sum();f['alternative_service_MWh']=np.minimum(dw,b[:,:,None]).sum(axis=(1,2));alternative.append(f)
 # Original end uses from cached table; source components for ablation only.
 import duckdb
 from input_mesh_alignment import components as getcomponents
 con=duckdb.connect();comp=getcomponents(con,ym,keys,'S1');con.close()
 for name,ids in [('office',[1]),('lighting_lunch',[2,3]),('hvac',[4,5])]:
  without=B-comp[:,:,ids].sum(axis=2)
  components.append(dict(ym=ym,omitted=name,service_loss_MWh=float(np.minimum(dw,B[:,:,None]).sum()-np.minimum(dw,np.maximum(without,0)[:,:,None]).sum())))
 print('spatial',ym,flush=True)
r=pd.concat(rows);r.to_parquet(O/'mesh_scale_month.parquet',index=False);alt=pd.concat(alternative);z=z.merge(alt[['mesh','ym','alternative_demand_MWh','alternative_service_MWh']],on=['mesh','ym']);pd.DataFrame(components).to_csv(O/'component_ablation.csv',index=False)
season=z.groupby(['mesh','pref','season'],as_index=False).agg(service_MWh=('service_MWh','sum'),surplus_MWh=('surplus_MWh','sum'),demand_MWh=('demand_MWh','sum'),alternative_demand_MWh=('alternative_demand_MWh','sum'),alternative_service_MWh=('alternative_service_MWh','sum'),midday_load_MWh=('midday_load_MWh','sum'),lon=('lon','first'),lat=('lat','first'),munic=('munic','first'),N_persons=('N_persons','first'),workers=('workers','first'))
pop=pd.read_parquet(ROOT/'data/derived/pv/wfh_spatial_weight.parquet')[['mesh','pop','working_pop']];pop.mesh=pop.mesh.astype(str);pv=pd.read_parquet(m.ALPHA)[['mesh','current_cap_kw','pot_cap_kw_B_IC']];pv.mesh=pv.mesh.astype(str)
season=season.merge(pop,on='mesh',how='left').merge(pv,on='mesh',how='left');season.to_parquet(O/'mesh_season.parquet',index=False)
annual=season.groupby('mesh',as_index=False).agg(service_MWh=('service_MWh','sum'),surplus_MWh=('surplus_MWh','sum'),demand_MWh=('demand_MWh','sum'),pref=('pref','first'),lon=('lon','first'),lat=('lat','first'),munic=('munic','first'),pop=('pop','first'),N_persons=('N_persons','first'),workers=('workers','first'));annual.to_parquet(O/'mesh_annual.parquet',index=False)
mu=annual.groupby('munic',as_index=False).sum(numeric_only=True);mu['match_percent']=100*mu.service_MWh/mu.surplus_MWh;mu.to_csv(O/'municipal_annual.csv',index=False)
cols=['pop','working_pop','current_cap_kw','pot_cap_kw_B_IC','midday_load_MWh'];loro=[];corr=[]
for seasonname,g in season.groupby('season'):
 g=g[np.isfinite(g[cols]).all(axis=1)&(g[cols]>=0).all(axis=1)].copy()
 X=np.column_stack([np.ones(len(g)),np.log1p(g[cols].to_numpy())]);Y=np.log1p(g.surplus_MWh.to_numpy());yr=Y-X@np.linalg.lstsq(X,Y,rcond=None)[0]
 for name,col in [('occupation_household','demand_MWh'),('equal_worker_reconstruction','alternative_demand_MWh')]:
  y=np.log1p(g[col]);r0=y-X@np.linalg.lstsq(X,y,rcond=None)[0];corr.append(dict(season=seasonname,reconstruction=name,partial_r=np.corrcoef(yr,r0)[0,1],meshes=len(g)))
 for pref in g.pref.unique():
  tr=g[g.pref.ne(pref)];te=g[g.pref.eq(pref)];x=np.log1p(tr[cols].to_numpy());xt=np.log1p(te[cols].to_numpy());mean=x.mean(axis=0);sd=np.maximum(x.std(axis=0),1e-12);x=np.column_stack([np.ones(len(tr)),(x-mean)/sd]);xt=np.column_stack([np.ones(len(te)),(xt-mean)/sd]);y=np.log1p(tr.service_MWh);yt=np.log1p(te.service_MWh);pr=xt@np.linalg.lstsq(x,y,rcond=None)[0];rmse=np.sqrt(np.mean((yt-pr)**2));base=np.sqrt(np.mean((yt-y.mean())**2));loro.append(dict(season=seasonname,pref=pref,test_n=len(te),skill=1-rmse/base,rmse=rmse,baseline_rmse=base))
pd.DataFrame(loro).to_csv(O/'geographic_holdout.csv',index=False);pd.DataFrame(corr).to_csv(O/'construction_association.csv',index=False)
# No significance assigned to model-construction correlation; Mantel withdrawn.
print('spatial complete',flush=True)
