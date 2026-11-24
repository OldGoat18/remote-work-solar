"""Five empirical spatial bias fields, kept separate from the PV residual sample.
Monthly error sums cannot recover half-hour residual paths. Transfer each observed
holdout bias field to hybrid daytime load as a deterministic scenario, not a CI.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));O=BASE/_asset_location('asset_0002');OLD=BASE.parent/_asset_location('asset_0023')
sys.path.insert(0,str(Path(__file__).resolve().parent));from run_technical import m,supp,lunch_schedule
import duckdb
from input_mesh_alignment import components
from lunch_constraints import prepare,evaluate_surplus
keys=pd.read_csv(OLD/'mesh_population_scope.csv',dtype={'mesh':str,'pref':str,'munic':str});mu=pd.read_csv(OLD/'policy_municipality_inputs.csv',dtype={'munic':str});mi=keys.munic.map(dict(zip(mu.munic,range(len(mu))))).fillna(-1).to_numpy(int)
sc=pd.read_csv(OLD/'policy_scenarios.csv');sid=sc.query("fraction==.25 and target=='high_surplus_q4' and rule=='TE'").scenario_id.iloc[0];al=pd.read_csv(OLD/'policy_allocations.csv',dtype={'munic':str});hi=al[al.scenario_id.eq(sid)].set_index('munic').reindex(mu.munic).scale.to_numpy();mg=pd.read_csv(O/'marginal_allocations.csv',dtype={'munic':str}).query("case=='L+0_P+0'").set_index('munic').reindex(mu.munic).x.to_numpy()+1
err=pd.read_parquet(ROOT/_asset_location('asset_0024'));err=err[err.model.eq('owner_scaled')];err.mesh=err.mesh.astype(str);factors={}
for h,g in err.groupby('held_out_month'):
 g=g.set_index('mesh').reindex(keys.mesh);den=g.target_MWh+g.signed_error_MWh;factors[str(h)]=np.divide(g.target_MWh.to_numpy(),den.to_numpy(),out=np.ones(len(g)),where=den.to_numpy()>1e-12)
rows=[];ranks=[];con=duckdb.connect()
for ym in m.MONTHS:
 p=O/f'load_residual_{ym}.csv'
 if p.exists():continue
 z=np.load(BASE/f"{_asset_location('asset_0017')}{ym}.npz");G=z['G'];L=z['L'];B=z['B'];cal=m.dates_frame().query(_asset_location('asset_0020'));wd=cal.daytype.eq('workday').to_numpy();comp=components(con,ym,keys,'S1');prepared=prepare(B,comp[:,:,3]);out=[]
 for h,factor in factors.items():
  ll=L*factor[:,None,None] if ym not in m.DIRECT_MONTHS else L;D=np.maximum(G-ll,0);dw=np.ascontiguousarray(D[:,:,wd]);u=np.minimum(dw,B[:,:,None]);local=u.sum();regional=np.minimum(np.maximum((G-ll).sum(axis=0),0),B.sum(axis=0)[:,None])[:,wd].sum();ideal=supp.local_schedules(dw,B)[:,2].sum();gain,_,_=evaluate_surplus(np.ascontiguousarray(dw[:,5:11,:]),*prepared);r=dict(source_month=h,ym=ym,local_MWh=local,regional_MWh=regional,ideal_gain_MWh=ideal-local,constrained_gain_MWh= gain[:,0].sum(),fixed_gain_MWh=gain[:,1].sum(),fixed_lunch_MWh=local+gain[:,1].sum())
  for name,scales in [('uniform',np.full(len(mu),1.25)),('high',hi),('marginal',mg)]:
   a=np.where(mi>=0,scales[np.maximum(mi,0)],1);r[name+'_increment_MWh']=(np.minimum(dw,B[:,:,None]*a[:,None,None])-u)[mi>=0].sum()
  out.append(r)
  rr=keys[['mesh','munic']].copy();rr['source_month']=h;rr['ym']=ym;rr['marginal_MWh']=(np.minimum(dw,B[:,:,None]*1.05)-u).sum(axis=(1,2));rr.groupby(['source_month','ym','munic'],as_index=False).marginal_MWh.sum().to_csv(O/f'load_priority_{ym}_{h}.csv',index=False)
  reg=pd.DataFrame(dict(date=np.repeat(cal.date.to_numpy(),20),slot=np.tile(np.arange(17,37),len(cal)),G=G.sum(axis=0).T.ravel(),L0=ll.sum(axis=0).T.ravel(),A=(B.sum(axis=0)[:,None]*wd).T.ravel()));reg.to_parquet(O/f'load_region_{ym}_{h}.parquet',index=False)
 pd.DataFrame(out).to_csv(p,index=False);print('load residual',ym,flush=True)
q=pd.concat([pd.read_csv(O/f'load_residual_{ym}.csv') for ym in m.MONTHS]).groupby('source_month').sum(numeric_only=True).drop(columns='ym');q['high_advantage_percent']=100*(q.high_increment_MWh/q.uniform_increment_MWh-1);q['marginal_advantage_percent']=100*(q.marginal_increment_MWh/q.uniform_increment_MWh-1);q.to_csv(O/'load_residual_summary.csv')
ranks=pd.concat([pd.read_csv(O/f'load_priority_{ym}_{h}.csv',dtype={'munic':str}) for ym in m.MONTHS for h in factors])
ranks=ranks.groupby(['source_month','munic'],as_index=False).marginal_MWh.sum().merge(mu[['munic','N_persons']],on='munic',how='inner')
ranks['kWh_per_day']=ranks.marginal_MWh*1000/(.05*243*ranks.N_persons)
ranks['rank']=ranks.groupby('source_month').kWh_per_day.rank(ascending=False,method='min')
ranks.to_csv(O/'load_priority_ranks.csv',index=False)
(O/'load_residual_manifest.json').write_text(json.dumps(dict(source='mesh_LV_month_holdout_error_sums.parquet',cases=list(factors),correction='held-out monthly observed/predicted energy ratio per mesh, applied coherently to 08:30-18:30 load on hybrid dates',interpretation='five transferred cold-month spatial-bias scenarios; not an empirical distribution of summer residual paths or a confidence interval',missing='date-slot mesh residuals, summer validation and paired load/PV errors'),indent=2))
