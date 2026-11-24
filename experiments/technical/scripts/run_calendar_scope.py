"""Reconcile retained experiments to the registered calendar and common energy ledger.
No new parameter grid or stochastic uncertainty experiment is introduced.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys,json,gc
import numpy as np,pandas as pd,duckdb
from numba import njit
BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists());D=BASE/_asset_location('asset_0002');OLD=BASE.parent/_asset_location('asset_0023');COPY=BASE/_asset_location('asset_0037');R1=COPY/_asset_location('asset_0038')
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'src')]
from run_technical import m,SRC,GRID
from summarize_technical import allocate_marginal
from wlpr.io.activity_calendar import calendar_frame,VERSION
SC={3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn',12:'Winter',1:'Winter',2:'Winter'}
SEASONS=['Spring','Summer','Autumn','Winter'];PV=[.5,.75,1.,1.25,1.5,2.,3.];ALPHA=[0,.025,.05,.1,.15,.2,.3,.4,.5,1,2,3,4,5,6.3753]

def physical_monthly():
 keys=pd.read_csv(OLD/'mesh_population_scope.csv',dtype={'mesh':str,'pref':str,'munic':str})
 dates=m.dates_frame();c=duckdb.connect();c.execute('set threads=4');c.execute("set memory_limit='3GB'");c.register('common_keys',keys[['pref','mesh']])
 for ym in m.MONTHS:
  dest=D/f'calendar_scope_pv_{ym}.parquet'
  if dest.exists():continue
  rows=[]
  for pref in ['11','12','13','14']:
   sql,_=m.month_source_sql(c,ym,dates,SRC/'load_reference_mesh_daytype_slot.parquet',SRC/'owner_load_reference_pref_daytype_slot.parquet',SRC/'observed_stock_yield_kernel_full_year.parquet')
   sql=sql.replace(_asset_location('asset_0018'),f"{_asset_location('asset_0016')}{pref}")
   if ym not in m.DIRECT_MONTHS:
    ref=m.qpath(SRC/'load_reference_mesh_daytype_slot.parquet');sql=sql.replace(f"FROM read_parquet('{ref}') r",f"FROM (SELECT * FROM read_parquet('{ref}') WHERE cast(pref AS varchar)='{pref}') r")
   fields=','.join(f'sum(greatest({p}*x.G-x.L0,0)) D{k}' for k,p in enumerate(PV))
   f=c.execute(sql+f' SELECT x.mesh,{fields},sum(x.L0) full_load_MWh FROM x JOIN common_keys k ON x.mesh=k.mesh AND x.pref=k.pref GROUP BY 1').df();rows.append(f)
   print('physical surplus',ym,pref,flush=True)
  z=pd.concat(rows);z['ym']=ym;z.to_parquet(dest,index=False)
 c.close()
 z=pd.concat([pd.read_parquet(D/f'calendar_scope_pv_{ym}.parquet') for ym in m.MONTHS]);z['season']=z.ym.str[4:].astype(int).map(SC)
 mm=pd.read_parquet(D/'mesh_month.parquet');mm['ym']=mm.ym.astype(str);mm.mesh=mm.mesh.astype(str);z=z.merge(mm[['mesh','ym','demand_MWh']],on=['mesh','ym'],validate='one_to_one')
 rows=[]
 for s,g in z.groupby('season'):
  for k,pv in enumerate(PV):
   den=g[f'D{k}'].sum()
   for alpha in ALPHA:
    for placement,service in [('seasonal_energy_ceiling',min(den,g.demand_MWh.sum()*alpha)),('monthly_residential_colocation',np.minimum(g[f'D{k}'],g.demand_MWh*alpha).sum())]:
     rows.append(dict(season=s,alpha=alpha,pv_multiplier=pv,placement=placement,matched_MWh=service,surplus_MWh=den,match_percent=100*service/den if den else 0,definition='positive surplus recomputed from pv*G-L0; A gated to 243 eligible dates; energy-aggregation upper bound'))
 pd.DataFrame(rows).to_csv(R1/'calendar_scope_parameter_grid.csv',index=False)
 # Read-back of the full48 denominator used by the spatial accounts.
 se=pd.read_parquet(D/'mesh_season.parquet');se.mesh=se.mesh.astype(str)
 chk=z.groupby(['mesh','season']).D2.sum().rename('recomputed_full48_surplus_MWh').reset_index().merge(se[['mesh','season','surplus_MWh']],on=['mesh','season'])
 chk['difference_MWh']=chk.recomputed_full48_surplus_MWh-chk.surplus_MWh
 chk.to_parquet(D/'calendar_scope_denominators.parquet',index=False)
 print('parameter grid complete',len(rows), 'denominator difference',chk.difference_MWh.abs().max(),flush=True)

def retained_random():
 z=pd.read_parquet(D/'mesh_season.parquet');rng=np.random.default_rng(20260804);draws=[];rows=[]
 for s in SEASONS:
  q=z[z.season.eq(s)].sort_values('mesh');d=q.surplus_MWh.to_numpy();a=q.demand_MWh.to_numpy();w=q.midday_load_MWh.clip(lower=0).to_numpy();w=w/w.sum()
  actual=np.minimum(d,a).sum()/d.sum()
  for model in ['uniform_permutation','load_weighted']:
   values=[]
   for j in range(2000):
    placed=rng.permutation(a) if model=='uniform_permutation' else np.bincount(rng.choice(len(a),size=len(a),p=w),weights=a,minlength=len(a))
    v=np.minimum(d,placed).sum()/d.sum();values.append(v);draws.append(dict(season=s,model=model,draw=j,matching_rate=v))
   low,med,high=np.quantile(values,[.025,.5,.975]);rows.append(dict(season=s,model=model,n=2000,actual_match=actual,interval_actual_match=q.service_MWh.sum()/d.sum(),null_mean=np.mean(values),null_sd=np.std(values,ddof=1),null_p025=low,null_median=med,null_p975=high,actual_percentile=np.mean(np.asarray(values)<=actual),operator='sum_mesh min(season_surplus, season_A) / sum_mesh season_surplus',seed=20260804))
  print('same-operator placement',s,flush=True)
 pd.DataFrame(rows).to_csv(R1/'behavior_seasonal_null_models.csv',index=False);pd.DataFrame(draws).to_parquet(R1/'behavior_seasonal_null_draws.parquet',index=False)

def regional_pairs():
 q=pd.read_parquet(D/'common48.parquet');q.date=q.date.astype(str);gate=q.date.isin(calendar_frame().query('is_workday_defacto==1').ymd.astype(str));a=q.template_A.to_numpy()*gate;g=q.G.to_numpy();l=q.L0.to_numpy()
 params=pd.read_csv(R1/'registered_parameter_pairs.csv').query("pair=='behavior__pv'")
 rows=[]
 for r in params.itertuples():
  surplus=np.maximum(r.pv*g-l,0);u=np.minimum(surplus,r.behavior*a);remaining=surplus-u
  rows.append(dict(pair=r.pair,behavior=r.behavior,pv=r.pv,negative_energy_GWh=remaining.sum()/1000,behavior_service_GWh=u.sum()/1000,eligible_GWh=surplus.sum()/1000,definition='regional residential net additional absorption; annual 366 dates, A on 243 dates; all48 intervals'))
 pd.DataFrame(rows).to_csv(R1/'calendar_scope_parameter_pairs.csv',index=False)

@njit
def score_allocations(D,B,mi,scales):
 out=np.zeros(scales.shape[0]);municipal=np.zeros((scales.shape[0],scales.shape[1]))
 for i in range(D.shape[0]):
  j=mi[i]
  if j<0:continue
  for t in range(B.shape[1]):
   v=np.sort(D[i,t]);cs=np.zeros(len(v)+1)
   for d in range(len(v)):cs[d+1]=cs[d]+v[d]
   for k in range(scales.shape[0]):
    b=B[i,t]*scales[k,j];n=np.searchsorted(v,b);municipal[k,j]+=cs[n]+(len(v)-n)*b
 for k in range(scales.shape[0]):out[k]=municipal[k].sum()
 return out,municipal

def policy_exact():
 mu=pd.read_csv(OLD/'policy_municipality_inputs.csv',dtype={'munic':str});sc=pd.read_csv(OLD/'policy_scenarios.csv');alloc=pd.read_csv(OLD/'policy_allocations.csv',dtype={'munic':str});matrix=alloc.pivot(index='scenario_id',columns='munic',values='scale').reindex(columns=mu.munic).to_numpy()
 grid=pd.read_parquet(D/'policy_grid_annual.parquet').query("case=='L+0_P+0'");grid.munic=grid.munic.astype(str);z=grid.pivot(index='x',columns='munic',values='service_MWh').reindex(index=GRID,columns=mu.munic).to_numpy();margs=[]
 for f in [.1,.25,.5]:
  x,v,gap=allocate_marginal(z,mu.N_persons.to_numpy(),f);matrix=np.vstack([matrix,1+x]);margs.append(dict(scenario_id=len(matrix)-1,fraction=f,target='all_municipalities',rule='MG',additional_person_days=f*243*mu.N_persons.sum(),total_expenditure_JPY=f*243*mu.N_persons.sum()*1050,grid_service_MWh=v,fractional_segment_gap_bound_MWh=gap))
 sums=np.zeros(len(matrix));municipal=np.zeros(matrix.shape)
 for ym in m.MONTHS:
  with np.load(BASE/f"{_asset_location('asset_0029')}{ym}.npz") as z:val,parts=score_allocations(z['D'],z['B'],z['mi'],matrix)
  sums+=val;municipal+=parts;print('policy72 + marginal3',ym,flush=True)
 rows=pd.concat([pd.DataFrame([dict(scenario_id=0,fraction=0,target='all_municipalities',rule='baseline',additional_person_days=0,total_expenditure_JPY=0)]),sc,pd.DataFrame(margs)],ignore_index=True)
 rows['baseline_MWh']=sums[0];rows['total_service_MWh']=sums[rows.scenario_id.astype(int)];rows['additional_service_MWh']=rows.total_service_MWh-sums[0];rows['JPY_per_kWh']=np.divide(rows.total_expenditure_JPY,1000*rows.additional_service_MWh,out=np.full(len(rows),np.nan),where=rows.additional_service_MWh>0);rows['kWh_per_1000_JPY']=np.divide(rows.additional_service_MWh*1e6,rows.total_expenditure_JPY,out=np.full(len(rows),np.nan),where=rows.total_expenditure_JPY>0)
 rows.to_csv(D/'calendar_scope_policy_exact.csv',index=False)
 pd.DataFrame(municipal,columns=mu.munic).rename_axis('scenario_id').to_parquet(D/'calendar_scope_policy_municipal.parquet')
 rows2=[]
 for k,s in enumerate(matrix):
  rows2.extend(dict(scenario_id=k,munic=mu.munic.iloc[j],scale=v,frequency=.349*v,additional_person_days=243*mu.N_persons.iloc[j]*(v-1)) for j,v in enumerate(s))
 pd.DataFrame(rows2).to_parquet(D/'calendar_scope_policy_allocations.parquet',index=False)

def frequency_caps():
 from run_technical import supp
 mu=pd.read_csv(OLD/'policy_municipality_inputs.csv',dtype={'munic':str});n=mu.N_persons.to_numpy();g=pd.read_parquet(D/'policy_grid_annual.parquet');g.munic=g.munic.astype(str)
 policy=pd.read_csv(D/'policy_comparison.csv');alloc=pd.read_csv(D/'marginal_allocations.csv',dtype={'munic':str});records=[];comparison=[]
 for frequency in [.5,.7]:
  case=f'frequency_{frequency:.2f}';amp=frequency/.349;xmax=min(1.,1/frequency-1);case_n=n*amp
  z=g[g['case'].eq(case)].pivot(index='x',columns='munic',values='service_MWh').reindex(index=GRID,columns=mu.munic).to_numpy();base=z[0].sum();cost=.25*243*n.sum()*1050
  high=supp.allocate(case_n*xmax,np.ones(len(n)),mu.high_surplus_q4.to_numpy(bool),.25/(amp*xmax))*xmax
  marginal,value,gap=allocate_marginal(z,case_n,.25/amp,x_max=xmax)
  oldx,oldvalue,_=allocate_marginal(z,case_n,.25/amp,x_max=1.)
  comparison.append(dict(case=case,old_max_frequency=float(frequency*(1+oldx.max())),new_max_frequency=float(frequency*(1+marginal.max())),old_additional_MWh=oldvalue-base,new_additional_MWh=value-base,old_JPY_per_kWh=cost/((oldvalue-base)*1000),new_JPY_per_kWh=cost/((value-base)*1000),same_added_person_days=.25*243*n.sum(),operator='piecewise linear grid lower bound in both cases'))
  for rule,x in [('uniform',np.full(len(n),.25/amp)),('high_surplus',high),('marginal_grid',marginal)]:
   service=sum(np.interp(x[j],GRID,z[:,j]) for j in range(len(n)));ix=policy['case'].eq(case)&policy.rule.eq(rule)
   for name,v in dict(additional_MWh=service-base,JPY_per_kWh=cost/((service-base)*1000),kWh_per_1000_JPY=(service-base)*1e6/cost,interpolation='concave grid lower bound; frequency capped at one eligible day').items():policy.loc[ix,name]=v
   if rule=='marginal_grid':
    policy.loc[ix,'fractional_segment_gap_bound_MWh']=gap
    xx=alloc['case'].eq(case);values=dict(zip(mu.munic,x));alloc.loc[xx,'x']=alloc.loc[xx,'munic'].map(values);alloc.loc[xx,'person_days']=243*alloc.loc[xx,'munic'].map(dict(zip(mu.munic,case_n)))*alloc.loc[xx,'x']
   records.extend(dict(case=case,rule=rule,munic=mu.munic.iloc[j],x=v,frequency=frequency*(1+v),added_person_days=243*case_n[j]*v) for j,v in enumerate(x))
 pd.DataFrame(comparison).to_csv(D/'calendar_scope_frequency_comparison.csv',index=False)
 policy.to_csv(D/'policy_comparison.csv',index=False);alloc.to_csv(D/'marginal_allocations.csv',index=False);pd.DataFrame(records).to_csv(D/'calendar_scope_frequency_allocations.csv',index=False)

def calendar_report():
 cal=calendar_frame();cal.to_csv(D/'calendar_scope_dates.csv',index=False)
 q=pd.read_csv(D/'metrics_all.csv');q['date']=q.date.astype(str);q=q[q['case'].eq('L+0_P+0')];# date labels are ISO-free YYYYMMDD keys
 # calendar_frame also has Timestamp date; avoid duplicate names.
 q=q.merge(cal[['ymd','is_workday_defacto','source']].assign(ymd=lambda x:x.ymd.astype(str)),left_on='date',right_on='ymd',validate='one_to_one')
 summary=q.groupby('day_class')[['demand_MWh','local_MWh','pool_MWh','regional_MWh','demand_template_MWh','local_template_MWh']].sum().reset_index();summary['dates']=summary.day_class.map(q.day_class.value_counts());summary.to_csv(D/'calendar_scope_summary.csv',index=False)
 r=dict(calendar_version=VERSION,dates=len(cal),eligible_dates=int(cal.is_workday_defacto.sum()),noneligible_dates=int((~cal.is_workday_defacto.astype(bool)).sum()),comparison_to_existing_primary_dates=int((q.is_workday_defacto.astype(bool)!=q.day_class.eq('eligible')).sum()),interpretation='registered representative workday A; no observed weekend attendance or nonstandard work schedule; G and ordinary L0 retained on all dates',changes=['single full-year calendar entrypoint','zero fallback outside registered dates','same operator in seasonal spatial random reference','generation-scaled PV before local positive clipping','exact72 equal-budget policy scenarios +3 marginal allocations'])
 (D/'calendar_scope_manifest.json').write_text(json.dumps(r,indent=2))

if __name__=='__main__':
 modes=sys.argv[1:] or ['calendar','regional','physical','random','policy','frequency']
 for mode in modes:dict(calendar=calendar_report,regional=regional_pairs,physical=physical_monthly,random=retained_random,policy=policy_exact,frequency=frequency_caps)[mode]()
