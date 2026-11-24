"""Summaries and equal-service storage design. All energy units MWh unless named."""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys,json,importlib.util
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,vstack
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));OUT=BASE/_asset_location('asset_0002');OLD=BASE.parent/_asset_location('asset_0023')
sys.path.insert(0,str(Path(__file__).resolve().parent));from run_technical import supp,GRID,m,score_grid


def minimum_storage(Q,H,target,kind='capture'):
 """Minimize E with P=E/4, empty endpoints, fixed target and continuous SOC."""
 Q=np.asarray(Q);H=np.asarray(H);n=len(Q);eta=np.sqrt(.9);t=np.arange(n);ei=3*n+1;N=ei+1
 rr=np.tile(t,4);cc=np.r_[t,n+t,2*n+t,2*n+t+1];vv=np.r_[-eta*np.ones(n),np.ones(n)/eta,-np.ones(n),np.ones(n)]
 eq=coo_matrix((vv,(rr,cc)),shape=(n,N)).tocsr()
 rows=np.r_[t,n+t,2*n+np.arange(n+1),np.full(n,3*n+1)];cols=np.r_[t,n+t,2*n+np.arange(n+1),t if kind=='capture' else n+t]
 vals=np.r_[np.ones(2*n),np.ones(n+1),-np.ones(n)]
 rows=np.r_[rows,np.arange(3*n+1)];cols=np.r_[cols,np.full(3*n+1,ei)];vals=np.r_[vals,np.full(2*n,-.125),-np.ones(n+1)]
 ub=coo_matrix((vals,(rows,cols)),shape=(3*n+2,N)).tocsr();rhs=np.r_[np.zeros(3*n+1),-target]
 bounds=list(zip(np.zeros(n),Q))+list(zip(np.zeros(n),H))+[(0,None)]*(n+1)+[(0,None)];bounds[2*n]=(0,0);bounds[3*n]=(0,0)
 obj=np.zeros(N);obj[ei]=1
 r=linprog(obj,A_ub=ub,b_ub=rhs,A_eq=eq,b_eq=np.zeros(n),bounds=bounds,method='highs')
 if not r.success:return dict(feasible=False,message=r.message),None
 E=r.x[ei];c=r.x[:n];d=r.x[n:2*n];soc=r.x[2*n:3*n+1]
 traj=pd.DataFrame(dict(charge_MWh=c,delivery_MWh=d,soc_start_MWh=soc[:-1],soc_end_MWh=soc[1:],Q_MWh=Q,H_MWh=H))
 return dict(feasible=True,capacity_MWh=E,power_MW=E/4,capture_MWh=c.sum(),delivery_MWh=d.sum(),loss_MWh=c.sum()-d.sum(),initial_SOC_MWh=soc[0],terminal_SOC_MWh=soc[-1],SOC_equivalent_cycles=eta*c.sum()/E,input_equivalent_cycles=c.sum()/E),traj


def allocate_marginal(z,n,f=.25,x_max=1.):
 # Concave piecewise-linear allocation with explicit eligible-day headroom.
 widths=np.maximum(0,np.minimum(np.diff(GRID),x_max-GRID[:-1]))
 slopes=np.diff(z,axis=0)/(np.diff(GRID)[:,None]*243*n)
 order=np.argsort(-slopes.ravel(),kind='stable');x=np.zeros(len(n));remaining=f*243*n.sum()
 for v in order:
  k,j=np.unravel_index(v,slopes.shape)
  amount=min(remaining,widths[k]*243*n[j]);x[j]+=amount/(243*n[j]);remaining-=amount
  if remaining<1e-6:break
 if remaining>1e-4:raise ValueError('Added activity exceeds remaining eligible-day capacity.')
 service=np.array([np.interp(x[j],GRID,z[:,j]) for j in range(len(n))]).sum()
 frac=(x/.05)%1;js=np.where((frac>1e-6)&(frac<1-1e-6))[0]
 gap=sum(z[min(int(x[j]/.05)+1,20),j]-z[int(x[j]/.05),j] for j in js)
 return x,service,gap


def main():
 q=pd.read_csv(OUT/'metrics_all.csv');q.groupby(['case','day_class']).sum(numeric_only=True).to_csv(OUT/'metric_calendar_summary.csv')
 q.groupby(['case','season']).sum(numeric_only=True).to_csv(OUT/'metric_season_summary.csv')
 pd.read_csv(OUT/'schedule_all.csv').groupby(['case','schedule']).service_MWh.sum().to_csv(OUT/'schedule_summary.csv')
 pd.read_csv(OUT/'components_all.csv').groupby('component').sum(numeric_only=True).to_csv(OUT/'component_summary.csv')
 muni=pd.read_csv(OLD/'policy_municipality_inputs.csv',dtype={'munic':str});n=muni.N_persons.to_numpy();alloc=pd.read_csv(OLD/'policy_allocations.csv',dtype={'munic':str});sc=pd.read_csv(OLD/'policy_scenarios.csv')
 selected=sc[(sc.fraction.eq(.25))&((sc.target.eq('all_municipalities')&sc.rule.eq('U'))|(sc.target.eq('high_surplus_q4')&sc.rule.eq('TE')))]
 g=pd.read_parquet(OUT/'policy_grid_annual.parquet');g.munic=g.munic.astype(str)
 rows=[];allocrows=[]
 for case,f in g.groupby('case'):
  z=f.pivot(index='x',columns='munic',values='service_MWh').reindex(index=GRID,columns=muni.munic).to_numpy();base=z[0].sum();cost=1050*243*.25*n.sum()
  amp=float(case.split('_')[1])/.349 if case.startswith('frequency_') else 1.
  case_n=n*amp
  # Frequency scenarios retain the original absolute added-person-day budget.
  high_x=(alloc[alloc.scenario_id.eq(int(selected[selected.rule.eq('TE')].scenario_id.iloc[0]))].set_index('munic').reindex(muni.munic).scale.to_numpy()-1)/amp
  xmax=min(1.,1/(.349*amp)-1)
  if case.startswith('frequency_'):
   high_x=supp.allocate(case_n*xmax,np.ones(len(n)),muni.high_surplus_q4.to_numpy(bool),.25/(amp*xmax))*xmax
  for name,x in [('uniform',np.full(len(n),.25/amp)),('high_surplus',high_x)]:
   total=sum(np.interp(x[j],GRID,z[:,j]) for j in range(len(n)))
   # The reference allocations are retained; their exact service is evaluated
   # below against the repaired interval templates, not the historical totals.
   inc=total-base;rows.append(dict(case=case,rule=name,baseline_MWh=base,additional_MWh=inc,cost_JPY=cost,JPY_per_kWh=cost/(inc*1000),kWh_per_1000_JPY=inc*1e6/cost,interpolation='exact original minima' if case=='L+0_P+0' else 'concave 0.05 grid lower bound'))
  x,total,gap=allocate_marginal(z,case_n,f=.25/amp,x_max=min(1.,1/(.349*amp)-1));inc=total-base
  rows.append(dict(case=case,rule='marginal_grid',baseline_MWh=base,additional_MWh=inc,cost_JPY=cost,JPY_per_kWh=cost/(inc*1000),kWh_per_1000_JPY=inc*1e6/cost,interpolation='piecewise-linear allocation lower bound',fractional_segment_gap_bound_MWh=gap))
  allocrows.extend(dict(case=case,munic=mu,x=x[j],person_days=243*case_n[j]*x[j],initial_marginal_kWh_per_person_day=(z[1,j]-z[0,j])/(.05*243*case_n[j])*1000) for j,mu in enumerate(muni.munic))
 pd.DataFrame(rows).to_csv(OUT/'policy_comparison.csv',index=False)
 allocations=pd.DataFrame(allocrows)
 allocations['marginal_rank']=allocations.groupby('case').initial_marginal_kWh_per_person_day.rank(ascending=False,method='min')
 allocations.to_csv(OUT/'marginal_allocations.csv',index=False)
 # Exact re-evaluation at selected marginal x; anchored and old policy remain fixed.
 x=pd.DataFrame(allocrows).query("case=='L+0_P+0'").set_index('munic').reindex(muni.munic).x.to_numpy();exact={name:0. for name in ['uniform','high_surplus','marginal_grid']}
 high=(alloc[alloc.scenario_id.eq(int(selected[selected.rule.eq('TE')].scenario_id.iloc[0]))].set_index('munic').reindex(muni.munic).scale.to_numpy()-1)
 for ym in m.MONTHS:
  z=np.load(BASE/f"{_asset_location('asset_0029')}{ym}.npz");D=z['D'];B=z['B'];mi=z['mi']
  for rule,xx in [('uniform',np.full(len(n),.25)),('high_surplus',high),('marginal_grid',x)]:
   mapped=np.where(mi>=0,1+xx[np.maximum(mi,0)],1)
   exact[rule]+=np.minimum(D,B[:,:,None]*mapped[:,None,None])[mi>=0].sum()
 for ref in rows:
  if ref['case']!='L+0_P+0':continue
  total=exact[ref['rule']];ref['exact_total_MWh']=total;ref['exact_additional_MWh']=total-ref['baseline_MWh']
  if ref['rule']!='marginal_grid':ref['additional_MWh']=ref['exact_additional_MWh'];ref['interpolation']='exact current interval minima at original allocation'
  ref['JPY_per_kWh']=ref['cost_JPY']/(ref['exact_additional_MWh']*1000);ref['kWh_per_1000_JPY']=ref['exact_additional_MWh']*1e6/ref['cost_JPY']
 pd.DataFrame(rows).to_csv(OUT/'policy_comparison.csv',index=False)
 # Historical all-date boundary reconciliation from its own 48-slot ledger.
 old=pd.read_parquet(ROOT/_asset_location('asset_0051')).sort_values(['date','slot']);old.date=old.date.astype(str)
 hist=[]
 for calendar_name,mask in [('all_366',np.ones(len(old),bool)),('eligible_243',old.date.isin(m.dates_frame().query("daytype=='workday'").date))]:
  for window,w in [('all_48',np.ones(len(old),bool)),('charge_06_15',old.slot.between(12,29))]:
   a=old[mask&w];net=np.maximum(a.future_generation_MWh-a.no_wfh_load_MWh,0)
   hist.append(dict(calendar=calendar_name,window=window,local_MWh=a.actual_absorbed_MWh.sum(),positive_pool_match_MWh=np.minimum(a.D_MWh,a.B_MWh).sum(),regional_net_surplus_MWh=net.sum(),regional_additional_MWh=np.minimum(net,a.B_MWh).sum(),demand_MWh=a.B_MWh.sum()))
 pd.DataFrame(hist).to_csv(OUT/'historical_boundary_reconciliation.csv',index=False)
 # Fixed common targets across no/new residential remote-work demand.
 storage=[]
 for case in json.loads((OUT/'scenario_definitions.json').read_text()):
  name=case['case'];q0=pd.read_parquet(OUT/'common48.parquet').rename(columns={'G':'future_generation_MWh','L0':'no_wfh_load_MWh','template_A':'B_MWh','positive_pool':'D_MWh','local_template':'actual_absorbed_MWh'});q0.date=q0.date.astype(str);hybrid=~q0.date.str[:6].isin(m.DIRECT_MONTHS);q0['future_generation_MWh']+=case['pv']*q0.GI;q0.loc[hybrid,'no_wfh_load_MWh']*=1+case['load'];q0=q0.set_index(['date','slot']);new=pd.concat([pd.read_parquet(OUT/f'regional_{ym}_{name}.parquet') for ym in m.MONTHS]);new.date=new.date.astype(str);new=new.set_index(['date','slot'])
  for col,newcol in [('future_generation_MWh','G'),('no_wfh_load_MWh','L0'),('B_MWh','A'),('D_MWh','positive_pool'),('actual_absorbed_MWh','local_service')]:q0.loc[new.index,col]=new[newcol]
  # A has support only at 17:36, so replacing these covers its whole support.
  q0=q0.reset_index();slot=q0.slot.to_numpy();gen=q0.future_generation_MWh.to_numpy();load=q0.no_wfh_load_MWh.to_numpy();a=q0.B_MWh.to_numpy();D=q0.D_MWh.to_numpy();u=q0.actual_absorbed_MWh.to_numpy();ch=(slot>=12)&(slot<30);dis=slot>=30
  # Target 100 GWh useful residential service, either direct+delivered or evening delivery alone.
  for scale in [0,1]:
   net=gen-load-scale*a;direct=np.minimum(np.maximum(gen-load,0),a).sum()*scale
   Q=np.maximum(net,0)*ch;H=np.maximum(-net,0)*dis
   for objective,target in [('combined_delivery_100GWh',max(0.,100000-direct)),('evening_delivery_100GWh',100000)]:
    result,tr=minimum_storage(Q,H,target,'delivery');result.update(case=name,participation=scale,objective=objective,target_MWh=100000,direct_MWh=direct,required_storage_delivery_MWh=target,available_charge_MWh=Q.sum(),charge_window='06:00-15:00',discharge_window='15:00-24:00',endpoint='empty/empty',duration_h=4)
    storage.append(result)
    if name=='L+0_P+0' and tr is not None:tr['date']=q0.date;tr['slot']=slot;tr.to_parquet(OUT/f'storage_{objective}_{scale}.parquet',index=False)
   if name=='L+0_P+0':
    Ql=np.maximum(D-scale*u,0)*ch;Hl=np.maximum(load+scale*a-gen+D-scale*u,0)*dis
    for kind in ['capture','delivery']:
     target0=float(q[q['case'].eq('L+0_P+0')].local_MWh.sum())
     eqres,eqtr=minimum_storage(Ql,Hl,target0,kind);eqres.update(case=name,participation=scale,objective='local_'+kind+'_equivalent',target_MWh=target0,direct_MWh=scale*u.sum(),required_storage_delivery_MWh=target0 if kind=='delivery' else .9*target0,available_charge_MWh=Ql.sum(),charge_window='06:00-15:00',discharge_window='15:00-24:00',endpoint='empty/empty',duration_h=4)
     storage.append(eqres)
     if eqtr is not None:eqtr['date']=q0.date;eqtr['slot']=slot;eqtr.to_parquet(OUT/f'local_{kind}_equivalent_{scale}.parquet',index=False)
  print('storage',name,flush=True)
 pd.DataFrame(storage).to_csv(OUT/'storage_common_target.csv',index=False)
 # Historical 270.732 MWh: same old all-date demand and target, no reinterpretation.
 slot=old.slot.to_numpy();D=old.D_MWh.to_numpy();u=old.actual_absorbed_MWh.to_numpy();net=old.no_wfh_load_MWh.to_numpy()+old.B_MWh.to_numpy()-old.future_generation_MWh.to_numpy()
 Q=(D-u)*((slot>=12)&(slot<30));H=np.maximum(net+D-u,0)*(slot>=30)
 hs,ht=minimum_storage(Q,H,u.sum(),'capture');ht['date']=old.date.to_numpy();ht['slot']=slot;ht.to_parquet(OUT/'historical_270_dispatch.parquet',index=False)
 daily=ht.groupby('date').agg(charge_MWh=('charge_MWh','sum'),delivery_MWh=('delivery_MWh','sum'),max_SOC_MWh=('soc_end_MWh','max'),terminal_SOC_MWh=('soc_end_MWh','last'));daily['full_cycle']=np.isclose(daily.max_SOC_MWh,hs['capacity_MWh'],atol=1e-5);daily.to_csv(OUT/'historical_270_daily.csv')
 hs.update(throughput_lower_bound_MWh=u.sum()*np.sqrt(.9)/366,dates_full_cycle=int(daily.full_cycle.sum()),dates_below_full_cycle=int((~daily.full_cycle).sum()),power_limited_intervals=int(np.isclose(ht.charge_MWh,hs['power_MW']*.5,atol=1e-6).sum()))
 (OUT/'historical_270_explanation.json').write_text(json.dumps(hs,indent=2))

if __name__=='__main__':main()
