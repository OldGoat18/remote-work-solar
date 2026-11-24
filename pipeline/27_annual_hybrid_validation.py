#!/usr/bin/env python3
"""Annual events, chronological SOC, joint resources and uncertainty diagnostics.

Hybrid months retain their evidence labels. System netting and local surplus
pooling are different services. Uniform multipliers are NOT registered S10.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/_asset_location('asset_0003')
OUT=ROOT/_asset_location('asset_0105')
SCALES=[0.,.5,1.,1.5,2.,3.,5.,6.3753]
SEED=20260916
DT=.5

def save(d,name):
    d.to_csv(OUT/name,index=False)

def load_ledger():
    d=pd.read_parquet(SOURCE/'seasonal_date_slot_fields.parquet').sort_values(['date','slot']).reset_index(drop=True)
    d['date']=d.date.astype(str)
    d['timestamp']=pd.to_datetime(d.date,format='%Y%m%d')+pd.to_timedelta(d.slot*30,unit='min')
    expected=pd.date_range('2023-04-01','2024-04-01',freq='30min',inclusive='left')
    assert np.array_equal(d.timestamp.to_numpy(),expected.to_numpy()),'Calendar gap or duplicate'
    fields=['D_MWh','B_MWh','actual_absorbed_MWh','no_wfh_load_MWh','future_generation_MWh']
    assert np.isfinite(d[fields]).all().all() and d[fields].min().min()>=-1e-8
    d['local_deficit_MWh']=d.no_wfh_load_MWh-d.future_generation_MWh+d.D_MWh
    assert d.local_deficit_MWh.min()>=-1e-6
    return d

def daily_events(d,net):
    rows=[]
    for date,ix in d.groupby('date').groups.items():
        x=np.asarray(net)[np.asarray(ix)];neg=x < -1e-8
        starts=np.flatnonzero(neg & ~np.r_[False,neg[:-1]])
        ends=np.flatnonzero(neg & ~np.r_[neg[1:],False])
        rows.append(dict(date=date,event_count=len(starts),event_day=bool(neg.any()),
                         total_negative_hours=neg.sum()*DT,
                         longest_within_day_event_h=max((ends-starts+1)*DT,default=0.),
                         max_negative_depth_MW=max(0.,-x.min()/DT),negative_energy_MWh=np.maximum(-x,0).sum(),
                         max_abs_halfhour_ramp_MW=np.max(np.abs(np.diff(x/DT))),ramp_15_to_18_MW=(x[36]-x[30])/DT,
                         season=d.loc[ix[0],'season'],evidence_tier=d.loc[ix[0],'evidence_tier']))
    return pd.DataFrame(rows)

def event_experiments(d):
    parts,summaries,balances=[],[],[]
    for scale in SCALES:
        a=d[f'A_{str(scale).replace(".","_")}'].to_numpy(float)
        net=d.no_wfh_load_MWh.to_numpy()+scale*d.B_MWh.to_numpy()-d.future_generation_MWh.to_numpy()
        residual=d.D_MWh.to_numpy()-a;deficit=net+residual
        assert min(residual.min(),deficit.min())>=-1e-6
        assert (a-scale*d.B_MWh.to_numpy()).max()<=1e-6
        if scale==1: assert np.allclose(a,d.actual_absorbed_MWh,rtol=1e-10,atol=1e-6)
        daily=daily_events(d,net);daily['behavior_scale']=scale
        daily['scenario_definition']='uniform multiplier of S1; not registered S10'
        parts.append(daily)
        for label,g in [('Annual',daily),*list(daily.groupby('season'))]:
            summaries.append(dict(period=label,behavior_scale=scale,dates=len(g),negative_dates=int(g.event_day.sum()),
                                  incidence_percent=100*g.event_day.mean(),max_depth_MW=g.max_negative_depth_MW.max(),
                                  negative_energy_GWh=g.negative_energy_MWh.sum()/1000,negative_hours=g.total_negative_hours.sum(),
                                  evidence_tiers=';'.join(sorted(g.evidence_tier.unique()))))
        balances.append(dict(behavior_scale=scale,absorption_GWh=a.sum()/1000,local_residual_GWh=residual.sum()/1000,
                             local_deficit_GWh=deficit.sum()/1000,max_balance_error_MWh=np.abs(net-(deficit-residual)).max()))
    save(pd.concat(parts,ignore_index=True),'annual_daily_events.csv')
    save(pd.DataFrame(summaries),'annual_event_summary.csv')
    save(pd.DataFrame(balances),'annual_local_energy_balance.csv')

def solve_dispatch(surplus,deficit,capacity,duration=4.,efficiency=.9,policy='perfect_foresight',endpoint='cyclic'):
    """MWh flows with n+1 SOC states including the actual terminal state."""
    surplus,deficit=np.asarray(surplus,float),np.asarray(deficit,float)
    assert np.all(np.minimum(surplus,deficit)<=1e-8),'Specify non-overlapping operational modes'
    eta=np.sqrt(efficiency);n=len(surplus)
    cmax=np.minimum(surplus,capacity/duration*DT);dmax=np.minimum(deficit,capacity/duration*DT)
    if policy=='greedy':
        charge,discharge,states=np.zeros(n),np.zeros(n),np.zeros(n+1)
        for t in range(n):
            charge[t]=min(cmax[t],max((capacity-states[t])/eta,0))
            discharge[t]=min(dmax[t],max(states[t]*eta,0))
            states[t+1]=states[t]+eta*charge[t]-discharge[t]/eta
    else:
        t=np.arange(n);rows=np.tile(t,4);cols=np.r_[t,n+t,2*n+t,2*n+t+1]
        vals=np.r_[np.full(n,-eta),np.full(n,1/eta),-np.ones(n),np.ones(n)]
        bounds=list(zip(np.zeros(n),cmax))+list(zip(np.zeros(n),dmax))+[(0,capacity)]*(n+1);nr=n
        if endpoint=='cyclic':
            rows=np.r_[rows,n,n];cols=np.r_[cols,2*n,3*n];vals=np.r_[vals,-1.,1.];nr+=1
        elif endpoint=='empty':
            bounds[2*n]=(0,0);bounds[3*n]=(0,0)
        else: raise ValueError(endpoint)
        matrix=coo_matrix((vals,(rows,cols)),shape=(nr,3*n+1)).tocsr()
        result=linprog(np.r_[-np.ones(n),np.zeros(2*n+1)],A_eq=matrix,b_eq=np.zeros(nr),bounds=bounds,method='highs')
        if not result.success: raise RuntimeError(result.message)
        charge,discharge,states=result.x[:n],result.x[n:2*n],result.x[2*n:]
    loss=(1-eta)*charge+(1/eta-1)*discharge
    err=np.diff(states)-eta*charge+discharge/eta
    summary=dict(captured_GWh=charge.sum()/1000,delivered_GWh=discharge.sum()/1000,loss_GWh=loss.sum()/1000,
                 initial_soc_MWh=states[0],terminal_soc_MWh=states[-1],max_soc_MWh=states.max(),min_soc_MWh=states.min(),
                 cyclic_error_MWh=abs(states[-1]-states[0]),max_step_error_MWh=np.abs(err).max(),
                 total_balance_error_MWh=abs(charge.sum()-discharge.sum()-loss.sum()-(states[-1]-states[0])),
                 simultaneous_MWh=np.minimum(charge,discharge).sum(),input_equivalent_cycles=charge.sum()/capacity)
    assert summary['max_step_error_MWh']<1e-6 and summary['total_balance_error_MWh']<1e-5
    assert summary['simultaneous_MWh']<1e-8 and states.min()>=-1e-6 and states.max()<=capacity+1e-6
    assert np.all(charge<=cmax+1e-6) and np.all(discharge<=dmax+1e-6)
    if policy!='greedy': assert summary['cyclic_error_MWh']<1e-6
    trajectory=pd.DataFrame(dict(charge_MWh=charge,discharge_MWh=discharge,loss_MWh=loss,
                                  soc_start_MWh=states[:-1],soc_end_MWh=states[1:],
                                  available_charge_MWh=surplus,available_discharge_MWh=deficit))
    return summary,trajectory

def storage_experiments(d):
    summaries,seasonal,trajectories=[],[],[]
    for scope in ['system_net_surplus','local_pool_fixed_windows']:
        for scale in [0.,1.]:
            net=d.no_wfh_load_MWh.to_numpy()+scale*d.B_MWh.to_numpy()-d.future_generation_MWh.to_numpy()
            if scope=='system_net_surplus': surplus=np.maximum(-net,0);deficit=np.maximum(net,0)
            else:
                actual=d.actual_absorbed_MWh.to_numpy()*scale
                surplus=(d.D_MWh.to_numpy()-actual)*(d.slot.to_numpy()>=12)*(d.slot.to_numpy()<30)
                deficit=(net+d.D_MWh.to_numpy()-actual)*(d.slot.to_numpy()>=30)
            for capacity in [106.027066707611,217.408120632172,1000.]:
                for duration in [2.,4.,8.]:
                    for policy,endpoint in [('perfect_foresight','cyclic'),('perfect_foresight','empty'),('greedy','empty_start_free_end')]:
                        run_id=len(summaries)
                        s,tr=solve_dispatch(surplus,deficit,capacity,duration,policy=policy,endpoint=endpoint)
                        cfg=dict(run_id=run_id,scope=scope,behavior_scale=scale,capacity_MWh=capacity,duration_h=duration,
                                 efficiency=.9,policy=policy,endpoint=endpoint)
                        summaries.append({**cfg,**s});tr['run_id']=run_id;tr['date']=d.date.to_numpy();tr['slot']=d.slot.to_numpy()
                        tr['season']=d.season.to_numpy();tr['evidence_tier']=d.evidence_tier.to_numpy()
                        for season,g in tr.groupby('season'):
                            delta=(g.soc_end_MWh-g.soc_start_MWh).sum()
                            seasonal.append({**cfg,'season':season,'charge_GWh':g.charge_MWh.sum()/1000,
                                             'discharge_GWh':g.discharge_MWh.sum()/1000,'loss_GWh':g.loss_MWh.sum()/1000,
                                             'soc_change_sum_MWh':delta,
                                             'balance_error_MWh':abs(g.charge_MWh.sum()-g.discharge_MWh.sum()-g.loss_MWh.sum()-delta),
                                             'evidence_tiers':';'.join(sorted(g.evidence_tier.unique()))})
                        trajectories.append(tr)
        print(f'Completed {scope} dispatch',flush=True)
    summary=pd.DataFrame(summaries);save(summary,'annual_storage_summary.csv');save(pd.DataFrame(seasonal),'seasonal_soc_closure.csv')
    pd.concat(trajectories,ignore_index=True).to_parquet(OUT/'annual_dispatch_trajectories.parquet',index=False)
    keys=['scope','capacity_MWh','duration_h','efficiency','policy','endpoint']
    paired=summary[summary.behavior_scale.eq(0)].merge(summary[summary.behavior_scale.eq(1)],on=keys,suffixes=('_only','_after'))
    for scope in paired.scope.unique():
        if scope=='system_net_surplus':
            n=d.no_wfh_load_MWh-d.future_generation_MWh
            behavior=(np.maximum(-n,0)-np.maximum(-n-d.B_MWh,0)).sum()/1000;total=np.maximum(-n,0).sum()/1000
        else:
            mask=d.slot.between(12,29);behavior=d.loc[mask,'actual_absorbed_MWh'].sum()/1000;total=d.loc[mask,'D_MWh'].sum()/1000
        ix=paired.scope.eq(scope);paired.loc[ix,'behavior_service_GWh']=behavior;paired.loc[ix,'eligible_surplus_GWh']=total
    paired['joint_capture_GWh']=paired.behavior_service_GWh+paired.captured_GWh_after
    paired['behavior_increment_after_storage_GWh']=paired.joint_capture_GWh-paired.captured_GWh_only
    paired['storage_increment_after_behavior_GWh']=paired.captured_GWh_after
    paired['interaction_GWh']=paired.captured_GWh_after-paired.captured_GWh_only
    assert (paired.joint_capture_GWh<=paired.eligible_surplus_GWh+1e-7).all()
    save(paired,'annual_joint_resource_comparison.csv')

def bootstrap_and_aggregation(d):
    rng=np.random.default_rng(SEED);draws=[]
    for season,g in d.groupby('season'):
        daily=g.groupby('date')[['D_MWh','actual_absorbed_MWh']].sum();n=len(daily)
        # Preserve chronological adjacent dates; circular blocks disclosed.
        a=daily.actual_absorbed_MWh.to_numpy();den=daily.D_MWh.to_numpy()
        for length in [1,7,14]:
            for draw in range(1000):
                starts=rng.integers(0,n,size=int(np.ceil(n/length)))
                ix=((starts[:,None]+np.arange(length))%n).ravel()[:n]
                draws.append(dict(season=season,block_days=length,draw=draw,matching_percent=100*a[ix].sum()/den[ix].sum(),
                                  interpretation='conditional circular date blocks; not parameter uncertainty'))
    b=pd.DataFrame(draws);save(b,'matching_date_block_draws.csv')
    save(b.groupby(['season','block_days']).matching_percent.agg(mean='mean',lo=lambda x:x.quantile(.025),hi=lambda x:x.quantile(.975)).reset_index(),'matching_date_block_intervals.csv')
    mesh=pd.read_parquet(SOURCE/'seasonal_mesh_fields.parquet');rows=[]
    for season,g in mesh.groupby('season'):
        D=g.surplus_MWh.to_numpy();B=g.behavior_supply_MWh.to_numpy()
        actual=g.behavior_absorbed_MWh.sum()/D.sum()*100;coarse=np.minimum(D,B).sum()/D.sum()*100
        vals=np.array([np.minimum(D,rng.permutation(B)).sum()/D.sum()*100 for _ in range(2000)])
        rows.append(dict(season=season,halfhour_actual_percent=actual,same_placement_seasonal_min_percent=coarse,
                         aggregation_only_gap_pp=coarse-actual,same_grain_null_mean_percent=vals.mean(),
                         actual_minus_null_same_grain_pp=coarse-vals.mean(),
                         monte_carlo_lower_tail_p=(1+(vals<=coarse).sum())/(len(vals)+1),
                         interpretation='seasonal-energy placement null only; not a half-hour or causal test'))
    save(pd.DataFrame(rows),'null_aggregation_bias_audit.csv')

def main():
    OUT.mkdir(parents=True,exist_ok=True);d=load_ledger();event_experiments(d);storage_experiments(d);bootstrap_and_aggregation(d)
    manifest=dict(script=str(Path(__file__).relative_to(ROOT)),seed=SEED,
                  input_sha256=hashlib.sha256((SOURCE/'seasonal_date_slot_fields.parquet').read_bytes()).hexdigest(),
                  dates=d.date.nunique(),slots=len(d),evidence_date_counts=d.groupby('evidence_tier').date.nunique().to_dict(),
                  scenario_boundary='6.3753 is an S1 multiplier, not registered S10',
                  storage_boundary='System netting and local fixed-window pool are distinct network-free services')
    (OUT/'annual_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
