#!/usr/bin/env python3
"""Evaluate >=11 actual parameter levels; never interpolate sparse outcomes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures/extra/v4.7/nc_completion'
SOURCE=ROOT/'outputs/figures/extra/legacy_34307_seasonal/seasonal_date_slot_fields.parquet'

@njit
def dispatch(surplus,deficit,cap,duration,efficiency):
    eta=np.sqrt(efficiency);limit=cap/duration*.5
    soc=0.;charge=0.;delivered=0.;loss=0.;maxerr=0.;violation=0.
    for t in range(len(surplus)):
        c=min(surplus[t],limit,max(0.,(cap-soc)/eta))
        u=min(deficit[t],limit,max(0.,soc*eta))
        end=soc+eta*c-u/eta
        maxerr=max(maxerr,abs(end-soc-eta*c+u/eta));violation=max(violation,-end,end-cap,min(c,u))
        charge+=c;delivered+=u;loss+=(1-eta)*c+(1/eta-1)*u;soc=end
    return charge,delivered,loss,soc,maxerr,violation,abs(charge-delivered-loss-soc)

BASE={'behavior':1.,'pv':1.,'load':1.,'rebound':0.,'efficiency':.9,'duration_h':4.,'capacity_MWh':217.408120632172}
GRIDS={'behavior':np.linspace(0,6.3753,11),'pv':np.linspace(.5,1.5,11),'load':np.linspace(.8,1.2,11),
       'rebound':np.linspace(0,.4,11),'efficiency':np.linspace(.70,1.,11),
       'duration_h':np.linspace(1,12,11),'capacity_MWh':np.linspace(50,1000,11)}

def run(d,p,window='18-22'):
    B=d.B_MWh.to_numpy()*p['behavior'];L=d.no_wfh_load_MWh.to_numpy()*p['load'];G=d.future_generation_MWh.to_numpy()*p['pv']
    slot=d.slot.to_numpy();windows={'00-08':slot<16,'09-15':(slot>=18)&(slot<30),'15-18':(slot>=30)&(slot<36),'18-22':(slot>=36)&(slot<44)}
    windows['Other']=~np.logical_or.reduce(list(windows.values()))
    mask=windows[window].reshape(366,48);budget=B.reshape(366,48).sum(axis=1)
    extra=(p['rebound']*budget[:,None]*mask/mask.sum(axis=1)[:,None]).ravel()
    net0=L-G;net=net0+B+extra
    eligible=np.maximum(-net0,0);remaining=np.maximum(-net,0)
    service=(eligible-np.maximum(-net0-B,0)).sum()
    storage=dispatch(remaining,np.maximum(net,0),p['capacity_MWh'],p['duration_h'],p['efficiency'])
    assert storage[4]<1e-6 and storage[5]<1e-6 and storage[6]<1e-5
    assert abs(extra.sum()-p['rebound']*B.sum())<1e-6
    energy=eligible.sum();physical=(eligible-remaining).sum()
    assert physical+storage[0]<=energy+1e-6
    return dict(**p,window=window,negative_dates=int((net.reshape(366,48)<-1e-8).any(axis=1).sum()),
                negative_energy_GWh=remaining.sum()/1000,eligible_GWh=energy/1000,
                behavior_service_GWh=service/1000,system_match_percent=100*service/energy if energy else 0.,
                added_rebound_GWh=extra.sum()/1000,physical_absorption_GWh=physical/1000,
                storage_capture_GWh=storage[0]/1000,storage_delivery_GWh=storage[1]/1000,
                loss_GWh=storage[2]/1000,terminal_soc_MWh=storage[3],max_step_error_MWh=storage[4],
                max_violation_MWh=storage[5],balance_error_MWh=storage[6],
                joint_capture_GWh=(physical+storage[0])/1000,max_net_import_MW=max(0,net.max()*2),
                max_abs_ramp_MW=np.max(np.abs(np.diff(net*2))))

def main():
    d=pd.read_parquet(SOURCE).sort_values(['date','slot']).reset_index(drop=True)
    assert len(d)==17568 and not d.duplicated(['date','slot']).any()
    rows=[]
    for name,values in GRIDS.items():
        for value in values:rows.append(dict(parameter=name,**run(d,{**BASE,name:float(value)})))
    a=pd.DataFrame(rows);a.to_csv(OUT/'dense_parameter_oat.csv',index=False)
    pairs=[]
    for x,y in [('behavior','pv'),('capacity_MWh','duration_h'),('rebound','load')]:
        for xx in GRIDS[x]:
            for yy in GRIDS[y]:pairs.append(dict(pair=x+'__'+y,**run(d,{**BASE,x:float(xx),y:float(yy)})))
    pd.DataFrame(pairs).to_csv(OUT/'dense_parameter_pairs.csv',index=False)
    timing=[]
    for r in GRIDS['rebound']:
        for window in ['00-08','09-15','15-18','18-22','Other']:
            timing.append(run(d,{**BASE,'rebound':float(r)},window))
    pd.DataFrame(timing).to_csv(OUT/'dense_rebound_timing.csv',index=False)
    runbase=run(d,BASE);old=pd.read_csv(OUT/'annual_event_summary.csv').query("period=='Annual' and behavior_scale==1").iloc[0]
    assert abs(runbase['negative_energy_GWh']-old.negative_energy_GWh)<1e-8 and runbase['negative_dates']==old.negative_dates
    # Eleven actual block lengths; conditional date uncertainty only.
    rng=np.random.default_rng(20260917);boot=[]
    for season,g in d.groupby('season'):
        daily=g.groupby('date')[['D_MWh','actual_absorbed_MWh']].sum().to_numpy();n=len(daily)
        for length in range(1,22,2):
            starts=rng.integers(0,n,size=(2000,int(np.ceil(n/length))))
            ix=((starts[:,:,None]+np.arange(length))%n).reshape(2000,-1)[:,:n]
            v=daily[ix].sum(axis=1);rates=100*v[:,1]/v[:,0]
            boot.append(dict(season=season,block_days=length,draws=2000,lo=np.quantile(rates,.025),
                             median=np.median(rates),hi=np.quantile(rates,.975)))
    pd.DataFrame(boot).to_csv(OUT/'dense_date_block_intervals.csv',index=False)
    manifest=dict(levels={k:list(map(float,v)) for k,v in GRIDS.items()},baseline=BASE,
                  one_factor_rows=len(rows),pair_rows=len(pairs),timing_rows=len(timing),
                  block_lengths=list(range(1,22,2)),block_draws_each=2000,
                  source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  scope='system-net events and empty-start/free-end greedy battery; mixed direct/hybrid annual ledger',
                  inference='deterministic sensitivity; NOT mesh-local matching, probability CI or calibrated errors',
                  thresholds='No outcome thresholds selected post hoc. Full response surfaces are reported.',
                  numerical_pass=True)
    (OUT/'dense_parameter_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
