#!/usr/bin/env python3
"""Same-grain spatial null, valid coalition accounting and system stress draws.

All draws are conditional model diagnostics. None identifies a causal WFH
effect or supplies an empirically calibrated parameter distribution.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'outputs/figures/extra/legacy_34307_seasonal'
OUT = ROOT/'outputs/figures/extra/v4.7/nc_completion'
SEED = 20260916
N_NULL = 2000

def matrix(mesh, column):
    return mesh.pivot(index=['pref','mesh'], columns='ym', values=column).sort_index().sort_index(axis=1)

def spatial_inputs(mesh):
    D = matrix(mesh,'surplus_MWh'); B = matrix(mesh,'behavior_supply_MWh')
    assert D.shape == (34307,12) and not D.isna().any().any()
    assert D.index.equals(B.index) and np.isfinite(B).all().all()
    keys = D.index.to_frame(index=False)
    s = keys.mesh.astype(str).str.zfill(9)
    q = s.str[8].astype(int)-1
    y = s.str[:2].astype(int)*160+s.str[4].astype(int)*20+s.str[6].astype(int)*2+q//2
    x = s.str[2:4].astype(int)*160+s.str[5].astype(int)*20+s.str[7].astype(int)*2+q%2
    keys['x_cell']=x;keys['y_cell']=y
    keys['block']=keys.pref.astype(str)+':'+(x//20).astype(str)+':'+(y//20).astype(str)
    keys['eligible']=B.sum(axis=1).to_numpy()>0
    return keys,D,B

def make_permutations(keys, n=N_NULL):
    """A single bijection per draw is reused for every month and slot."""
    rng=np.random.default_rng(SEED)
    # Explicit eligibility: positive model B somewhere in this year. This
    # does not certify household or job eligibility at a new destination.
    groups={
        'global_unrestricted': [np.arange(len(keys))],
        'within_pref_eligible': list(keys.groupby(['pref','eligible']).indices.values()),
        'within_spatial_block_eligible': list(keys.groupby(['block','eligible']).indices.values()),
    }
    for model,ixs in groups.items():
        for draw in range(n):
            p=np.arange(len(keys),dtype=np.int32)
            for ix in ixs: p[ix]=rng.permutation(ix)
            yield model,draw,p

def monthly_spatial_null(mesh):
    keys,df,bf=spatial_inputs(mesh);D=df.to_numpy();B=bf.to_numpy();den=D.sum(axis=0)
    identity=np.minimum(D,B).sum(axis=0)
    halfhour=matrix(mesh,'behavior_absorbed_MWh').sum(axis=0).to_numpy()
    draws=[]; audit=[]
    for model,draw,p in make_permutations(keys):
        moved=B[p]
        vals=np.minimum(D,moved).sum(axis=0)
        if draw in [0,N_NULL-1]:
            assert np.array_equal(np.sort(p),np.arange(len(keys)))
            err=np.max(np.abs(moved.sum(axis=0)-den*0-B.sum(axis=0)))
            assert err<1e-7
            audit.append(dict(model=model,draw=draw,max_month_budget_error_MWh=err,
                              trajectory_mapping_sha256=hashlib.sha256(p.tobytes()).hexdigest()))
        draws.extend(dict(model=model,draw=draw,ym=ym,match_percent=100*v/d)
                     for ym,v,d in zip(df.columns,vals,den))
    draws=pd.DataFrame(draws);rows=[]
    for j,ym in enumerate(df.columns):
        for model,g in draws[draws.ym.eq(ym)].groupby('model'):
            v=g.match_percent.to_numpy();actual=100*identity[j]/den[j]
            k=int((v<=actual).sum());p=(k+1)/(len(v)+1)
            rows.append(dict(ym=ym,model=model,draws=len(v),actual_same_grain_percent=actual,
                             halfhour_absorption_percent=100*halfhour[j]/den[j],
                             aggregation_gap_pp=100*(identity[j]-halfhour[j])/den[j],
                             null_mean_percent=v.mean(),null_p025_percent=np.quantile(v,.025),
                             null_p975_percent=np.quantile(v,.975),lower_tail_p=p,
                             two_sided_p=min(1,2*min(p,(1+(v>=actual).sum())/(len(v)+1))),
                             lower_tail_mc_se=np.sqrt(p*(1-p)/(len(v)+1)),
                             extreme_count=k,identity_error_MWh=0.,
                             grain='min of monthly mesh energies; not half-hour matching'))
    pd.DataFrame(rows).to_csv(OUT/'monthly_spatial_null_summary.csv',index=False)
    draws.to_parquet(OUT/'monthly_spatial_null_draws.parquet',index=False)
    pd.DataFrame(audit).to_csv(OUT/'spatial_permutation_budget_audit.csv',index=False)
    print('Monthly trajectory permutations complete',flush=True)

def coalition_values(d,mesh,k):
    """Finite magnitude scaling M, pooled space S, free within-season time T.

    Each coalition relaxes exactly its named constraints, so every order is
    a path through the SAME set function. No wall silently relaxes another.
    """
    D=d.D_MWh.to_numpy();B=d.B_MWh.to_numpy()
    md=mesh.groupby(['pref','mesh']).surplus_MWh.sum().to_numpy()
    mb=mesh.groupby(['pref','mesh']).behavior_supply_MWh.sum().to_numpy()
    vals={}
    for mag in [False,True]:
        scale=k if mag else 1.
        bit=1 if mag else 0
        col='A_'+str(float(scale)).replace('.','_')
        vals[bit]=float(d[col].sum())
        vals[bit|2]=float(np.minimum(D,scale*B).sum())
        vals[bit|4]=float(np.minimum(md,scale*mb).sum())
        vals[bit|6]=float(min(D.sum(),scale*B.sum()))
    return vals

def decomposition_order(mesh,d):
    rows=[];coalitions=[]
    for season,g in d.groupby('season'):
        mg=mesh[mesh.season.eq(season)]
        for k in [2.,6.3753]:
            vals=coalition_values(g,mg,k);den=g.D_MWh.sum()
            for mask,val in vals.items():
                coalitions.append(dict(season=season,magnitude_scale=k,coalition_mask=mask,absorption_GWh=val/1000))
            for order in itertools.permutations([('magnitude',1),('spatial',2),('temporal',4)]):
                mask=0;gains={}
                for name,bit in order:
                    gains[name]=(vals[mask|bit]-vals[mask])/1000
                    assert gains[name]>=-1e-7
                    mask|=bit
                rows.append(dict(season=season,magnitude_scale=k,order=' > '.join(x[0] for x in order),
                                 actual_GWh=vals[0]/1000,final_relaxed_GWh=vals[7]/1000,
                                 residual_after_all_relaxations_GWh=(den-vals[7])/1000,
                                 **{f'{name}_gain_GWh':v for name,v in gains.items()},
                                 order_sum_error_GWh=sum(gains.values())-(vals[7]-vals[0])/1000,
                                 interpretation='finite magnitude scaling, space pooling, season-wide time pooling; conditional accounting'))
    out=pd.DataFrame(rows);out.to_csv(OUT/'three_wall_order_sensitivity.csv',index=False)
    pd.DataFrame(coalitions).to_csv(OUT/'three_wall_coalition_values.csv',index=False)
    out.groupby(['season','magnitude_scale'])[[c for c in out if c.endswith('_gain_GWh')]].agg(['mean','min','max']).to_csv(OUT/'three_wall_shapley_order_summary.csv')

@njit
def greedy_summary(surplus,deficit,capacity,efficiency):
    eta=np.sqrt(efficiency);limit=capacity/8;state=0.;C=0.;U=0.;loss=0.;err=0.;violation=0.
    for t in range(len(surplus)):
        c=min(surplus[t],limit,max((capacity-state)/eta,0.))
        u=min(deficit[t],limit,max(state*eta,0.))
        end=state+eta*c-u/eta
        err=max(err,abs(end-state-eta*c+u/eta))
        violation=max(violation,-end,end-capacity,min(c,u))
        C+=c;U+=u;loss+=(1-eta)*c+(1/eta-1)*u;state=end
    return C,U,loss,state,err,violation,abs(C-U-loss-state)

def system_ensemble_row(d,p):
    B=d.B_MWh.to_numpy()*p['behavior_multiplier']
    before=p['load_multiplier']*d.no_wfh_load_MWh.to_numpy()-p['pv_multiplier']*d.future_generation_MWh.to_numpy()
    # Energy-conserving evening rebound: each day's r * B energy in 18-22.
    rebound=np.zeros((366,48));rebound[:,36:44]=p['rebound_rate']*B.reshape(366,48).sum(axis=1)[:,None]/8
    after=before+B+rebound.ravel()
    eligible=np.maximum(-before,0)
    behavior_service=(eligible-np.maximum(-before-B,0)).sum()
    surplus=np.maximum(-after,0);deficit=np.maximum(after,0)
    only=greedy_summary(eligible,np.maximum(before,0),217.408120632172,p['round_trip_efficiency'])
    joint=greedy_summary(surplus,deficit,217.408120632172,p['round_trip_efficiency'])
    physical=(eligible-surplus).sum()
    return dict(**p,system_behavior_matching_percent=100*behavior_service/eligible.sum() if eligible.sum() else 0.,
                behavior_service_GWh=behavior_service/1000,rebound_energy_GWh=rebound.sum()/1000,
                expected_rebound_GWh=p['rebound_rate']*B.sum()/1000,
                negative_dates=int((after.reshape(366,48)<-1e-8).any(axis=1).sum()),
                negative_energy_GWh=surplus.sum()/1000,eligible_surplus_GWh=eligible.sum()/1000,
                storage_only_capture_GWh=only[0]/1000,joint_storage_capture_GWh=joint[0]/1000,
                joint_physical_capture_GWh=(physical+joint[0])/1000,
                joint_storage_delivery_GWh=joint[1]/1000,terminal_soc_MWh=joint[3],
                max_step_error_MWh=joint[4],max_violation_MWh=joint[5],balance_error_MWh=joint[6],
                interpretation='system-net service; assumption draws, not empirical probability intervals')

def correlated_ensemble(d,n=4000):
    d=d.sort_values(['date','slot']).reset_index(drop=True)
    assert len(d)==366*48
    corr=np.array([[1,-.2,-.1,0,0],[-.2,1,.25,0,0],[-.1,.25,1,.15,0],[0,0,.15,1,-.1],[0,0,0,-.1,1]])
    bounds={'behavior_multiplier':(.75,1.35),'pv_multiplier':(.9,1.1),'load_multiplier':(.9,1.1),
            'rebound_rate':(0,.4),'round_trip_efficiency':(.81,.95)}
    rows=[]
    anchor={k:1. for k in bounds};anchor.update(rebound_rate=0.,round_trip_efficiency=.9)
    rows.append(dict(batch=-1,draw=-1,**system_ensemble_row(d,anchor)))
    for batch in [0,1]:
        u=qmc.LatinHypercube(5,seed=SEED+17+batch).random(n)
        z=norm.ppf(np.clip(u,1e-10,1-1e-10))@np.linalg.cholesky(corr).T
        uniforms=norm.cdf(z)
        for i in range(n):
            p={k:lo+(hi-lo)*uniforms[i,j] for j,(k,(lo,hi)) in enumerate(bounds.items())}
            rows.append(dict(batch=batch,draw=i,**system_ensemble_row(d,p)))
    out=pd.DataFrame(rows);out.to_csv(OUT/'joint_assumption_ensemble.csv',index=False)
    result=[]
    for col in ['system_behavior_matching_percent','negative_energy_GWh','joint_storage_capture_GWh']:
        a=out[out.batch.eq(0)][col].quantile([.025,.5,.975]).to_numpy()
        b=out[out.batch.eq(1)][col].quantile([.025,.5,.975]).to_numpy()
        for q,x,y in zip([.025,.5,.975],a,b): result.append(dict(metric=col,quantile=q,batch0=x,batch1=y,absolute_difference=abs(x-y)))
    pd.DataFrame(result).to_csv(OUT/'assumption_ensemble_batch_stability.csv',index=False)
    return {'bounds':bounds,'latent_gaussian_correlation':corr.tolist(),
            'provenance':'Analyst-defined stress ranges and dependence; NOT survey SEs or calibrated uncertainty',
            'scope':'System-net events and chronological greedy SOC. No local mesh matching claim.',
            'draws':2*n,'independent_batches':2,'anchor_rows':1,'rebound':'daily budget allocated 18-22'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mesh=pd.read_parquet(SRC/'seasonal_mesh_month_fields.parquet')
    d=pd.read_parquet(SRC/'seasonal_date_slot_fields.parquet')
    monthly_spatial_null(mesh);decomposition_order(mesh,d)
    manifest=correlated_ensemble(d)
    manifest.update(seed=SEED,null_draws_per_model=N_NULL,monthly_null_grain='mesh x month; fixed all-year trajectory mapping',
                    same_grain_rule='monthly null compared ONLY to min(monthly D,monthly B); half-hour result separate',
                    magnitude_order_rule='coalition set function, finite scaling 2 and 6.3753; residual retained')
    (OUT/'spatial_order_uncertainty_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Spatial, coalition and system-assumption experiments complete',flush=True)

if __name__=='__main__': main()
