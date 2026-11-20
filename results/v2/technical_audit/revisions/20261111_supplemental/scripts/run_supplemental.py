"""Read-only source reconstruction for spatial, scheduling and policy experiments.

All outputs are versioned here; no source ledger or plotting input is overwritten.
Policy participation is an explicitly modelled increment above the S1 activity
profile, paid per added person-day, with the same annual calendar for all rules.
"""
from pathlib import Path
import sys, json, calendar, gc
import numpy as np
import pandas as pd
import duckdb
from numba import njit, prange, set_num_threads

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src')]
from run_v47_halfhour_spatial import reconstruct_month
from run_v47_spatial_order_uncertainty import spatial_inputs
from wlpr.wfh_state.state_layer import mesh_state
BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'data'; OUT.mkdir(exist_ok=True)
SRC = ROOT/'outputs/figures/extra/legacy_34307_seasonal'
FIG = ROOT/'results/v2/technical_audit/figures_stage9'
HOLIDAYS = {'20230429','20230503','20230504','20230505','20230717','20230811',
 '20230918','20230923','20231009','20231103','20231123','20231229','20231230',
 '20231231','20240101','20240102','20240103','20240108','20240211','20240212','20240223','20240320'}
TARGETS=['all_municipalities','high_surplus_q4','low_surplus_q4','high_relief_q4','low_relief_q4','tokyo_prefecture_subset']
RULES=['U','TE','SW','RW']; FRACTIONS=[.10,.25,.50]

def allocate(n, weight, target, fraction):
    """Equal-cost added activity with x in [0,1]; target priority then overflow."""
    x=np.zeros(len(n)); remaining=fraction*n.sum()
    for mask in [target & (n>0), (~target) & (n>0)]:
        if remaining<=1e-7: break
        cap=n[mask].sum()
        if cap==0: continue
        spend=min(remaining,cap)
        w=np.maximum(weight[mask],0)
        if w.sum()==0: w=np.ones(len(w))
        # Zero-weight locations enter after positive-weight locations saturate.
        w=np.maximum(w,np.max(w)*1e-12)
        lo=0.;hi=max(1.,1./w.min())
        for _ in range(90):
            mid=(lo+hi)/2
            if np.sum(n[mask]*np.minimum(1.,mid*w))<spend: lo=mid
            else: hi=mid
        x[mask]=np.minimum(1.,(lo+hi)/2*w); remaining-=np.sum(n[mask]*x[mask])
    return x

@njit(parallel=True)
def policy_scores(D, prefix, B, multipliers, municipal_index, nm):
    ns=multipliers.shape[0]; nd=D.shape[2]
    out=np.zeros((ns,nm))
    for s in prange(ns):
        for i in range(D.shape[0]):
            j=municipal_index[i]
            if j<0: continue
            value=0.
            for t in range(D.shape[1]):
                b=B[i,t]*multipliers[s,j];lo=0;hi=nd
                while lo<hi:
                    mid=(lo+hi)//2
                    if D[i,t,mid]<b:lo=mid+1
                    else:hi=mid
                value+=prefix[i,t,lo]+(nd-lo)*b
            out[s,j]+=value
    return out

@njit(parallel=True)
def local_schedules(D,B):
    """Exact per-mesh permutation expectation and rank upper bound, 16 slots."""
    out=np.zeros((D.shape[2],3))
    # Parallel over days, keeping reductions deterministic within each day.
    for d in prange(D.shape[2]):
        for i in range(D.shape[0]):
            fixed=0.;anch=0.;mean=0.;direct=0.
            for t in (0,17,18,19): fixed+=min(D[i,t,d],B[i,t])
            ds=np.sort(D[i,1:17,d]);bs=np.sort(B[i,1:17])
            for t in range(16):
                anch+=min(D[i,t+1,d],B[i,t+1]);direct+=min(ds[t],bs[t])
                for v in range(16):mean+=min(ds[t],bs[v])/16
            out[d,0]+=fixed+anch;out[d,1]+=fixed+mean;out[d,2]+=fixed+direct
    return out

def main():
    set_num_threads(4)
    monthly=pd.read_parquet(SRC/'seasonal_mesh_month_fields.parquet')
    keys,_,_=spatial_inputs(monthly)
    complete=monthly.groupby('mesh').source_mesh_present.min().eq(1)
    keys=keys[keys.mesh.isin(complete[complete].index)].reset_index(drop=True)
    state=mesh_state()[['mesh','N_persons','workers']]
    keys=keys.merge(state,on='mesh',how='left',validate='one_to_one')
    keys[['N_persons','workers']]=keys[['N_persons','workers']].fillna(0)
    bridge=pd.read_parquet(ROOT/'data/frozen/pv/repos_mesh_munic_bpg.parquet')[['mesh','munic','B_PG']]
    bridge['mesh']=bridge.mesh.astype(str);bridge['munic']=bridge.munic.astype(str)
    bridge=bridge.sort_values('B_PG',ascending=False).drop_duplicates('mesh')
    keys=keys.merge(bridge[['mesh','munic']],on='mesh',how='left',validate='one_to_one')
    membership=pd.read_csv(FIG/'R5/redesign_spatial_M5_v1/data/M5_municipality_membership.csv',dtype={'munic':str})
    muni=membership[['munic','pref','surplus_MWh','service_GWh',*TARGETS]].copy()
    counts=keys.groupby('munic')[['N_persons','workers']].sum()
    muni=muni.join(counts,on='munic');n=muni.N_persons.to_numpy()
    muni['baseline_person_days']=243*n
    muni.to_csv(OUT/'policy_municipality_inputs.csv',index=False)
    lookup={v:i for i,v in enumerate(muni.munic)}
    mi=keys.munic.map(lookup).fillna(-1).to_numpy(np.int64)
    scenarios=[]; multipliers=[np.ones(len(muni))]
    for f in FRACTIONS:
        for target in TARGETS:
            mask=muni[target].to_numpy(bool)
            for rule in RULES:
                if rule=='U': x=np.full(len(n),f)
                else:
                    w=np.ones(len(n)) if rule=='TE' else muni.surplus_MWh.to_numpy() if rule=='SW' else muni.service_GWh.to_numpy()
                    # Priority sets use their stated weights; overflow is uniform.
                    w=np.where(mask,w,1.)
                    x=allocate(n,w,mask,f)
                scenarios.append(dict(scenario_id=len(scenarios)+1,fraction=f,target=target,rule=rule,
                   additional_person_days=float(243*np.dot(n,x)),total_expenditure_JPY=float(1050*243*np.dot(n,x))))
                multipliers.append(1+x)
    multipliers=np.array(multipliers)
    sc=pd.DataFrame(scenarios);sc.to_csv(OUT/'policy_scenarios.csv',index=False)
    pd.DataFrame([dict(scenario_id=k,munic=m,scale=a,additional_person_days=243*n[j]*(a-1))
       for k,row in enumerate(multipliers) for j,(m,a) in enumerate(zip(muni.munic,row))]).to_csv(OUT/'policy_allocations.csv',index=False)
    coords=pd.read_csv(FIG/'R2/reference_swapped_guides/data/mesh_season.csv',usecols=['mesh','lon','lat'],dtype={'mesh':str}).drop_duplicates('mesh')
    keys=keys.merge(coords,on='mesh',how='left',validate='one_to_one')
    groups={}
    for r in [.5,1.,2.,5.,10.,20.]:
        xy=np.column_stack([np.floor(keys.lon/(r/91)),np.floor(keys.lat/(r/111))])
        _,idx=np.unique(xy,axis=0,return_inverse=True);groups[r]=idx
    keys.to_csv(OUT/'mesh_population_scope.csv',index=False)
    con=duckdb.connect();con.execute("SET threads=4");con.execute("SET memory_limit='5GB'")
    (BASE/'tmp').mkdir(exist_ok=True);con.execute(f"SET temp_directory='{BASE/'tmp'}'")
    for ym in sorted(monthly.ym.unique()):
        ym=str(ym);done=OUT/f'month_{ym}.json'
        if done.exists():print('reusing',ym,flush=True);continue
        D,B,seen,audit=reconstruct_month(ym,keys[['pref','mesh']],con)
        nd=D.shape[2];dates=pd.date_range(ym+'01',periods=nd)
        work=np.array([d.dayofweek<5 and d.strftime('%Y%m%d') not in HOLIDAYS for d in dates])
        season=monthly.loc[monthly.ym.eq(ym),'season'].iloc[0]
        local=local_schedules(np.ascontiguousarray(D[:,:,work]),B)
        Ds=D.sum(axis=0);Bs=B.sum(axis=0)
        sched=[]
        for k,d in enumerate(dates[work]):
            j=np.flatnonzero(work)[k];sv=Ds[:,j];dv=Bs
            fixed=np.minimum(sv[[0,17,18,19]],dv[[0,17,18,19]]).sum()
            pooled=[np.minimum(sv,dv).sum(),fixed+np.minimum(sv[1:17,None],dv[None,1:17]).mean(axis=1).sum(),fixed+np.minimum(np.sort(sv[1:17]),np.sort(dv[1:17])).sum()]
            for scope,values in [('mesh_local',local[k]),('regional_pool',pooled)]:
                for name,val in zip(['anchored','permutation_expectation','pv_directed'],values):
                    sched.append(dict(date=d.strftime('%Y%m%d'),season=season,scope=scope,schedule=name,service_MWh=float(val),demand_MWh=float(B.sum())))
        pd.DataFrame(sched).to_csv(OUT/f'schedule_{ym}.csv',index=False)
        spatial=[]
        for r,idx in groups.items():
            ds=np.zeros((idx.max()+1,20,nd));bs=np.zeros((idx.max()+1,20))
            np.add.at(ds,idx,D);np.add.at(bs,idx,B)
            spatial.append(dict(ym=ym,season=season,cell_side_km=r,simultaneous_cell_MWh=float(np.minimum(ds,bs[:,:,None]).sum()),mesh_local_MWh=float(np.minimum(D,B[:,:,None]).sum())))
            del ds,bs
        pd.DataFrame(spatial).to_csv(OUT/f'spatial_{ym}.csv',index=False)
        sorted_d=np.sort(D[:,:,work],axis=2)
        prefix=np.concatenate([np.zeros((*sorted_d.shape[:2],1)),np.cumsum(sorted_d,axis=2)],axis=2)
        score=policy_scores(sorted_d,prefix,B,multipliers,mi,len(muni))
        pd.DataFrame([dict(ym=ym,scenario_id=k,munic=m,total_service_MWh=float(v),additional_service_MWh=float(v-score[0,j]))
          for k,row in enumerate(score) for j,(m,v) in enumerate(zip(muni.munic,row))]).to_csv(OUT/f'policy_{ym}.csv',index=False)
        base=monthly[monthly.ym.eq(ym)&monthly.mesh.isin(keys.mesh)]
        info=dict(ym=ym,meshes=len(keys),workdays=int(work.sum()),local_MWh=float(np.minimum(D,B[:,:,None]).sum()),
                  source_local_MWh=float(base.behavior_absorbed_MWh.sum()),demand_MWh=float(B.sum()*nd),
                  source_demand_MWh=float(base.behavior_supply_MWh.sum()),missing_people_demand_MWh=float(B[keys.N_persons.eq(0)].sum()*nd),audit=audit)
        done.write_text(json.dumps(info,indent=2));print('completed',ym,info['local_MWh'],flush=True)
        del D,B,seen,sorted_d,prefix;gc.collect()
    con.close()
    schedule=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('schedule_202*.csv'))]);schedule.to_csv(OUT/'schedule_daily.csv',index=False)
    schedule.groupby(['season','scope','schedule'],as_index=False)[['service_MWh','demand_MWh']].sum().to_csv(OUT/'schedule_seasonal.csv',index=False)
    schedule.groupby(['scope','schedule'],as_index=False)[['service_MWh','demand_MWh']].sum().to_csv(OUT/'schedule_annual.csv',index=False)
    spatial=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('spatial_202*.csv'))]);spatial.groupby(['season','cell_side_km'],as_index=False)[['simultaneous_cell_MWh','mesh_local_MWh']].sum().to_csv(OUT/'spatial_simultaneous.csv',index=False)
    policy=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('policy_202*.csv'))]);pm=policy.groupby(['scenario_id','munic'],as_index=False)[['total_service_MWh','additional_service_MWh']].sum();pm.to_csv(OUT/'policy_contributions.csv',index=False)
    sums=pm.groupby('scenario_id',as_index=False)[['total_service_MWh','additional_service_MWh']].sum()
    sums=sc.merge(sums,on='scenario_id');sums['service_per_JPY_kWh']=sums.additional_service_MWh*1000/sums.total_expenditure_JPY
    sums['service_per_person_day_kWh']=sums.additional_service_MWh*1000/sums.additional_person_days
    sums.to_csv(OUT/'policy_equal_budget.csv',index=False)
    print(sums[sums.fraction.eq(.25)].to_string(index=False),flush=True)

if __name__=='__main__':main()
