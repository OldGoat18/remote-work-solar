"""Targeted approved recalculation: no storage, allocation or spatial reruns.

Reuses all 11 deterministic cases, 100 recorded PV paths and five load fields.
Only scheduling columns change in existing results. Full-precision daily gains,
constraint diagnostics and compact before/after results are separately retained.
"""
from pathlib import Path
import json, gc, hashlib
import numpy as np
import pandas as pd
import duckdb
from numba import set_num_threads
from run_technical import m
from input_mesh_alignment import components
from lunch_constraints import VERSION, prepare, candidate_lunch, evaluate_surplus, evaluate_pv
from extend_pv_storage_window import residual_source, selected_factors

R=Path(__file__).resolve().parents[1];D=R/'data';ROOT=next(p for p in R.parents if (p/'pyproject.toml').exists())
BEFORE=D/'scheduling_before_revision.json'

def main():
    set_num_threads(4)
    if not BEFORE.exists():
        before={name:pd.read_csv(D/name).to_dict('records') for name in
                ['schedule_summary.csv','fixed_lunch_summary.csv','pv_residual_propagation.csv','load_residual_summary.csv']}
        BEFORE.write_text(json.dumps(before,indent=2)+'\n')
    keys=pd.read_csv(R.parent/'20261111_supplemental/data/mesh_population_scope.csv',dtype={'mesh':str,'pref':str})
    pi=keys.pref.map({'11':0,'12':1,'13':2,'14':3}).to_numpy(np.int64)
    cases=json.loads((D/'scenario_definitions.json').read_text())
    err=pd.read_parquet(ROOT/'outputs/figures/extra/v4.7/nc_completion/mesh_LV_month_holdout_error_sums.parquet')
    err=err[err.model.eq('owner_scaled')].copy();err.mesh=err.mesh.astype(str);load_fields={}
    for h,z in err.groupby('held_out_month'):
        z=z.set_index('mesh').reindex(keys.mesh);den=z.target_MWh+z.signed_error_MWh
        load_fields[str(h)]=np.divide(z.target_MWh.to_numpy(),den.to_numpy(),out=np.ones(len(z)),where=den.to_numpy()>1e-12)
    pools,source=residual_source();con=duckdb.connect();con.execute('set threads=4')
    daily=[];fixed_rows=[];constraints=[];pv_rows=[];load_rows=[];definition_rows=[]
    for ym in m.MONTHS:
        cal=m.dates_frame().query('ym==@ym');wd=cal.daytype.eq('workday').to_numpy();dates=cal.date.to_numpy()[wd]
        season=m.SEASON_BY_MONTH[int(ym[4:])];nd=int(wd.sum())
        with np.load(R/f'tmp/inputs_{ym}.npz') as z:
            B=z['B'];G=np.ascontiguousarray(z['G'][:,5:11,:][:,:,wd])
            L=np.ascontiguousarray(z['L'][:,5:11,:][:,:,wd]);GI=np.ascontiguousarray(z['GI'][:,5:11,:][:,:,wd])
        C=components(con,ym,keys,'S1')[:,:,3];cand,valid,moved,retained,fixed=prepare(B,C)
        profiles,_,_,_=candidate_lunch(C);active=C.sum(axis=1)>0
        for k in range(5):
            allowed=valid[:,k];peak=C[:,5:11].max(axis=1)
            constraints.append(dict(ym=ym,shift_min=(k-2)*30,active_meshes=int(active.sum()),
                eligible_dates=nd,feasible_active_meshes=int((allowed&active).sum()),
                peak_rejected_active_meshes=int((~allowed&active).sum()),
                source_lunch_MWh=float(C.sum()*nd),permitted_source_share=.5,
                moved_if_selected_MWh=float(moved[allowed,k].sum()*nd),
                boundary_retained_if_selected_MWh=float(retained[allowed,k].sum()*nd),
                max_component_energy_difference_MWh=float(np.abs(profiles[:,k].sum(axis=1)-C[:,5:11].sum(axis=1)).max()),
                max_feasible_peak_excess_MWh=float(np.maximum(profiles[allowed,k].max(axis=1)-peak[allowed],0).max()),
                outside_window_lunch_MWh=float((C[:,:5].sum()+C[:,11:].sum())*nd)))
        definition_rows.extend(dict(ym=ym,slot=t+17,original_lunch_MWh=float(C[:,t].sum()*nd),
            fixed_lunch_MWh=float(profiles[np.arange(len(B)),fixed,t-5].sum()*nd) if 5<=t<=10 else float(C[:,t].sum()*nd)) for t in range(20))
        schedule=pd.read_csv(D/f'schedule_{ym}.csv',dtype={'date':str})
        for case in cases:
            a=case['a'];d=np.maximum(G+case['pv']*GI-L*(1+(case['load'] if ym not in m.DIRECT_MONTHS else 0)),0)
            gains,counts,movement=evaluate_surplus(d,cand*a,valid,moved*a,retained*a,fixed)
            anchor=schedule[(schedule.case==case['case'])&schedule.schedule.eq('anchored')].set_index('date').reindex(dates.astype(str)).service_MWh.to_numpy()
            lookup=dict(zip(dates.astype(str),anchor+gains[:,0]))
            loc=(schedule.case==case['case'])&schedule.schedule.eq('lunch_constrained')
            schedule.loc[loc,'service_MWh']=schedule.loc[loc,'date'].map(lookup)
            fixed_rows.append(dict(case=case['case'],ym=ym,anchored_MWh=float(anchor.sum()),
                fixed_lunch_MWh=float(anchor.sum()+gains[:,1].sum()),gain_MWh=float(gains[:,1].sum())))
            for j,date in enumerate(dates):
                row=dict(case=case['case'],ym=ym,date=str(date),season=season,anchored_MWh=anchor[j],
                    constrained_gain_MWh=gains[j,0],fixed_gain_MWh=gains[j,1],
                    constrained_MWh=anchor[j]+gains[j,0],fixed_lunch_MWh=anchor[j]+gains[j,1],
                    upper_shifted_source_MWh=movement[j,0],upper_boundary_retained_MWh=movement[j,1],
                    fixed_shifted_source_MWh=movement[j,2],fixed_boundary_retained_MWh=movement[j,3],
                    fixed_fallback_meshes=int((fixed==2).sum()))
                row.update({f'upper_shift_{(k-2)*30:+d}_mesh_days':int(counts[j,k]) for k in range(5)})
                daily.append(row)
        schedule.to_csv(D/f'schedule_{ym}.csv',index=False)
        print('deterministic scheduling',ym,flush=True)
        fac=np.ascontiguousarray(selected_factors(ym,pools)[:,wd,:,22:28])
        gain,count,movement=evaluate_pv(G,L,GI,pi,fac,cand,valid,moved,retained,fixed)
        pv=pd.read_csv(D/f'residual_totals_{ym}.csv').sort_values('draw')
        pv['constrained_MWh']=pv.local_MWh.to_numpy()+gain[:,0]
        pv['fixed_lunch_MWh']=pv.local_MWh.to_numpy()+gain[:,1]
        pv['fixed_gain_MWh']=gain[:,1]
        pv.to_csv(D/f'residual_totals_{ym}.csv',index=False)
        for k in range(len(fac)):
            row=dict(draw=k,ym=ym,constrained_gain_MWh=gain[k,0],fixed_gain_MWh=gain[k,1],
                upper_shifted_source_MWh=movement[k,0],upper_boundary_retained_MWh=movement[k,1],
                fixed_shifted_source_MWh=movement[k,2],fixed_boundary_retained_MWh=movement[k,3])
            row.update({f'upper_shift_{(j-2)*30:+d}_mesh_days':int(count[k,j]) for j in range(5)})
            pv_rows.append(row)
        loads=pd.read_csv(D/f'load_residual_{ym}.csv',dtype={'source_month':str})
        for h,factor in load_fields.items():
            d=np.maximum(G-L*(factor[:,None,None] if ym not in m.DIRECT_MONTHS else 1),0)
            gain,counts,movement=evaluate_surplus(d,cand,valid,moved,retained,fixed)
            loc=loads.source_month.eq(h)
            loads.loc[loc,'constrained_gain_MWh']=gain[:,0].sum()
            loads.loc[loc,'fixed_gain_MWh']=gain[:,1].sum()
            loads.loc[loc,'fixed_lunch_MWh']=loads.loc[loc,'local_MWh']+gain[:,1].sum()
            load_rows.append(dict(source_month=h,ym=ym,constrained_gain_MWh=float(gain[:,0].sum()),fixed_gain_MWh=float(gain[:,1].sum())))
        loads.to_csv(D/f'load_residual_{ym}.csv',index=False)
        print('existing PV paths and load fields',ym,flush=True)
        del G,L,GI,C,fac;gc.collect()
    con.close()
    pd.DataFrame(daily).to_csv(D/'scheduling_revision_daily.csv',index=False)
    pd.DataFrame(constraints).to_csv(D/'scheduling_constraints.csv',index=False)
    pd.DataFrame(definition_rows).to_csv(D/'scheduling_lunch_profiles.csv',index=False)
    pd.DataFrame(pv_rows).to_csv(D/'scheduling_pv_monthly.csv',index=False)
    pd.DataFrame(load_rows).to_csv(D/'scheduling_load_monthly.csv',index=False)
    fixed=pd.DataFrame(fixed_rows);fixed.to_csv(D/'fixed_lunch_monthly.csv',index=False)
    fixed.groupby('case')[['anchored_MWh','fixed_lunch_MWh','gain_MWh']].sum().to_csv(D/'fixed_lunch_summary.csv')
    sche=pd.concat([pd.read_csv(D/f'schedule_{ym}.csv') for ym in m.MONTHS]);sche.to_csv(D/'schedule_all.csv',index=False)
    sche.groupby(['case','schedule']).service_MWh.sum().to_csv(D/'schedule_summary.csv')
    # Update affected annual columns only; preserve unrelated conclusions exactly.
    pv_month=pd.concat([pd.read_csv(D/f'residual_totals_{ym}.csv') for ym in m.MONTHS])
    sums=pv_month.groupby('draw')[['constrained_MWh','fixed_lunch_MWh','fixed_gain_MWh']].sum()
    q=pd.read_csv(D/'pv_residual_propagation.csv').set_index('draw')
    for col in sums:q[col]=sums[col]
    q['constrained_gain_MWh']=q.constrained_MWh-q.local_MWh
    q.to_csv(D/'pv_residual_propagation.csv');q.quantile([.05,.5,.95]).to_csv(D/'pv_residual_percentiles.csv')
    loads=pd.read_csv(D/'load_residual_summary.csv',dtype={'source_month':str}).set_index('source_month')
    sums=pd.concat([pd.read_csv(D/f'load_residual_{ym}.csv',dtype={'source_month':str}) for ym in m.MONTHS]).groupby('source_month')[['constrained_gain_MWh','fixed_gain_MWh','fixed_lunch_MWh']].sum()
    for col in sums:loads[col]=sums[col]
    loads.to_csv(D/'load_residual_summary.csv')
    old=json.loads(BEFORE.read_text());before=pd.DataFrame(old['schedule_summary.csv']).pivot(index='case',columns='schedule',values='service_MWh')
    after=sche.groupby(['case','schedule']).service_MWh.sum().unstack()
    comp=pd.DataFrame({'anchored_MWh':after.anchored,'old_constrained_MWh':before.lunch_constrained,
        'new_constrained_MWh':after.lunch_constrained,'old_constrained_gain_MWh':before.lunch_constrained-before.anchored,
        'new_constrained_gain_MWh':after.lunch_constrained-after.anchored,
        'old_fixed_gain_MWh':pd.DataFrame(old['fixed_lunch_summary.csv']).set_index('case').gain_MWh,
        'new_fixed_gain_MWh':fixed.groupby('case').gain_MWh.sum()})
    comp.to_csv(D/'scheduling_before_after.csv')
    ref=pd.DataFrame(daily).query("case=='L+0_P+0'")
    manifest=dict(version=VERSION,date='2026-11-14',status='complete',meshes=len(keys),eligible_dates=243,
        energy_unit='MWh per half-hour',source_slots=list(range(22,28)),destination_slots=list(range(22,28)),
        shifts_minutes=[-60,-30,0,30,60],movable_fraction=.5,boundary_rule='retain out-of-window energy at its original interval',
        peak_cap='each mesh-month original lunch half-hour maximum',cap_identity='engineering scenario',
        fixed_rule='+30 min with boundary retention; identity fallback if peak cap fails; independent of PV',
        upper_rule='best feasible candidate for each mesh-date with perfect supply information',
        tie_rule='identity, then -30,+30,-60,+60; improvement tolerance 1e-12 MWh per mesh-day',
        deterministic_cases=len(cases),pv_paths=100,load_fields=5,pv_source=source,
        block_hashes={ym:hashlib.sha256((D/f'residual_blocks_{ym}.csv').read_bytes()).hexdigest() for ym in m.MONTHS},
        diagnostics=pd.DataFrame(constraints).groupby('shift_min').agg(
            feasible_mesh_months=('feasible_active_meshes','sum'),rejected_mesh_months=('peak_rejected_active_meshes','sum'),
            max_energy_difference_MWh=('max_component_energy_difference_MWh','max'),max_peak_excess_MWh=('max_feasible_peak_excess_MWh','max')).reset_index().to_dict('records'),
        reference_movement={c:float(ref[c].sum()) for c in ref.columns if 'shifted_source' in c or 'boundary_retained' in c or 'mesh_days' in c},
        approved_scope_completed=True,necessary_inputs_missing=[],unaffected='original/ideal/permutation schedules, participation, G/L/A, storage and allocation results',
        tests_or_safety_reviews_run=False)
    (D/'scheduling_revision_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(comp.to_string());print(q[['constrained_gain_MWh','fixed_gain_MWh']].quantile([.05,.5,.95]).to_string())

if __name__=='__main__':main()
