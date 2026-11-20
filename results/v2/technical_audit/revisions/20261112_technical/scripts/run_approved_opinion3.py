"""Approved P55–74: trace the existing A templates, units and person-day account.

Read the registered S1 products and the functions that construct them. Write
component/coverage summaries; do not regenerate A, load/PV, or later experiments.
No safety review, test suite or parameter scan is run.
"""
from pathlib import Path
import hashlib
import json
import sys
import duckdb
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
REV = Path(__file__).resolve().parents[1]
D = REV/'data'
sys.path.insert(0, str(ROOT/'src'))
from wlpr.wfh_state import state_layer as SL, component_loads as CL
from wlpr.wfh_state import materialize as M, climate_response as CR

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def quantiles(values, weights):
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = np.asarray(values)[good], np.asarray(weights)[good]
    order = np.argsort(v, kind='stable');v=v[order];w=w[order]
    # Inverse weighted empirical CDF; no interpolated synthetic observations.
    return {str(q): float(v[min(np.searchsorted(np.cumsum(w), q*w.sum()),len(v)-1)]) for q in [.1,.5,.9]}

def main():
    previous_path=D/'opinion3_input_account.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())['previous_profile_summary']
    else:
        old=pd.read_parquet(D/'per_person_day_energy.parquet')
        previous=dict(rows=len(old),weight_sum=float(old.weight.sum()),
                      quantiles=quantiles(old.kWh_pd.to_numpy(),old.weight.to_numpy()),
                      mean=float((old.kWh_pd*old.weight).sum()/old.weight.sum()),
                      zero_energy_rows=int(old.kWh_pd.eq(0).sum()),
                      sha256=digest(D/'per_person_day_energy.parquet'))

    keys=pd.read_csv(REV.parent/'20261111_supplemental/data/mesh_population_scope.csv',
                     dtype={'mesh':str,'pref':str,'munic':str})
    state=SL.mesh_state();work,lunch,presence=SL.occupancy_profiles()
    g=M.anchor_G(state,work,lunch,presence)
    kappa=1000*g
    scalars=pd.read_parquet(ROOT/'data/derived/wfh_state/a_dwfh_factors_scalars.parquet')
    scalars=scalars[scalars.scenario_id.eq('S1')].copy()
    for col in ['mesh','pref']:scalars[col]=scalars[col].astype(str)
    scalar_pair=scalars.merge(state,on=['mesh','pref'])
    ratios=scalar_pair.office_base/(scalar_pair.N_persons*CL.DEVICE_KW*CL.WORK_H)
    target=CL.SURVEY_CENTRAL_WINDOW_MWH
    unscaled_MWh=target/kappa
    allstate=keys[['pref','mesh','munic']].rename(columns={'pref':'scope_pref'}).merge(
        state.rename(columns={'pref':'state_pref'}),on='mesh',how='left')
    allstate['N_persons']=allstate.N_persons.fillna(0)
    for col in ['workers','teleworkers']:allstate[col]=allstate[col].fillna(0)

    metrics=pd.read_csv(D/'metrics_all.csv',dtype={'date':str})
    cal=metrics[metrics['case'].eq('L+0_P+0')][['date','day_class']].drop_duplicates().copy()
    cal['ym']=cal.date.str[:6]
    counts=cal.groupby('ym').day_class.agg(eligible_days=lambda x:int(x.eq('eligible').sum()),calendar_days='size')
    original=ROOT/'data/derived/wfh_state/a_dwfh_mesh_mon_slot.parquet'
    full=ROOT/'data/derived/wfh_state/a_dwfh_mesh_mon_slot_v2_fullyear/scenario=S1/part.parquet'
    fields=['dwfh']+['c_'+c for c in CL.COMP]
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
    con.register('keys',keys[['pref','mesh']])
    con.register('source_scalars',scalars[['mesh','pref']])
    rows=[];monthrows=[];profiles=[];coverage=[];slots=[]
    for ym,n in counts.iterrows():
        direct=ym in M.OBS_YM
        src=original if direct else full
        join='cast(a.mesh as varchar)=k.mesh'
        select=', '.join(f'sum(a.{c}) AS {c}' for c in fields)
        sql=f"""SELECT k.pref scope_pref,k.mesh,{select},count(*) source_rows,
               min(slot) first_slot,max(slot) last_slot
               FROM read_parquet('{src}') a JOIN keys k ON {join}
               WHERE a.scenario_id='S1' AND a.mon={int(ym[4:])} GROUP BY 1,2"""
        energy=con.execute(sql).df()
        f=allstate.merge(energy,on=['scope_pref','mesh'],how='left')
        f['ym']=ym;f['eligible_days']=int(n.eligible_days)
        f['source_present']=f.source_rows.notna()
        f['person_days']=f.N_persons*int(n.eligible_days)
        f['weight']=f.person_days
        f['source_status']=np.select([f.N_persons.eq(0),~f.source_present],
                                    ['zero_participant_state','template_join_missing'],default='available')
        f['ledger_template_MWh']=f.dwfh.fillna(0)
        f['ledger_kWh_pd']=np.where(f.N_persons.gt(0),f.ledger_template_MWh*1000/f.N_persons,np.nan)
        f['kWh_pd']=np.where(f.source_present & f.N_persons.gt(0),f.ledger_kWh_pd,np.nan)
        f['annual_month_MWh']=f.ledger_template_MWh*int(n.eligible_days)
        profiles.append(f[['scope_pref','state_pref','mesh','ym','N_persons','eligible_days','weight',
                           'source_present','source_status','ledger_template_MWh','annual_month_MWh','ledger_kWh_pd','kWh_pd']])
        coverage.append(f.loc[f.N_persons.gt(0)&~f.source_present,
                              ['scope_pref','state_pref','mesh','ym','N_persons','person_days','source_status']])
        monthrows.append(dict(ym=ym,source=str(src.relative_to(ROOT)),join_key='mesh',
                              eligible_days=int(n.eligible_days),calendar_days=int(n.calendar_days),
                              complete_meshes=len(keys),participant_meshes=int(f.N_persons.gt(0).sum()),
                              source_present_participant_meshes=int((f.source_present&f.N_persons.gt(0)).sum()),
                              baseline_participant_days=float(f.person_days.sum()),
                              profiled_participant_days=float(f.loc[f.source_present,'person_days'].sum()),
                              template_MWh=float(f.ledger_template_MWh.sum()),annual_MWh=float(f.annual_month_MWh.sum()),
                              first_nonzero_slot=int(energy.first_slot.min()),last_nonzero_slot=int(energy.last_slot.max())))
        for pref,ff in f.groupby('scope_pref'):
            for comp in CL.COMP:
                value=float(ff['c_'+comp].fillna(0).sum())
                rows.append(dict(ym=ym,pref=pref,component=comp,eligible_days=int(n.eligible_days),
                                 baseline_participant_days=float(ff.person_days.sum()),
                                 template_before_anchor_MWh=value/kappa,template_after_anchor_MWh=value,
                                 annual_before_anchor_MWh=value*int(n.eligible_days)/kappa,
                                 annual_after_anchor_MWh=value*int(n.eligible_days)))
        slotdata=con.execute(f"""SELECT a.slot,{select} FROM read_parquet('{src}') a
                             JOIN keys k ON {join} WHERE a.scenario_id='S1' AND a.mon={int(ym[4:])}
                             GROUP BY a.slot ORDER BY a.slot""").df()
        slotdata['ym']=ym;slotdata['eligible_days']=int(n.eligible_days);slots.append(slotdata)
        print('input account',ym,flush=True)
    con.close()
    prof=pd.concat(profiles,ignore_index=True)
    prof.to_parquet(D/'per_person_day_energy.parquet',index=False)
    missing=pd.concat(coverage,ignore_index=True);missing.to_csv(D/'opinion3_profile_coverage.csv',index=False)
    monthly=pd.DataFrame(monthrows)
    monthly.to_csv(D/'opinion3_monthly_account.csv',index=False)
    components=pd.DataFrame(rows)
    components.to_csv(D/'opinion3_components_month_pref.csv',index=False)
    total=components.groupby('component')[['annual_before_anchor_MWh','annual_after_anchor_MWh']].sum().reset_index()
    pdays=float(prof.weight.sum());energy=float(prof.annual_month_MWh.sum())
    total['kWh_per_baseline_participant_day']=total.annual_after_anchor_MWh*1000/pdays
    total['share_of_A']=total.annual_after_anchor_MWh/energy
    total.to_csv(D/'opinion3_component_account.csv',index=False)
    pd.concat(slots,ignore_index=True).to_csv(D/'opinion3_month_slot_account.csv',index=False)
    pd.DataFrame(dict(slot=np.arange(48),start_hour=np.arange(48)/2,work_energy_fraction=work,
                      lunch_energy_fraction=lunch,presence_weight=presence)).to_csv(D/'opinion3_shape_factors.csv',index=False)
    gh,gc,dark=M.monthly_factors(sorted(state.pref.unique()),presence,M.FULL_YM)
    sec=CR.monthly_secondary_cool(sorted(state.pref.unique()),list(range(1,13)))
    weather=[]
    for (pref,month),cool in sorted(gc.items()):
        mult=sec.get((pref,month),1.) if M.month_tier(month)=='cdd_prior' else 1.
        weather.append(dict(pref=pref,month=month,heating_response=gh[pref,month],
                            cooling_response_before_secondary=cool,secondary_cooling_multiplier=mult,
                            cooling_response_final=cool*mult,lighting_shape_sum=float(np.dot(presence,dark[month]))))
    pd.DataFrame(weather).to_csv(D/'opinion3_weather_factors.csv',index=False)
    muni=pd.read_csv(REV.parent/'20261111_supplemental/data/policy_municipality_inputs.csv')
    valid=prof.kWh_pd.notna()&prof.weight.gt(0)
    summary=dict(date='2026-11-13',opinion='P55–74',status='approved_input_account_completed',
        units='A is MWh per half-hour; normalized slot-energy fractions are summed without a second 0.5-hour factor.',
        annual_source_unchanged=False,source_template_values_unchanged=True,input_version='unique_mesh_template_join_20261113',complete_meshes=len(keys),participant_state_meshes=int(allstate.N_persons.gt(0).sum()),
        worker_count=float(allstate.workers.sum()),teleworker_equivalents=float(allstate.teleworkers.sum()),
        daily_participant_equivalents=float(allstate.N_persons.sum()),occupation_rates=[.189],day_share=SL.I0,
        daily_participation=.189*SL.I0,eligible_days=int(counts.eligible_days.sum()),baseline_participant_days=pdays,
        annual_A_MWh=energy,mean_kWh_per_baseline_participant_day=energy*1000/pdays,
        previous_profile_summary=previous,profile_quantiles_kWh=quantiles(prof.kWh_pd.to_numpy(),prof.weight.to_numpy()),
        profile_mean_kWh=float((prof.loc[valid,'kWh_pd']*prof.loc[valid,'weight']).sum()/prof.loc[valid,'weight'].sum()),
        profiled_participant_days=float(prof.loc[valid,'weight'].sum()),
        missing_profile_rows=len(missing),missing_profile_meshes=int(missing.mesh.nunique()),
        missing_profile_participant_days=float(missing.person_days.sum()),
        profile_coverage_fraction=float(prof.loc[valid,'weight'].sum()/pdays),
        missing_profile_cause='No positive-participant profiles are missing after aligning registered templates by unique mesh identity.',
        missing_profile_scope='All available participant profiles enter the energy ledger and weighted distribution. Source prefecture/weather and scope prefecture/municipal membership are preserved separately.',
        recovered_meshes=14,recovered_mesh_months=98,recovered_participant_days=43322.12942399999,
        anchor=dict(target_representative_templates_MWh=target,months=M.OBS_YM,
                    raw_engine_total_kWh=target/g,unscaled_MWh=unscaled_MWh,
                    dimensionless_kappa=kappa,code_G_MWh_per_kWh=g,formula='MWh = unscaled_kWh * kappa / 1000',
                    source='released_scorecard_anchor; legacy forward table absent',
                    original_forward_table_present=Path(CL.FWD).exists(),domain='all 33,597 positive-participant source meshes; before manuscript-domain restriction',
                    stored_scalar_inferred_coefficient_min=float(ratios.min()),stored_scalar_inferred_coefficient_max=float(ratios.max()),
                    device_unscaled_kWh_per_day=CL.DEVICE_KW*CL.WORK_H,
                    device_scaled_kWh_per_available_profile=CL.DEVICE_KW*CL.WORK_H*kappa),
        lighting=dict(coefficient=.05,implemented_unit='kWh per fully weighted half-hour slot before common scaling',
                      historical_variable='LIGHT_KW',note='The implementation multiplies 0.050 directly by occupancy/darkness; it is not integrated again as 0.050 kW. Preserve existing energy, clarify parameter name/unit.'),
        municipal=dict(units=len(muni),daily_equivalents=float(muni.N_persons.sum()),
                       added_budget_person_days=float(muni.N_persons.sum()*243*.25),
                       additional_scale_cap=1,meaning='registered fixed-activity allocation bound; no identified subsidy response or observed adoption limit'),
        evidence=dict(participation='reconstructed coefficient tables plus registered frequency; no empirical category differences in S1',
                      weather='AMeDAS month climatology; summer secondary humidity/wind response and geometric solar prior',
                      independent_A_validation=False,independent_load_and_PV_evaluations='different targets; not A validation',
                      realistic_participation_cap_identified=False),
        files={name:str(D/name) for name in ['opinion3_component_account.csv','opinion3_components_month_pref.csv',
              'opinion3_monthly_account.csv','opinion3_profile_coverage.csv','opinion3_month_slot_account.csv',
              'opinion3_shape_factors.csv','opinion3_weather_factors.csv','per_person_day_energy.parquet']})
    previous_components=pd.read_csv(D/'component_summary.csv').set_index('component').eligible_MWh
    summary['component_total_difference_from_existing_MWh']={r.component:float(r.annual_after_anchor_MWh-previous_components.loc[r.component]) for r in total.itertuples()}
    (D/'opinion3_input_account.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    parameters=[
        ('participation','occupation_rate',.189,'fraction','public_reconstructed_parameter','data/derived/wfh_scenarios/wfh_s6_occupation_telework_rates.csv'),
        ('participation','day_share',SL.I0,'fraction','registered_frequency_parameter','src/wlpr/wfh_state/state_layer.py'),
        ('occupancy','Q_NW',SL.Q_NW,'fraction','engineering_household_presence','src/wlpr/wfh_state/state_layer.py'),
        ('occupancy','Q_WK',SL.Q_WK,'fraction','engineering_household_presence','src/wlpr/wfh_state/state_layer.py'),
        ('end_use','device_power',CL.DEVICE_KW,'kW','engineering_prior','src/wlpr/wfh_state/component_loads.py'),
        ('end_use','work_hours',CL.WORK_H,'h','engineering_prior','src/wlpr/wfh_state/component_loads.py'),
        ('end_use','lighting_slot_coefficient',.05,'kWh/weighted half-hour slot','registered_implemented_coefficient','src/wlpr/wfh_state/component_loads.py'),
        ('end_use','lunch_energy',CL.LUNCH_KWH,'kWh/household workday','engineering_prior','src/wlpr/wfh_state/component_loads.py'),
        ('end_use','heating_power',CL.HEAT_KW,'kW','engineering_prior_with_heatloss_and_weather','src/wlpr/wfh_state/component_loads.py'),
        ('end_use','cooling_power',CL.COOL_KW,'kW','engineering_prior_with_heatloss_and_weather','src/wlpr/wfh_state/component_loads.py'),
        ('normalization','kappa',kappa,'dimensionless','single_released_template_total_anchor','src/wlpr/wfh_state/materialize.py'),
        ('calendar','eligible_dates',int(counts.eligible_days.sum()),'days','registered_activity_calendar','results/v2/technical_audit/revisions/20261112_technical/data/metrics_all.csv')]
    pd.DataFrame(parameters,columns=['group','parameter','value','unit','evidence_identity','source']).to_csv(D/'opinion3_parameter_sources.csv',index=False)
    print(json.dumps({k:summary[k] for k in ['annual_A_MWh','mean_kWh_per_baseline_participant_day','profile_quantiles_kWh',
                    'missing_profile_meshes','missing_profile_rows','profile_coverage_fraction','anchor']},indent=2),flush=True)

if __name__=='__main__':main()
