"""Approved P85–96: fixed-reference E/A account and signed paired absorption.
Uses the current complete mesh domain, common calendar and registered PV scenario.
No source baseline is changed; no test suite or unrelated experiments are run.
"""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
from run_clipping_cause import REV, ROOT, DATA, m, INPUT_VERSION, source_sql, calendar_frame

AGG = """
count(*) records, sum(E) E_MWh,sum(A) A_MWh,sum(L0) L0_MWh,
sum(G) G_MWh,sum(Lobs) Lsrc_MWh,sum(clip_energy) C_MWh,
sum(source_identity_gap) source_identity_gap_MWh,
max(abs(L0+E-Lobs-clip_energy-source_identity_gap)) restoration_max_error_MWh,
sum(least(G,L0)) U0_local_MWh,
sum(least(G,L0+E)) UE_local_MWh,sum(least(G,L0+A)) UA_local_MWh,
sum(least(greatest(G-L0,0),E)) S_E_local_MWh,
sum(local_service) S_A_local_MWh,
sum(local_service-least(greatest(G-L0,0),E)) T_local_MWh,
sum(greatest(local_service-least(greatest(G-L0,0),E),0)) T_positive_local_MWh,
sum(least(local_service-least(greatest(G-L0,0),E),0)) T_negative_local_MWh,
sum(least(G,L0+E)-least(G,Lobs)) source_correction_local_MWh,
sum(surplus) positive_surplus_MWh,
sum(least(E,A)) E_A_overlap_MWh,
sum(CASE WHEN A=0 THEN E ELSE 0 END) E_without_A_MWh,
sum(CASE WHEN E=0 THEN A ELSE 0 END) A_without_E_MWh
"""

def main():
    keys=pd.read_csv(REV.parent/'20261111_supplemental/data/mesh_population_scope.csv',
                     dtype={'mesh':str,'pref':str,'munic':str})
    cal=calendar_frame();cal['date']=cal.date.astype(str)
    dates=pd.to_datetime(cal.date)
    cal['day_class']=np.where(cal.daytype.eq('workday'),'eligible',
                    np.where(dates.dt.dayofweek>=5,'weekend','excluded_weekday'))
    con=duckdb.connect()
    con.execute('SET threads=4');con.execute("SET memory_limit='4GB'")
    con.execute('SET preserve_insertion_order=false')
    con.register('scope',keys[['mesh','pref']]);con.register('cal',cal[['date','day_class']])
    con.execute(f"CREATE TEMP TABLE templates AS SELECT cast(mesh AS varchar) mesh,mon,slot,dwfh FROM read_parquet('{m.WFH_REGISTERED}') WHERE scenario_id='S1'")
    con.execute(f"CREATE TEMP TABLE alpha AS SELECT cast(mesh AS varchar) mesh,cast(pref AS varchar) pref,increment_kw_a015 FROM read_parquet('{m.ALPHA}')")
    kernel=ROOT/'outputs/figures/extra/legacy_34307_seasonal/observed_stock_yield_kernel_full_year.parquet'
    con.execute(f"CREATE TEMP TABLE kernel AS SELECT cast(pref AS varchar) pref,cast(ym AS varchar) ym,slot,yield_kWh_per_kW_30min FROM read_parquet('{kernel}')")
    temporal=[];spatial=[]
    for ym in sorted(m.DIRECT_MONTHS):
        z=con.execute(source_sql(ym)+f"""SELECT CASE WHEN grouping(mesh)=0 THEN 'mesh' ELSE 'time' END grain,
          pref,mesh,date,slot,day_class,{AGG} FROM scores
          GROUP BY GROUPING SETS ((pref,mesh,day_class),(pref,date,slot,day_class))""").df()
        z['ym']=ym
        temporal.append(z[z.grain.eq('time')].drop(columns=['grain','mesh']))
        spatial.append(z[z.grain.eq('mesh')].drop(columns=['grain','date','slot']))
        print(json.dumps(dict(month=ym,local_E_GWh=float(temporal[-1].S_E_local_MWh.sum()/1000),
               local_A_GWh=float(temporal[-1].S_A_local_MWh.sum()/1000)),ensure_ascii=False),flush=True)
    con.close()
    time=pd.concat(temporal,ignore_index=True)
    mesh=pd.concat(spatial,ignore_index=True)
    mesh.to_parquet(DATA/'e_a_transition_mesh_month.parquet',index=False)
    time.to_parquet(DATA/'e_a_reference_pref_intervals.parquet',index=False)
    values=time.select_dtypes(include='number').columns.difference(['slot'])
    ops={c:'max' if c.endswith('max_error_MWh') else 'sum' for c in values}
    profile=time.groupby(['pref','ym','slot','day_class'],as_index=False).agg(
        **{c:(c,op) for c,op in ops.items()},date_count=('date','nunique'))
    for c in ['E_MWh','A_MWh']:profile[c.replace('_MWh','_mean_date_MWh')]=profile[c]/profile.date_count
    profile.to_csv(DATA/'e_a_reference_profile.csv',index=False)
    region=time.groupby(['date','slot','day_class','ym'],as_index=False).agg(ops)
    region['U0_regional_MWh']=np.minimum(region.G_MWh,region.L0_MWh)
    region['UE_regional_MWh']=np.minimum(region.G_MWh,region.L0_MWh+region.E_MWh)
    region['UA_regional_MWh']=np.minimum(region.G_MWh,region.L0_MWh+region.A_MWh)
    for tag in ['E','A']:
        region[f'S_{tag}_regional_MWh']=np.minimum(np.maximum(region.G_MWh-region.L0_MWh,0),region[f'{tag}_MWh'])
        region[f'pool_{tag}_MWh']=np.minimum(region.positive_surplus_MWh,region[f'{tag}_MWh'])
        region[f'residual_{tag}_MWh']=np.maximum(region.G_MWh-region.L0_MWh-region[f'{tag}_MWh'],0)
        region[f'positive_service_{tag}']=region[f'S_{tag}_regional_MWh']>1e-10
    region['T_regional_MWh']=region.S_A_regional_MWh-region.S_E_regional_MWh
    region['source_correction_regional_MWh']=region.UE_regional_MWh-np.minimum(region.G_MWh,region.Lsrc_MWh)
    region['pool_difference_MWh']=region.pool_A_MWh-region.pool_E_MWh
    region['regional_net_surplus_before_MWh']=np.maximum(region.G_MWh-region.L0_MWh,0)
    region.to_parquet(DATA/'e_a_transition_region_intervals.parquet',index=False)
    results=[]
    domains=[('direct_152_dates',region),('eligible_99_dates',region[region.day_class.eq('eligible')])]
    domains += [('calendar_'+k,z) for k,z in region.groupby('day_class')]
    for period,z in domains:
        for ac in ['local','regional']:
            results.append(dict(period=period,scope='complete_34305_meshes',accounting=ac,
             dates=int(z.date.nunique()),records=int(z.records.sum()),E_MWh=float(z.E_MWh.sum()),A_MWh=float(z.A_MWh.sum()),
             C_MWh=float(z.C_MWh.sum()),source_identity_gap_MWh=float(z.source_identity_gap_MWh.sum()),
             restoration_max_error_MWh=float(z.restoration_max_error_MWh.max()),
             S_E_MWh=float(z[f'S_E_{ac}_MWh'].sum()),S_A_MWh=float(z[f'S_A_{ac}_MWh'].sum()),
             transition_MWh=float(z[f'T_{ac}_MWh'].sum()),
             source_correction_MWh=float(z[f'source_correction_{ac}_MWh'].sum()),
             U0_MWh=float(z[f'U0_{ac}_MWh'].sum()),UE_MWh=float(z[f'UE_{ac}_MWh'].sum()),UA_MWh=float(z[f'UA_{ac}_MWh'].sum())))
        results.append(dict(period=period,scope='complete_34305_meshes',accounting='positive_pool_upper_bound',
             dates=int(z.date.nunique()),records=int(z.records.sum()),
             E_MWh=float(z.E_MWh.sum()),A_MWh=float(z.A_MWh.sum()),
             S_E_MWh=float(z.pool_E_MWh.sum()),S_A_MWh=float(z.pool_A_MWh.sum()),
             transition_MWh=float(z.pool_difference_MWh.sum())))
    pd.DataFrame(results).to_csv(DATA/'e_a_transition_services.csv',index=False)
    monthly=[]
    for ym,z in region.groupby('ym'):
        for ac in ['local','regional']:
            monthly.append(dict(ym=ym,accounting=ac,dates=z.date.nunique(),
             eligible_dates=z.loc[z.day_class.eq('eligible'),'date'].nunique(),
             E_MWh=z.E_MWh.sum(),A_MWh=z.A_MWh.sum(),
             S_E_MWh=z[f'S_E_{ac}_MWh'].sum(),S_A_MWh=z[f'S_A_{ac}_MWh'].sum(),
             transition_MWh=z[f'T_{ac}_MWh'].sum(),
             source_correction_MWh=z[f'source_correction_{ac}_MWh'].sum()))
    pd.DataFrame(monthly).to_csv(DATA/'e_a_transition_monthly.csv',index=False)
    info=dict(date='2026-11-14',status='calculated_fixed_reference_retained',
      input_version=INPUT_VERSION,units='MWh per half-hour before summation',
      scope_meshes=len(keys),dates=int(region.date.nunique()),eligible_dates=int(region.loc[region.day_class.eq('eligible'),'date'].nunique()),
      source=str(m.BASE.relative_to(ROOT)),A=str(m.WFH_REGISTERED.relative_to(ROOT)),
      script=str(Path(__file__).relative_to(ROOT)),
      definition='S_A=U(L0+A)-U(L0); S_E=U(L0+E)-U(L0); T=S_A-S_E. Same fixed L0 and G.',
      regional_active_E_intervals=int(region.positive_service_E.sum()),
      regional_active_A_intervals=int(region.positive_service_A.sum()),
      regional_active_E_dates=int(region.loc[region.positive_service_E,'date'].nunique()),
      regional_active_A_dates=int(region.loc[region.positive_service_A,'date'].nunique()),
      regional_positive_transition_MWh=float(region.T_regional_MWh.clip(lower=0).sum()),
      regional_negative_transition_MWh=float(region.T_regional_MWh.clip(upper=0).sum()),
      local_positive_record_transition_MWh=float(region.T_positive_local_MWh.sum()),
      local_negative_record_transition_MWh=float(region.T_negative_local_MWh.sum()),
      results=results,baseline_changed=False,tests_run=False,safety_review_run=False)
    (REV/'reports/e_a_reference_calculation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
