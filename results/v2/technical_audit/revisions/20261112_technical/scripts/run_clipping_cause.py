"""P85–96: direct-period negative-source and reconstruction decomposition.

Reads the current complete-mesh ledger and registered A; does not change the
baseline, observations, or annual experiment outputs. All energies are MWh per
half-hour before summation. The unclipped reference is diagnostic only.
"""
from pathlib import Path
import json
import sys

import duckdb
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = next(p for p in REV.parents if (p / 'pyproject.toml').exists())
DATA = REV / 'data'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from input_mesh_alignment import m, INPUT_VERSION
from wlpr.io.activity_calendar import calendar_frame

TOL = 1e-10


def source_sql(ym):
    return f"""WITH joined AS (
      SELECT cast(b.pref AS varchar) pref,cast(b.mesh AS varchar) mesh,
        cast(b.date AS varchar) date,b.slot,c.day_class,
        b.L_owner_gross O,b.L_nonowner_import V,b.E_wfh E,
        b.L_all_gross_equiv Lobs,b.L_all_import,b.owner_import,
        b.import_scope_mismatch,b.imputed_mix,
        coalesce(w.dwfh,0) B,
        CASE WHEN c.day_class='eligible' THEN coalesce(w.dwfh,0) ELSE 0 END A,
        b.G_pv+coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000 G,
        CASE WHEN b.L_owner_gross+b.L_nonowner_import>0
          THEN least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0) ELSE 0 END q
      FROM read_parquet('{m.BASE}/pref=*/ym={ym}/*.parquet') b
      JOIN scope s ON cast(b.mesh AS varchar)=s.mesh AND cast(b.pref AS varchar)=s.pref
      JOIN cal c ON cast(b.date AS varchar)=c.date
      LEFT JOIN templates w ON cast(b.mesh AS varchar)=w.mesh AND b.slot=w.slot AND w.mon={int(ym[4:])}
      LEFT JOIN alpha a ON cast(b.mesh AS varchar)=a.mesh AND cast(b.pref AS varchar)=a.pref
      LEFT JOIN kernel k ON cast(b.pref AS varchar)=k.pref AND k.ym='{ym}' AND b.slot=k.slot
    ), remainders AS (
      SELECT *,O-q*E ro,V-(1-q)*E rn,O+V-E Lraw,
        greatest(-O,0)+greatest(-V,0) source_negative_energy,
        O+V-Lobs source_identity_gap FROM joined
    ), reference AS (
      SELECT *,greatest(ro,0)+greatest(rn,0) L0,
        greatest(-ro,0)+greatest(-rn,0) clip_energy,
        CASE WHEN O < -{TOL} OR V < -{TOL} THEN 'source_negative'
             WHEN ro < -{TOL} OR rn < -{TOL} THEN 'removal_induced'
             ELSE 'no_material_clip' END cause,
        CASE WHEN Lobs>=0 THEN least(A,Lobs) END Astar,
        CASE WHEN Lobs>=0 THEN Lobs-least(A,Lobs) END Lstar
      FROM remainders
    ), scores AS (
      SELECT *,least(greatest(G-L0,0),A) local_service,
        least(greatest(G-Lraw,0),A) unclipped_service,
        least(greatest(G-(Lraw+source_negative_energy),0),A) source_only_service,
        least(greatest(G-Lstar,0),Astar) conserving_service,
        L0+A-Lobs reconstruction_gap,
        greatest(G-L0,0) surplus,
        CASE WHEN Lobs>=0 THEN greatest(G-Lstar,0) END conserving_surplus
      FROM reference
    ) """


AGG = f"""
 count(*) records,
 count(*) filter(where O < -{TOL}) owner_source_negative_count,
 count(*) filter(where V < -{TOL}) nonowner_source_negative_count,
 count(*) filter(where ro < -{TOL}) owner_clip_count,
 count(*) filter(where rn < -{TOL}) nonowner_clip_count,
 count(*) filter(where O>=-{TOL} AND ro<-{TOL}) owner_removal_induced_count,
 count(*) filter(where V>=-{TOL} AND rn<-{TOL}) nonowner_removal_induced_count,
 count(*) filter(where q<0 OR q>1) share_outside_unit_count,
 count(*) filter(where O+V<=0) share_zero_denominator_count,
 count(*) filter(where Lobs<0) negative_total_count,
 count(*) filter(where E<-{TOL}) negative_E_count,
 count(*) filter(where imputed_mix=1) imputed_mix_count,
 count(*) filter(where import_scope_mismatch=1) source_mismatch_flag_count,
 count(*) filter(where A>Lobs AND Lobs>=0) A_above_observed_count,
 count(*) filter(where reconstruction_gap>{TOL}) reconstruction_positive_count,
 count(*) filter(where reconstruction_gap<-{TOL}) reconstruction_negative_count,
 sum(greatest(-O,0)) owner_source_negative_MWh,
 sum(greatest(-V,0)) nonowner_source_negative_MWh,
 sum(greatest(-ro,0)) owner_clip_MWh,
 sum(greatest(-rn,0)) nonowner_clip_MWh,
 sum(clip_energy) clipping_MWh,
 sum(source_negative_energy) source_negative_MWh,
 sum(clip_energy-source_negative_energy) removal_extra_clipping_MWh,
 sum(source_identity_gap) source_identity_gap_MWh,
 sum(abs(source_identity_gap)) source_identity_abs_gap_MWh,
 sum(A-E) component_difference_MWh,
 sum(E) removed_MWh,sum(A) added_MWh,sum(B) ungated_template_MWh,
 sum(Lobs) observed_total_MWh,sum(L0) reference_MWh,sum(G) generation_MWh,
 sum(reconstruction_gap) reconstruction_gap_MWh,
 sum(abs(reconstruction_gap)) reconstruction_abs_gap_MWh,
 sum(L0+B-Lobs) ungated_reconstruction_gap_MWh,
 max(abs(reconstruction_gap-(source_identity_gap+A-E+clip_energy))) identity_max_error_MWh,
 sum(local_service) local_service_MWh,sum(unclipped_service) unclipped_service_MWh,
 sum(source_only_service-unclipped_service) source_clipping_service_effect_MWh,
 sum(local_service-source_only_service) removal_clipping_service_effect_MWh,
 sum(local_service-unclipped_service) clipping_service_effect_MWh,
 sum(surplus) positive_surplus_MWh,
 sum(CASE WHEN Lobs>=0 THEN local_service END) valid_total_local_service_MWh,
 sum(conserving_service) conserving_service_MWh,
 sum(Astar) conserving_A_MWh,sum(Lstar) conserving_L0_MWh,
 sum(conserving_surplus) conserving_positive_surplus_MWh,
 sum(CASE WHEN Lobs>=0 THEN greatest(A-Lobs,0) END) template_above_load_MWh,
 sum(CASE WHEN Lobs>=0 THEN G END) valid_total_G_MWh,
 sum(CASE WHEN Lobs>=0 THEN A END) valid_total_A_MWh,
 sum(CASE WHEN Lobs>=0 THEN L0 END) valid_total_L0_MWh,
 sum(CASE WHEN Lobs>=0 THEN surplus END) valid_total_positive_surplus_MWh,
 max(CASE WHEN Lobs>=0 THEN abs(Lstar+Astar-Lobs) END) conserving_identity_max_error_MWh
"""


def aggregate(frame, by):
    numeric = frame.select_dtypes(include='number').columns.difference(['slot'])
    ops = {column: 'max' if 'max_error' in column else 'sum' for column in numeric}
    return frame.groupby(by, dropna=False, as_index=False).agg(ops)


def main():
    keys = pd.read_csv(REV.parent / '20261111_supplemental/data/mesh_population_scope.csv', dtype={'mesh': str, 'pref': str, 'munic': str})
    cal = calendar_frame()
    cal['date'] = cal.date.astype(str)
    dates = pd.to_datetime(cal.date)
    cal['day_class'] = np.where(cal.daytype.eq('workday'), 'eligible', np.where(dates.dt.dayofweek>=5, 'weekend', 'excluded_weekday'))
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='4GB'")
    con.execute('SET preserve_insertion_order=false')
    con.register('scope', keys[['mesh', 'pref']])
    con.register('cal', cal[['date', 'day_class']])
    con.execute(f"CREATE TEMP TABLE templates AS SELECT cast(mesh AS varchar) mesh,mon,slot,dwfh FROM read_parquet('{m.WFH_REGISTERED}') WHERE scenario_id='S1'")
    con.execute(f"CREATE TEMP TABLE alpha AS SELECT cast(mesh AS varchar) mesh,cast(pref AS varchar) pref,increment_kw_a015 FROM read_parquet('{m.ALPHA}')")
    kernel = ROOT / 'outputs/figures/extra/legacy_34307_seasonal/observed_stock_yield_kernel_full_year.parquet'
    con.execute(f"CREATE TEMP TABLE kernel AS SELECT cast(pref AS varchar) pref,cast(ym AS varchar) ym,slot,yield_kWh_per_kW_30min FROM read_parquet('{kernel}')")
    times, spaces, events = [], [], []
    for ym in sorted(m.DIRECT_MONTHS):
        sql = source_sql(ym)
        grouped = con.execute(sql + f"""SELECT CASE WHEN grouping(mesh)=0 THEN 'mesh' ELSE 'time' END grain,
          pref,mesh,date,slot,day_class,cause,{AGG} FROM scores
          GROUP BY GROUPING SETS ((pref,mesh,day_class,cause),(pref,date,slot,day_class,cause))""").df()
        grouped['ym'] = ym
        temporal = grouped[grouped.grain.eq('time')].drop(columns=['grain', 'mesh'])
        spatial = grouped[grouped.grain.eq('mesh')].drop(columns=['grain', 'date', 'slot'])
        times.append(temporal); spaces.append(spatial)
        temporal.to_csv(DATA / f'clipping_cause_time_{ym}.csv', index=False)
        spatial.to_parquet(DATA / f'clipping_cause_mesh_{ym}.parquet', index=False)
        event = con.execute(sql + f"""SELECT pref,mesh,date,slot,day_class,cause,O,V,E,q,ro,rn,Lobs,
          L_all_import,owner_import,import_scope_mismatch,imputed_mix,A,B,G,L0,
          source_negative_energy,clip_energy,source_identity_gap,reconstruction_gap,
          local_service,unclipped_service,source_only_service,conserving_service,Astar,Lstar
          FROM scores WHERE O<-{TOL} OR V<-{TOL} OR ro<-{TOL} OR rn<-{TOL} OR q<0 OR q>1 OR Lobs<0
          ORDER BY pref,mesh,date,slot""").df()
        events.append(event)
        print(json.dumps(dict(month=ym,records=int(temporal.records.sum()),events=len(event),
                              clip_MWh=float(temporal.clipping_MWh.sum()),service_MWh=float(temporal.local_service_MWh.sum())), ensure_ascii=False), flush=True)
    con.close()
    time = pd.concat(times, ignore_index=True)
    space = pd.concat(spaces, ignore_index=True)
    event = pd.concat(events, ignore_index=True)
    time.to_csv(DATA / 'clipping_cause_time.csv', index=False)
    space.to_parquet(DATA / 'clipping_cause_space.parquet', index=False)
    event.to_parquet(DATA / 'clipping_cause_events.parquet', index=False)
    summary = aggregate(time, ['day_class', 'cause'])
    summary.to_csv(DATA / 'clipping_cause_summary.csv', index=False)
    for name, by in [('month', ['ym', 'cause']), ('prefecture', ['pref', 'cause']), ('slot', ['slot', 'cause'])]:
        aggregate(time, by).to_csv(DATA / f'clipping_cause_{name}.csv', index=False)
    period = aggregate(time.assign(period='direct_152_dates'), ['period'])
    eligible = aggregate(time[time.day_class.eq('eligible')].assign(period='eligible_99_dates'), ['period'])
    totals = pd.concat([period, eligible], ignore_index=True)
    totals.to_csv(DATA / 'clipping_reconstruction_totals.csv', index=False)
    intervals = aggregate(time, ['date', 'slot', 'day_class'])
    intervals['regional_increment_MWh'] = np.minimum(np.maximum(intervals.generation_MWh-intervals.reference_MWh,0),intervals.added_MWh)
    intervals['pool_MWh'] = np.minimum(intervals.positive_surplus_MWh,intervals.added_MWh)
    intervals['valid_total_regional_increment_MWh'] = np.minimum(np.maximum(intervals.valid_total_G_MWh-intervals.valid_total_L0_MWh,0),intervals.valid_total_A_MWh)
    intervals['conserving_regional_increment_MWh'] = np.minimum(np.maximum(intervals.valid_total_G_MWh-intervals.conserving_L0_MWh,0),intervals.conserving_A_MWh)
    intervals['conserving_pool_MWh'] = np.minimum(intervals.conserving_positive_surplus_MWh,intervals.conserving_A_MWh)
    intervals.to_parquet(DATA / 'clipping_cause_region_intervals.parquet', index=False)
    monthly = intervals.assign(ym=intervals.date.str[:6]).groupby('ym')[['added_MWh','removed_MWh','clipping_MWh',
        'reconstruction_gap_MWh','local_service_MWh','conserving_service_MWh','regional_increment_MWh',
        'conserving_regional_increment_MWh']].sum()
    monthly['local_change_pct'] = 100*(monthly.conserving_service_MWh/monthly.local_service_MWh-1)
    monthly['regional_change_pct'] = 100*(monthly.conserving_regional_increment_MWh/monthly.regional_increment_MWh.replace(0,np.nan)-1)
    monthly.to_csv(DATA / 'clipping_reconstruction_monthly.csv')
    region_unclipped = np.minimum(np.maximum(intervals.generation_MWh-(intervals.reference_MWh-intervals.clipping_MWh),0),intervals.added_MWh)
    active = intervals.regional_increment_MWh > TOL
    active_star = intervals.conserving_regional_increment_MWh > TOL
    threshold_detail = dict(regional_clipping_effect_MWh=float((intervals.regional_increment_MWh-region_unclipped).sum()),
        registered_active_intervals=int(active.sum()),conserving_active_intervals=int(active_star.sum()),
        registered_active_dates=int(intervals.loc[active,'date'].nunique()),conserving_active_dates=int(intervals.loc[active_star,'date'].nunique()),
        lost_positive_intervals=int((active&~active_star).sum()),gained_positive_intervals=int((~active&active_star).sum()),
        pool_change_MWh=float(intervals.conserving_pool_MWh.sum()-intervals.pool_MWh.sum()))
    services = []
    for scenario, local, pool, regional, demand, reference in [
        ('registered', 'local_service_MWh','pool_MWh','regional_increment_MWh','added_MWh','reference_MWh'),
        ('conserving_valid_total', 'conserving_service_MWh','conserving_pool_MWh','conserving_regional_increment_MWh','conserving_A_MWh','conserving_L0_MWh')]:
        services.append(dict(scenario=scenario,local_MWh=float(intervals[local].sum()),pool_MWh=float(intervals[pool].sum()),
                             regional_MWh=float(intervals[regional].sum()),A_MWh=float(intervals[demand].sum()),L0_MWh=float(intervals[reference].sum())))
    pd.DataFrame(services).to_csv(DATA / 'clipping_reconstruction_services.csv', index=False)
    manifest = dict(date='2026-11-13',opinion='P85–96',status='calculated_current_baseline_unchanged',
                    script=str(Path(__file__).resolve().relative_to(ROOT)),input_version=INPUT_VERSION,
                    scope_meshes=len(keys),dates=int(time.date.nunique()),eligible_dates=int(time.loc[time.day_class.eq('eligible'),'date'].nunique()),
                    source=str(m.BASE.relative_to(ROOT)),template=str(m.WFH_REGISTERED.relative_to(ROOT)),
                    units='MWh per half-hour before summation',negative_count_tolerance_MWh=TOL,
                    clipping_energy='Exact max(-remainder,0); numerical tolerance used for event counts only.',
                    identity='L0+A-Lobs = (O+V-Lobs)+(A-E)+max(-ro,0)+max(-rn,0)',
                    diagnostic='Unclipped Lraw may be negative; service difference is arithmetic attribution, not an alternative feasible household load.',
                    conserving='Astar=min(A,Lobs), Lstar=Lobs-Astar only on nonnegative Lobs; same G and calendar; main baseline not replaced.',
                    totals=totals.to_dict('records'),services=services,regional_threshold_detail=threshold_detail,
                    event_meshes=int(event.mesh.nunique()),event_dates=int(event.date.nunique()),
                    tests_run=False,safety_review_run=False)
    (REV / 'reports/clipping_cause_calculation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
