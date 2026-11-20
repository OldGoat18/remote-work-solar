"""P55–74: conditional A construction contrasts on the repaired mesh ledger.

Four scientifically defined cases, not a parameter scan: registered shared-use
response; no incremental shared use; full shared-use activation; and removal of
the inherited common normalization. Same people, dates, G and L0 in every case.
The contrasts are model experiments, not independently observed treatment effects.
No safety review or test suite is run; current reference arrays are read only.
"""
from pathlib import Path
import gc
import json
import sys
import duckdb
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = next(p for p in REV.parents if (p / 'pyproject.toml').exists())
D = REV / 'data'
sys.path.insert(0, str(ROOT / 'src'))
from wlpr.wfh_state import state_layer as SL
from input_mesh_alignment import components, m, INPUT_VERSION
from run_approved_opinion3 import quantiles

CASES = ['shared_already_served', 'registered', 'full_shared_activation', 'engineering_unscaled']
COMPONENTS = ['office_device', 'lighting', 'lunch', 'heating', 'cooling', 'other']


def main():
    account = json.loads((D / 'opinion3_input_account.json').read_text())
    kappa = account['anchor']['dimensionless_kappa']
    keys = pd.read_csv(REV.parent / '20261111_supplemental/data/mesh_population_scope.csv',
                       dtype={'mesh': str, 'pref': str, 'munic': str})
    state = SL.mesh_state().set_index('mesh').reindex(keys.mesh)
    persons = state.N_persons.fillna(0).to_numpy()
    vacancy = state.vacancy_margin.fillna(1).to_numpy()
    if np.any((vacancy <= 0) & (persons > 0)):
        raise ValueError('A full-activation contrast requires nonzero registered vacancy margins.')
    cal = m.dates_frame()
    q = pd.read_parquet(D / 'common48.parquet').sort_values(['date', 'slot']).copy()
    q['date'] = q.date.astype(str)
    q['ym'] = q.date.str[:6]
    q['eligible'] = q.date.isin(set(cal.loc[cal.daytype.eq('workday'), 'date'].astype(str)))
    con = duckdb.connect()
    con.execute('SET threads=2')
    templates, local_rows, component_rows = [], [], []
    profiles = {case: [] for case in CASES}
    identity = 0.0
    for ym in m.MONTHS:
        raw = components(con, ym, keys, 'S1')
        comp = raw[:, :, 1:]
        shared = comp[:, :, 1:].sum(axis=2)
        full_shared = shared / vacancy[:, None]
        # Exactly the current factors: shared-use coefficient times vacancy.
        identity = max(identity, float(np.abs(raw[:, :, 0] - comp[:, :, 0] - vacancy[:, None] * full_shared).max()))
        scenario_components = {
            'registered': comp,
            'shared_already_served': np.concatenate([comp[:, :, :1], np.zeros_like(comp[:, :, 1:])], axis=2),
            'full_shared_activation': np.concatenate([comp[:, :, :1], comp[:, :, 1:] / vacancy[:, None, None]], axis=2),
            'engineering_unscaled': comp / kappa,
        }
        dates = cal.loc[cal.ym.eq(ym)]
        ndays = int(dates.daytype.eq('workday').sum())
        totals = {case: v.sum(axis=2) for case, v in scenario_components.items()}
        for case in CASES:
            v = scenario_components[case]
            A = totals[case]
            device = v[:, :, 0]
            templates.extend(dict(case=case, ym=ym, slot=t+17,
                                  A_template_MWh=float(A[:, t].sum()),
                                  device_template_MWh=float(device[:, t].sum())) for t in range(20))
            for ci, label in enumerate(COMPONENTS):
                component_rows.append(dict(case=case, ym=ym, component=label,
                    eligible_days=ndays, annual_MWh=float(v[:, :, ci].sum()*ndays),
                    participant_days=float(persons.sum()*ndays)))
            good = persons > 0
            profiles[case].append((A[good].sum(axis=1)*1000/persons[good], persons[good]*ndays))
        with np.load(REV / f'tmp/inputs_{ym}.npz') as z:
            G, L = z['G'], z['L']
        for di, day in enumerate(dates.itertuples(index=False)):
            if day.daytype != 'workday':
                local_rows.extend(dict(case=case, date=str(day.date), local_MWh=0.) for case in CASES)
                continue
            surplus = np.maximum(G[:, :, di] - L[:, :, di], 0)
            for case in CASES:
                local_rows.append(dict(case=case, date=str(day.date),
                                       local_MWh=float(np.minimum(surplus, totals[case]).sum())))
        print('A state counterfactual', ym, flush=True)
        del raw, comp, scenario_components, totals, G, L, surplus
        gc.collect()
    con.close()
    template = pd.DataFrame(templates)
    local = pd.DataFrame(local_rows)
    components_out = pd.DataFrame(component_rows)
    components_out.to_csv(D / 'opinion3_counterfactual_components.csv', index=False)
    intervals, summaries = [], []
    for case in CASES:
        f = q.merge(template.loc[template.case.eq(case)].drop(columns='case'), on=['ym', 'slot'], how='left', validate='many_to_one')
        f['case'] = case
        f['A_MWh'] = f.A_template_MWh.fillna(0) * f.eligible
        f['office_device_reduction_MWh'] = f.device_template_MWh.fillna(0) * f.eligible
        f['net_electricity_change_MWh'] = f.A_MWh - f.office_device_reduction_MWh
        f['positive_pool_match_MWh'] = np.minimum(f.positive_pool, f.A_MWh)
        f['regional_residential_absorption_MWh'] = np.minimum(np.maximum(f.G-f.L0, 0), f.A_MWh)
        f['paired_region_absorption_MWh'] = np.minimum(f.G, f.L0+f.A_MWh) - np.minimum(f.G, f.L0+f.office_device_reduction_MWh)
        f['paired_external_supply_change_MWh'] = f.net_electricity_change_MWh - f.paired_region_absorption_MWh
        columns = ['A_MWh', 'office_device_reduction_MWh', 'net_electricity_change_MWh',
                   'positive_pool_match_MWh', 'regional_residential_absorption_MWh',
                   'paired_region_absorption_MWh', 'paired_external_supply_change_MWh']
        intervals.append(f[['case','date','slot','eligible']+columns])
        daily = f.groupby('date')[columns].sum().merge(local.loc[local.case.eq(case), ['date','local_MWh']], on='date')
        x = np.concatenate([a for a, _ in profiles[case]])
        w = np.concatenate([b for _, b in profiles[case]])
        qs = quantiles(x, w)
        for calendar, mask in [('annual_366', np.ones(len(f), bool)), ('eligible_243', f.eligible.to_numpy())]:
            totals = f.loc[mask, columns].sum().to_dict()
            totals.update(case=case, calendar=calendar, dates=int(f.loc[mask].date.nunique()),
                          complete_meshes=len(keys), participant_days=float(w.sum()),
                          local_MWh=float(daily.local_MWh.sum()),
                          kWh_per_person_day=float(totals['A_MWh']*1000/w.sum()),
                          p10_kWh=qs['0.1'], median_kWh=qs['0.5'], p90_kWh=qs['0.9'])
            summaries.append(totals)
    pd.concat(intervals, ignore_index=True).to_parquet(D / 'opinion3_counterfactual_intervals.parquet', index=False)
    local.to_csv(D / 'opinion3_counterfactual_local_daily.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(D / 'opinion3_counterfactual_summary.csv', index=False)

    # Count-based bounds only. They do not identify employer permission or willingness.
    caps = []
    for label, mask in [('complete_domain', np.ones(len(keys), bool)),
                        ('municipal_domain', keys.munic.notna().to_numpy())]:
        workers = float(state.workers.fillna(0).to_numpy()[mask].sum())
        teleworkers = float(state.teleworkers.fillna(0).to_numpy()[mask].sum())
        baseline = float(persons[mask].sum())
        caps.append(dict(scope=label, workers=workers, registered_teleworker_equivalents=teleworkers,
                         baseline_daily_equivalents=baseline, baseline_person_days=baseline*243,
                         registered_person_days_at_frequency_one=teleworkers*243,
                         frequency_only_additional_person_days=(teleworkers-baseline)*243,
                         all_worker_count_person_day_bound=workers*243,
                         model_additional_scale_cap=1., implied_frequency_at_double_baseline=2*SL.I0,
                         maximum_scale_at_same_teleworker_pool=1/SL.I0,
                         fixed_budget_implied_mean_frequency=1.25*SL.I0))
    cap_frame = pd.DataFrame(caps)
    cap_frame.to_csv(D / 'opinion3_participation_count_bounds.csv', index=False)
    report = dict(date='2026-11-13', opinion='P55–74', status='calculations_completed',
        input_version=INPUT_VERSION, units='MWh per half-hour; sum slots without another 0.5 factor',
        source_scope=str((REV.parent/'20261111_supplemental/data/mesh_population_scope.csv').relative_to(ROOT)),
        scientific_question='How do registered household shared-use assumptions and the inherited normalization determine A and its absorption?',
        definitions={
            'A_registered':'D + v*C, where v=vacancy_margin and C is the existing shared-use template before its vacancy multiplier',
            'shared_already_served':'A=D; all shared end uses have zero additional energy for the participant task',
            'full_shared_activation':'A=D+C; all registered shared uses are newly activated, retaining H_wfh and household sharing',
            'engineering_unscaled':'A=(D+v*C)/kappa; paired workplace devices are D/kappa in this case',
            'residential_states':'L_home_onsite=L0; L_home_remote=L0+A_case',
            'paired_states':'L_region_onsite=L0+D_case; L_region_remote=L0+A_case in the residential-plus-task account',
            'interpretation':'Conditional marginal-response contrasts at fixed L0, not alternative observed populations or a resimulated building thermal baseline',
            'fixed_office_background':'B=0 defines the tracked account; unchanged B>=0 gives a paired gain between zero and the reported value',
        },
        unchanged=['G','L0','366-date calendar','243 eligible dates','participants','frequency','source weather and housing','unique-mesh repair','current baseline A'],
        no_new_data_or_probability_distribution=True,
        max_reference_factorization_residual_MWh=identity,
        prior_mesh_repair='All 98 recovered mesh-month templates are included; no repeat repair or baseline change.',
        calibration=dict(kappa=kappa, target_MWh=account['anchor']['target_representative_templates_MWh'],
                         unscaled_templates_MWh=account['anchor']['unscaled_MWh'],
                         original_target_derivation_recovered=False,
                         existing_related_material=['outputs/scorecards/wfh_anchor_scorecard.csv',
                            'data/frozen/normalized_inputs/wfh_survey_anchored_pref.parquet',
                            'data/derived/products/wfh_survey_anchored_component_product.parquet'],
                         role='These products concern fitted component allocation or released rates; the unscaled contrast quantifies normalization dependence.'),
        not_run=['independent A observational validation','unidentified office HVAC savings','evening reheat with arbitrary fraction','new residual sampling','parameter scan','tests','safety review'],
        remaining=['Independent A treatment observations','Original derivation of 12,534.9 MWh target','Empirical feasible participation and employer/willingness constraints'],
        summary=summary.to_dict('records'),participation_bounds=caps)
    record_path=REV/'reports/opinion3_counterfactual.json'
    if record_path.exists():
        prior=json.loads(record_path.read_text())
        if 'manuscript_changes' in prior:report['manuscript_changes']=prior['manuscript_changes']
    record_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(summary.loc[summary.calendar.eq('annual_366'),['case','A_MWh','kWh_per_person_day','local_MWh','regional_residential_absorption_MWh','paired_region_absorption_MWh']].to_string(index=False), flush=True)
    print(cap_frame.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
