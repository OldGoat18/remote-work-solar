"""A-construction preview: fixed-policy, common-target storage, equal-energy timing.

Read existing templates and repaired inputs only. Write a separate downstream
result set; never update registered scientific results or formal figures.
Four deterministic assumptions are not observational replicates or a CI.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import gc
import hashlib
import json
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from numba import set_num_threads

REV = (next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'))
ROOT = next(p for p in REV.parents if (p / 'pyproject.toml').exists())
D = REV / _asset_location('asset_0002')
OUT = D / _asset_location('asset_0030')
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'src')]
from input_mesh_alignment import components, m, INPUT_VERSION
from run_technical import supp
from summarize_technical import minimum_storage
from lunch_constraints import prepare, candidate_lunch, evaluate_surplus, VERSION
from wlpr.wfh_state import state_layer as SL

CASES = ['shared_already_served', 'registered', 'full_shared_activation', 'engineering_unscaled']
RULES = ['baseline', 'U', 'TE', 'MG']
IDS = [0, 25, 30, 74]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    set_num_threads(4)
    keys = pd.read_csv(REV.parent / _asset_location('asset_0012'),
                       dtype={'mesh': str, 'pref': str, 'munic': str})
    municipalities = pd.read_csv(REV.parent / _asset_location('asset_0025'),
                                  dtype={'munic': str})
    mi = keys.munic.map(dict(zip(municipalities.munic, range(len(municipalities))))).fillna(-1).to_numpy(np.int64)
    state = SL.mesh_state().set_index('mesh').reindex(keys.mesh)
    vacancy = state.vacancy_margin.fillna(1).to_numpy()
    assert np.all(vacancy > 0)
    kappa = json.loads((D / 'opinion3_input_account.json').read_text())['anchor']['dimensionless_kappa']
    allocations = pd.read_parquet(D / 'calendar_scope_policy_allocations.parquet')
    multipliers = np.stack([allocations[allocations.scenario_id.eq(sid)].set_index('munic')
                           .reindex(municipalities.munic).scale.to_numpy() for sid in IDS])
    assert np.isfinite(multipliers).all() and np.max(multipliers) <= 2 + 1e-10
    policy_meta = pd.read_csv(D / 'calendar_scope_policy_exact.csv').set_index('scenario_id').loc[IDS]
    assert np.ptp(policy_meta.iloc[1:].additional_person_days) < .001
    assert np.ptp(policy_meta.iloc[1:].total_expenditure_JPY) < .1
    cal = m.dates_frame()
    con = duckdb.connect()
    con.execute('SET threads=2')
    policy_rows, schedule_rows, constraint_rows, monthly_rows = [], [], [], []
    input_records = []
    for ym in m.MONTHS:
        month_start = time.time()
        dates = cal[cal.ym.eq(ym) & cal.daytype.eq('workday')].date.astype(str).to_numpy()
        raw = components(con, ym, keys, 'S1')
        comp = raw[:, :, 1:]
        total = {
            'shared_already_served': comp[:, :, 0],
            'registered': comp.sum(axis=2),
            'full_shared_activation': comp[:, :, 0] + comp[:, :, 1:].sum(axis=2) / vacancy[:, None],
            'engineering_unscaled': comp.sum(axis=2) / kappa,
        }
        lunch = {
            'shared_already_served': np.zeros_like(raw[:, :, 3]),
            'registered': raw[:, :, 3],
            'full_shared_activation': raw[:, :, 3] / vacancy[:, None],
            'engineering_unscaled': raw[:, :, 3] / kappa,
        }
        cache = REV / f"{_asset_location('asset_0029')}{ym}.npz"
        with np.load(cache) as z:
            surplus = z['D']
            registered_B = z['B']
            assert np.array_equal(z['mi'], mi)
        assert surplus.shape == (len(keys), 20, len(dates))
        assert np.max(np.abs(total['registered'] - registered_B)) < 1e-12
        sorted_surplus = np.sort(surplus, axis=2)
        prefix = np.concatenate([np.zeros((*surplus.shape[:2], 1)), np.cumsum(sorted_surplus, axis=2)], axis=2)
        lunch_surplus = np.ascontiguousarray(surplus[:, 5:11, :])
        season = m.SEASON_BY_MONTH[int(ym[4:])]
        for case in CASES:
            B = np.ascontiguousarray(total[case])
            C = np.ascontiguousarray(lunch[case])
            scores = supp.policy_scores(sorted_surplus, prefix, B, multipliers, mi, len(municipalities))
            for j, mu in enumerate(municipalities.munic):
                for k, rule in enumerate(RULES):
                    policy_rows.append(dict(case=case, ym=ym, munic=mu, rule=rule,
                                            service_MWh=float(scores[k, j]), scale=float(multipliers[k, j])))
            schedules = supp.local_schedules(surplus, B)
            candidates, valid, moved, retained, fixed = prepare(B, C)
            gains, counts, movement = evaluate_surplus(lunch_surplus, candidates, valid, moved, retained, fixed)
            profiles, _, _, _ = candidate_lunch(C)
            energy_error = float(np.max(np.abs(profiles.sum(axis=2) - C[:, 5:11].sum(axis=1)[:, None])))
            peak = C[:, 5:11].max(axis=1)
            peak_excess = float(np.maximum(profiles.max(axis=2) - peak[:, None], 0)[valid].max())
            assert energy_error < 1e-12 and peak_excess < 1e-12
            assert np.all(gains[:, 0] >= -1e-10)
            if case == 'shared_already_served':
                assert np.max(np.abs(gains)) == 0
            for k, date in enumerate(dates):
                anchor = schedules[k, 0]
                schedule_rows.append(dict(case=case, ym=ym, date=date, season=season,
                    anchored_MWh=float(anchor), permutation_MWh=float(schedules[k, 1]),
                    ideal_MWh=float(schedules[k, 2]), constrained_MWh=float(anchor + gains[k, 0]),
                    fixed_MWh=float(anchor + gains[k, 1]), constrained_gain_MWh=float(gains[k, 0]),
                    fixed_gain_MWh=float(gains[k, 1])))
            constraint_rows.append(dict(case=case, ym=ym, source_lunch_MWh=float(C.sum()*len(dates)),
                max_energy_error_MWh=energy_error, max_feasible_peak_excess_MWh=peak_excess,
                lunch_outside_window_MWh=float(C[:, :5].sum()+C[:, 11:].sum()),
                fixed_identity_fallback_meshes=int((fixed == 2).sum()),
                rejected_mesh_shift_pairs=int((~valid).sum())))
            monthly_rows.append(dict(case=case, ym=ym, eligible_dates=len(dates),
                A_MWh=float(B.sum()*len(dates)), local_MWh=float(schedules[:, 0].sum()),
                municipal_baseline_MWh=float(scores[0].sum())))
        input_records.append(dict(path=str(cache.relative_to(ROOT)), sha256=sha(cache),
                                  shape=list(surplus.shape)))
        # Persist completed months so an interruption does not erase evidence.
        pd.DataFrame(policy_rows).to_parquet(OUT / 'policy_monthly_municipal.parquet', index=False)
        pd.DataFrame(schedule_rows).to_csv(OUT / 'scheduling_daily.csv', index=False)
        pd.DataFrame(constraint_rows).to_csv(OUT / 'scheduling_constraints.csv', index=False)
        pd.DataFrame(monthly_rows).to_csv(OUT / 'input_monthly.csv', index=False)
        print(f'{ym}: all four constructions, policy and schedules ({time.time()-month_start:.1f}s)', flush=True)
        del raw, comp, total, lunch, surplus, sorted_surplus, prefix, lunch_surplus, B, C, profiles
        gc.collect()
    con.close()
    policy = pd.DataFrame(policy_rows).groupby(['case', 'rule']).service_MWh.sum().unstack().reindex(CASES)
    policy_out = []
    for case in CASES:
        baseline = float(policy.loc[case, 'baseline'])
        uniform_increment = float(policy.loc[case, 'U'] - baseline)
        for rule, sid in zip(RULES[1:], IDS[1:]):
            increment = float(policy.loc[case, rule] - baseline)
            meta = policy_meta.loc[sid]
            cost = float(meta.total_expenditure_JPY)
            policy_out.append(dict(case=case, rule=rule, allocation_source_scenario_id=sid,
                baseline_MWh=baseline, total_service_MWh=float(policy.loc[case, rule]),
                additional_service_MWh=increment, additional_person_days=float(meta.additional_person_days),
                total_cost_JPY=cost, full_cost_JPY_per_kWh=cost/(increment*1000),
                advantage_over_uniform_pct=100*(increment/uniform_increment-1),
                fixed_registered_allocation=True))
    pd.DataFrame(policy_out).to_csv(OUT / 'policy_summary.csv', index=False)
    schedule = pd.DataFrame(schedule_rows)
    schedule_cols = ['anchored_MWh', 'permutation_MWh', 'ideal_MWh', 'constrained_MWh',
                     'fixed_MWh', 'constrained_gain_MWh', 'fixed_gain_MWh']
    schedule_summary = schedule.groupby('case')[schedule_cols].sum().reindex(CASES)
    schedule_summary.to_csv(OUT / 'scheduling_summary.csv')
    schedule.groupby(['case', 'season'])[schedule_cols].sum().to_csv(OUT / 'scheduling_seasonal.csv')
    reference_schedule = pd.read_csv(D / 'schedule_summary.csv').query("case=='L+0_P+0'").set_index('schedule').service_MWh
    reference_checks = []
    for old, new in [('anchored', 'anchored_MWh'), ('permutation_expectation', 'permutation_MWh'),
                     ('ideal_sorting', 'ideal_MWh'), ('lunch_constrained', 'constrained_MWh')]:
        error = float(schedule_summary.loc['registered', new] - reference_schedule[old])
        assert abs(error) < 1e-5, (old, error)
        reference_checks.append(dict(metric=new, difference_MWh=error))
    old_fixed = pd.read_csv(D / 'fixed_lunch_summary.csv').set_index('case').loc['L+0_P+0', 'gain_MWh']
    assert abs(schedule_summary.loc['registered', 'fixed_gain_MWh'] - old_fixed) < 1e-5
    for row in policy_out:
        if row['case'] == 'registered':
            error = row['additional_service_MWh'] - policy_meta.loc[row['allocation_source_scenario_id'], 'additional_service_MWh']
            assert abs(error) < 1e-5, error
            reference_checks.append(dict(metric='policy_' + row['rule'], difference_MWh=error))
    annual = pd.DataFrame(monthly_rows).groupby('case')[['A_MWh', 'local_MWh']].sum()
    old_cases = pd.read_csv(D / 'opinion3_counterfactual_summary.csv').query("calendar=='annual_366'").set_index('case')
    for case in CASES:
        for field in ['A_MWh', 'local_MWh']:
            error = float(annual.loc[case, field] - old_cases.loc[case, field])
            assert abs(error) < 1e-5, (case, field, error)
            reference_checks.append(dict(metric=case+'_'+field, difference_MWh=error))

    # Full chronological LP, retaining the registered objectives and constraints.
    common = pd.read_parquet(D / 'common48.parquet').sort_values(['date', 'slot']).reset_index(drop=True)
    source_a = pd.read_parquet(D / 'opinion3_counterfactual_intervals.parquet')
    idx = pd.MultiIndex.from_frame(common[['date', 'slot']])
    G, L = common.G.to_numpy(), common.L0.to_numpy()
    slots = common.slot.to_numpy()
    ch, dh = (slots >= 12) & (slots < 30), slots >= 30
    storage_rows, trajectories, diagnostics = [], [], []
    for case in ['no_A'] + CASES:
        A = np.zeros(len(common)) if case == 'no_A' else source_a[source_a.case.eq(case)].set_index(['date', 'slot']).reindex(idx).A_MWh.to_numpy()
        assert np.isfinite(A).all() and np.all(A >= 0)
        direct = float(np.minimum(np.maximum(G-L, 0), A).sum())
        net = G-L-A
        Q, H = np.maximum(net, 0)*ch, np.maximum(-net, 0)*dh
        for objective in ['combined_delivery_100GWh', 'evening_delivery_100GWh']:
            target = max(0., 100000-direct) if objective.startswith('combined') else 100000.
            answer, traj = minimum_storage(Q, H, target, 'delivery')
            answer.update(case=case, objective=objective, total_target_MWh=100000., direct_MWh=direct,
                          required_storage_delivery_MWh=target, available_charge_MWh=float(Q.sum()))
            storage_rows.append(answer)
            if traj is not None:
                capacity, power = answer['capacity_MWh'], answer['power_MW']
                checks = dict(case=case, objective=objective,
                    max_balance_error_MWh=float(np.max(np.abs(traj.soc_end_MWh-traj.soc_start_MWh-
                        np.sqrt(.9)*traj.charge_MWh+traj.delivery_MWh/np.sqrt(.9)))),
                    charge_bound_excess_MWh=float(np.maximum(traj.charge_MWh-np.minimum(Q, power*.5), 0).max()),
                    discharge_bound_excess_MWh=float(np.maximum(traj.delivery_MWh-np.minimum(H, power*.5), 0).max()),
                    capacity_bound_excess_MWh=float(np.maximum(traj.soc_end_MWh-capacity, 0).max()),
                    target_shortfall_MWh=float(max(0, target-traj.delivery_MWh.sum())),
                    endpoint_max_abs_MWh=float(max(abs(traj.soc_start_MWh.iloc[0]), abs(traj.soc_end_MWh.iloc[-1]))))
                assert max(v for k,v in checks.items() if k not in ['case','objective']) < 1e-5, checks
                diagnostics.append(checks)
                traj['date'], traj['slot'] = common.date, slots
                traj['case'], traj['objective'] = case, objective
                trajectories.append(traj)
            print('storage', case, objective, answer.get('capacity_MWh'), 'feasible', answer['feasible'], flush=True)
            if case == 'no_A':
                # Same Q, H and target for the two baseline objectives; retain two labels.
                copy = dict(answer, objective='evening_delivery_100GWh')
                storage_rows.append(copy)
                if traj is not None:
                    copy_traj = traj.copy(); copy_traj['objective'] = 'evening_delivery_100GWh'; trajectories.append(copy_traj)
                break
    storage = pd.DataFrame(storage_rows)
    baseline = storage[storage.case.eq('no_A')].set_index('objective').capacity_MWh
    storage['capacity_change_MWh'] = storage.capacity_MWh - storage.objective.map(baseline)
    storage.to_csv(OUT / 'storage_summary.csv', index=False)
    pd.concat(trajectories, ignore_index=True).to_parquet(OUT / 'storage_trajectories.parquet', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / 'storage_constraints.csv', index=False)
    original_storage = pd.read_csv(D / 'storage_common_target.csv').query("case=='L+0_P+0'")
    for case, a in [('no_A', 0), ('registered', 1)]:
        for obj in baseline.index:
            new = storage[storage.case.eq(case) & storage.objective.eq(obj)].iloc[0]
            old = original_storage[original_storage.participation.eq(a) & original_storage.objective.eq(obj)].iloc[0]
            error = float(new.capacity_MWh-old.capacity_MWh)
            assert abs(error) < 1e-4, (case, obj, error)
            reference_checks.append(dict(metric='capacity_'+case+'_'+obj, difference_MWh=error))
    paths = [D/'common48.parquet', D/'opinion3_counterfactual_intervals.parquet',
             D/'opinion3_counterfactual_summary.csv', D/'opinion3_input_account.json',
             D/'calendar_scope_policy_allocations.parquet', D/'calendar_scope_policy_exact.csv',
             Path(__file__), (Path(__file__).resolve().parent / 'lunch_constraints.py'), (Path(__file__).resolve().parent / 'summarize_technical.py'),
             (Path(__file__).resolve().parent / 'input_mesh_alignment.py'), (ROOT / 'experiments/supplemental/scripts/run_supplemental.py')]
    manifest = dict(date='2026-11-16', status='computed_for_preview', cases=CASES,
        input_version=INPUT_VERSION, meshes=len(keys), municipal_meshes=int((mi >= 0).sum()),
        municipalities=len(municipalities), dates=366, eligible_dates=243, kappa=kappa,
        policy={'allocation_scenario_ids':IDS[1:], 'fixed_across_constructions':True,
                'reoptimized':False, 'own_case_baseline':True, 'fraction':.25,
                'payment_JPY_per_added_day':1000, 'administration_fraction':.05},
        scheduling={'constraint_version':VERSION, 'same_energy_within_each_case':True,
                    'same_people_not_same_total_energy_between_cases':True,
                    'all_block_bounds':'16 half-hours, 09:00–16:30, other slots fixed',
                    'lunch_window':'11:00–14:00, maximum 50% movement, original peak, retain boundaries'},
        storage={'solver':'same full chronological LP as registered results', 'roundtrip_efficiency':.9,
                 'duration_h':4, 'charge_window':'06:00–15:00', 'discharge_window':'15:00–24:00',
                 'endpoints':'empty/empty', 'targets_MWh':100000, 'total_configurations':len(storage),
                 'feasible_configurations':int(storage.feasible.sum())},
        uncertainty='Deterministic constructions, no CI, significance test or empirical response fit.',
        source_files=input_records+[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],
        registered_reference_checks=reference_checks, elapsed_seconds=time.time()-started,
        not_run=['Alternative-reference 214-date extension', 'New PV residual draws',
                 'A-specific policy reoptimization', 'Formal figure publication'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(policy_out).to_string(index=False),flush=True)
    print(schedule_summary.to_string(),flush=True)
    print(storage[['case','objective','feasible','capacity_MWh','capacity_change_MWh']].to_string(index=False),flush=True)
    print('COMPLETE',time.time()-started,flush=True)


if __name__ == '__main__':
    run()
