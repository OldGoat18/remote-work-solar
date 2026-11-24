"""Approved P29–52: paired work-device relocation using registered components.

This is a conditional model experiment, not an office-building measurement.
The same participant device task takes place inside the four-prefecture region,
at the same half-hours, at work before and at home afterwards. Other office loads
are fixed. Their level is unidentified: numerical absorption is for the explicitly
tracked residential-plus-participant-device account (fixed other office load=0).
Its additional absorption bounds the result with any nonnegative fixed office
load. No office geography, PV, occupancy-response coefficients or errors are made
up. Only the registered base case is evaluated; no new parameter scan or tests.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import gc
import json
import sys
import duckdb
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists())
BASE = (next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'))
OUT = BASE / _asset_location('asset_0002')
sys.path.insert(0, str(ROOT / 'tools'))
from input_mesh_alignment import components, m


def main():
    target = OUT / 'opinion2_relocation_summary.csv'
    # The current approved copy is overwritten after the input-key repair;
    # its previous numerical summary is retained in the repair change record.
    keys = pd.read_csv(BASE.parent / _asset_location('asset_0012'),
                       dtype={'mesh': str, 'pref': str, 'munic': str})
    cal = m.dates_frame()
    common = pd.read_parquet(OUT / 'common48.parquet').sort_values(['date', 'slot'])
    common['date'] = common.date.astype(str)
    common['ym'] = common.date.str[:6]
    common['eligible'] = common.date.isin(set(cal.loc[cal.daytype.eq('workday'), 'date'].astype(str)))
    con = duckdb.connect()
    con.execute('SET threads=4')
    slotrows, localrows = [], []
    for ym in m.MONTHS:
        comp = components(con, ym, keys, 'S1')
        A, D = comp[:, :, 0], comp[:, :, 1]
        if np.min(A-D) < -1e-8:
            raise ValueError('Device component exceeds total registered residential increment.')
        slotrows.extend(dict(ym=ym, slot=t+17, device_template_MWh=float(v))
                        for t, v in enumerate(D.sum(axis=0)))
        dates = cal.loc[cal.ym.eq(ym)]
        with np.load(BASE / f"{_asset_location('asset_0017')}{ym}.npz") as z:
            G, L = z['G'], z['L']
        for di, row in enumerate(dates.itertuples(index=False)):
            if row.daytype != 'workday':
                continue
            surplus = np.maximum(G[:, :, di]-L[:, :, di], 0)
            device_local = np.minimum(surplus, D).sum()
            full_local = np.minimum(surplus, A).sum()
            localrows.append(dict(date=str(row.date), ym=ym,
                                  device_relocation_home_local_MWh=float(device_local),
                                  all_enduses_home_local_MWh=float(full_local),
                                  remaining_enduses_after_device_local_MWh=float(full_local-device_local)))
        print('paired device-transfer account', ym, flush=True)
        del comp, A, D, G, L, surplus
        gc.collect()
    con.close()
    q = common.merge(pd.DataFrame(slotrows), on=['ym', 'slot'], how='left', validate='many_to_one')
    q['device_MWh'] = q.device_template_MWh.fillna(0) * q.eligible
    q['residential_increment_MWh'] = q.template_A * q.eligible
    q['other_residential_MWh'] = q.residential_increment_MWh-q.device_MWh
    q['onsite_tracked_load_MWh'] = q.L0+q.device_MWh
    q['remote_device_only_tracked_load_MWh'] = q.L0+q.device_MWh
    q['remote_all_enduses_tracked_load_MWh'] = q.L0+q.residential_increment_MWh
    q['residential_reference_absorption_MWh'] = np.minimum(q.G, q.L0)
    q['onsite_tracked_absorption_MWh'] = np.minimum(q.G, q.onsite_tracked_load_MWh)
    q['remote_device_only_absorption_MWh'] = np.minimum(q.G, q.remote_device_only_tracked_load_MWh)
    q['remote_all_enduses_absorption_MWh'] = np.minimum(q.G, q.remote_all_enduses_tracked_load_MWh)
    q['residential_addition_absorption_gain_MWh'] = q.remote_all_enduses_absorption_MWh-q.residential_reference_absorption_MWh
    q['office_devices_baseline_absorption_MWh'] = q.onsite_tracked_absorption_MWh-q.residential_reference_absorption_MWh
    q['paired_device_only_absorption_gain_MWh'] = q.remote_device_only_absorption_MWh-q.onsite_tracked_absorption_MWh
    q['paired_all_enduses_absorption_gain_MWh'] = q.remote_all_enduses_absorption_MWh-q.onsite_tracked_absorption_MWh
    q['onsite_tracked_external_supply_MWh'] = np.maximum(q.onsite_tracked_load_MWh-q.G, 0)
    q['remote_all_enduses_external_supply_MWh'] = np.maximum(q.remote_all_enduses_tracked_load_MWh-q.G, 0)
    q['paired_external_supply_change_MWh'] = q.remote_all_enduses_external_supply_MWh-q.onsite_tracked_external_supply_MWh
    q['onsite_tracked_surplus_MWh'] = np.maximum(q.G-q.onsite_tracked_load_MWh, 0)
    q['remote_all_enduses_surplus_MWh'] = np.maximum(q.G-q.remote_all_enduses_tracked_load_MWh, 0)
    q = q.drop(columns=['device_template_MWh'])
    q.to_parquet(OUT/'opinion2_relocation_intervals.parquet', index=False)
    local = pd.DataFrame(localrows)
    local.to_csv(OUT/'opinion2_relocation_local_daily.csv', index=False)
    cols = [c for c in q if c.endswith('_MWh')]
    monthly = q.groupby('ym')[cols].sum().join(local.groupby('ym').sum(numeric_only=True))
    monthly.to_csv(OUT/'opinion2_relocation_monthly.csv')
    persons = float(json.loads((OUT/'input_provenance.json').read_text())['participant_equivalents'])
    rows = []
    for calendar, use in [('annual_366', np.ones(len(q), bool)), ('eligible_243', q.eligible.to_numpy())]:
        g = q.loc[use]
        totals = g[cols].sum().to_dict()
        totals.update(local.sum(numeric_only=True).to_dict())
        totals.update(calendar=calendar, dates=int(g.date.nunique()), complete_meshes=len(keys),
                      participant_days=persons*int(cal.daytype.eq('workday').sum()),
                      assumption='identical device task, same time, both work locations inside domain; fixed other office loads',
                      absorption_scope='residential-plus-participant-device tracked account; other fixed office load excluded')
        totals['net_electricity_change_MWh'] = totals['other_residential_MWh']
        totals['additional_office_saving_for_energy_neutrality_MWh'] = totals['other_residential_MWh']
        totals['additional_office_saving_for_energy_neutrality_kWh_per_person_day'] = totals['other_residential_MWh']*1000/totals['participant_days']
        totals['device_kWh_per_person_day'] = totals['device_MWh']*1000/totals['participant_days']
        totals['device_share_of_residential_increment'] = totals['device_MWh']/totals['residential_increment_MWh']
        totals['regional_absorption_gain_lower_bound_with_fixed_office_load_MWh'] = 0.0
        totals['regional_absorption_gain_upper_bound_with_fixed_office_load_MWh'] = totals['paired_all_enduses_absorption_gain_MWh']
        rows.append(totals)
    summary = pd.DataFrame(rows)
    summary.to_csv(target, index=False)
    # This is a numerical identity recorded in the scientific result, not a test suite.
    identities = dict(
        max_interval_energy_balance_residual_MWh=float(np.abs(q.other_residential_MWh-q.paired_all_enduses_absorption_gain_MWh-q.paired_external_supply_change_MWh).max()),
        max_interval_pure_relocation_load_difference_MWh=float(np.abs(q.remote_device_only_tracked_load_MWh-q.onsite_tracked_load_MWh).max()),
        max_interval_absorption_decomposition_residual_MWh=float(np.abs(q.residential_addition_absorption_gain_MWh-q.office_devices_baseline_absorption_MWh-q.paired_all_enduses_absorption_gain_MWh).max()))
    definitions = dict(
        opinion='P29–52', date='2026-11-12', status='conditional_device_relocation_experiment_completed',
        units='MWh per half-hour; MWh totals; convert to GWh by dividing by 1000',
        source_components='registered S1 month-slot files selected by tools/run_halfhour_spatial.py:components',
        source_scope=_asset_location('asset_0031'),
        source_reference='configured asset and configured asset; existing complete-mesh base case',
        assumptions=[
            'Same participant-device electricity D at work and at home, in the same intervals; devices stop at work when the participant works at home.',
            'The paired participant tasks at both locations are inside the four-prefecture accounting domain; no workplace mesh destinations are invented.',
            'Other office lighting/HVAC/server loads B are fixed, with unknown nonnegative level; their response is zero in this scenario.',
            'The full residential increment A includes D; H=A-D is the additional residential non-device energy.',
            'Same G, L0, participant-day budget and 243-day gate; all 366 baseline dates remain.',
            'Numerical regional absorption includes residential reference and participant devices; fixed office load B=0 specifies this partial account, not an office-building baseline estimate.',
            'For B>=0 unchanged, total energy change is sum(H), while additional regional absorption lies between zero and the partial-account value; no probability distribution is attached.',
            'Local values are residential-location absorption gains only, not a summed office-and-home local balance.',
            'No thermal simulation, commuting/emissions, grid curtailment or new scheduling experiment is added.'
        ],
        equations={
            'devices_only':'L_on=L0+B+D; L_remote_devices=L0+B+D; delta_E=delta_U_region=0',
            'full_enduses':'L_on=L0+B+D; L_remote=L0+B+A; H=A-D',
            'net_energy':'delta_E=sum(A-D), invariant to unchanged B',
            'regional_absorption':'sum(min(G,L0+B+A)-min(G,L0+B+D))',
            'fixed_office_bound':'0 <= delta_U(B) <= delta_U(0) for A>=D and B>=0',
            'local_device':'sum_i,d,t min(max(G_i-L0_i,0),D_i)',
            'local_full':'sum_i,d,t min(max(G_i-L0_i,0),A_i)',
            'local_sequence':'full_local = device_local + remaining_after_device_local; order-specific, not unique causal end-use attribution',
            'external_supply':'delta_Q=delta_E-delta_U; positive means more required supply from outside the tracked PV account'
        },
        unsolved=['Office-building shared-load response and within-region workplace linkage are not independently identified.',
                  'A full office baseline and office PV are not supplied; complete city absorption is not point-estimated.',
                  'No observed office saving, emissions reduction or avoided curtailment is claimed.'],
        numerical_identities=identities)
    (OUT/'opinion2_relocation_definitions.json').write_text(json.dumps(definitions, ensure_ascii=False, indent=2)+'\n')
    print(summary[['calendar','device_MWh','net_electricity_change_MWh','device_relocation_home_local_MWh',
                   'paired_all_enduses_absorption_gain_MWh','paired_external_supply_change_MWh',
                   'additional_office_saving_for_energy_neutrality_kWh_per_person_day']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
