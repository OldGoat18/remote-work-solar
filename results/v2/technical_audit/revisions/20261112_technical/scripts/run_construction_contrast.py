"""P232–239: conserved spatial reconstruction, service response and shared controls.

Reuse the released calendar-corrected mesh/season service integrations. Read only
the cached A templates to retain the prefecture–month–slot conservation account.
No other experiment, prediction fit, bootstrap or parameter sweep is executed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = next(p for p in REV.parents if (p / 'pyproject.toml').exists())
D = REV / 'data'
SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
CONTROLS = ['pop', 'working_pop', 'current_cap_kw', 'pot_cap_kw_B_IC', 'midday_load_MWh']


def main():
    s = pd.read_parquet(D / 'mesh_season.parquet')
    keys = pd.read_csv(REV.parent / '20261111_supplemental/data/mesh_population_scope.csv',
                       dtype={'mesh': str, 'pref': str, 'munic': str})
    calendar = pd.read_csv(D / 'metrics_all.csv', dtype={'date': str})
    calendar = calendar[calendar['case'].eq('L+0_P+0')].copy()
    calendar['ym'] = calendar.date.str[:6]
    counts = calendar[calendar.day_class.eq('eligible')].groupby('ym').size()

    # The same positive worker weights as spatial_revision.py; restore all 48
    # slots explicitly. A has zero support outside slots 17–36 in these inputs.
    conservation = []
    month_totals = {}
    for ym in sorted(calendar.ym.unique()):
        with np.load(REV / f'tmp/inputs_{ym}.npz') as cache:
            b = cache['B']
        if b.shape != (len(keys), 20):
            raise ValueError('Cached A dimensions differ from the registered mesh order.')
        alt = np.zeros_like(b)
        for pref, indices in keys.groupby('pref', sort=True).groups.items():
            indices = np.asarray(indices)
            w = keys.loc[indices, 'workers'].to_numpy(float)
            total = b[indices].sum(axis=0)
            if w.sum() <= 0:
                raise ValueError(f'No worker denominator for prefecture {pref}.')
            alt[indices] = w[:, None] / w.sum() * total
            actual = alt[indices].sum(axis=0)
            for slot in range(48):
                a0 = float(total[slot - 17]) if 17 <= slot <= 36 else 0.0
                a1 = float(actual[slot - 17]) if 17 <= slot <= 36 else 0.0
                conservation.append(dict(ym=ym, pref=pref, slot=slot,
                    eligible_dates=int(counts[ym]), reference_MWh_per_eligible_interval=a0,
                    worker_MWh_per_eligible_interval=a1, difference_MWh=a1-a0))
        month_totals[ym] = (np.pad(b.sum(axis=0), (17, 11)),
                            np.pad(alt.sum(axis=0), (17, 11)))
        print('Conserved A template:', ym, flush=True)
    conservation = pd.DataFrame(conservation)
    conservation.to_csv(D / 'construction_contrast_conservation.csv', index=False)

    # Both aggregate accounts use the same dates, generation and reference load.
    intervals = pd.read_parquet(D / 'common48.parquet')
    intervals['date'] = intervals.date.astype(str)
    intervals['ym'] = intervals.date.str[:6]
    intervals = intervals.merge(calendar[['date', 'season', 'day_class']], on='date', validate='many_to_one')
    active = intervals.day_class.eq('eligible').to_numpy()
    intervals['reference_A_MWh'] = [month_totals[ym][0][int(t)] for ym, t in zip(intervals.ym, intervals.slot)]
    intervals['worker_A_MWh'] = [month_totals[ym][1][int(t)] for ym, t in zip(intervals.ym, intervals.slot)]
    intervals[['reference_A_MWh', 'worker_A_MWh']] *= active[:, None]
    net = np.maximum(intervals.G.to_numpy() - intervals.L0.to_numpy(), 0)
    for name in ['reference', 'worker']:
        intervals[f'{name}_regional_MWh'] = np.minimum(net, intervals[f'{name}_A_MWh'])
        intervals[f'{name}_pool_MWh'] = np.minimum(intervals.positive_pool, intervals[f'{name}_A_MWh'])
    intervals[['date', 'slot', 'season', 'day_class', 'reference_A_MWh', 'worker_A_MWh',
        'reference_regional_MWh', 'worker_regional_MWh', 'reference_pool_MWh', 'worker_pool_MWh']].to_csv(
        D / 'construction_contrast_regional.csv', index=False)

    totals = []
    for season in SEASONS + ['Annual']:
        g = s if season == 'Annual' else s[s.season.eq(season)]
        q = intervals if season == 'Annual' else intervals[intervals.season.eq(season)]
        u0 = float(g.service_MWh.sum()); u1 = float(g.alternative_service_MWh.sum())
        totals.append(dict(season=season, meshes=g.mesh.nunique(),
            calendar_dates=q.date.nunique(), eligible_dates=q.loc[q.day_class.eq('eligible'), 'date'].nunique(),
            reference_A_MWh=float(g.demand_MWh.sum()), worker_A_MWh=float(g.alternative_demand_MWh.sum()),
            reference_local_MWh=u0, worker_local_MWh=u1, delta_local_MWh=u1-u0,
            delta_local_percent=100*(u1-u0)/u0,
            reference_regional_MWh=float(q.reference_regional_MWh.sum()),
            worker_regional_MWh=float(q.worker_regional_MWh.sum()),
            reference_pool_MWh=float(q.reference_pool_MWh.sum()),
            worker_pool_MWh=float(q.worker_pool_MWh.sum())))
    totals = pd.DataFrame(totals)
    totals.to_csv(D / 'construction_contrast_service.csv', index=False)
    by_pref = s.groupby(['pref', 'season'], as_index=False).agg(
        meshes=('mesh', 'size'), reference_A_MWh=('demand_MWh', 'sum'),
        worker_A_MWh=('alternative_demand_MWh', 'sum'),
        reference_local_MWh=('service_MWh', 'sum'), worker_local_MWh=('alternative_service_MWh', 'sum'))
    by_pref['delta_local_MWh'] = by_pref.worker_local_MWh - by_pref.reference_local_MWh
    by_pref.to_csv(D / 'construction_contrast_prefecture.csv', index=False)

    coefficients = []
    for season in SEASONS:
        g = s[s.season.eq(season)].copy()
        fields = CONTROLS + ['surplus_MWh', 'demand_MWh', 'alternative_demand_MWh']
        mask = np.isfinite(g[fields]).all(axis=1) & g[fields].ge(0).all(axis=1)
        g = g.loc[mask].copy()
        fixed = pd.get_dummies(g.pref.astype(str), prefix='pref', drop_first=True, dtype=float)
        x = np.column_stack([np.ones(len(g)), np.log1p(g[CONTROLS].to_numpy()), fixed.to_numpy()])
        y = np.log1p(g.surplus_MWh.to_numpy())
        yr = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        for name, field in [('reference', 'demand_MWh'), ('worker', 'alternative_demand_MWh')]:
            z = np.log1p(g[field].to_numpy())
            zr = z - x @ np.linalg.lstsq(x, z, rcond=None)[0]
            tol = np.finfo(float).eps * max(x.shape) * max(np.linalg.norm(z), np.linalg.norm(y), 1)
            identifiable = bool(np.linalg.norm(zr) > tol and np.linalg.norm(yr) > tol)
            coefficients.append(dict(season=season, construction=name, meshes=len(g),
                excluded_meshes=int((~mask).sum()), partial_r=float(np.corrcoef(yr, zr)[0, 1]) if identifiable else np.nan,
                demand_log_sd=float(z.std()), demand_residual_sd=float(zr.std()),
                demand_residual_variance_fraction=float(zr.var()/z.var()) if z.var() else np.nan,
                surplus_residual_sd=float(yr.std()),
                surplus_residual_variance_fraction=float(yr.var()/y.var()) if y.var() else np.nan,
                numerically_identifiable=identifiable, design_rank=int(np.linalg.matrix_rank(x)),
                design_columns='intercept;'+';'.join('log1p('+v+')' for v in CONTROLS)+';'+';'.join(fixed.columns)))
    coefficients = pd.DataFrame(coefficients)
    wide = coefficients.pivot(index='season', columns='construction', values='partial_r')
    coefficients['delta_r_worker_minus_reference'] = coefficients.season.map(wide.worker-wide.reference)
    coefficients.to_csv(D / 'construction_contrast_association.csv', index=False)

    unique = s.drop_duplicates('mesh')
    provenance = dict(
        date='2026-11-16', opinion='P232–239',
        design='Deterministic within-prefecture spatial allocation contrast; original energy integrations reused.',
        units='MWh per half-hour before aggregation; GWh shown in figure and prose',
        service_source='mesh_season.parquet; spatial_revision.py integrates min(max(G-L0,0),A) on eligible dates',
        template_source='tmp/inputs_YYYYMM.npz:B; slots 17–36, zero A outside support',
        date_source='metrics_all.csv, case L+0_P+0',
        workers_source='T001193 sum of 11 occupation columns through state_layer.mesh_state; retained-state coverage; missing scope rows set to zero',
        worker_weight_file='revisions/20261111_supplemental/data/mesh_population_scope.csv:workers',
        control_working_pop_source='wfh_spatial_weight.parquet:working_pop = pop*wage_share; missing demographics use prefecture median (or 1 if unavailable)',
        workers_equal_working_pop=bool(np.array_equal(unique.workers, unique.working_pop)),
        worker_sum=float(unique.workers.sum()), working_pop_sum=float(unique.working_pop.sum()),
        five_controls=CONTROLS, prefecture_fixed_effects='11 reference; indicator columns for 12,13,14',
        correlation='Pearson correlation of residual log1p seasonal demand and surplus; identical complete-case mask and controls in both arms',
        interval_uncertainty='None; deterministic construction contrast, no p values or confidence intervals',
        maximum_prefecture_slot_energy_difference_MWh=float(conservation.difference_MWh.abs().max()),
        annual_service=totals.iloc[-1].to_dict(), correlations=coefficients.to_dict('records'),
        baseline_replaced=False, geographic_prediction_refit=False,
        tests_or_safety_reviews_run=False, status='calculation_complete_awaiting_figure_and_manuscript')
    (REV / 'reports/construction_contrast_calculation.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    print(totals.to_string(index=False))
    print(coefficients[['season','construction','partial_r','demand_residual_variance_fraction','meshes']].to_string(index=False))


if __name__ == '__main__':
    main()
