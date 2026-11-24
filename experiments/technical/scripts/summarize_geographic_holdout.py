"""Describe the existing 16 geographic folds without fitting another model.

P240–246: target and the five existing covariates only; the source fold scores
are retained. Distribution summaries are descriptive, not causal attributions.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

REV = (next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'))
D = REV / _asset_location('asset_0002')
SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
PREFS = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}
CONTROLS = ['pop', 'working_pop', 'current_cap_kw', 'pot_cap_kw_B_IC', 'midday_load_MWh']
FIELDS = ['service_MWh'] + CONTROLS
UNITS = dict(zip(FIELDS, ['MWh/mesh/season', 'persons/mesh', 'persons/mesh',
                         'kW/mesh', 'kW/mesh', 'MWh/mesh/season']))


def describe(x):
    quantiles = np.quantile(x, [.05, .25, .5, .75, .95], method='linear')
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=0)),
                minimum=float(x.min()), maximum=float(x.max()),
                zero_n=int((x == 0).sum()), zero_fraction=float((x == 0).mean()),
                **dict(zip(['q05', 'q25', 'q50', 'q75', 'q95'], map(float, quantiles))))


def main():
    source = D / 'mesh_season.parquet'
    mesh = pd.read_parquet(source)
    mesh['pref'] = mesh.pref.astype(str)
    folds = pd.read_csv(D / 'geographic_holdout.csv', dtype={'pref': str})
    distributions, contrasts, accounts = [], [], []
    for season in SEASONS:
        raw = mesh[mesh.season.eq(season)]
        # Same complete-control selection as spatial_revision.py, lines 37–44.
        g = raw[np.isfinite(raw[CONTROLS]).all(axis=1) & (raw[CONTROLS] >= 0).all(axis=1)]
        for pref, name in PREFS.items():
            tr, te = g[g.pref.ne(pref)], g[g.pref.eq(pref)]
            saved = folds[folds.season.eq(season) & folds.pref.eq(pref)].iloc[0]
            if len(te) != saved.test_n:
                raise ValueError(f'Current sample differs from saved fold: {season}/{pref}')
            accounts.append(dict(season=season, pref=pref, prefecture=name,
                train_n=len(tr), test_n=len(te), excluded_meshes=len(raw)-len(g),
                rmse=float(saved.rmse), baseline_rmse=float(saved.baseline_rmse),
                skill=float(saved.skill), relative_rmse_change_percent=-100*float(saved.skill),
                target='log1p(service_MWh)', baseline='training target mean'))
            for field in FIELDS:
                values = {}
                for split, group in [('train', tr), ('held_out', te)]:
                    raw_values = group[field].to_numpy(float)
                    for scale, x in [('raw', raw_values), ('log1p', np.log1p(raw_values))]:
                        stats = describe(x)
                        values[(split, scale)] = stats
                        distributions.append(dict(season=season, pref=pref, prefecture=name,
                            split=split, variable=field, scale=scale,
                            unit=UNITS[field] if scale == 'raw' else 'log1p of numerical value in '+UNITS[field],
                            **stats))
                train, held = values['train', 'log1p'], values['held_out', 'log1p']
                contrasts.append(dict(season=season, pref=pref, prefecture=name, variable=field,
                    train_log_mean=train['mean'], held_out_log_mean=held['mean'],
                    log_mean_difference=held['mean']-train['mean'],
                    log_mean_shift_in_train_sd=(held['mean']-train['mean'])/train['sd'],
                    train_raw_mean=values['train', 'raw']['mean'],
                    held_out_raw_mean=values['held_out', 'raw']['mean'],
                    train_zero_fraction=train['zero_fraction'], held_out_zero_fraction=held['zero_fraction']))
    pd.DataFrame(distributions).to_csv(D / 'geographic_holdout_distributions.csv', index=False)
    pd.DataFrame(contrasts).to_csv(D / 'geographic_holdout_distribution_contrasts.csv', index=False)
    account = pd.DataFrame(accounts)
    account.to_csv(D / 'geographic_holdout_account.csv', index=False)
    report = dict(opinion='P240–246', status='descriptive_aggregation_complete',
        sources={str(f.relative_to(REV)): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in [source, D / 'geographic_holdout.csv']},
        implementation_reference='scripts/spatial_revision.py, geographic holdout loop',
        mean_skill=float(account.skill.mean()), negative_folds=int((account.skill < 0).sum()),
        folds=len(account), prefectures=PREFS, seasons=SEASONS,
        target='log1p of seasonal mesh-local service expressed in MWh',
        controls=CONTROLS, sample_selection='finite nonnegative five controls, identical to saved folds',
        mean_skill_aggregation='Unweighted arithmetic mean over 16 prefecture–season skills; not pooled RMSE reduction',
        model='Existing OLS, intercept, log1p controls standardized with training means and population SDs',
        baseline='Constant mean of log1p seasonal service across the other three prefectures',
        inference='Target and covariate distributions only; no causal attribution or significance tests',
        distributions='Linear empirical quantiles; population SD; exact zero on original service scale',
        model_refitted=False, source_scores_changed=False, extra_features=False,
        safety_reviews_or_test_suites_run=False, missing_inputs_for_approved_scope=[])
    (REV / _asset_location('asset_0050')).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(account.to_string(index=False))
    print('\nTARGET DISTRIBUTIONS')
    print(pd.DataFrame(contrasts).query("variable == 'service_MWh'").to_string(index=False))
    print('\nTOKYO COVARIATES')
    print(pd.DataFrame(contrasts).query("pref == '13' and variable != 'service_MWh'").to_string(index=False))


if __name__ == '__main__':
    main()
