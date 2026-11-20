"""Supplementary figure — raw-mobility independent validation. (1) Across all prefs, foot-traffic presence is
consistently NEGATIVELY correlated with residential midday load (people out -> lower home load): an independent,
non-electricity corroboration of the agent occupancy mechanism. (2) Adding raw daily presence does NOT make the midday
residual predictable out-of-sample (LOMO-CV R^2 stays negative) -> the day-variability floor is robust even to raw 5-min
mobility.

Ported verbatim (numerics unchanged) from the source `figures/mobility_validation.py`.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from wlpr.io.paths import input_path, out_path

PN = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}


def build_mobility_figure():
    test = pd.read_csv(out_path("scorecards", "mobility_midday_test_by_pref.csv"))
    mob = pd.read_csv(input_path("mobility_presence_panel"))   # frozen raw-derived presence panel
    diag = pd.read_csv(out_path("scorecards", "midday_diag_perday.csv"))
    mob['pref'] = mob.pref.astype(str); mob['date'] = mob.date.astype(str)
    diag['pref'] = diag.pref.astype(str); diag['d'] = diag.d.astype(str)
    df = diag.merge(mob, left_on=['pref', 'd'], right_on=['pref', 'date'])
    plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold'})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # A presence vs residential midday load, per pref (within-pref standardised), consistent negative slope
    for p in ['13', '14', '11', '12']:
        g = df[df.pref == p]
        if len(g) < 5:
            continue
        x = (g.midday_presence - g.midday_presence.mean())/g.midday_presence.std()
        y = (g.midday_obs - g.midday_obs.mean())/g.midday_obs.std()
        ax[0].scatter(x, y, s=10, alpha=0.5, label=PN[p])
    ax[0].axhline(0, c='k', lw=0.6); ax[0].axvline(0, c='k', lw=0.6)
    ax[0].set_xlabel('daily midday presence (z, within pref)'); ax[0].set_ylabel('daily midday load (z)')
    ax[0].set_title('A  Foot-traffic ↔ residential midday load\n(consistent NEGATIVE: people out → lower home load)'); ax[0].legend(fontsize=8)

    # B correlation bars per pref
    ax[1].bar([PN[str(p)] for p in test.pref], test.corr_presence_obs, color='tab:purple', alpha=0.8)
    ax[1].axhline(0, c='k', lw=0.8); ax[1].set_ylabel('corr(presence, midday load)')
    ax[1].set_title('B  Independent occupancy validation\n(all four prefs negative)')

    # C LOMO-CV R^2: weather-only vs +presence (still negative -> floor robust)
    x = np.arange(len(test)); w = 0.38
    ax[2].bar(x-w/2, test.r2_weather_only, w, label='weather + calendar', color='tab:gray')
    col = 'r2_weather_plus_presW' if 'r2_weather_plus_presW' in test.columns else 'r2_weather_plus_presence'
    ax[2].bar(x+w/2, test[col], w, label='+ raw mobility (resid-weighted)', color='tab:red')
    ax[2].axhline(0, c='k', lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels([PN[str(p)] for p in test.pref])
    ax[2].set_ylabel('midday-residual LOMO-CV R²')
    ax[2].set_title('C  Mobility does NOT break the floor\n(held-out R² stays < 0)'); ax[2].legend(fontsize=8)

    fig.suptitle('Supplementary — raw mobility: independent occupancy validation, and a robust day-variability floor', fontsize=12.5, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(out_path("figures", "figS_mobility_validation.png"), dpi=150, bbox_inches='tight')
    plt.savefig(out_path("figures", "figS_mobility_validation.pdf"), bbox_inches='tight'); plt.close()
    return dict(ok=True)


def main():
    return build_mobility_figure()


if __name__ == "__main__":
    main()
