"""Figure 2 — calibration ablation: 'is it just calibration?' answered. Shows that NO post-hoc calibration improves
held-out (baseline ~= black-box ~= component-aware), the midday residual is dominated by irreducible day-scatter (negative
LOMO-CV R^2), and that ONLY the component-aware model yields physically-interpretable structure (HVAC-HDD, housing-HVAC,
WFH survey gradient) — which calibration-only models cannot produce.

Ported verbatim (numerics unchanged) from the source `configured asset`. Reads only scorecards (out_path).
"""

from wlpr.io.bindings import asset as _asset_location
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from wlpr.io.paths import out_path

PN = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}


def _read(n):
    p = out_path(_asset_location('asset_0058'), n); return pd.read_csv(p) if os.path.exists(p) else None


def build_ablation_figure():
    abl = _read("bmscal_ablation_summary.csv"); expl = _read("midday_explainability.csv")
    bymon = _read("midday_diag_by_month.csv"); gsc = _read("grid_component_scorecard.csv"); wsc = _read("wfh_anchor_scorecard.csv")
    plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold'})
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9)); ax = ax.ravel()

    # A held-out NMAE: baseline / cal-only / component-aware, for midday + all-day
    a = abl.iloc[0]
    models = ['baseline', 'cal-only\n(black-box)', 'component-\naware']
    mid = [a.midday_baseline, a.midday_cal_only, a.midday_component_aware]
    alld = [a.allday_baseline, a.allday_cal_only, a.allday_component_aware]
    x = np.arange(3)
    ax[0].bar(x-0.18, mid, 0.36, label='midday', color='tab:orange')
    ax[0].bar(x+0.18, alld, 0.36, label='all-day', color='tab:blue')
    ax[0].set_xticks(x); ax[0].set_xticklabels(models); ax[0].set_ylabel('held-out NMAE (LOMO)')
    ax[0].set_title('A  No calibration improves held-out\n(all variants ≈ baseline)'); ax[0].legend(fontsize=8)
    for i, v in enumerate(mid): ax[0].text(x[i]-0.18, v+0.0004, f'{v:.3f}', ha='center', fontsize=7)

    # B LOMO-CV R^2 by pref (all negative -> no generalizable residual signal)
    if expl is not None:
        ax[1].bar([PN[str(p)] for p in expl.pref], expl.r2_LOMO_cv, color='tab:red', alpha=0.7, label='LOMO-CV R²')
        ax[1].bar([PN[str(p)] for p in expl.pref], expl.r2_insample, color='gray', alpha=0.35, label='in-sample R² (overfit)')
        ax[1].axhline(0, color='k', lw=0.8); ax[1].set_ylabel('R²')
        ax[1].set_title('B  Midday residual unpredictable out-of-sample\n(LOMO-CV R² < 0 — overfit floor)'); ax[1].legend(fontsize=8)

    # C systematic vs scatter by month (the residual is mostly scatter)
    if bymon is not None:
        xm = np.arange(len(bymon))
        ax[2].bar(xm, bymon.scatter_std, 0.6, color='tab:purple', alpha=0.6, label='day scatter (std)')
        ax[2].plot(xm, bymon.systematic_bias.abs(), '-o', color='k', label='|systematic bias|')
        ax[2].set_xticks(xm); ax[2].set_xticklabels(bymon.month); ax[2].set_ylabel('midday residual')
        ax[2].set_title('C  Residual = dominant day-scatter + tiny systematic bias'); ax[2].legend(fontsize=8)

    # D physics only component-aware provides
    ax[3].axis('off')
    hvac_hdd = float(gsc[gsc.metric == 'HVAC_HDD_corr'].value.iloc[0]) if gsc is not None else float('nan')
    det_hvac = float(gsc[gsc.metric == 'corr_detached_hvac'].value.iloc[0]) if gsc is not None else float('nan')
    aft = {m.split('_')[-1]: v for m, v in zip(wsc.metric, wsc.value) if m.startswith('wfh_share_after_')} if wsc is not None else {}
    rank = " > ".join(f"{PN[p]} {aft[p]:.2f}%" for p in ['13', '14', '12', '11'] if p in aft)
    txt = ("D  What ONLY the component-aware model yields\n"
           "    (a calibration-only twin cannot produce these)\n\n"
           f"  • HVAC component ↔ HDD      r = +{hvac_hdd:.2f}\n"
           f"  • detached-house → HVAC     r = +{det_hvac:.2f}\n"
           f"  • WFH cross-pref (survey-anchored):\n      {rank}\n"
           "  • non-circular census denominators\n"
           "  • component energy split (HVAC / cooling /\n      non-thermal / WFH) at 500 m × day × slot\n\n"
           "  ⇒ Calibration is at its floor; the bottom-up\n      structure carries the explanatory load.")
    ax[3].text(0.02, 0.98, txt, va='top', ha='left', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', fc='#eef5ee', ec='#2e8b57'))

    fig.suptitle('Figure 2 — Calibration ablation: the twin is not a calibration crutch', fontsize=13.5, fontweight='bold', y=0.99)
    plt.tight_layout(); plt.savefig(out_path(_asset_location('asset_0059'), "fig2_calibration_ablation.png"), dpi=150, bbox_inches='tight')
    plt.savefig(out_path(_asset_location('asset_0059'), "fig2_calibration_ablation.pdf"), bbox_inches='tight'); plt.close()
    return dict(hvac_hdd=hvac_hdd, det_hvac=det_hvac)


def main():
    return build_ablation_figure()


if __name__ == "__main__":
    main()
