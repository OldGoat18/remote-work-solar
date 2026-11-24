"""BMS-CAL step 3 — component-aware, constraint-aware calibration (NOT black-box residual scaling).

Ported verbatim (numerics unchanged) from the source `configured asset`.

Idea: the product load is already daily-LV-closed; its residual is a SHAPE error concentrated at midday. We re-estimate,
per (cell = pref x building-class, daypart), a bounded multiplier on each PHYSICAL component so the recombined load fits
observed within that daypart — then renormalise each day to preserve the (validated) daily total. So the correction can
only move load along component directions (HVAC / cooling / non-thermal / WFH), each bounded, and never changes the daily
level. Fit per held-out month (LOMO); adopt a cell only if held-out midday improves with whole-day no-harm.

Ablation companions (to prove components matter):
  - cal_only: a black-box per-(cell,daypart) single scalar on TOTAL load (no component structure) — same level-preserving
    renormalisation, so the only difference is component-awareness.
We report midday/overall NMAE for: baseline / cal_only / component_aware, plus whether physics (HVAC-HDD etc.) is kept.
"""

from wlpr.io.bindings import asset as _asset_location
import os
import numpy as np, pandas as pd, duckdb
from scipy.optimize import lsq_linear
from wlpr.io.paths import input_path, out_path, derived_path

MLAB = {'202310': 'Oct', '202311': 'Nov', '202312': 'Dec', '202401': 'Jan', '202402': 'Feb'}
ALLM = list(MLAB)
DAYPARTS = {'morning': list(range(12, 19)), 'midday': list(range(19, 31)),
            'evening': list(range(31, 43)), 'late_night': list(range(43, 48)) + list(range(0, 12))}
COMPS = ['E_hvac', 'E_cool', 'E_nonth', 'E_wfh']
# physical bounds per component multiplier (WFH tightest = survey-bounded; HVAC/nonthermal moderate; cool small in winter)
BOUNDS = {'E_hvac': (0.7, 1.4), 'E_cool': (0.5, 1.5), 'E_nonth': (0.8, 1.2), 'E_wfh': (0.8, 1.2)}
RIDGE = 0.5   # pull multipliers toward 1 (constraint-aware, avoids overfit)


def _con():
    tmp = out_path(_asset_location('asset_0085'), _asset_location('asset_0146')); os.makedirs(tmp, exist_ok=True)
    c = duckdb.connect(); c.execute("SET memory_limit='10GB'"); c.execute("SET threads=6")
    c.execute(f"SET temp_directory='{tmp}'"); c.execute("SET preserve_insertion_order=false")
    return c


def _cell_panel():
    """aggregate obs + components to (cell, ym, slot), mean per day. cell = pref x building-class."""
    SH = input_path("closure_surface_share").replace("\\", "/")
    HOUS = input_path("census_housing_T001194").replace("\\", "/")
    PROD = derived_path(_asset_location('asset_0107'), "grid_component_daily_product.parquet").replace("\\", "/")
    con = _con()
    con.execute(f"""create view bclass as select mesh g,
        case when T001194007/nullif(T001194001,0)>=0.6 then 'detached'
             when T001194009/nullif(T001194001,0)>=0.6 then 'apartment' else 'mixed' end bclass from '{HOUS}'""")
    # obs per (cell, ym, slot) summed, plus day count for per-day mean
    obs = con.execute(f"""select e.pref, coalesce(b.bclass,'imputed') bclass, substr(e.d,1,6) ym, e.slot,
            sum(e.obs) o, count(distinct e.d) nd
        from (select g,pref,d,slot,obs from '{SH}') e left join bclass b on e.g=b.g
        group by 1,2,3,4""").df()
    comp = con.execute(f"""select p.pref, coalesce(b.bclass,'imputed') bclass, substr(p.d,1,6) ym, p.slot,
            sum(p.E_hvac) E_hvac, sum(p.E_cool) E_cool, sum(p.E_nonth) E_nonth, sum(p.E_wfh) E_wfh, sum(p.load_mwh) pred
        from '{PROD}' p left join bclass b on p.g=b.g group by 1,2,3,4""").df()
    df = obs.merge(comp, on=['pref', 'bclass', 'ym', 'slot'])
    for c in COMPS + ['o', 'pred']:
        df[c] = df[c] / df.nd   # per-day mean profile
    return df


def _fit_daypart(Xtr, ytr, bounds):
    """bounded ridge least squares: min ||X b - y||^2 + RIDGE*||b-1||^2, b in bounds. returns b (len = ncomp)."""
    n = Xtr.shape[1]
    A = np.vstack([Xtr, np.sqrt(RIDGE) * np.eye(n)])
    bvec = np.concatenate([ytr, np.sqrt(RIDGE) * np.ones(n)])
    lo = np.array([bounds[c][0] for c in COMPS]); hi = np.array([bounds[c][1] for c in COMPS])
    res = lsq_linear(A, bvec, bounds=(lo, hi), max_iter=200)
    return res.x


def run_calibration():
    df = _cell_panel()
    cells = df[['pref', 'bclass']].drop_duplicates().values.tolist()
    # accumulate per-cell-month held-out predictions for baseline / cal_only / component_aware
    recs = []
    adopt = {}   # (pref,bclass) -> dict(slot-> per-component betas) for the final full-product apply (use all-months fit)
    for pref, bclass in cells:
        sub = df[(df.pref == pref) & (df.bclass == bclass)]
        if sub.o.sum() <= 0:
            continue
        for ho in ALLM:
            tr = sub[sub.ym != ho]; te = sub[sub.ym == ho]
            if len(te) == 0 or len(tr) == 0:
                continue
            te = te.set_index('slot').reindex(range(48)).fillna(0)
            base = te['pred'].values; obs = te['o'].values
            ca = base.copy(); co = base.copy()
            for dp, slots in DAYPARTS.items():
                trd = tr[tr.slot.isin(slots)]
                Xtr = trd[COMPS].values; ytr = trd['o'].values
                if Xtr.sum() == 0 or ytr.sum() == 0:
                    continue
                # component-aware: bounded ridge betas per component
                b = _fit_daypart(Xtr, ytr, BOUNDS)
                Xte = te.loc[slots, COMPS].values
                ca[slots] = Xte @ b
                # cal-only: single scalar on TOTAL (no components), bounded same midday range, ridge to 1
                s_num = (trd['pred'].values * ytr).sum(); s_den = (trd['pred'].values ** 2).sum() + RIDGE
                s = (s_num + RIDGE) / s_den
                s = float(np.clip(s, 0.7, 1.4))
                co[slots] = te.loc[slots, 'pred'].values * s
            # level-preserving renormalisation to the original daily total (pure shape correction)
            for arr in (ca, co):
                tot0 = base.sum()
                if arr.sum() > 0:
                    arr *= tot0 / arr.sum()
            O = obs.sum()
            def dnmae(pred, slots=None):
                idx = slots if slots is not None else range(48)
                return float(np.abs(pred[list(idx)] - obs[list(idx)]).sum() / max(obs[list(idx)].sum(), 1e-9))
            recs.append(dict(pref=pref, bclass=bclass, ym=ho, demand=O,
                             base_all=dnmae(base), cal_all=dnmae(co), ca_all=dnmae(ca),
                             base_mid=dnmae(base, DAYPARTS['midday']), cal_mid=dnmae(co, DAYPARTS['midday']),
                             ca_mid=dnmae(ca, DAYPARTS['midday'])))
    R = pd.DataFrame(recs)
    # demand-weighted held-out summary
    def wm(col): return float((R[col] * R.demand).sum() / R.demand.sum())
    summary = pd.DataFrame([dict(
        n_cellmonths=len(R),
        midday_baseline=wm('base_mid'), midday_cal_only=wm('cal_mid'), midday_component_aware=wm('ca_mid'),
        allday_baseline=wm('base_all'), allday_cal_only=wm('cal_all'), allday_component_aware=wm('ca_all'),
    )])
    # per-cell no-harm adoption decision (component-aware): adopt where held-out midday improves AND all-day no-worse
    cell_dec = R.groupby(['pref', 'bclass']).apply(lambda g: pd.Series(dict(
        base_mid=float((g.base_mid*g.demand).sum()/g.demand.sum()),
        ca_mid=float((g.ca_mid*g.demand).sum()/g.demand.sum()),
        base_all=float((g.base_all*g.demand).sum()/g.demand.sum()),
        ca_all=float((g.ca_all*g.demand).sum()/g.demand.sum()),
        demand=float(g.demand.sum())))).reset_index()
    cell_dec['adopt'] = (cell_dec.ca_mid < cell_dec.base_mid - 1e-4) & (cell_dec.ca_all <= cell_dec.base_all + 1e-4)
    R.to_csv(out_path(_asset_location('asset_0058'), "bmscal_heldout_by_cellmonth.csv"), index=False)
    cell_dec.to_csv(out_path(_asset_location('asset_0058'), "bmscal_cell_decision.csv"), index=False)
    summary.to_csv(out_path(_asset_location('asset_0058'), "bmscal_ablation_summary.csv"), index=False)
    return dict(summary=summary, cell_dec=cell_dec, R=R)


def main():
    cal = run_calibration(); cs = cal["summary"].iloc[0]
    print(f"[15] calibration ablation midday NMAE: baseline {cs.midday_baseline:.4f} / cal-only {cs.midday_cal_only:.4f} "
          f"/ component-aware {cs.midday_component_aware:.4f}  (adopt {int(cal['cell_dec'].adopt.sum())}/{len(cal['cell_dec'])} cells)")
    return cal


if __name__ == "__main__":
    main()
