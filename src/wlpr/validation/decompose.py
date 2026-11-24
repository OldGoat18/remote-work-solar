"""Error decomposition — the analytical foundation for BMS-CAL (answers: where is the gap?).

Ported verbatim (numerics unchanged) from the source `configured asset`. From the closure surface (obs) and the
component product (pred + component split), decompose the twin error across:
  grain:   system / pref / building-cell (pref x detached-heavy|apartment-heavy|mixed) / grid
  time:    overall + per month
  metric:  level vs shape (split), peak timing/magnitude, valley, ramp, daypart NMAE, bias
  attribution: residual vs HDD (HVAC-driven), and per-component energy share

level/shape split (the key diagnostic): for each (grain, day)
  level error  = |Σ_slot pred - Σ_slot obs| / Σ_slot obs            (daily total wrong)
  shape error  = Σ_slot |pred_s * (Σobs/Σpred) - obs_s| / Σ_slot obs (within-day distribution wrong, level removed)
DuckDB-streamed (never pandas-loads 244 M rows). The headline system daily NMAE ~0.039 / rep-day ~0.023 comes from
run_decomposition()'s system grain over the component product + frozen closure surface.
"""

from wlpr.io.bindings import asset as _asset_location
import os
import numpy as np, pandas as pd, duckdb
from wlpr.io.paths import input_path, out_path, derived_path

PN = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}
MLAB = {'202310': 'Oct', '202311': 'Nov', '202312': 'Dec', '202401': 'Jan', '202402': 'Feb'}


def _con():
    tmp = out_path(_asset_location('asset_0085'), _asset_location('asset_0147')); os.makedirs(tmp, exist_ok=True)
    c = duckdb.connect(); c.execute("SET memory_limit='10GB'"); c.execute("SET threads=6")
    c.execute(f"SET temp_directory='{tmp}'"); c.execute("SET preserve_insertion_order=false")
    return c


def run_decomposition():
    SH = input_path("closure_surface_share").replace("\\", "/")
    HOUS = input_path("census_housing_T001194").replace("\\", "/")
    PROD = derived_path(_asset_location('asset_0107'), "grid_component_daily_product.parquet").replace("\\", "/")
    con = _con()
    # building class per mesh (detached-heavy / apartment-heavy / mixed)
    con.execute(f"""create view bclass as
      select mesh g, case when T001194007/nullif(T001194001,0) >= 0.6 then 'detached'
                          when T001194009/nullif(T001194001,0) >= 0.6 then 'apartment'
                          else 'mixed' end as bclass
      from '{HOUS}'""")
    # base: obs + pred per (g,pref,d,slot) straight from the closure surface (pred = pv*oshare = product load_mwh)
    con.execute(f"create view E as select g, pref, d, substr(d,1,6) ym, slot, obs, pv*oshare pred from '{SH}'")

    out = {}

    # ---------- per (grain, day): daily totals + slot arrays for level/shape/peak/ramp ----------
    # system + pref daily totals
    sysd = con.execute("""select pref, d, ym, slot, sum(obs) o, sum(pred) p from E group by 1,2,3,4""").df()
    def grain_metrics(df, keys):
        """df has columns keys + slot,o,p (per-slot sums). compute per-day level/shape then average (demand-wt)."""
        rows = []
        for kv, g in df.groupby(keys):
            g = g.sort_values('slot')
            # per day
            recs = []
            for d, gd in g.groupby('d'):
                o = gd.set_index('slot').o.reindex(range(48)).fillna(0).values
                p = gd.set_index('slot').p.reindex(range(48)).fillna(0).values
                O, P = o.sum(), p.sum()
                if O <= 0:
                    continue
                level = abs(P - O) / O
                bias = (P - O) / O
                shape = np.abs(p * (O / P if P > 0 else 0) - o).sum() / O
                nmae = np.abs(p - o).sum() / O
                # peak (evening 17:00+ = slot>=34)
                ev = slice(34, 48)
                pk_o_i, pk_p_i = 34 + np.argmax(o[ev]), 34 + np.argmax(p[ev])
                pk_time = (pk_o_i == pk_p_i)
                pk_within1 = (abs(pk_o_i - pk_p_i) <= 1)
                pk_mag = (p[pk_p_i] - o[pk_o_i]) / o[pk_o_i] if o[pk_o_i] > 0 else 0
                # valley (overnight min 0-12 = slot 0..11)
                vmin_o, vmin_p = o[:12].min(), p[:12].min()
                valley = (vmin_p - vmin_o) / vmin_o if vmin_o > 0 else 0
                # ramp = max hourly rise (morning) error
                ramp_o = np.diff(o).max(); ramp_p = np.diff(p).max()
                ramp = (ramp_p - ramp_o) / ramp_o if ramp_o > 0 else 0
                recs.append((O, nmae, level, shape, pk_time, pk_within1, pk_mag, valley, ramp, bias))
            if not recs:
                continue
            R = np.array(recs, float); w = R[:, 0]; wsum = w.sum()
            wavg = lambda c: float((R[:, c] * w).sum() / wsum)
            kvd = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
            kvd.update(dict(demand=float(wsum), nmae=wavg(1), level_nmae=wavg(2), shape_nmae=wavg(3),
                            peak_time_exact=wavg(4), peak_time_within1=wavg(5), peak_mag_err=wavg(6),
                            valley_err=wavg(7), ramp_err=wavg(8), bias=wavg(9), n_days=len(recs)))
            rows.append(kvd)
        return pd.DataFrame(rows)

    # system (collapse pref)
    sysonly = sysd.groupby(['d', 'ym', 'slot'], as_index=False)[['o', 'p']].sum(); sysonly['grain'] = 'system'
    m_sys = grain_metrics(sysonly.assign(k='all'), ['k']).assign(grain='system', key='all').drop(columns=['k'])
    m_pref = grain_metrics(sysd, ['pref']).assign(grain='pref'); m_pref['key'] = m_pref.pref.map(PN)
    out['by_level'] = pd.concat([m_sys, m_pref[m_sys.columns.tolist()] if set(m_sys.columns).issubset(m_pref.columns) else m_pref], ignore_index=True)

    # monthly (system + pref)
    m_sys_mo = grain_metrics(sysonly, ['ym']).assign(grain='system'); m_sys_mo['key'] = m_sys_mo.ym.map(MLAB)
    m_pref_mo = grain_metrics(sysd, ['pref', 'ym']).assign(grain='pref')
    m_pref_mo['key'] = m_pref_mo.pref.map(PN) + '/' + m_pref_mo.ym.map(MLAB)
    out['by_month'] = pd.concat([m_sys_mo, m_pref_mo], ignore_index=True)

    # building-cell (pref x bclass) daily metrics — LEFT join so grids without housing census fold into 'imputed'
    cell = con.execute("""select e.pref, coalesce(b.bclass,'imputed') bclass, e.d, e.ym, e.slot, sum(e.obs) o, sum(e.pred) p
                          from E e left join bclass b on e.g=b.g group by 1,2,3,4,5""").df()
    m_cell = grain_metrics(cell, ['pref', 'bclass']).assign(grain='cell')
    m_cell['key'] = m_cell.pref.map(PN) + '/' + m_cell.bclass
    out['by_cell'] = m_cell

    # ---------- grid-level NMAE (per grid over all days) + demand + pref + bclass ----------
    grid = con.execute(f"""
      select e.g, e.pref, coalesce(b.bclass,'imputed') bclass, sum(abs(e.pred-e.obs)) ae, sum(e.obs) dem
      from E e left join bclass b on e.g=b.g group by 1,2,3""").df()
    grid['gn'] = grid.ae / grid.dem.replace(0, np.nan)
    grid = grid.dropna(subset=['gn'])
    grid.to_parquet(out_path(_asset_location('asset_0058'), "decomp_grid_nmae.parquet"), index=False)
    def wq(df, q):
        v = df.gn.values; w = df.dem.values; o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
        return float(v[o][min(np.searchsorted(cw, q), len(v) - 1)])
    out['grid_summary'] = pd.DataFrame([dict(
        grids=len(grid), grid_nmae_P50=wq(grid, .5), grid_nmae_P75=wq(grid, .75),
        grid_nmae_P90=wq(grid, .9), grid_nmae_P99=wq(grid, .99),
        high_demand_nmae=float(np.average(grid.nlargest(int(len(grid)*0.1), 'dem').gn,
                                           weights=grid.nlargest(int(len(grid)*0.1), 'dem').dem)))])

    # ---------- residual vs HDD attribution (is the residual HVAC-driven?) ----------
    # daily system residual (signed) vs daily HDD proxy (use Tokyo pref mean obs trough as cold proxy not available here;
    # instead correlate daily |level| error with month as a coarse season tag)
    dd = sysonly.groupby('d').apply(lambda x: pd.Series({
        'o': x.o.sum(), 'p': x.p.sum()})).reset_index()
    dd['resid'] = (dd.p - dd.o) / dd.o; dd['ym'] = dd.d.str[:6]
    out['resid_by_month'] = dd.groupby('ym').resid.agg(['mean', 'std']).reset_index().assign(month=lambda x: x.ym.map(MLAB))

    # ---------- daypart NMAE + monthly-profile NMAE (system) ----------
    DAYPARTS = {'morning': range(12, 19), 'midday': range(19, 31), 'evening': range(33, 43),
                'late_night': list(range(43, 48)) + list(range(0, 11))}
    piv = sysonly.pivot_table(index='d', columns='slot', values=['o', 'p']).fillna(0)
    dp_rows = []
    for name, slots in DAYPARTS.items():
        slots = list(slots)
        o = piv['o'][slots].values; p = piv['p'][slots].values
        Od = o.sum(axis=1); num = np.abs(p - o).sum(axis=1)
        nm = float((num[Od > 0] / Od[Od > 0] * Od[Od > 0]).sum() / Od[Od > 0].sum())  # per-day daypart NMAE (incl. day scatter)
        # mean-profile daypart NMAE: error of the AVERAGE daily profile (shape fidelity; excludes irreducible day scatter)
        om = o.mean(axis=0); pm = p.mean(axis=0)
        nm_mp = float(np.abs(pm - om).sum() / om.sum()) if om.sum() > 0 else np.nan
        dp_rows.append(dict(daypart=name, nmae=nm, meanprofile_nmae=nm_mp))
    out['daypart'] = pd.DataFrame(dp_rows)
    # monthly-profile NMAE: NMAE of each month's MEAN daily profile, demand-weighted across months
    mp = []
    for ym, g in sysonly.groupby('ym'):
        pr = g.groupby('slot')[['o', 'p']].mean()
        O = pr.o.sum(); nm = np.abs(pr.p - pr.o).sum() / O if O > 0 else np.nan
        mp.append((ym, nm, O))
    mpa = np.array([(m[1], m[2]) for m in mp], float)
    # evening ramp (afternoon trough -> evening peak rise, slots 30..40 = 15:00-20:00), demand-wt over days
    er_o = np.maximum.reduce([np.diff(piv['o'][list(range(30, 41))].values, axis=1)], 0).max(axis=1)
    er_p = np.maximum.reduce([np.diff(piv['p'][list(range(30, 41))].values, axis=1)], 0).max(axis=1)
    Od = piv['o'][list(range(30, 41))].values.sum(axis=1)
    er_err = float((((er_p - er_o) / np.where(er_o > 0, er_o, np.nan)) * Od).sum() / Od.sum())
    out['headline'] = pd.DataFrame([dict(
        monthly_profile_nmae=float((mpa[:, 0] * mpa[:, 1]).sum() / mpa[:, 1].sum()),
        evening_ramp_err=er_err)])

    # component energy shares (attribution context) from the product
    comp = con.execute(f"""select substr(d,1,6) ym, sum(E_hvac) hv, sum(E_cool) co, sum(E_nonth) nt, sum(E_wfh) wf,
                           sum(load_mwh) tot from '{PROD}' group by 1""").df()
    for c in ['hv', 'co', 'nt', 'wf']:
        comp[c+'_share'] = comp[c] / comp.tot * 100
    comp['month'] = comp.ym.map(MLAB)
    out['component_shares'] = comp

    # write all tables
    for name, df in out.items():
        df.to_csv(out_path(_asset_location('asset_0058'), f"decomp_{name}.csv"), index=False)
    return out


def run_gates(out: dict) -> None:
    """Headline accuracy gates: system daily NMAE ~0.039, rep-day (monthly-profile) NMAE ~0.023 (exit 1 on fail)."""
    fails = []
    def ck(name, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {d}")
        if not ok:
            fails.append(name)
    sysrow = out["by_level"][out["by_level"].grain == "system"].iloc[0]
    repday = float(out["headline"].monthly_profile_nmae.iloc[0])
    ck("1 system daily NMAE <= 0.05 (~0.039)", sysrow.nmae <= 0.05, f"nmae={sysrow.nmae:.4f}")
    ck("2 rep-day (monthly-profile) NMAE <= 0.04 (~0.023)", repday <= 0.04, f"nmae={repday:.4f}")
    ck("3 grid P50 present", out["grid_summary"].iloc[0].grid_nmae_P50 > 0)
    if fails:
        raise SystemExit("decompose: GATE FAILS -> " + ", ".join(fails))


def main() -> dict:
    out = run_decomposition()
    sysrow = out["by_level"][out["by_level"].grain == "system"].iloc[0]
    print(f"[15] system daily NMAE {sysrow.nmae:.4f} (level {sysrow.level_nmae:.4f} / shape {sysrow.shape_nmae:.4f}) "
          f"bias {sysrow.bias:+.4f} peak±1 {sysrow.peak_time_within1:.2f}")
    print(f"[15] rep-day (monthly-profile) NMAE {out['headline'].monthly_profile_nmae.iloc[0]:.4f}")
    run_gates(out)
    print("decompose: ALL GATES PASS")
    return out


if __name__ == "__main__":
    main()
