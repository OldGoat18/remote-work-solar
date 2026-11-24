"""Step 02 — grid × date × slot LOAD-FIT component decomposition (HVAC / cooling / non-thermal / WFH).

Ported verbatim (numerics unchanged) from the source `configured asset`. Rebuilds the component
product FROM FROZEN INPUTS ONLY (closure surface + census housing + demographics + denominators + weather +
LV load). Product-grade closure: per (g,d,slot) the four components sum to load_mwh exactly; all-zero-weight
slots assign the whole load to the non-thermal residual; grids without housing census get the pref-mean mix
(imputed_mix=1). The 244 M-row join is streamed to parquet via DuckDB COPY.

The WFH mesh weight is households × wage_share, normalised within each prefecture.
The prefecture-level design separately scales the work-activity basis by aggregate
wfh_eligible. The survey cross-pref anchor is the SEPARATE step 03; the baseline
ledger uses this unanchored product. Component closure is imposed by allocation
and is distinct from the raw held-out total-load prediction accuracy.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import datetime
import os

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from wlpr.io.paths import derived_path, input_path, model_config, out_path

from wlpr.io.activity_calendar import EXCLUDED_DATES
HOL = EXCLUDED_DATES
STN = {'11': '埼玉熊谷', '12': '千葉', '13': '東京', '14': '神奈川横浜'}


def build_component_product(mode: str = "full") -> dict:
    cfg = model_config()
    HL_D = float(cfg["heat_loss"]["HL_detached"]); HL_R = float(cfg["heat_loss"]["HL_rowhouse"])
    ALL = ["202310", "202311", "202312", "202401", "202402"]
    W_OWN = {k: float(v) for k, v in cfg["weather"]["pop_weight_own_station"].items()}
    TAU = float(cfg["agent_prior"]["pooling_tau"]); NB = int(cfg["agent_prior"]["env_bumps"])
    hdd_base = float(cfg["weather"]["hdd_base_c"]); cdd_base = float(cfg["weather"]["cdd_base_c"])

    SHP = input_path("closure_surface_share")
    HOUS = input_path("census_housing_T001194")
    DEMO = input_path("demographic_mesh_panel")
    MESH = input_path("mesh_denominator_panel")
    AME = input_path("weather_amedas")
    STD = input_path("lv_load_hourly")
    tmp = out_path(_asset_location('asset_0085'), _asset_location('asset_0106')); os.makedirs(tmp, exist_ok=True)

    con = duckdb.connect(); con.execute("SET memory_limit='10GB'"); con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{tmp}'"); con.execute("SET preserve_insertion_order=false")

    smoke = (mode == "smoke")
    smoke_pref = "13"; smoke_months = ("202310",)

    # ---- housing panel + heat-loss ----
    HJ = con.execute(f"""select mesh, pref_code pref, T001194001 hh, T001194002 own_hh, T001194007 detached,
        T001194008 rowhouse, T001194009 apartment from '{HOUS}'""").df()
    HJ['pref'] = HJ.pref.astype(str)
    for c_ in ['hh', 'own_hh', 'detached', 'rowhouse', 'apartment']:
        HJ[c_] = pd.to_numeric(HJ[c_], errors='coerce')
    HJ['det_sh'] = HJ.detached / HJ.hh; HJ['row_sh'] = HJ.rowhouse / HJ.hh
    HJ['apt_sh'] = HJ.apartment / HJ.hh; HJ['own_sh'] = HJ.own_hh / HJ.hh
    HJ['heatloss'] = HJ.det_sh * HL_D + HJ.row_sh * HL_R + HJ.apt_sh * 1.0
    DM = pd.read_parquet(DEMO)[['mesh', 'mean_hh_size', 'wage_share']]
    H = HJ.merge(DM, on='mesh', how='left')
    for col in ['heatloss', 'det_sh', 'own_sh', 'mean_hh_size', 'wage_share']:
        H[col] = H[col].fillna(H.pref.map(H.groupby('pref')[col].mean()))
    H['hh'] = H.hh.fillna(H.pref.map(H.groupby('pref').hh.mean()))
    H['w_hvac'] = H.hh * H.heatloss; H['w_nth'] = H.hh * H.mean_hh_size; H['w_wfh'] = H.hh * H.wage_share
    for ch in ['w_hvac', 'w_nth', 'w_wfh']:
        H[ch + '_n'] = H[ch] / H.pref.map(H.groupby('pref')[ch].mean())
    Hn = H[['mesh', 'w_hvac_n', 'w_nth_n', 'w_wfh_n']].rename(
        columns={'w_hvac_n': 'HVAC_n', 'w_nth_n': 'person_n', 'w_wfh_n': 'wfh_n'})
    where_sh = f"where pref='{smoke_pref}' and substr(d,1,6) in ('{smoke_months[0]}')" if smoke else ""
    GR = con.execute(f"select distinct cast(g as varchar) mesh, cast(pref as varchar) pref from '{SHP}' {where_sh}").df()
    Mden = GR.merge(Hn, on='mesh', how='left')
    Mden['imputed_mix'] = Mden.HVAC_n.isna().astype('int8')
    for c in ['HVAC_n', 'person_n', 'wfh_n']:
        Mden[c] = Mden[c].fillna(1.0)
    con.register('Mden', Mden)
    H['pv_suit'] = H.det_sh * H.own_sh   # PV roof-access scenario index (NOT load)
    H[['mesh', 'pref', 'det_sh', 'apt_sh', 'own_sh', 'heatloss', 'pv_suit']]\
        .to_parquet(derived_path(_asset_location('asset_0107'), "housing_mesh_panel.parquet"), index=False)

    # ---- census mechanism denominators ----
    MM = pd.read_parquet(MESH)[['mesh', 'pref', 'conditioned_dwelling', 'person', 'wfh_eligible']]
    MM['pref'] = MM.pref.astype(str)
    pdg = MM.groupby('pref')[['conditioned_dwelling', 'person', 'wfh_eligible']].sum()
    nrm = {c: pdg[c].mean() for c in pdg.columns}
    DEN_P = {p: {'HVAC': pdg.loc[p, 'conditioned_dwelling'] / nrm['conditioned_dwelling'],
                 'person': pdg.loc[p, 'person'] / nrm['person'],
                 'wfh': pdg.loc[p, 'wfh_eligible'] / nrm['wfh_eligible']} for p in STN}

    # ---- weather -> 48-slot temperature, HDD (AMeDAS UTC -> +9h JST) ----
    ALLSTN = set(STN.values()) | {'東京'}
    th = [con.execute(f"select strftime(cast(timestamp+INTERVAL 9 HOUR as DATE),'%Y%m%d') d, "
                      f"hour(timestamp+INTERVAL 9 HOUR) hr, avg(temp) t from '{AME}' where station='{s}' "
                      f"group by 1,2").df().assign(station=s) for s in ALLSTN]
    HH = pd.concat(th); stationT = {}
    for (s, d), g in HH.groupby(['station', 'd']):
        hrs = g.set_index('hr').t.reindex(range(24)).interpolate().bfill().ffill().values
        stationT[(s, d)] = np.interp(np.arange(48) * 0.5, np.arange(24), hrs, period=24)
    slotT, dHDD = {}, {}
    for p, s in STN.items():
        for d in set(d for (st, d) in stationT if st == s):
            if ('東京', d) in stationT and (s, d) in stationT:
                slotT[(p, d)] = W_OWN[p] * stationT[(s, d)] + (1 - W_OWN[p]) * stationT[('東京', d)]
                dHDD[(p, d)] = np.clip(hdd_base - slotT[(p, d)].mean(), 0, None)

    hod = np.arange(48) / 2.0; gauss = lambda c, w: np.exp(-0.5 * ((hod - c) / w) ** 2)
    B_act = ((hod >= 9) & (hod <= 17)).astype(float) * gauss(13, 3.0); B_eve = 1 / (1 + np.exp(-(hod - 16)))
    BUMPS = [gauss(c, 1.25) for c in np.linspace(0.5, 23.5, NB)]
    HVACN = ['HVAC', 'HVAC_eve', 'COOL']; ENVN = [f'env{k:02d}' for k in range(NB)]; CN = HVACN + ['wfh'] + ENVN

    def shapes(p, d):
        T = slotT[(p, d)]; hdd = np.clip(hdd_base - T, 0, None); cdd = np.clip(T - cdd_base, 0, None)
        c = {'HVAC': hdd, 'HVAC_eve': hdd * B_eve, 'COOL': cdd, 'wfh': B_act}
        for k in range(NB):
            c[f'env{k:02d}'] = BUMPS[k]
        return c

    def Xrow(p, d, wk):
        sh = shapes(p, d); Dp = DEN_P[p]; cols = []
        for n in CN:
            ch = 'HVAC' if n in HVACN else ('wfh' if n == 'wfh' else 'person'); s = sh[n].copy()
            if wk and n == 'wfh':
                s = np.zeros(48)
            cols.append(Dp[ch] * s)
        return np.column_stack(cols)

    LV = "voltage='低圧' and flow='順' and pref_code in ('11','12','13','14')"
    panel = con.execute(f"select pref_code pref,date_str d,(cast(hour_str as int)-1) slot, sum(kwh)/1e3 mwh "
                        f"from '{STD}' where {LV} group by 1,2,3").df()
    panel['pref'] = panel.pref.astype(str)
    is_wk = lambda d: datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])).weekday() >= 5 or d in HOL
    ym = lambda d: d[:6]; days = {p: sorted([d for (pp, d) in slotT if pp == p]) for p in STN}

    def ovf(p, d):
        y = panel[(panel.pref == p) & (panel.d == d)].sort_values('slot').mwh.values
        return y if len(y) == 48 else None

    NPc = len(CN)

    def collect(months, wk, pref=None):
        Xs, Ys = [], []
        for p in ([pref] if pref else STN):
            for d in days[p]:
                if ym(d) not in months or is_wk(d) != wk:
                    continue
                y = ovf(p, d)
                if y is None:
                    continue
                Xs.append(Xrow(p, d, wk)); Ys.append(y)
        return (np.vstack(Xs), np.concatenate(Ys)) if Xs else (None, None)

    iHV = [CN.index(n) for n in ['HVAC', 'HVAC_eve']]; iCO = CN.index('COOL')
    iE = [CN.index(n) for n in ENVN]; iW = CN.index('wfh'); Arows = []
    for ho in ALL:
        for wk in (False, True):
            tr = [m for m in ALL if m != ho]; X, Y = collect(tr, wk); mu = nnls(X, Y)[0]
            for p in STN:
                Xp, Yp = collect(tr, wk, p)
                bp = nnls(np.vstack([Xp, TAU * np.eye(NPc)]), np.concatenate([Yp, TAU * mu]))[0] if Xp is not None else mu
                for d in days[p]:
                    if ym(d) != ho or is_wk(d) != wk:
                        continue
                    sh = shapes(p, d)
                    a_hv = sum(bp[i] * sh[CN[i]] for i in iHV); a_co = bp[iCO] * sh['COOL']
                    a_no = sum(bp[i] * sh[CN[i]] for i in iE); a_wf = (bp[iW] * sh['wfh']) if not wk else np.zeros(48)
                    for s in range(48):
                        Arows.append((p, d, s, float(a_hv[s]), float(a_co[s]), float(a_no[s]), float(a_wf[s])))
    A = pd.DataFrame(Arows, columns=['pref', 'd', 'slot', 'a_hv', 'a_co', 'a_no', 'a_wf'])
    con.register('A4', A)

    con.execute(f"create view SH as select g,pref,d,slot,oshare,pv from '{SHP}' {where_sh}")
    con.execute("""create table comp as
      select s.pref,s.d,s.slot,s.g,(s.pv*s.oshare) gtot, m.imputed_mix,
             m.HVAC_n*a.a_hv hv, m.HVAC_n*a.a_co co, m.person_n*a.a_no nth, m.wfh_n*a.a_wf wf
      from SH s join Mden m on s.g=m.mesh and s.pref=m.pref join A4 a on a.pref=s.pref and a.d=s.d and a.slot=s.slot""")
    PROD = derived_path(_asset_location('asset_0107'), "grid_component_daily_product.parquet")
    con.execute(f"""COPY (
      select g, pref, d, slot, imputed_mix, gtot as load_mwh,
             case when tt>0 then gtot*hv/tt  else 0    end as E_hvac,
             case when tt>0 then gtot*co/tt  else 0    end as E_cool,
             case when tt>0 then gtot*nth/tt else gtot end as E_nonth,
             case when tt>0 then gtot*wf/tt  else 0    end as E_wfh
      from (select *, (hv+co+nth+wf) tt from comp)
    ) TO '{PROD}' (FORMAT parquet, COMPRESSION zstd)""")
    nrows = con.execute(f"select count(*) from '{PROD}'").fetchone()[0]

    # ---- verification + scorecard ----
    nnull = con.execute(f"select count(*) from '{PROD}' where E_hvac is null or E_cool is null or "
                        f"E_nonth is null or E_wfh is null").fetchone()[0]
    nneg = con.execute(f"select count(*) from '{PROD}' where E_hvac<0 or E_cool<0 or E_nonth<0 or E_wfh<0").fetchone()[0]
    chk = con.execute(f"select sum(load_mwh), sum(E_hvac+E_cool+E_nonth+E_wfh) from '{PROD}'").fetchone()
    rowmax = con.execute(f"select max(abs(load_mwh-(E_hvac+E_cool+E_nonth+E_wfh))) from '{PROD}'").fetchone()[0]
    ngrid = con.execute(f"select count(distinct g) from '{PROD}'").fetchone()[0]
    nimp = con.execute(f"select count(distinct g) from '{PROD}' where imputed_mix=1").fetchone()[0]
    full_closure = con.execute(f"select sum(pv*oshare) from SH").fetchone()[0]
    tot = con.execute(f"select sum(E_hvac) h,sum(E_cool) c,sum(E_nonth) n,sum(E_wfh) w,sum(load_mwh) t from '{PROD}'").fetchone()
    CD = con.execute(f"select pref,d, sum(E_hvac) E_hvac, sum(load_mwh) E_tot from '{PROD}' group by 1,2").df()
    con.register('dh', pd.DataFrame([(p, d, dHDD[(p, d)]) for (p, d) in dHDD], columns=['pref', 'd', 'hdd']))
    CDh = CD.merge(con.execute("select * from dh").df(), on=['pref', 'd']); CDh['hvac_share'] = CDh.E_hvac / CDh.E_tot
    r_hd = float(np.corrcoef(CDh.hvac_share, CDh.hdd)[0, 1]) if len(CDh) > 2 else float('nan')
    gs = con.execute(f"select g, sum(E_hvac)/nullif(sum(load_mwh),0) hvf, sum(load_mwh) tt from '{PROD}' "
                     f"group by 1 having sum(load_mwh)>0").df()
    gs = gs.merge(H[['mesh', 'det_sh']].rename(columns={'mesh': 'g'}), on='g', how='left').dropna(subset=['det_sh', 'hvf'])
    r_dh = float(np.corrcoef(gs.det_sh, gs.hvf)[0, 1]) if len(gs) > 2 else float('nan')

    reldiff = abs(chk[0] - chk[1]) / chk[0]
    sc = pd.DataFrame([
        ('HL_d', HL_D), ('mode', mode),
        ('share_HVAC', tot[0] / tot[4] * 100), ('share_cooling', tot[1] / tot[4] * 100),
        ('share_nonthermal', tot[2] / tot[4] * 100), ('share_WFH', tot[3] / tot[4] * 100),
        ('HVAC_HDD_corr', r_hd), ('corr_detached_hvac', r_dh),
        ('product_rows', nrows), ('load_bearing_meshes', ngrid),
        ('imputed_mix_meshes', nimp), ('demand_coverage_pct_of_full_closure', chk[0] / full_closure * 100),
        ('null_component_rows', nnull), ('negative_component_rows', nneg),
        ('per_row_max_abs_load_minus_components', rowmax), ('agg_reldiff', reldiff),
    ], columns=['metric', 'value'])
    sc.to_csv(out_path(_asset_location('asset_0058'), "grid_component_scorecard.csv"), index=False)
    return dict(rows=nrows, meshes=ngrid, imputed=nimp, nnull=int(nnull), nneg=int(nneg),
                reldiff=float(reldiff), rowmax=float(rowmax),
                coverage=chk[0] / full_closure * 100, hvac_hdd=r_hd, det_hvac=r_dh,
                shares=(tot[0] / tot[4] * 100, tot[2] / tot[4] * 100, tot[3] / tot[4] * 100))


def run_gates(r: dict) -> None:
    """Closure + sanity gates (exit 1 on fail)."""
    fails = []
    def ck(name, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {d}")
        if not ok:
            fails.append(name)
    ck("1 four components close to load (reldiff<1e-6)", r["reldiff"] < 1e-6, f"reldiff={r['reldiff']:.2e}")
    ck("2 per-row max |load-Σcomp| ~ 0", r["rowmax"] < 1e-6, f"rowmax={r['rowmax']:.2e}")
    ck("3 no null components", r["nnull"] == 0)
    ck("4 no negative components", r["nneg"] == 0)
    ck("5 HVAC↔HDD positive (climate signal)", (r["hvac_hdd"] > 0) or np.isnan(r["hvac_hdd"]), f"{r['hvac_hdd']:.3f}")
    if fails:
        raise SystemExit("component_product: GATE FAILS -> " + ", ".join(fails))


def main(mode: str = "full") -> None:
    r = build_component_product(mode)
    print(f"[02] rows={r['rows']:,} meshes={r['meshes']:,} (imputed {r['imputed']:,}) | "
          f"reldiff={r['reldiff']:.2e} rowmax={r['rowmax']:.2e} | "
          f"shares HVAC {r['shares'][0]:.1f}% nonth {r['shares'][1]:.1f}% WFH {r['shares'][2]:.1f}%")
    run_gates(r)
    print("02_component_product: ALL GATES PASS")


if __name__ == "__main__":
    import os as _os
    main(_os.environ.get("WLPR_MODE", "full"))
