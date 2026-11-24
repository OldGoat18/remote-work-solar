"""Figure 1 — main validation (paper main text). 6 panels: daily tracking, monthly profiles, grid NMAE CDF,
spatial error map, day x hour residual heatmap, component physics. Reads the frozen closure surface (obs) + the
component product + the decomposition grid table. DuckDB-streamed.

Ported verbatim (numerics unchanged) from the source validation implementation."""

from wlpr.io.bindings import asset as _asset_location
import os
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wlpr.io.paths import input_path, out_path, derived_path

MLAB = {'202310': 'Oct', '202311': 'Nov', '202312': 'Dec', '202401': 'Jan', '202402': 'Feb'}
ALLM = list(MLAB)


def _mesh_latlon(code):
    code = str(code); p = int(code[0:2]); u = int(code[2:4]); q = int(code[4]); v = int(code[5]); r = int(code[6]); w = int(code[7]); m = int(code[8])
    lat = p*40/60 + q*5/60 + r*30/3600 + ((m-1)//2)*15/3600 + 7.5/3600
    lon = (u+100) + v*7.5/60 + w*45/3600 + ((m-1) % 2)*22.5/3600 + 11.25/3600
    return lat, lon


def build_main_figure():
    SH = input_path("closure_surface_share").replace("\\", "/")
    HOUS = input_path("census_housing_T001194").replace("\\", "/")
    PROD = derived_path(_asset_location('asset_0107'), "grid_component_daily_product.parquet").replace("\\", "/")
    grid = pd.read_parquet(out_path(_asset_location('asset_0058'), "decomp_grid_nmae.parquet"))
    tmp = out_path(_asset_location('asset_0085'), _asset_location('asset_0113')); os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{tmp}'"); con.execute("SET preserve_insertion_order=false")
    con.execute(f"create view E as select d, substr(d,1,6) ym, slot, sum(obs) o, sum(pv*oshare) p from '{SH}' group by 1,2,3")
    sysd = con.execute("select * from E").df()
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 10, 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(17, 10)); gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.26)
    CM = plt.cm.viridis

    # A daily total tracking
    ax = fig.add_subplot(gs[0, 0]); dt = sysd.groupby('d')[['o', 'p']].sum().sort_index()
    ax.plot(range(len(dt)), dt.o.values, lw=1.3, color='#222', label='observed')
    ax.plot(range(len(dt)), dt.p.values, lw=0.9, ls='--', color='tab:red', label='twin')
    nm = np.abs(dt.p - dt.o).sum() / dt.o.sum()
    mb = [i for i, x in enumerate(dt.index) if x[6:8] == '01']
    for i, m in zip([0]+mb, ALLM): ax.text(i+2, dt.o.max()*0.97, MLAB[m], fontsize=7, color='gray')
    ax.set_title(f'A  Daily total: obs vs twin (NMAE {nm:.3f})'); ax.set_xlabel('day'); ax.set_ylabel('MWh/day'); ax.legend(fontsize=7)

    # B monthly mean daily profiles
    ax = fig.add_subplot(gs[0, 1])
    for i, m in enumerate(ALLM):
        pr = sysd[sysd.ym == m].groupby('slot')[['o', 'p']].mean()
        ax.plot(pr.index/2, pr.o, lw=1.5, color=CM(i/4.2), label=MLAB[m])
        ax.plot(pr.index/2, pr.p, lw=0.9, ls='--', color=CM(i/4.2))
    ax.set_title('B  Monthly daily profile (obs solid / twin dash)'); ax.set_xlabel('hour'); ax.set_ylabel('MWh/30min'); ax.legend(fontsize=7, ncol=5)

    # C grid NMAE CDF (demand-weighted) + percentile guides
    ax = fig.add_subplot(gs[0, 2])
    g = grid.sort_values('gn'); cw = np.cumsum(g.dem.values)/g.dem.sum()
    ax.plot(g.gn.values, cw, lw=2.2, color='tab:blue', label='calibrated twin')
    for q, t in [(.5, 'P50'), (.9, 'P90')]:
        x = g.gn.values[np.searchsorted(cw, q)]; ax.axhline(q, ls=':', c='k', alpha=.4); ax.text(x, q-0.05, f'{t}={x:.3f}', fontsize=7)
    ax.set_xlim(0, 0.25); ax.set_title('C  Per-grid NMAE CDF (demand-weighted)'); ax.set_xlabel('grid NMAE'); ax.set_ylabel('cum. demand'); ax.legend(fontsize=8)

    # D spatial error map
    ax = fig.add_subplot(gs[1, 0]); ll = np.array([_mesh_latlon(x) for x in grid.g]); grid2 = grid.assign(lat=ll[:, 0], lon=ll[:, 1])
    w = grid2[(grid2.lat > 34.95) & (grid2.lat < 36.45) & (grid2.lon > 138.95) & (grid2.lon < 140.95)]
    s = ax.scatter(w.lon, w.lat, c=w.gn.clip(upper=0.15), s=2.5, cmap='RdYlGn_r', marker='s', linewidths=0)
    ax.set_aspect(1.23); ax.set_title('D  Per-grid NMAE map (500 m)'); ax.set_xlabel('lon'); ax.set_ylabel('lat')
    plt.colorbar(s, ax=ax, fraction=0.046, pad=0.02, label='grid NMAE')

    # E day x hour residual heatmap
    ax = fig.add_subplot(gs[1, 1]); sysd['rel'] = (sysd.p - sysd.o)/sysd.o
    H = sysd.pivot_table(index='d', columns='slot', values='rel').sort_index()
    im = ax.imshow(H.values, aspect='auto', cmap='RdBu_r', vmin=-0.25, vmax=0.25, origin='lower', extent=[0, 24, 0, len(H)])
    ax.set_title('E  Relative residual heatmap (day x hour)'); ax.set_xlabel('hour'); ax.set_ylabel('day')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='(twin-obs)/obs')

    # F component physics: detached share -> grid HVAC share
    ax = fig.add_subplot(gs[1, 2])
    gh = con.execute(f"""select g, sum(E_hvac)/nullif(sum(load_mwh),0) hvf from '{PROD}' group by 1""").df()
    hp = con.execute(f"select mesh g, T001194007/nullif(T001194001,0) det_sh from '{HOUS}'").df()
    gh = gh.merge(hp, on='g').dropna()
    ax.scatter(gh.det_sh*100, gh.hvf*100, s=2, alpha=0.05, color='tab:red')
    bins = np.linspace(0, 1, 21); gh['b'] = pd.cut(gh.det_sh, bins)
    bm = gh.groupby('b', observed=True).hvf.mean()*100; bc = [(b.left+b.right)/2*100 for b in bm.index]
    ax.plot(bc, bm.values, '-o', color='darkred', lw=2, ms=3, label='mean')
    rdh = np.corrcoef(gh.det_sh, gh.hvf)[0, 1]
    ax.set_title(f'F  Housing -> HVAC component (r=+{rdh:.2f})'); ax.set_xlabel('detached-house share %'); ax.set_ylabel('grid HVAC share %'); ax.legend(fontsize=8)

    fig.suptitle('Figure 1 — Daily x 500 m component-resolved load twin: main validation', fontsize=13.5, fontweight='bold', y=0.98)
    plt.savefig(out_path(_asset_location('asset_0059'), "fig1_main_validation.png"), dpi=150, bbox_inches='tight')
    plt.savefig(out_path(_asset_location('asset_0059'), "fig1_main_validation.pdf"), bbox_inches='tight'); plt.close()
    return dict(daily_nmae=float(nm), det_hvac=float(rdh))


def main():
    f1 = build_main_figure()
    print(f"[15] fig1 daily NMAE {f1['daily_nmae']:.4f} det->HVAC +{f1['det_hvac']:.3f}")
    return f1


if __name__ == "__main__":
    main()
