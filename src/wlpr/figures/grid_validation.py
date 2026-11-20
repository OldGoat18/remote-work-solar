"""Figure 3 — 500 m grid-level validation and risk bands. Shows the grid error map, error vs exposure (low-exposure =
sparse/noisy not systematic), high-demand grids tracking, the low-exposure risk band, and per-pref x building-type grid
NMAE — so the twin is not strong only in Tokyo or only in apartments.

Ported verbatim (numerics unchanged) from the source `figures/grid_validation.py`.
"""
import os
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wlpr.io.paths import input_path, out_path

PN = {'11': 'Saitama', '12': 'Chiba', '13': 'Tokyo', '14': 'Kanagawa'}
MLAB = {'202310': 'Oct', '202311': 'Nov', '202312': 'Dec', '202401': 'Jan', '202402': 'Feb'}


def _mesh_latlon(code):
    code = str(code); p = int(code[0:2]); u = int(code[2:4]); q = int(code[4]); v = int(code[5]); r = int(code[6]); w = int(code[7]); m = int(code[8])
    lat = p*40/60 + q*5/60 + r*30/3600 + ((m-1)//2)*15/3600 + 7.5/3600
    lon = (u+100) + v*7.5/60 + w*45/3600 + ((m-1) % 2)*22.5/3600 + 11.25/3600
    return lat, lon


def build_grid_figure():
    grid = pd.read_parquet(out_path("scorecards", "decomp_grid_nmae.parquet"))
    SH = input_path("closure_surface_share").replace("\\", "/")
    tmp = out_path("_tmp", "_duck_g3"); os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{tmp}'"); con.execute("SET preserve_insertion_order=false")
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 10, 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(17, 9.5)); gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

    # A grid error map
    ax = fig.add_subplot(gs[0, 0]); ll = np.array([_mesh_latlon(x) for x in grid.g]); g2 = grid.assign(lat=ll[:, 0], lon=ll[:, 1])
    w = g2[(g2.lat > 34.95) & (g2.lat < 36.45) & (g2.lon > 138.95) & (g2.lon < 140.95)]
    s = ax.scatter(w.lon, w.lat, c=w.gn.clip(upper=0.15), s=2.5, cmap='RdYlGn_r', marker='s', linewidths=0)
    ax.set_aspect(1.23); ax.set_title('A  Per-grid NMAE map (500 m)'); ax.set_xlabel('lon'); ax.set_ylabel('lat')
    plt.colorbar(s, ax=ax, fraction=0.046, pad=0.02, label='grid NMAE')

    # B grid NMAE vs exposure (binned median) — low exposure = noisy
    ax = fig.add_subplot(gs[0, 1]); gg = grid.copy()
    ax.scatter(gg.dem, gg.gn, s=2, alpha=0.05, color='tab:blue'); ax.set_xscale('log'); ax.set_ylim(0, 0.3)
    gg['b'] = pd.qcut(gg.dem.rank(method='first'), 20, labels=False); bm = gg.groupby('b').agg(d=('dem', 'median'), n=('gn', 'median'))
    ax.plot(bm.d, bm.n, '-o', color='darkblue', lw=2, ms=3, label='median')
    ax.set_title('B  Grid NMAE vs exposure (low-exposure = sparse/noisy)'); ax.set_xlabel('grid seasonal demand (MWh, log)'); ax.set_ylabel('grid NMAE'); ax.legend(fontsize=8)

    # C high-demand top grids: mean daily profile obs vs twin
    ax = fig.add_subplot(gs[0, 2]); top = grid.sort_values('dem', ascending=False).head(6).g.tolist()
    gl = ','.join(f"'{x}'" for x in top)
    ts = con.execute(f"select g, slot, sum(obs) o, sum(pv*oshare) p from '{SH}' where g in ({gl}) group by 1,2").df()
    for i, gid in enumerate(top):
        t = ts[ts.g == gid].groupby('slot')[['o', 'p']].mean()
        ax.plot(t.index/2, t.o, lw=1.3, color=plt.cm.tab10(i)); ax.plot(t.index/2, t.p, lw=0.9, ls='--', color=plt.cm.tab10(i))
    ax.set_title('C  Top-6 highest-demand grids (obs solid / twin dash)'); ax.set_xlabel('hour'); ax.set_ylabel('MWh/30min')

    # D risk band: low-exposure grid NMAE distribution
    ax = fig.add_subplot(gs[1, 0]); q = grid.dem.quantile(0.2)
    lo = grid[grid.dem <= q]; hi = grid[grid.dem > q]
    ax.hist(hi.gn.clip(upper=0.25), bins=50, alpha=0.6, density=True, color='tab:green', label=f'top-80% exposure (med {hi.gn.median():.3f})')
    ax.hist(lo.gn.clip(upper=0.25), bins=50, alpha=0.6, density=True, color='tab:red', label=f'low-20% exposure RISK BAND (med {lo.gn.median():.3f})')
    ax.set_title('D  Low-exposure risk band (sparse observation)'); ax.set_xlabel('grid NMAE'); ax.set_ylabel('density'); ax.legend(fontsize=8)

    # E per-pref x building-type weighted NMAE heatmap
    ax = fig.add_subplot(gs[1, 1])
    piv = grid.groupby(['pref', 'bclass']).apply(lambda x: np.average(x.gn, weights=x.dem)).unstack()
    order = ['detached', 'mixed', 'apartment', 'imputed']
    piv = piv.reindex(columns=[c for c in order if c in piv.columns])
    im = ax.imshow(piv.values, aspect='auto', cmap='YlOrRd', vmin=0.03, vmax=0.09)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=20); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([PN[p] for p in piv.index])
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            if not np.isnan(piv.values[r, c]): ax.text(c, r, f'{piv.values[r,c]:.3f}', ha='center', va='center', fontsize=8)
    ax.set_title('E  Grid NMAE by pref × building type'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # F demand-weighted CDF with percentile callouts
    ax = fig.add_subplot(gs[1, 2]); g = grid.sort_values('gn'); cw = np.cumsum(g.dem.values)/g.dem.sum()
    ax.plot(g.gn.values, cw, lw=2.2, color='tab:blue')
    for qq in (.5, .75, .9):
        x = g.gn.values[np.searchsorted(cw, qq)]; ax.axhline(qq, ls=':', c='k', alpha=.4); ax.text(x, qq-0.06, f'P{int(qq*100)}={x:.3f}', fontsize=7)
    ax.set_xlim(0, 0.2); ax.set_title('F  Grid NMAE CDF (demand-weighted)'); ax.set_xlabel('grid NMAE'); ax.set_ylabel('cum. demand')

    fig.suptitle('Figure 3 — 500 m grid-level validation and risk bands', fontsize=13.5, fontweight='bold', y=0.99)
    plt.tight_layout(); plt.savefig(out_path("figures", "fig3_grid_validation.png"), dpi=150, bbox_inches='tight')
    plt.savefig(out_path("figures", "fig3_grid_validation.pdf"), bbox_inches='tight'); plt.close()
    return dict(p50=float(g.gn.values[np.searchsorted(cw, .5)]), lo_risk_med=float(lo.gn.median()))


def main():
    f3 = build_grid_figure()
    print(f"[15] fig3 grid P50 {f3['p50']:.4f} low-exposure risk-band median {f3['lo_risk_med']:.4f}")
    return f3


if __name__ == "__main__":
    main()
