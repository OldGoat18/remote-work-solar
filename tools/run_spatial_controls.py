#!/usr/bin/env python3
"""Spatial block intervals and conditional associations, with no iid mesh p-values."""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_spatial_order_uncertainty import spatial_inputs, SEED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/_asset_location('asset_0105')
SRC=ROOT/_asset_location('asset_0003')

def residual_corr(moment,n_controls):
    xx=moment[:n_controls,:n_controls];xy=moment[:n_controls,n_controls:]
    ss=moment[n_controls:,n_controls:]-xy.T@np.linalg.pinv(xx)@xy
    return float(ss[0,1]/np.sqrt(max(ss[0,0]*ss[1,1],1e-20)))

def main():
    mesh=pd.read_parquet(SRC/'seasonal_mesh_month_fields.parquet');keys,_,_=spatial_inputs(mesh)
    pop=pd.read_parquet(ROOT/_asset_location('asset_0049'))[['mesh','pref','pop','working_pop']]
    pv=pd.read_parquet(ROOT/_asset_location('asset_0199'))[['mesh','pref','current_cap_kw','pot_cap_kw_B_IC']]
    for d in [pop,pv]:
        d['mesh']=d.mesh.astype(str);d['pref']=d.pref.astype(str)
        assert not d.duplicated(['pref','mesh']).any()
    controls=keys.merge(pop,on=['pref','mesh'],how='left',validate='one_to_one').merge(pv,on=['pref','mesh'],how='left',validate='one_to_one')
    fields=['pop','working_pop','current_cap_kw','pot_cap_kw_B_IC','midday_load_MWh']
    rng=np.random.default_rng(SEED+71);results=[];boot=[];coverage=[]
    for season,g in mesh.groupby('season'):
        f=g.groupby(['pref','mesh'],as_index=False)[['surplus_MWh','behavior_supply_MWh','behavior_absorbed_MWh','midday_load_MWh']].sum()
        f=f.merge(controls,on=['pref','mesh'],how='left',validate='one_to_one')
        valid=np.isfinite(f[fields]).all(axis=1)&(f[fields]>=0).all(axis=1)
        coverage.append(dict(season=season,total_meshes=len(f),complete_covariates=int(valid.sum()),excluded=int((~valid).sum())))
        f=f[valid].reset_index(drop=True)
        z=np.log1p(f[fields].to_numpy());z=(z-z.mean(axis=0))/np.maximum(z.std(axis=0),1e-12)
        pref=pd.get_dummies(f.pref,drop_first=True).to_numpy()
        X=np.column_stack([np.ones(len(f)),z,pref]);n_controls=X.shape[1]
        Y=np.log1p(f[['behavior_supply_MWh','surplus_MWh']].to_numpy())
        design=np.column_stack([X,Y]);moment=design.T@design
        rho=residual_corr(moment,n_controls)
        unadjusted=np.corrcoef(Y.T)[0,1]
        for width in [10,20,40]:
            # 500 m Japanese grid is rectangular: widths are 5/10/20 km
            # nominal, approximately 4.6/9.3/18.5 km north-south.
            block=f.pref.astype(str)+':'+(f.x_cell//width).astype(str)+':'+(f.y_cell//width).astype(str)
            groups=list(f.groupby(block).indices.values());nb=len(groups)
            moments=np.stack([design[ix].T@design[ix] for ix in groups])
            energies=np.array([[f.behavior_absorbed_MWh.iloc[ix].sum(),f.surplus_MWh.iloc[ix].sum()] for ix in groups])
            correlations=[];matching=[]
            for draw in range(2000):
                weights=np.bincount(rng.integers(0,nb,nb),minlength=nb)
                mm=np.einsum('b,bij->ij',weights,moments)
                r=residual_corr(mm,n_controls);en=weights@energies;rate=100*en[0]/en[1]
                correlations.append(r);matching.append(rate)
                boot.append(dict(season=season,block_cells=width,draw=draw,partial_log_correlation=r,local_match_percent=rate))
            results.append(dict(season=season,block_cells=width,blocks=nb,meshes=len(f),
                                unadjusted_log_correlation=unadjusted,partial_log_correlation=rho,
                                partial_r_p025=np.quantile(correlations,.025),partial_r_p975=np.quantile(correlations,.975),
                                matching_percent=100*f.behavior_absorbed_MWh.sum()/f.surplus_MWh.sum(),
                                matching_p025=np.quantile(matching,.025),matching_p975=np.quantile(matching,.975),
                                controls='log1p population,working population,current PV,rooftop PV potential,midday load; prefecture FE',
                                interpretation='conditional log-linear spatial association; block resampling, not causal'))
    pd.DataFrame(results).to_csv(OUT/'spatial_partial_association.csv',index=False)
    pd.DataFrame(boot).to_parquet(OUT/'spatial_block_bootstrap.parquet',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'spatial_covariate_coverage.csv',index=False)
    print(pd.DataFrame(results)[['season','block_cells','partial_log_correlation','partial_r_p025','partial_r_p975']].to_string(index=False))

if __name__=='__main__':main()
