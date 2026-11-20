"""Propagate the recovered 14 meshes on the already-recorded residual paths.

Only missing-template contributions are added. G, L0, residual block draws and
fixed allocations are retained. Regional minima are recomputed from full-domain
totals, not from the sum of mesh-level regional differences. No new samples.
"""
from pathlib import Path
import json,sys,runpy,gc
import numpy as np
import pandas as pd
import duckdb
from numba import set_num_threads
R=Path(__file__).resolve().parents[1];D=R/'data';ROOT=next(p for p in R.parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(R/'scripts'))
from run_technical import m,supp,lunch_schedule
from input_mesh_alignment import components
from propagate_pv_residuals import evaluate

REPORT=R/'reports/opinion3_mesh_repair.json'
def mark(kind,ym):
    r=json.loads(REPORT.read_text());r.setdefault(kind,[]).append(ym)
    REPORT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')

def main():
    set_num_threads(4)
    old=R.parent/'20261111_supplemental/data'
    keys=pd.read_csv(old/'mesh_population_scope.csv',dtype={'mesh':str,'munic':str,'pref':str})
    muni=pd.read_csv(old/'policy_municipality_inputs.csv',dtype={'munic':str})
    mapping=pd.read_csv(D/'opinion3_mesh_key_repair.csv',dtype={'mesh':str,'ym':str})
    take=keys.mesh.isin(mapping.mesh).to_numpy();small=keys.loc[take].reset_index(drop=True)
    mi=small.munic.map(dict(zip(muni.munic,range(len(muni))))).fillna(-1).to_numpy(np.int64)
    pi=small.pref.map({'11':0,'12':1,'13':2,'14':3}).to_numpy(np.int64)
    alloc=pd.read_csv(old/'policy_allocations.csv',dtype={'munic':str});sc=pd.read_csv(old/'policy_scenarios.csv')
    sid=sc.query("fraction==.25 and target=='high_surplus_q4' and rule=='TE'").scenario_id.iloc[0]
    high=alloc[alloc.scenario_id.eq(sid)].set_index('munic').reindex(muni.munic).scale.to_numpy()
    now=pd.read_csv(D/'marginal_allocations.csv',dtype={'munic':str}).query("case=='L+0_P+0'").set_index('munic').reindex(muni.munic)
    previous=pd.DataFrame(json.loads(REPORT.read_text())['before_marginal_allocations']).query("case=='L+0_P+0'").set_index('munic').reindex(muni.munic)
    if not np.allclose(now.x,previous.x,rtol=0,atol=1e-12):
        raise ValueError('Changed allocations require full policy reevaluation, not a missing-mesh-only update.')
    marginal=now.x.to_numpy()+1
    raw=pd.read_parquet(ROOT/'outputs/figures/extra/v4.7/nc_completion/owner_year2_holdout_predictions.parquet')
    raw=raw[raw.model.eq('month_slot')&raw.target.eq('generation_kWh')&raw.pref.astype(int).isin([11,12,13,14])].copy()
    raw['dt']=pd.to_datetime(raw.timestamp,format='%Y%m%d %H:%M');raw['date']=raw.dt.dt.normalize();raw['slot']=raw.dt.dt.hour*2+raw.dt.dt.minute//30
    raw['season']=raw.dt.dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})
    raw=raw[raw.slot.between(17,36)]
    raw['relative_error']=(raw.observed_kWh-raw.prediction_kWh)/raw.groupby(['season','pref','slot']).prediction_kWh.transform('mean')
    paths=raw.pivot(index='date',columns=['pref','slot'],values='relative_error').sort_index().reindex(columns=pd.MultiIndex.from_product([[11,12,13,14],range(17,37)]))
    err=pd.read_parquet(ROOT/'outputs/figures/extra/v4.7/nc_completion/mesh_LV_month_holdout_error_sums.parquet');err=err[err.model.eq('owner_scaled')];err.mesh=err.mesh.astype(str)
    factors={}
    for h,f in err.groupby('held_out_month'):
        f=f.set_index('mesh').reindex(small.mesh);den=f.target_MWh+f.signed_error_MWh
        factors[str(h)]=np.divide(f.target_MWh.to_numpy(),den.to_numpy(),out=np.ones(len(f)),where=den.to_numpy()>1e-12)
    con=duckdb.connect()
    for ym in sorted(mapping.ym.unique()):
        state=json.loads(REPORT.read_text())
        with np.load(R/f'tmp/inputs_{ym}.npz') as z:
            G=z['G'][take];L=z['L'][take];GI=z['GI'][take];B=z['B'][take]
        cal=m.dates_frame().query('ym==@ym');work=cal.daytype.eq('workday').to_numpy();comp=components(con,ym,small,'S1')
        if ym not in state.get('pv_months_applied',[]):
            blocks=pd.read_csv(D/f'residual_blocks_{ym}.csv');fac=np.zeros((100,len(cal),4,20))
            for row in blocks.itertuples():
                dates=pd.date_range(row.source_date,periods=row.length,freq='D')
                fac[row.draw,row.start_position:row.start_position+row.length]=np.nan_to_num(paths.reindex(dates).to_numpy()).reshape(row.length,4,20)
            vals,mu,sub=evaluate(G,L,GI,B,pi,mi,work,fac,np.full(len(muni),1.25),high,marginal,len(muni),comp[:,:,3])
            with np.load(D/f'residual_arrays_{ym}.npz') as oldarrays:
                reg=oldarrays['regional'].copy();municipal=oldarrays['municipal'].copy();dates=oldarrays['dates'].copy()
            olda=reg[:,:,:,2].copy();net=np.maximum(reg[:,:,:,0]-reg[:,:,:,1],0)
            vals[:,1]=(np.minimum(net,olda+sub[:,:,:,2])-np.minimum(net,olda)).sum(axis=(1,2))
            reg[:,:,:,2]+=sub[:,:,:,2];reg[:,:,:,4]+=sub[:,:,:,4];municipal+=mu
            totals=pd.read_csv(D/f'residual_totals_{ym}.csv').sort_values('draw')
            cols=['local_MWh','regional_MWh','ideal_MWh','uniform_increment_MWh','high_increment_MWh','marginal_increment_MWh','constrained_MWh']
            totals[cols]=totals[cols].to_numpy()+vals
            totals.to_csv(D/f'residual_totals_{ym}.csv',index=False)
            np.savez_compressed(D/f'residual_arrays_{ym}.npz',municipal=municipal,regional=reg,dates=dates)
            mark('pv_months_applied',ym);print('existing PV paths updated',ym,flush=True)
        if ym not in state.get('load_months_applied',[]):
            totals=pd.read_csv(D/f'load_residual_{ym}.csv',dtype={'source_month':str})
            for h,factor in factors.items():
                dw=np.ascontiguousarray(np.maximum(G-L*factor[:,None,None],0)[:,:,work]);u=np.minimum(dw,B[:,:,None]);local=float(u.sum())
                idx=totals.source_month.eq(h)
                totals.loc[idx,'local_MWh']+=local
                totals.loc[idx,'ideal_gain_MWh']+=supp.local_schedules(dw,B)[:,2].sum()-local
                constrained,_=lunch_schedule(dw,B,comp[:,:,3]);totals.loc[idx,'constrained_gain_MWh']+=constrained.sum()-local
                for rule,scales in [('uniform',np.full(len(muni),1.25)),('high',high),('marginal',marginal)]:
                    scale=np.where(mi>=0,scales[np.maximum(mi,0)],1)
                    totals.loc[idx,rule+'_increment_MWh']+=(np.minimum(dw,B[:,:,None]*scale[:,None,None])-u)[mi>=0].sum()
                priority=pd.read_csv(D/f'load_priority_{ym}_{h}.csv',dtype={'munic':str})
                gains=(np.minimum(dw,B[:,:,None]*1.05)-u).sum(axis=(1,2))
                add=pd.DataFrame({'munic':small.munic,'gain':gains}).groupby('munic').gain.sum()
                priority['marginal_MWh']+=priority.munic.map(add).fillna(0)
                region=pd.read_parquet(D/f'load_region_{ym}_{h}.parquet');region.date=region.date.astype(str)
                delta=B.sum(axis=0)[region.slot.to_numpy(int)-17]*region.date.isin(set(cal.loc[cal.daytype.eq('workday'),'date'])).to_numpy()
                net=np.maximum(region.G.to_numpy()-region.L0.to_numpy(),0)
                totals.loc[idx,'regional_MWh']+=(np.minimum(net,region.A.to_numpy()+delta)-np.minimum(net,region.A.to_numpy())).sum()
                region['A']+=delta
                priority.to_csv(D/f'load_priority_{ym}_{h}.csv',index=False);region.to_parquet(D/f'load_region_{ym}_{h}.parquet',index=False)
            totals.to_csv(D/f'load_residual_{ym}.csv',index=False);mark('load_months_applied',ym);print('existing load fields updated',ym,flush=True)
        del G,L,GI,B,comp;gc.collect()
    con.close()
    # Existing monthly outputs are present, so these calls only refresh aggregates.
    from propagate_pv_residuals import main as pv_summary
    pv_summary()
    runpy.run_path(str(R/'scripts/load_residual_scenarios.py'),run_name='__main__')
    from residual_storage import main as storage_summary
    storage_summary()

if __name__=='__main__':main()
