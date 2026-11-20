"""Complete the existing 100 PV paths to 48 slots; paired common-target storage.

No resampling: residual_blocks_YYYYMM.csv is the authoritative date mapping.
Only added-rooftop generation receives the existing standardized owner residual.
The original twenty-slot paths and all nine deterministic scenarios are retained.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import duckdb

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
REV=Path(__file__).resolve().parents[1];D=REV/'data';SRC=ROOT/'outputs/figures/extra/legacy_34307_seasonal'
sys.path.insert(0,str(REV/'scripts'))
from run_technical import m
from residual_storage import design
from summarize_technical import minimum_storage

N=100;ETA=np.sqrt(.9);TARGET=100000.
PREFS=[11,12,13,14];MISSING=list(range(17))+list(range(37,48))
OBJECTIVES=['combined_delivery_100GWh','evening_delivery_100GWh']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def residual_source():
    path=ROOT/'outputs/figures/extra/v4.7/nc_completion/owner_year2_holdout_predictions.parquet'
    f=pd.read_parquet(path)
    f=f[f.model.eq('month_slot')&f.target.eq('generation_kWh')&f.pref.astype(int).isin(PREFS)].copy()
    f['dt']=pd.to_datetime(f.timestamp,format='%Y%m%d %H:%M')
    f['date']=f.dt.dt.normalize();f['slot']=f.dt.dt.hour*2+f.dt.dt.minute//30
    f['season']=f.dt.dt.month.map(m.SEASON_BY_MONTH);f['pref']=f.pref.astype(int)
    denom=f.groupby(['season','pref','slot']).prediction_kWh.transform('mean')
    zero=denom.eq(0);bad=zero&((f.observed_kWh!=0)|(f.prediction_kWh!=0))
    if bad.any() or f[['observed_kWh','prediction_kWh']].isna().any().any():
        raise ValueError('A residual cell lacks a valid normalization; do not impute it.')
    f['residual']=np.divide(f.observed_kWh-f.prediction_kWh,denom,out=np.zeros(len(f)),where=~zero)
    pools={}
    for season,z in f.groupby('season'):
        a=z.pivot(index='date',columns=['pref','slot'],values='residual').sort_index()
        a=a.reindex(columns=pd.MultiIndex.from_product([PREFS,range(48)]))
        if a.isna().any().any():raise ValueError('Incomplete 48-slot joint-prefecture residual date.')
        pools[season]=(a.index,a.to_numpy().reshape(-1,4,48))
    return pools,dict(path=str(path.relative_to(ROOT)),sha256=digest(path),rows=len(f),
        dates=int(f.date.nunique()),slots=48,prefectures=PREFS,zero_denominator_rows=int(zero.sum()),
        missing_residual_cells=0,period=[str(f.date.min().date()),str(f.date.max().date())])

def selected_factors(ym,pools):
    season=m.SEASON_BY_MONTH[int(ym[4:])];dates,err=pools[season]
    blocks=pd.read_csv(D/f'residual_blocks_{ym}.csv');nd=len(m.dates_frame().query('ym==@ym'))
    factors=np.full((N,nd,4,48),np.nan);coverage=np.zeros((N,nd),np.uint8)
    for row in blocks.itertuples():
        source=pd.date_range(row.source_date,periods=row.length)
        ix=dates.get_indexer(source)
        if np.any(ix<0):raise ValueError(f'Missing original source block: {ym}, {row.draw}')
        sl=slice(row.start_position,row.start_position+row.length)
        factors[row.draw,sl]=err[ix];coverage[row.draw,sl]+=1
    if not np.all(coverage==1) or not np.isfinite(factors).all():raise ValueError('Incomplete path mapping')
    return factors

def sum_clipped(G,GI,factors):
    """Exact sum_i max(G_i + GI_i*f,0), including mesh-level zero clipping.

    Sort G/GI once instead of expanding mesh x 100 paths. For a residual f,
    only ratios greater than -f contribute. This preserves the old operator.
    """
    active=GI>0;g=G[active];inc=GI[active]
    order=np.argsort(g/inc);ratio=(g/inc)[order]
    cs_g=np.r_[0.,np.cumsum(g[order])];cs_i=np.r_[0.,np.cumsum(inc[order])]
    cut=np.searchsorted(ratio,-factors,side='right')
    total=(cs_g[-1]-cs_g[cut])+factors*(cs_i[-1]-cs_i[cut])+np.maximum(G[~active],0).sum()
    unclipped=G.sum()+GI.sum()*factors
    return total,np.maximum(total-unclipped,0)

def extend_paths(pools):
    common=pd.read_parquet(D/'common48.parquet').sort_values(['date','slot']).reset_index(drop=True)
    common.date=common.date.astype(str)
    keys=pd.read_csv(REV.parent/'20261111_supplemental/data/mesh_population_scope.csv',dtype={'mesh':str,'pref':str})
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='3GB'")
    con.register('common_keys',keys[['mesh','pref']])
    all_g=[];evidence=[];reconciliation=[];offset=0
    for ym in m.MONTHS:
        fac=selected_factors(ym,pools);nd=fac.shape[1]
        base=common[common.date.str.startswith(ym)].G.to_numpy().reshape(nd,48)
        generated=np.broadcast_to(base,(N,nd,48)).copy();old=np.load(D/f'residual_arrays_{ym}.npz')
        generated[:,:,17:37]=old['regional'][:,:,:,0]
        extension=np.zeros((N,nd,48));original=np.zeros((nd,48));clip=np.zeros_like(extension)
        cache=D/f'pv_storage_extension_{ym}.npz'
        if cache.exists():
            z=np.load(cache);extension=z['generation'];original=z['reference'];clip=z['clipping']
        else:
            for pi,pref in enumerate(PREFS):
                sql,_=m.month_source_sql(con,ym,m.dates_frame(),SRC/'load_reference_mesh_daytype_slot.parquet',
                    SRC/'owner_load_reference_pref_daytype_slot.parquet',SRC/'observed_stock_yield_kernel_full_year.parquet')
                sql=sql.replace('pref=*',f'pref={pref}')
                inc='coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000.0'
                sql=sql.replace('coalesce(w.dwfh,0) B',f'coalesce(w.dwfh,0) B, {inc} GI')
                if ym not in m.DIRECT_MONTHS:
                    ref=m.qpath(SRC/'load_reference_mesh_daytype_slot.parquet')
                    sql=sql.replace(f"FROM read_parquet('{ref}') r",f"FROM (SELECT * FROM read_parquet('{ref}') WHERE cast(pref AS varchar)='{pref}') r")
                q=con.execute(sql+''' SELECT x.date,x.slot,x.G,x.GI FROM x
                    JOIN common_keys k ON x.mesh=k.mesh AND x.pref=k.pref
                    WHERE x.slot NOT BETWEEN 17 AND 36''').df()
                q['day']=q.date.astype(str).str[-2:].astype(int)-1
                for (day,slot),a in q.groupby(['day','slot'],sort=False):
                    v,c=sum_clipped(a.G.to_numpy(),a.GI.to_numpy(),fac[:,day,pi,slot])
                    extension[:,day,slot]+=v;clip[:,day,slot]+=c;original[day,slot]+=a.G.sum()
                print(f'extended {ym} pref {pref}: {len(q):,} rows',flush=True)
            np.savez_compressed(cache,generation=extension,reference=original,clipping=clip)
        error=float(np.max(np.abs(original[:,MISSING]-base[:,MISSING])))
        reconciliation.append(dict(ym=ym,reference_max_abs_difference_MWh=error,
            old_twenty_slots_retained=True,path_file_sha256=digest(D/f'residual_blocks_{ym}.csv')))
        if error>1e-6:raise ValueError(f'Generation reconstruction differs from current common48: {ym}, {error}')
        generated[:,:,MISSING]=extension[:,:,MISSING];all_g.append(generated)
        for k in range(N):
            early=np.maximum(generated[k,:,12:17]-common[common.date.str.startswith(ym)].L0.to_numpy().reshape(nd,48)[:,12:17],0).sum()
            evidence.append(dict(ym=ym,draw=k,early_generation_change_MWh=float((generated[k,:,12:17]-base[:,12:17]).sum()),
                early_charge_available_MWh=float(early),new_slot_clipping_MWh=float(clip[k].sum()),
                new_slot_generation_change_MWh=float((generated[k,:,MISSING]-base[:,MISSING].T).sum())))
        offset+=nd;print(f'completed full-window paths {ym}',flush=True)
    con.close();G=np.concatenate(all_g,axis=1)
    np.savez_compressed(D/'pv_residual_storage_full48.npz',generation=G,dates=common.date.to_numpy()[::48].astype('U8'),slots=np.arange(48),draws=np.arange(N))
    pd.DataFrame(evidence).to_csv(D/'pv_storage_window_exposure.csv',index=False)
    return common,G,reconciliation

def unlimited_delivery(Q,H):
    stock=0.;delivery=0.
    for q,h in zip(Q.ravel(),H.ravel()):
        stock+=ETA*q;out=min(h,ETA*stock);stock-=out/ETA;delivery+=out
    return delivery

def solve_paths(common,G):
    L=common.L0.to_numpy().reshape(-1,48);slots=np.arange(48)
    wd=common.date.isin(m.dates_frame().query("daytype=='workday'").date).to_numpy().reshape(-1,48)
    A=common.template_A.to_numpy().reshape(-1,48)*wd
    ch=(slots>=12)&(slots<30);dh=slots>=30;rows=[];daily=[]
    for k in range(N):
        direct=float(np.minimum(np.maximum(G[k]-L,0),A).sum())
        for participation in [0,1]:
            net=G[k]-L-participation*A;Q=np.maximum(net,0)*ch;H=np.maximum(-net,0)*dh
            max_delivery=unlimited_delivery(Q,H)
            for objective in OBJECTIVES:
                needed=max(0.,TARGET-participation*direct) if objective==OBJECTIVES[0] else TARGET
                ans=design(Q,H,needed)
                ans.update(draw=k,participation=participation,objective=objective,source='pv_full48',target_MWh=TARGET,
                    direct_MWh=participation*direct,required_storage_delivery_MWh=needed,
                    available_charge_MWh=float(Q.sum()),early_available_charge_MWh=float(Q[:,12:17].sum()),
                    unlimited_capacity_power_delivery_MWh=max_delivery,
                    charge_window='06:00-15:00',discharge_window='15:00-24:00',duration_h=4,
                    charge_efficiency=ETA,discharge_efficiency=ETA,initial_SOC_MWh=0.,terminal_SOC_MWh=0.)
                ans['infeasibility_reason']='' if ans['feasible'] else ('Insufficient chronological charge/discharge energy' if max_delivery<needed-1e-6 else 'Chronological capacity-power constraints')
                if ans['feasible']:
                    E=ans['capacity_MWh'];dc=.9*np.minimum(Q,E/8).sum(axis=1);dd=np.minimum(H,E/8).sum(axis=1)
                    if ans.get('solver','').startswith('daily-exact'):
                        delivered=np.minimum.reduce([np.full(len(Q),ETA*E),dc,dd])
                        # At the capacity optimum this equals the target; distribute only numerical surplus.
                        delivered*=needed/delivered.sum() if delivered.sum()>0 else 0
                        ans.update(capture_MWh=float(delivered.sum()/.9),delivery_MWh=float(delivered.sum()),loss_MWh=float(delivered.sum()/9),
                            SOC_equivalent_cycles=float(delivered.sum()/(ETA*E)) if E else 0,
                            nightly_full_store_emptying_min_margin_MWh=float((dd-ETA*E).min()),
                            capacity_limited_days=int(np.sum((ETA*E<=dc+1e-6)&(ETA*E<=dd+1e-6))),
                            charge_limited_days=int(np.sum(dc<ETA*E-1e-6)),
                            evening_demand_limited_days=int(np.sum(dd<np.minimum(dc,ETA*E)-1e-6)))
                        for d,value in enumerate(delivered):
                            daily.append(dict(draw=k,participation=participation,objective=objective,date=common.date.iloc[d*48],
                                delivery_MWh=value,capture_MWh=value/.9,loss_MWh=value/9,max_SOC_MWh=value/ETA,terminal_SOC_MWh=0.,
                                charge_limit_delivery_MWh=dc[d],discharge_limit_MWh=dd[d],capacity_limit_delivery_MWh=E*ETA))
                rows.append(ans)
        if k%10==0:print('storage full48 path',k,flush=True)
    result=pd.DataFrame(rows);result.to_csv(D/'pv_residual_storage_full48.csv',index=False)
    pd.DataFrame(daily).to_parquet(D/'pv_residual_storage_full48_daily.parquet',index=False)
    old=pd.read_csv(D/'residual_storage.csv').query("source=='pv'")
    metrics=['capacity_MWh','power_MW','capture_MWh','delivery_MWh','loss_MWh','direct_MWh']
    paired=[];comparisons=[]
    for label,frame in [('partial20',old),('full48',result)]:
        for objective,z in frame.groupby('objective'):
            no=z[z.participation.eq(0)].set_index('draw');yes=z[z.participation.eq(1)].set_index('draw')
            for k in range(N):
                row=dict(coverage=label,objective=objective,draw=k,no_A_feasible=bool(no.loc[k,'feasible']),with_A_feasible=bool(yes.loc[k,'feasible']))
                row['paired_feasible']=row['no_A_feasible'] and row['with_A_feasible']
                for metric in metrics:row['delta_'+metric]=yes.loc[k,metric]-no.loc[k,metric]
                paired.append(row)
    pair=pd.DataFrame(paired);pair.to_csv(D/'pv_residual_storage_window_pairs.csv',index=False)
    key=['draw','participation','objective'];merged=result.merge(old,on=key,suffixes=('_full48','_partial20'))
    for metric in metrics:merged['full_minus_partial_'+metric]=merged[metric+'_full48']-merged[metric+'_partial20']
    merged.to_csv(D/'pv_residual_storage_window_comparison.csv',index=False)
    summaries=[]
    for label,frame in [('partial20',old),('full48',result)]:
        for (objective,part),z in frame.groupby(['objective','participation']):
            for metric in metrics:
                vals=z.loc[z.feasible,metric]
                summaries.append(dict(coverage=label,objective=objective,participation=part,metric=metric,n=len(z),feasible=int(z.feasible.sum()),
                    **{name:float(vals.quantile(p)) for name,p in [('p05',.05),('median',.5),('p95',.95)]},minimum=float(vals.min()),maximum=float(vals.max())))
        for objective,z in pair[pair.coverage.eq(label)].groupby('objective'):
            for metric in metrics:
                vals=z.loc[z.paired_feasible,'delta_'+metric]
                summaries.append(dict(coverage=label,objective=objective,participation='with_minus_without',metric=metric,n=len(z),feasible=int(z.paired_feasible.sum()),
                    **{name:float(vals.quantile(p)) for name,p in [('p05',.05),('median',.5),('p95',.95)]},minimum=float(vals.min()),maximum=float(vals.max())))
    summary=pd.DataFrame(summaries);summary.to_csv(D/'pv_residual_storage_full48_summary.csv',index=False)
    result.loc[~result.feasible].to_csv(D/'pv_residual_storage_full48_infeasible.csv',index=False)
    return result,summary

def window_binding_days():
    """Explain the paired window effect at the earlier solved capacity."""
    c=pd.read_parquet(D/'common48.parquet').sort_values(['date','slot']);c.date=c.date.astype(str)
    z=np.load(D/'pv_residual_storage_full48.npz');G=z['generation'];dates=z['dates']
    L=c.L0.to_numpy().reshape(-1,48);base=c.G.to_numpy().reshape(-1,48)
    gate=c.date.isin(m.dates_frame().query("daytype=='workday'").date).to_numpy().reshape(-1,48)
    A=c.template_A.to_numpy().reshape(-1,48)*gate
    old=pd.read_csv(D/'residual_storage.csv').query("source=='pv'")
    slots=np.arange(48);ch=(slots>=12)&(slots<30);dis=slots>=30;rows=[]
    for k in range(N):
        previous=G[k].copy();previous[:,MISSING]=base[:,MISSING]
        for r in old[old.draw.eq(k)].itertuples():
            if not r.feasible:continue
            E=r.capacity_MWh;before=previous-L-r.participation*A;after=G[k]-L-r.participation*A
            Q0=np.maximum(before,0)*ch;Q1=np.maximum(after,0)*ch
            H0=np.maximum(-before,0)*dis;H1=np.maximum(-after,0)*dis
            input0=.9*np.minimum(Q0,E/8).sum(axis=1);input1=.9*np.minimum(Q1,E/8).sum(axis=1)
            evening0=np.minimum(H0,E/8).sum(axis=1);evening1=np.minimum(H1,E/8).sum(axis=1)
            bound0=np.minimum.reduce([np.full(len(dates),E*ETA),input0,evening0])
            bound1=np.minimum.reduce([np.full(len(dates),E*ETA),input1,evening1])
            early=Q1[:,12:17].sum(axis=1)
            for day in np.where((early>1e-8)|(np.abs(bound1-bound0)>1e-8))[0]:
                rows.append(dict(draw=k,objective=r.objective,participation=r.participation,date=dates[day],
                    earlier_capacity_MWh=E,early_charge_available_MWh=early[day],
                    old_daily_delivery_bound_MWh=bound0[day],full_daily_delivery_bound_MWh=bound1[day],
                    added_daily_delivery_bound_MWh=bound1[day]-bound0[day],
                    capacity_limited_before=bool(E*ETA<=min(input0[day],evening0[day])+1e-6),
                    daily_bound_changed=bool(abs(bound1[day]-bound0[day])>1e-7)))
    q=pd.DataFrame(rows);q.to_csv(D/'pv_storage_window_binding_days.csv',index=False)
    q.groupby(['objective','participation']).agg(early_opportunity_path_dates=('date','size'),
        capacity_limited_before=('capacity_limited_before','sum'),delivery_bound_changed=('daily_bound_changed','sum'),
        added_delivery_bound_MWh=('added_daily_delivery_bound_MWh','sum')).to_csv(D/'pv_storage_window_binding_summary.csv')
    return q

def main():
    pools,source=residual_source();common,G,recon=extend_paths(pools);result,summary=solve_paths(common,G)
    window_binding_days()
    manifest=dict(date='2026-11-14',source=source,draws=N,seed_of_original_paths=20261112,block_days=7,
        new_sampling=False,path_definition='Existing residual_blocks_YYYYMM.csv, same source date for all prefectures and 48 slots',
        normalization='(observed-predicted)/(season-prefecture-slot mean prediction)',
        transfer='Added rooftops only; mesh-level max(G+GI*residual,0) applied before aggregation',
        clipping_aggregation='Exact sorted G/GI tail sums, algebraically identical to mesh-by-path evaluation',
        load_and_A='Fixed current common48, A on 243 eligible dates; 366 chronological dates',
        interval_units='MWh per half-hour; P=E/4 h; half-hour input/output bound E/8 MWh',
        original_twenty_slot_values='Copied without recalculation',interpretation='Conditional PV residual-transfer range; no joint G/L/A confidence interval',
        reference_reconstruction=recon,solver_counts=result.solver.value_counts().to_dict(),
        targets_GWh=[100,100],target_definitions=OBJECTIVES,feasible=int(result.feasible.sum()),cases=len(result),
        infeasible=int((~result.feasible).sum()),nine_deterministic_cases='Retained unchanged',
        outputs=['pv_residual_storage_full48.npz','pv_residual_storage_full48.csv','pv_residual_storage_full48_summary.csv',
            'pv_residual_storage_full48_daily.parquet','pv_residual_storage_full48_infeasible.csv','pv_residual_storage_window_pairs.csv',
            'pv_residual_storage_window_comparison.csv','pv_storage_window_exposure.csv',
            'pv_storage_window_binding_days.csv','pv_storage_window_binding_summary.csv'],
        dispatch_construction='Each day input and output use proportional fractions of min(Q,E/8) and min(H,E/8); daily delivery/eta is peak SOC, daily terminal SOC zero')
    (D/'pv_residual_storage_full48_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary[(summary.coverage=='full48')&summary.metric.isin(['capacity_MWh','power_MW'])].to_string(index=False))

if __name__=='__main__':main()
