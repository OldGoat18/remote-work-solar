"""Recover all 48 intervals on exactly the common complete mesh domain."""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys
import pandas as pd,duckdb,numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));O=BASE/_asset_location('asset_0002');SRC=ROOT/_asset_location('asset_0003')
sys.path.insert(0,str(Path(__file__).resolve().parent));from run_technical import m
keys=pd.read_csv(BASE.parent/_asset_location('asset_0012'),dtype={'mesh':str,'pref':str})
c=duckdb.connect();c.execute('set threads=2');c.execute("set memory_limit='3GB'");c.register('common_keys',keys[['pref','mesh']]);dates=m.dates_frame()
for ym in m.MONTHS:
 p=O/f'common48_{ym}.parquet'
 if p.exists():
  if ym not in m.DIRECT_MONTHS:
   q=pd.read_parquet(p);q.date=q.date.astype(str)
   with np.load(BASE/f"{_asset_location('asset_0017')}{ym}.npz") as z:
    B=z['B'];local=np.minimum(np.maximum(z['G']-z['L'],0),B[:,:,None]).sum(axis=0)
   active=q.slot.between(17,36);slots=q.loc[active,'slot'].to_numpy(int)-17;days=q.loc[active,'date'].str[-2:].astype(int).to_numpy()-1
   q.loc[active,'template_A']=B.sum(axis=0)[slots]
   q.loc[active,'local_template']=local[slots,days]
   q.to_parquet(p,index=False)
  continue
 rows=[]
 for pref in ['11','12','13','14']:
  sql,_=m.month_source_sql(c,ym,dates,SRC/'load_reference_mesh_daytype_slot.parquet',SRC/'owner_load_reference_pref_daytype_slot.parquet',SRC/'observed_stock_yield_kernel_full_year.parquet');sql=sql.replace(_asset_location('asset_0018'),f"{_asset_location('asset_0016')}{pref}");inc='coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000.0';sql=sql.replace('coalesce(w.dwfh,0) B',f'coalesce(w.dwfh,0) B, {inc} GI')
  if ym not in m.DIRECT_MONTHS:
   ref=m.qpath(SRC/'load_reference_mesh_daytype_slot.parquet');sql=sql.replace(f"FROM read_parquet('{ref}') r",f"FROM (SELECT * FROM read_parquet('{ref}') WHERE cast(pref AS varchar)='{pref}') r")
  f=c.execute(sql+''' SELECT x.date,x.slot,sum(x.G) G,sum(x.L0) L0,sum(x.GI) GI,sum(x.B) template_A,sum(greatest(x.G-x.L0,0)) positive_pool,sum(least(greatest(x.G-x.L0,0),x.B)) local_template FROM x JOIN common_keys k ON x.mesh=k.mesh AND x.pref=k.pref GROUP BY 1,2''').df();rows.append(f)
 pd.concat(rows).groupby(['date','slot'],as_index=False).sum().to_parquet(p,index=False);print('common48',ym,flush=True)
pd.concat([pd.read_parquet(O/f'common48_{ym}.parquet') for ym in m.MONTHS]).sort_values(['date','slot']).to_parquet(O/'common48.parquet',index=False)
