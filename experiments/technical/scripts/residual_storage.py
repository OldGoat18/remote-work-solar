"""Equal-target capacity propagation for empirical PV paths / load-bias cases.
Uses the exact daily throughput form where charging precedes discharge and each
night can empty a full store; verifies this condition at the solved capacity.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));O=BASE/_asset_location('asset_0002')
sys.path.insert(0,str(Path(__file__).resolve().parent));from run_technical import m
from summarize_technical import minimum_storage
eta=np.sqrt(.9)

def design(Q,H,delivered):
 Q=Q.reshape(366,48);H=H.reshape(366,48)
 if delivered<0:return dict(feasible=True,capacity_MWh=0,power_MW=0,delivery_MWh=0,capture_MWh=0,loss_MWh=0)
 if .9*Q.sum()<delivered-1e-6:return dict(feasible=False,capacity_MWh=np.nan,power_MW=np.nan)
 def supply(E):return np.minimum.reduce([np.full(366,E*eta),.9*np.minimum(Q,E/8).sum(axis=1),np.minimum(H,E/8).sum(axis=1)]).sum()
 lo=0.;hi=1000.
 while supply(hi)<delivered:
  hi*=2
  if hi>1e8:return dict(feasible=False,capacity_MWh=np.nan,power_MW=np.nan)
 for _ in range(45):
  mid=(lo+hi)/2
  if supply(mid)<delivered:lo=mid
  else:hi=mid
 # With full-store discharge available every night, no interday carry can
 # increase annual throughput; daily formula is then the chronological optimum.
 full_empty=np.minimum(H,hi/8).sum(axis=1)>=hi*eta-1e-6
 if not full_empty.all():
  ans,_=minimum_storage(Q.ravel(),H.ravel(),delivered,'delivery');ans['solver']='chronological LP';return ans
 return dict(feasible=True,capacity_MWh=hi,power_MW=hi/4,delivery_MWh=delivered,capture_MWh=delivered/.9,loss_MWh=delivered/.9-delivered,solver='daily-exact, nightly full-store emptying verified')

def main():
 c=pd.read_parquet(O/'common48.parquet').sort_values(['date','slot']).reset_index(drop=True);dates=c.date.astype(str).to_numpy();slots=c.slot.to_numpy();idx=pd.MultiIndex.from_arrays([dates,slots]);wd=np.isin(dates,m.dates_frame().query("daytype=='workday'").date);baseG=c.G.to_numpy();baseL=c.L0.to_numpy();A=c.template_A.to_numpy()*wd;ch=(slots>=12)&(slots<30);dh=slots>=30;rows=[]
 for mode in ['pv','load']:
  for k in (range(100) if mode=='pv' else ['202310','202311','202312','202401','202402']):
   G=baseG.copy();L=baseL.copy()
   for ym in m.MONTHS:
    if mode=='pv':
     z=np.load(O/f'residual_arrays_{ym}.npz');r=z['regional'][k];dd=z['dates'];ix=idx.get_indexer(pd.MultiIndex.from_arrays([np.repeat(dd,20),np.tile(np.arange(17,37),len(dd))]));G[ix]=r[:,:,0].ravel();L[ix]=r[:,:,1].ravel()
    else:
     q=pd.read_parquet(O/f'load_region_{ym}_{k}.parquet');ix=idx.get_indexer(pd.MultiIndex.from_frame(q[['date','slot']]));G[ix]=q.G;L[ix]=q.L0
   direct=np.minimum(np.maximum(G-L,0),A).sum()
   for a in [0,1]:
    net=G-L-a*A;Q=np.maximum(net,0)*ch;H=np.maximum(-net,0)*dh
    for objective,target in [('combined_delivery_100GWh',100000-a*direct),('evening_delivery_100GWh',100000)]:
     ans=design(Q,H,target);ans.update(source=mode,draw=k,participation=a,objective=objective,direct_MWh=a*direct);rows.append(ans)
   if int(k)%10==0:print(mode,k,flush=True)
  pd.DataFrame(rows).to_csv(O/'residual_storage.csv',index=False)
 q=pd.DataFrame(rows);out=[]
 for (source,obj),g in q.groupby(['source','objective']):
  p=g.pivot(index='draw',columns='participation',values='capacity_MWh');delta=p[1]-p[0];out.append(dict(source=source,objective=obj,n=len(p),paired_feasible=int(delta.notna().sum()),delta_p05=delta.quantile(.05),delta_median=delta.median(),delta_p95=delta.quantile(.95),delta_min=delta.min(),delta_max=delta.max()))
 pd.DataFrame(out).to_csv(O/'residual_storage_summary.csv',index=False)
if __name__=='__main__':main()
