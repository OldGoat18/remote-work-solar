"""Season-stratified seven-day PV residual block transfer, separately from load.
Observed year-two owner residual paths retain common prefecture/date/slot errors.
Transfer to future rooftops is an explicit scenario assumption; percentile ranges
are conditional propagated residual ranges, not joint G/L/A confidence intervals.
"""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));OUT=BASE/_asset_location('asset_0002')
sys.path.insert(0,str(Path(__file__).resolve().parent));from run_technical import m
from lunch_constraints import candidate_lunch,one_gain
N=100;SEED=20261112
@njit(parallel=True)
def evaluate(G,L,GI,B,pi,mi,work,fac,uniform,high,marg,nm,lunch,with_fixed=False):
 profiles,valid,_,_=candidate_lunch(lunch)
 candidates=B[:,None,5:11]-lunch[:,None,5:11]+profiles
 candidates[:,2,:]=B[:,5:11]
 nd=G.shape[2];tot=np.zeros((len(fac),8 if with_fixed else 7));municipal=np.zeros((len(fac),nm));reg=np.zeros((len(fac),nd,20,5))
 for k in prange(len(fac)):
  for d in range(nd):
   gtot=np.zeros(20);ltot=np.zeros(20);btot=np.zeros(20);ptot=np.zeros(20);utot=np.zeros(20)
   for i in range(G.shape[0]):
    D=np.empty(20)
    for t in range(20):
     g=max(G[i,t,d]+GI[i,t,d]*fac[k,d,pi[i],t],0.);D[t]=max(g-L[i,t,d],0);gtot[t]+=g;ltot[t]+=L[i,t,d];ptot[t]+=D[t]
     if work[d]:
      b=B[i,t];u=min(D[t],b);tot[k,0]+=u;btot[t]+=b;utot[t]+=u
      j=mi[i]
      if j>=0:
       municipal[k,j]+=min(D[t],b*1.05)-u
       tot[k,3]+=min(D[t],b*uniform[j])-u;tot[k,4]+=min(D[t],b*high[j])-u;tot[k,5]+=min(D[t],b*marg[j])-u
    if work[d]:
     sd=np.sort(D[1:17]);sb=np.sort(B[i,1:17]);v=0.
     for t in range(16):v+=min(sd[t],sb[t])
     for t in [0,17,18,19]:v+=min(D[t],B[i,t])
     tot[k,2]+=v
     gain,fixed_gain,_=one_gain(D[5:11],candidates[i],valid[i],3 if valid[i,3] else 2)
     original=0.
     for t in range(20):original+=min(D[t],B[i,t])
     tot[k,6]+=original+gain
     if with_fixed:tot[k,7]+=original+fixed_gain
   for t in range(20):
    a=btot[t] if work[d] else 0.;tot[k,1]+=min(max(gtot[t]-ltot[t],0.),a)
    reg[k,d,t,0]=gtot[t];reg[k,d,t,1]=ltot[t];reg[k,d,t,2]=a;reg[k,d,t,3]=ptot[t];reg[k,d,t,4]=utot[t]
 return tot,municipal,reg

def main():
 set_num_threads(4);rng=np.random.default_rng(SEED)
 keys=pd.read_csv(BASE.parent/_asset_location('asset_0012'),dtype={'mesh':str,'munic':str});muni=pd.read_csv(BASE.parent/_asset_location('asset_0025'),dtype={'munic':str});mi=keys.munic.map(dict(zip(muni.munic,range(len(muni))))).fillna(-1).to_numpy(np.int64);pi=keys.pref.map({11:0,12:1,13:2,14:3}).to_numpy(np.int64)
 oldalloc=pd.read_csv(BASE.parent/_asset_location('asset_0026'),dtype={'munic':str});sc=pd.read_csv(BASE.parent/_asset_location('asset_0027'));sid=sc.query("fraction==.25 and target=='high_surplus_q4' and rule=='TE'").scenario_id.iloc[0]
 high=oldalloc[oldalloc.scenario_id.eq(sid)].set_index('munic').reindex(muni.munic).scale.to_numpy();mg=pd.read_csv(OUT/'marginal_allocations.csv',dtype={'munic':str}).query("case=='L+0_P+0'").set_index('munic').reindex(muni.munic).x.to_numpy()+1
 raw=pd.read_parquet(ROOT/_asset_location('asset_0019'));raw=raw[raw.model.eq('month_slot') & raw.target.eq('generation_kWh') & raw.pref.astype(int).isin([11,12,13,14])].copy();raw['dt']=pd.to_datetime(raw.timestamp,format='%Y%m%d %H:%M');raw['date']=raw.dt.dt.normalize();raw['slot']=raw.dt.dt.hour*2+raw.dt.dt.minute//30;raw['season']=raw.dt.dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'});raw=raw[raw.slot.between(17,36)]
 # Within each season+pref+slot normalize by the mean predicted yield.
 denom=raw.groupby(['season','pref','slot']).prediction_kWh.transform('mean');raw['relative_error']=(raw.observed_kWh-raw.prediction_kWh)/denom
 pools={}
 for season,ff in raw.groupby('season'):
  x=ff.pivot(index='date',columns=['pref','slot'],values='relative_error').sort_index();x=x.reindex(columns=pd.MultiIndex.from_product([[11,12,13,14],range(17,37)]));pools[season]=(x.index,np.nan_to_num(x.to_numpy()).reshape(-1,4,20))
 rows=[];municipal_total=np.zeros((N,len(muni)));reglist=[]
 import duckdb
 from input_mesh_alignment import components
 con=duckdb.connect()
 for ym in m.MONTHS:
  done=OUT/f'residual_totals_{ym}.csv'
  if done.exists():continue
  z=np.load(BASE/f"{_asset_location('asset_0017')}{ym}.npz");G=z['G'];L=z['L'];GI=z['GI'];B=z['B'];cal=m.dates_frame().query(_asset_location('asset_0020'));work=cal.daytype.eq('workday').to_numpy();season=raw.loc[raw.dt.dt.month.eq(int(ym[4:])), 'season'].iloc[0];dates,err=pools[season];nd=len(cal);fac=np.zeros((N,nd,4,20));starts=[s for s in range(len(dates)-6) if (dates[s+6]-dates[s]).days==6]
  sample=[]
  for k in range(N):
   pos=0
   while pos<nd:
    st=int(rng.choice(starts));n=min(7,nd-pos);fac[k,pos:pos+n]=err[st:st+n];sample.append(dict(ym=ym,draw=k,start_position=pos,source_date=str(dates[st].date()),length=n));pos+=n
  comp=components(con,ym,keys.astype({'pref':str}),'S1')
  vals,mu,reg=evaluate(G,L,GI,B,pi,mi,work,fac,np.full(len(muni),1.25),high,mg,len(muni),comp[:,:,3],True);pd.DataFrame(vals,columns=['local_MWh','regional_MWh','ideal_MWh','uniform_increment_MWh','high_increment_MWh','marginal_increment_MWh','constrained_MWh','fixed_lunch_MWh']).assign(draw=np.arange(N),ym=ym).to_csv(done,index=False)
  np.savez_compressed(OUT/f'residual_arrays_{ym}.npz',municipal=mu,regional=reg,dates=cal.date.to_numpy().astype('U8'))
  pd.DataFrame(sample).to_csv(OUT/f'residual_blocks_{ym}.csv',index=False);print('residual',ym,flush=True)
 q=pd.concat([pd.read_csv(OUT/f'residual_totals_{ym}.csv') for ym in m.MONTHS]);q=q.groupby('draw').sum(numeric_only=True).drop(columns='ym');q['ideal_gain_MWh']=q.ideal_MWh-q.local_MWh;q['constrained_gain_MWh']=q.constrained_MWh-q.local_MWh;q['fixed_gain_MWh']=q.fixed_lunch_MWh-q.local_MWh;q['high_advantage_percent']=100*(q.high_increment_MWh/q.uniform_increment_MWh-1);q['marginal_advantage_percent']=100*(q.marginal_increment_MWh/q.uniform_increment_MWh-1);q.to_csv(OUT/'pv_residual_propagation.csv')
 mu=sum(np.load(OUT/f'residual_arrays_{ym}.npz')['municipal'] for ym in m.MONTHS);rate=mu/(.05*243*muni.N_persons.to_numpy());top=np.argsort(-rate,axis=1)[:,:20];pd.DataFrame(dict(munic=muni.munic,top20_frequency=[np.mean(np.any(top==j,axis=1)) for j in range(len(muni))],mean_marginal_kWh_per_day=rate.mean(axis=0)*1000)).to_csv(OUT/'pv_residual_priority_frequency.csv',index=False)
 q.quantile([.05,.5,.95]).to_csv(OUT/'pv_residual_percentiles.csv')
 previous=json.loads((OUT/'pv_residual_manifest.json').read_text()) if (OUT/'pv_residual_manifest.json').exists() else {}
 previous.update(dict(draws=N,seed=SEED,block_days=7,stratification='season',spatial_dependence='same observed four-prefecture date path jointly resampled',cross_input_dependence='PV only; no artificial pairing with load validation in another year',transfer='year-two owner aggregate standardized residual transferred coherently to rooftop increment within prefecture; transfer assumption, not mesh validation',interpretation='conditional PV residual-transfer ranges; no joint confidence coverage',normalization='observed minus predicted divided by season/prefecture/slot mean prediction',load_and_A='fixed',source=str(ROOT/_asset_location('asset_0019'))))
 previous['scheduling_definition']='strict_lunch_window_peak_20261114'
 (OUT/'pv_residual_manifest.json').write_text(json.dumps(previous,indent=2))
if __name__=='__main__':main()
