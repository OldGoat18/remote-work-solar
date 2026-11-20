from pathlib import Path
import sys
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];O=BASE/'data';sys.path.insert(0,str(BASE/'scripts'));from run_technical import m
q=pd.read_parquet(O/'common48.parquet');wd=set(m.dates_frame().query("daytype=='workday'").date);rows=[]
for date,g in q.groupby('date'):
 if str(date) not in wd:continue
 g=g.sort_values('slot');B=g.template_A.to_numpy();net=np.maximum(g.G-g.L0,0).to_numpy();pool=g.positive_pool.to_numpy();win=np.arange(18,34);fix=np.ones(48,bool);fix[win]=False
 for scope,S in [('positive_pool',pool),('regional_net',net)]:
  fixed=np.minimum(S[fix],B[fix]).sum();rows.append(dict(date=date,scope=scope,anchored_MWh=np.minimum(S,B).sum(),ideal_MWh=fixed+np.minimum(np.sort(S[win]),np.sort(B[win])).sum(),permutation_MWh=fixed+np.minimum(S[win,None],B[None,win]).sum()/16))
d=pd.DataFrame(rows);d.to_csv(O/'regional_schedule_daily.csv',index=False);d.groupby('scope').sum(numeric_only=True).to_csv(O/'regional_schedule_summary.csv');print(d.groupby('scope').sum(numeric_only=True))
