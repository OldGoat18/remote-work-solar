"""Preannounced lunch-only rule uses no PV forecast or realized PV information.
50% of each lunch slot delayed by 30 min; window-edge energy stays in place.
Destinations are slots 22..27; original lunch peak is the engineering cap.
If the capped +30 min rule is infeasible, retain the original profile.
Movable fraction is an explicit engineering scenario, not measured compliance.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,duckdb
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=Path(__file__).resolve().parents[1];O=BASE/'data'
sys.path.insert(0,str(BASE/'scripts'));from run_technical import m
from input_mesh_alignment import components
from lunch_constraints import fixed_profile
keys=pd.read_csv(BASE.parent/'20261111_supplemental/data/mesh_population_scope.csv',dtype={'mesh':str,'pref':str});cases=json.loads((O/'scenario_definitions.json').read_text());c=duckdb.connect();rows=[]
for ym in m.MONTHS:
 z=np.load(BASE/f'tmp/inputs_{ym}.npz');B=z['B'];C=components(c,ym,keys,'S1')[:,:,3];b=fixed_profile(B,C);assert np.max(abs(b.sum(axis=1)-B.sum(axis=1)))<1e-10
 wd=m.dates_frame().query('ym==@ym').daytype.eq('workday').to_numpy()
 for case in cases:
  L=z['L']*(1+case['load'] if ym not in m.DIRECT_MONTHS else 1);G=z['G']+case['pv']*z['GI'];D=np.maximum(G-L,0)
  anchored=np.minimum(D[:,:,wd],(B*case['a'])[:,:,None]).sum();fixed=np.minimum(D[:,:,wd],(b*case['a'])[:,:,None]).sum();rows.append(dict(case=case['case'],ym=ym,anchored_MWh=anchored,fixed_lunch_MWh=fixed,gain_MWh=fixed-anchored))
 print('fixed lunch',ym,flush=True)
pd.DataFrame(rows).to_csv(O/'fixed_lunch_monthly.csv',index=False);pd.DataFrame(rows).groupby('case').sum(numeric_only=True).to_csv(O/'fixed_lunch_summary.csv')
