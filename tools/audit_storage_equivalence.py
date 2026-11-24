#!/usr/bin/env python3
"""Check the saved legacy trajectory and recompute a restricted same-target bracket."""

from wlpr.io.bindings import asset as _asset_location
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/_asset_location('asset_0105')
spec=importlib.util.spec_from_file_location('annual',ROOT/'pipeline/27_annual_hybrid_validation.py')
annual=importlib.util.module_from_spec(spec);spec.loader.exec_module(annual)

def main():
    legacy=ROOT/_asset_location('asset_0192')
    p=ROOT/_asset_location('asset_0193')
    tr=pd.read_csv(p)
    simultaneous=np.minimum(tr.charge_MWh,tr.discharge_MWh)
    audit=dict(source=str(p.relative_to(ROOT)),slots=len(tr),
               simultaneous_positive_slots=int((simultaneous>1e-8).sum()),
               simultaneous_energy_GWh=float(simultaneous.sum()/1000),
               charged_GWh=float(tr.charge_MWh.sum()/1000),
               interpretation='Relaxed pooled trajectory allows contemporaneous charge and discharge. This is not a verified stationary single-battery or feeder deployment.',
               bound_direction='At fixed capacity perfect foresight bounds capture above; inverse required capacity bounds below. A feasible heuristic bounds minimum required capacity above within the same model.')
    d=pd.read_parquet(legacy/'behavior_date_slot_fields.parquet').sort_values(['date','slot'])
    target=float(pd.read_csv(legacy/'storage_equivalence.csv').target_behavior_relief_GWh.iloc[0])
    surplus=d.D.to_numpy()*d.slot.between(12,29).to_numpy()
    deficit=np.maximum(d.net_load_MW.to_numpy(),0)*.5*(d.slot.to_numpy()>=30)
    rows=[]
    for policy,endpoint in [('perfect_foresight','cyclic'),('greedy','empty_start_free_end')]:
        lo,hi=0.,106.027066707611
        while annual.solve_dispatch(surplus,deficit,hi,policy=policy,endpoint=endpoint)[0]['captured_GWh']<target:
            hi*=2
            if hi>1e7: raise RuntimeError('Target unattainable under fixed windows')
        for _ in range(28):
            mid=(lo+hi)/2
            s,_=annual.solve_dispatch(surplus,deficit,mid,policy=policy,endpoint=endpoint)
            if s['captured_GWh']>=target: hi=mid
            else:lo=mid
        s,t=annual.solve_dispatch(surplus,deficit,hi,policy=policy,endpoint=endpoint)
        assert abs(s['captured_GWh']-target)<1e-5
        assert s['cyclic_error_MWh']<1e-6,'Endpoint mismatch: do not report capacity bracket'
        rows.append(dict(policy=policy,capacity_MWh=hi,power_MW=hi/4,target_GWh=target,
                         capacity_bracket_width_MWh=hi-lo,**s,
                         scope='152-date local-surplus pool; charge 06:00-15:00, discharge 15:00-24:00; no network',
                         comparison_boundary='Restricted operating model; not a replacement for unrestricted or feeder-optimal capacity'))
        t['date']=d.date.to_numpy();t['slot']=d.slot.to_numpy()
        t.to_parquet(OUT/f'restricted_equivalence_{policy}.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'restricted_storage_equivalence.csv',index=False)
    (OUT/'legacy_storage_physics_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2));print(pd.DataFrame(rows)[['policy','capacity_MWh','captured_GWh']].to_string(index=False))

if __name__=='__main__':main()
