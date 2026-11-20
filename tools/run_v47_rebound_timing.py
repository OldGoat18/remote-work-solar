#!/usr/bin/env python3
"""Same rebound energy across alternative timing windows, system scope only."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures/extra/v4.7/nc_completion'
spec=importlib.util.spec_from_file_location('annual',ROOT/'pipeline/27_v47_annual_hybrid_validation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    d=m.load_ledger();slot=d.slot.to_numpy()
    # Half-open intervals. Other contains 08-09 and 22-24, exactly covers 48 slots.
    windows={'00-08':slot<16,'09-15':(slot>=18)&(slot<30),
             '15-18':(slot>=30)&(slot<36),'18-22':(slot>=36)&(slot<44)}
    windows['Other']=~np.logical_or.reduce(list(windows.values()))
    assert np.all(sum(w.astype(int) for w in windows.values())==1)
    budget=d.groupby('date').B_MWh.transform('sum').to_numpy()
    net=d.no_wfh_load_MWh.to_numpy()+d.B_MWh.to_numpy()-d.future_generation_MWh.to_numpy()
    base_events=m.daily_events(d,net)
    rows=[];trajectories=[]
    for rate in [0.,.1,.2,.4]:
        for name,mask in windows.items():
            count=pd.Series(mask,index=d.index).groupby(d.date).transform('sum').to_numpy()
            added=rate*budget*mask/count
            assert abs(added.sum()-rate*d.B_MWh.sum())<1e-6
            updated=net+added
            physical_extra=np.maximum(-net,0)-np.maximum(-updated,0)
            ev=m.daily_events(d,updated)
            s,tr=m.solve_dispatch(np.maximum(-updated,0),np.maximum(updated,0),217.408120632172)
            run_id=len(rows)
            rows.append(dict(run_id=run_id,rebound_rate=rate,window=name,
                             added_energy_GWh=added.sum()/1000,
                             expected_added_GWh=rate*d.B_MWh.sum()/1000,
                             additional_physical_PV_absorption_GWh=physical_extra.sum()/1000,
                             negative_dates=int(ev.event_day.sum()),
                             negative_energy_GWh=ev.negative_energy_MWh.sum()/1000,
                             mean_abs_halfhour_ramp_MW=ev.max_abs_halfhour_ramp_MW.mean(),
                             mean_ramp_15_to_18_change_MW=(ev.ramp_15_to_18_MW-base_events.ramp_15_to_18_MW).mean(),
                             credited_additional_WFH_relocation_GWh=0.,**s,
                             evidence='model sensitivity on mixed annual ledger; timing and rebound magnitude not observed',
                             storage='system-net service; 217.408 MWh,4h,90%,cyclic perfect foresight'))
            tr['run_id']=run_id;tr['date']=d.date.to_numpy();tr['slot']=slot
            tr['added_demand_MWh']=added;tr['additional_physical_absorption_MWh']=physical_extra
            trajectories.append(tr)
    pd.DataFrame(rows).to_csv(OUT/'rebound_equal_energy_timing.csv',index=False)
    pd.concat(trajectories,ignore_index=True).to_parquet(OUT/'rebound_timing_trajectories.parquet',index=False)

if __name__=='__main__':main()
