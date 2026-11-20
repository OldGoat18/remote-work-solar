"""Small independent oracles for the new scientific computations."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('annual',ROOT/'pipeline/27_v47_annual_hybrid_validation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_two_slot_energy_oracle():
    # One MWh charge, 81% round-trip efficiency => .81 delivered, .19 loss.
    s,t=m.solve_dispatch([1,0],[0,2],2,duration=.5,efficiency=.81,endpoint='empty')
    assert s['captured_GWh']*1000==pytest.approx(1)
    assert s['delivered_GWh']*1000==pytest.approx(.81)
    assert s['loss_GWh']*1000==pytest.approx(.19)
    assert t.soc_end_MWh.iloc[0]==pytest.approx(.9)
    assert t.soc_end_MWh.iloc[-1]==pytest.approx(0)

def test_terminal_state_not_last_slot_start():
    _,t=m.solve_dispatch([1,0],[0,2],2,duration=.5,efficiency=.81)
    assert t.soc_start_MWh.iloc[-1]>0
    assert t.soc_end_MWh.iloc[-1]==pytest.approx(t.soc_start_MWh.iloc[0])

def test_reject_simultaneous_pool_without_operational_modes():
    with pytest.raises(AssertionError): m.solve_dispatch([1],[1],1)

def test_no_load_sink_no_empty_endpoint_capture():
    s,_=m.solve_dispatch([1,1],[0,0],2,endpoint='empty')
    assert s['captured_GWh']==pytest.approx(0)

def test_daily_incidence_counts_days_and_retains_zero_days():
    d=pd.DataFrame({'date':['20230401']*48+['20230402']*48,
                    'season':['Spring']*96,'evidence_tier':['test']*96})
    net=np.ones(96);net[2:4]=-1;net[8:11]=-2
    r=m.daily_events(d,net)
    assert len(r)==2 and r.event_day.sum()==1
    assert r.event_count.tolist()==[2,0]
    assert r.total_negative_hours.tolist()==[2.5,0]
    assert r.negative_energy_MWh.tolist()==[8,0]
    assert r.max_negative_depth_MW.tolist()==[4,0]


sys.path.insert(0,str(ROOT/'tools'))
import run_v47_spatial_order_uncertainty as spatial
import run_v47_halfhour_spatial as halfhour


def test_exact_sorted_surplus_preserves_halfhour_null():
    # Monthly min would be 6 rather than 3 for mesh zero. The oracle must
    # preserve this aggregation gap and the complete donor slot trajectory.
    D=np.array([[[0.,3.,3.],[1.,0.,4.]],[[2.,0.,1.],[0.,2.,0.]]])
    B=np.array([[1.,2.],[4.,0.]])
    s=np.sort(D,axis=2)
    prefix=np.concatenate([np.zeros((2,2,1)),np.cumsum(s,axis=2)],axis=2)
    maps=np.array([[0,1],[1,0]],dtype=np.int32)
    got=halfhour.score_maps(s,prefix,B,maps)
    expected=[sum(min(D[i,t,j],B[p[i],t]) for i in range(2) for t in range(2) for j in range(3)) for p in maps]
    assert got==pytest.approx(expected)


def test_coalitions_relax_only_named_constraint():
    D=np.array([[3.,0.],[0.,3.]])
    B=np.array([[0.,1.],[1.,0.]])
    d=pd.DataFrame({'D_MWh':D.sum(axis=0),'B_MWh':B.sum(axis=0),
                    'A_1_0':np.minimum(D,B).sum(axis=0),'A_2_0':np.minimum(D,2*B).sum(axis=0)})
    mesh=pd.DataFrame({'pref':['a','a'],'mesh':['0','1'],'surplus_MWh':D.sum(axis=1),
                       'behavior_supply_MWh':B.sum(axis=1)})
    v=spatial.coalition_values(d,mesh,2.)
    assert v[0]==0 and v[1]==0  # Scaling cannot create co-location.
    assert v[2]==2 and v[4]==2 and v[6]==2
    assert v[3]==4 and v[5]==4 and v[7]==4


def test_fast_dispatch_independent_energy_oracle():
    r=spatial.greedy_summary(np.array([1.,0.]),np.array([0.,2.]),8.,.81)
    assert r[0]==pytest.approx(1.)
    assert r[1]==pytest.approx(.81)
    assert r[2]==pytest.approx(.19)
    assert r[3]==pytest.approx(0.)
