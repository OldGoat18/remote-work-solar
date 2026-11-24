"""Trace the actual owner/non-owner removal, clipping and re-addition."""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import sys,json,importlib.util
import numpy as np,pandas as pd,duckdb
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());BASE=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'));OUT=BASE/_asset_location('asset_0002')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_halfhour_spatial import m
from wlpr.wfh_state.state_layer import mesh_state
con=duckdb.connect();con.execute('set threads=2');con.execute("set memory_limit='2GB'")
cal=m.dates_frame();con.register('cal',cal)
scope=str(m.SCOPE);reg=str(m.WFH_REGISTERED)
rows=[];meshes=[]
for ym in sorted(m.DIRECT_MONTHS):
 p=OUT/f'clipping_{ym}.csv'
 if p.exists():continue
 sql=f"""WITH a AS (
 SELECT b.*,coalesce(w.dwfh,0) B,c.daytype,
 CASE WHEN b.L_owner_gross+b.L_nonowner_import>0 THEN least(b.L_owner_gross/(b.L_owner_gross+b.L_nonowner_import),1.0) ELSE 0 END owner_fraction
 FROM read_parquet('{m.BASE}{_asset_location('asset_0009')}{ym}{_asset_location('asset_0010')}') b
 JOIN read_parquet('{scope}') s ON cast(s.mesh as varchar)=cast(b.mesh as varchar) AND cast(s.pref as varchar)=cast(b.pref as varchar) AND s.load_status='positive_aggregate_load'
 LEFT JOIN read_parquet('{reg}') w ON cast(w.mesh as varchar)=cast(b.mesh as varchar) AND w.mon={int(ym[4:])} AND w.slot=b.slot AND w.scenario_id='S1'
 JOIN cal c ON c.date=cast(b.date as varchar)),
 q AS (SELECT *,L_owner_gross-E_wfh*owner_fraction ro,L_nonowner_import-E_wfh*(1-owner_fraction) rn FROM a),
 r AS (SELECT *,greatest(ro,0)+greatest(rn,0) L0,greatest(-ro,0)+greatest(-rn,0) excess FROM q)
 """
 common='''count(*) records,count(*) filter(where ro < -1e-10) owner_clip_count,count(*) filter(where rn < -1e-10) nonowner_clip_count,
 sum(greatest(-ro,0)) owner_clip_MWh,sum(greatest(-rn,0)) nonowner_clip_MWh,sum(excess) clipping_MWh,
 sum(E_wfh) removed_MWh,sum(B) template_added_MWh,sum(CASE WHEN daytype='workday' THEN B ELSE 0 END) calendar_added_MWh,
 sum(L0+B-L_all_gross_equiv) original_template_gap_MWh,
 sum(L0+CASE WHEN daytype='workday' THEN B ELSE 0 END-L_all_gross_equiv) calendar_reconstruction_gap_MWh,
 sum(least(greatest(G_pv-L0,0),B)-least(greatest(G_pv-(L_all_gross_equiv-E_wfh),0),B)) current_PV_service_clipping_effect_MWh'''
 con.execute(sql+f"SELECT pref,date,slot,daytype,{common} FROM r GROUP BY 1,2,3,4").df().to_csv(p,index=False)
 con.execute(sql+f"SELECT pref,mesh,{common} FROM r GROUP BY 1,2").df().to_parquet(OUT/f'clipping_mesh_{ym}.parquet',index=False)
 # Re-addition mismatch and a load-conserving template-removal counterfactual,
 # evaluated with the same expanded PV used in the manuscript.
 alpha=m.qpath(m.ALPHA);kernel=m.qpath(BASE.parents[2]/_asset_location('asset_0014'))
 kernel=m.qpath(ROOT/_asset_location('asset_0015'))
 f=con.execute(sql+f'''SELECT r.pref,r.date,r.slot,
 sum(least(greatest(r.G_pv+coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000-r.L0,0),r.B)) old_service,
 sum(least(greatest(r.G_pv+coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000-(r.L_all_gross_equiv-r.E_wfh),0),r.B)) unclipped_service,
 sum(least(greatest(r.G_pv+coalesce(a.increment_kw_a015,0)*coalesce(k.yield_kWh_per_kW_30min,0)/1000-greatest(r.L_all_gross_equiv-r.B,0),0),least(r.B,r.L_all_gross_equiv))) conserving_service,
 sum(greatest(r.B-r.L_all_gross_equiv,0)) template_above_load
 FROM r LEFT JOIN read_parquet('{alpha}') a ON cast(a.mesh as varchar)=cast(r.mesh as varchar) AND cast(a.pref as varchar)=cast(r.pref as varchar)
 LEFT JOIN read_parquet('{kernel}') k ON cast(k.pref as varchar)=cast(r.pref as varchar) AND k.ym='{ym}' AND k.slot=r.slot
 WHERE r.daytype='workday' GROUP BY 1,2,3''').df();f.to_csv(OUT/f'reconstruction_effect_{ym}.csv',index=False)
 print('audited',ym,flush=True)
pd.concat([pd.read_csv(OUT/f'clipping_{ym}.csv') for ym in sorted(m.DIRECT_MONTHS)]).to_csv(OUT/'clipping_time_distribution.csv',index=False)
pd.concat([pd.read_parquet(OUT/f'clipping_mesh_{ym}.parquet') for ym in sorted(m.DIRECT_MONTHS)]).groupby(['pref','mesh'],as_index=False).sum(numeric_only=True).to_parquet(OUT/'clipping_space_distribution.parquet',index=False)
pd.concat([pd.read_csv(OUT/f'reconstruction_effect_{ym}.csv') for ym in sorted(m.DIRECT_MONTHS)]).to_csv(OUT/'reconstruction_effect.csv',index=False)
state=mesh_state();keys=pd.read_csv(BASE.parent/_asset_location('asset_0012'),dtype={'mesh':str});state=state[state.mesh.isin(keys.mesh)]
state.to_parquet(OUT/'participant_state.parquet',index=False)
owner=pd.read_csv(ROOT/_asset_location('asset_0013'));owner=owner[owner.pref.isin([11,12,13,14]) & owner.month.between(202304,202403)]
counts=owner.groupby(['pref','month']).contracts.agg(['min','max']).reset_index();counts.to_csv(OUT/'owner_contract_coverage.csv',index=False)
alpha=pd.read_parquet(m.ALPHA);alpha=alpha[alpha.mesh.astype(str).isin(keys.mesh)]
summary=dict(participant_equivalents=float(state.N_persons.sum()),workers=float(state.workers.sum()),teleworkers=float(state.teleworkers.sum()),mean_daily_participation=float(state.N_persons.sum()/state.workers.sum()),day_share=.349,source_owner_contract_min=int(counts.groupby('month')['min'].sum().min()),source_owner_contract_max=int(counts.groupby('month')['max'].sum().max()),rooftop_potential_GW=float(alpha.pot_cap_kw_B_IC.sum()/1e6),current_GW=float(alpha.current_cap_kw.sum()/1e6),future_GW=float(alpha.future_cap_kw_a015.sum()/1e6),increment_GW=float(alpha.increment_kw_a015.sum()/1e6),household_sample='Source is aggregate LV statistics and PV-owner contracts, not an individual household panel; unique individuals/non-owner household count not supplied.')
(OUT/'input_provenance.json').write_text(json.dumps(summary,indent=2));print(summary)
