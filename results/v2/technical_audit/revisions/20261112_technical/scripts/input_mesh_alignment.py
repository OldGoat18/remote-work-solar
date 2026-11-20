"""Join registered WFH templates by their unique mesh identity.

Scope prefecture labels describe the residential ledger; template prefecture
labels describe the existing participant/weather construction. They are retained
as separate attributes, not required to agree. No template values, population,
weather factors, municipal membership, G or L0 are reassigned by this repair.
Historical producer scripts and their released outputs remain unchanged.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(ROOT/'tools'))
from run_v47_halfhour_spatial import m, COMP

INPUT_VERSION='unique_mesh_template_join_20261113'
if not hasattr(m,'_original_month_source_sql'):
    m._original_month_source_sql=m.month_source_sql

def month_source_sql(*args,**kwargs):
    sql,evidence=m._original_month_source_sql(*args,**kwargs)
    sql=sql.replace('ON cast(w.mesh AS varchar)=r.mesh AND cast(w.pref AS varchar)=r.pref',
                    'ON cast(w.mesh AS varchar)=r.mesh')
    return sql,evidence

m.month_source_sql=month_source_sql

def components(con,ym,keys,scenario):
    direct=ym in m.DIRECT_MONTHS and scenario=='S1'
    source=m.WFH_REGISTERED if direct else ROOT/f'data/derived/wfh_state/a_dwfh_mesh_mon_slot_v2_fullyear/scenario={scenario}/part.parquet'
    fields=['dwfh']+COMP
    f=con.execute(f"SELECT cast(mesh AS varchar) mesh,slot,{','.join(fields)} FROM read_parquet('{source}') WHERE mon={int(ym[4:])} AND scenario_id='{scenario}'").df()
    if not keys.mesh.astype(str).is_unique or f.duplicated(['mesh','slot']).any():
        raise ValueError('Template alignment requires unique scope meshes and mesh-slot source rows.')
    ix=pd.Index(keys.mesh.astype(str)).get_indexer(f.mesh);good=ix>=0
    f=f.loc[good];ix=ix[good];t=f.slot.to_numpy(int)-17
    if np.any((t<0)|(t>=20)):raise ValueError('Registered template outside the existing twenty-slot support.')
    result=np.zeros((len(keys),20,len(fields)));result[ix,t]=f[fields].to_numpy()
    return result
