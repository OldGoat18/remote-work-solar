"""Canonical a_dwfh emitter — projects the v2 cell-native engine output to the canonical schema.

Ports ONLY the materialize_a_dwfh() function from the source
`configured asset`. The BTM-twin half of that builder (the per-row PV
self-consumption SQL, the eta / kappa / theta-marginal overlay, and its future-PV ledger dependency) is
EXCLUDED — it is the PV/BTM user-state twin, out of scope for this reproduction.

This reads the v2 a_dwfh per-scenario parquet (configured asset),
projects it to the canonical legacy schema (scenario_id, mesh, mon, slot, dwfh, c_*), validates
  - months ⊆ the validated window {10,11,12,1,2}
  - S1 window total ≈ 12534.9 MWh (the survey-central anchor)
and writes the single canonical product configured asset

    PYTHONPATH=src python -m wlpr.wfh_state.emit_adwfh [S1 S8 ...]
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os
import sys

import pandas as pd

from wlpr.io.paths import derived_path
from wlpr.wfh_state.component_loads import COMP
from wlpr.wfh_state.scenario_downscaling import SCEN

# Cell-native a_dwfh emission is the active source for this model component.
V2_ADWFH_DIR = derived_path(_asset_location('asset_0055'), _asset_location('asset_0071'))
ADWFH = derived_path(_asset_location('asset_0055'), "a_dwfh_mesh_mon_slot.parquet")
OBS_MONTHS = [10, 11, 12, 1, 2]
SURVEY_CENTRAL_S1_TOTAL_MWH = 12534.9  # S1 a_dwfh window total (survey-central anchor); v2 reproduces it bit-for-bit
ALL_SCEN = [s[0] for s in SCEN]


def materialize_a_dwfh(scenarios):
    """Source A's per-(scenario,mesh,mon,slot) dWFH from the v2 cell-native state-component-climate engine
    (configured asset emit), projecting to the canonical ADWFH schema so every downstream consumer is unchanged.

    For S1 (the survey-central anchor) and the uniform-scaling scenarios (S2/S3 frequency, S6/S7) the v2 a_dwfh
    is bit-identical to the retired in-line foundation materialization. For the industry/policy scenarios
    (S4/S5/S8/S9/S10) v2 applies the G1 industry-resolved within-prefecture downscaling -- it uses the T001193
    industry signal the old global multiplier ignored -- so those a_dwfh differ slightly (more principled,
    within the LV-unidentifiable WFH spatial-allocation band)."""
    frames = []
    for sid in scenarios:
        part = os.path.join(V2_ADWFH_DIR, f"{_asset_location('asset_0153')}{sid}", "part.parquet")
        if not os.path.exists(part):
            raise SystemExit(f"cutover: v2 a_dwfh missing for {sid} ({part}); run pipeline/06 first "
                             "(emit_main builds it before this emitter)")
        d = pd.read_parquet(part)
        d["scenario_id"] = sid
        frames.append(d[["scenario_id", "mesh", "mon", "slot", "dwfh", *[f"c_{k}" for k in COMP]]])
    out = pd.concat(frames, ignore_index=True)
    bad = set(int(x) for x in out.mon.unique()) - set(OBS_MONTHS)
    assert not bad, f"v2 a_dwfh carries non-window months {sorted(bad)} (expected validated window {OBS_MONTHS})"
    if "S1" in scenarios:                 # anchor guard: S1 must reproduce the survey-central window total
        s1tot = float(out[out.scenario_id == "S1"].dwfh.sum())
        assert abs(s1tot - SURVEY_CENTRAL_S1_TOTAL_MWH) < 1.0, \
            f"v2 S1 a_dwfh total {s1tot:.1f} MWh != survey-central anchor {SURVEY_CENTRAL_S1_TOTAL_MWH}"
    os.makedirs(os.path.dirname(ADWFH), exist_ok=True)
    out.to_parquet(ADWFH, index=False)
    print(f"[a_dwfh<-v2] {len(out):,} rows ({len(scenarios)} scen x {len(OBS_MONTHS)} mon) "
          f"from the v2 cell-native engine -> {ADWFH}", flush=True)
    return out


def main():
    scenarios = [a for a in sys.argv[1:] if a.startswith("S")] or ALL_SCEN
    print(f"=== canonical a_dwfh emit (v2 cell-native engine -> legacy schema) | scenarios={scenarios} ===")
    materialize_a_dwfh(scenarios)


if __name__ == "__main__":
    main()
