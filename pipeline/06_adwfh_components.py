"""Step 06 — WFH state-component-climate a_dwfh builder (thin wrapper).

All model logic lives in configured asset (state_layer / climate_response / component_loads /
scenario_downscaling / household_cells / materialize). Emits the v2 factored a_dwfh: the validated-window
drop-in (reproduces the survey-central S1 anchor 12534.9 MWh) plus the full-year evidence-tiered extension,
under configured asset
"""
from wlpr.io import paths
from wlpr.wfh_state import materialize

if __name__ == "__main__":
    paths.assert_not_source_tree()
    materialize.emit_main()        # validated window: factored scalars + weather + a_dwfh_mesh_mon_slot_v2 (S1-S10)
    materialize.emit_full_year()   # 12-month evidence-tiered a_dwfh (summer cooling; no annual-truth)
    print('[configured asset] v2 a_dwfh emitted -> configured asset (validated + full-year tiered)')
