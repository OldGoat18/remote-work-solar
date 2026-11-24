"""Step 07 — canonical a_dwfh emit (thin wrapper).

Projects the v2 cell-native a_dwfh (step 06) to the canonical legacy schema, validates the window-month
subset + the S1 survey-central anchor (12534.9 MWh), and writes the single canonical product
configured asset
"""
from wlpr.io import paths
from wlpr.wfh_state import emit_adwfh

if __name__ == "__main__":
    paths.assert_not_source_tree()
    emit_adwfh.main()
