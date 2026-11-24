"""Step 14 — mobility SI validation of the WFH twin (thin wrapper).

Default: runs the validation (reads the two frozen diurnal-presence parquets + derived deps, writes figures +
scorecard, asserts the four gates). Set WLPR_MOBILITY_EXTRACT=1 to first run the heavy one-time raw streaming
pass over configured asset (idempotent / skip-if-exists) that produces those presence parquets.
"""
import os

from wlpr.io import paths
from wlpr.mobility import extract_presence, wfh_validation

if __name__ == "__main__":
    paths.assert_not_source_tree()
    if os.environ.get("WLPR_MOBILITY_EXTRACT") == "1":
        extract_presence.main()
    wfh_validation.main()
