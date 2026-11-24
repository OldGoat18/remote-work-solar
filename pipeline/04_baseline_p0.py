"""Step 04 — build the P0-corrected mesh × date × slot baseline ledger (thin wrapper)."""
from wlpr.baseline import ledger_p0
from wlpr.io import paths

if __name__ == "__main__":
    paths.assert_not_source_tree()
    ledger_p0.main()
