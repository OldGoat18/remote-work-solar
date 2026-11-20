"""Step 01 — parse the restricted self-consumption zip into the metered PV-owner ledger (thin wrapper)."""
from wlpr.io import paths
from wlpr.pv import owner_ledger

if __name__ == "__main__":
    paths.assert_not_source_tree()
    owner_ledger.main()
