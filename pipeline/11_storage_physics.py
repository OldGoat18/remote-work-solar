"""Step 11 — storage physical layer L0/L1 (no price), built on the mesh × date × slot net-load (thin wrapper)."""
from wlpr.io import paths
from wlpr.scenario import storage_physics

if __name__ == "__main__":
    paths.assert_not_source_tree()
    storage_physics.main()
