"""Step 08 — mesh × date × slot net-load reconstruction (thin wrapper)."""
from wlpr.io import paths
from wlpr.scenario import netload_mesh_date

if __name__ == "__main__":
    paths.assert_not_source_tree()
    netload_mesh_date.main()
