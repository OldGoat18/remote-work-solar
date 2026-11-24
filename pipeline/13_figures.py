"""Step 13 — build the three RESULT figures (Fig 1 load reshaping, Fig 2 net-load/PV, Fig 3 spatial matching).

Thin wrapper: guards against running inside the source tree, then runs each builder's main() (each writes its
figure outputs using private bindings and asserts its own gates)."""
from wlpr.figures import load_reshaping, netload_pv, spatial_matching
from wlpr.io import paths

if __name__ == "__main__":
    paths.assert_not_source_tree()
    load_reshaping.main()
    netload_pv.main()
    spatial_matching.main()
