"""Rebuild the missing mesh closure surface and pref-level LV twin from Standard Statistics."""
from wlpr.io import paths
from wlpr.io.closure_rebuild import main

if __name__ == "__main__":
    paths.assert_not_source_tree()
    main()
