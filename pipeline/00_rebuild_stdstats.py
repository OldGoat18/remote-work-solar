"""Rebuild the low-voltage load input from the staged Standard Statistics ZIP."""
from wlpr.io import paths
from wlpr.io.stdstats import main


if __name__ == "__main__":
    paths.assert_not_source_tree()
    main()
