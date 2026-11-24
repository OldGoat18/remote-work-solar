"""Step 02 — build the grid × date × slot load-fit component product (thin wrapper)."""
import os

from wlpr.baseline import component_product
from wlpr.io import paths

if __name__ == "__main__":
    paths.assert_not_source_tree()
    component_product.main(os.environ.get("WLPR_MODE", "full"))
