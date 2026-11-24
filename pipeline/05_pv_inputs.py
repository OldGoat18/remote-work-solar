"""Step 05 — build the PV scenario inputs: obs-stock yield kernel, alpha rooftop capacity, WFH spatial weight.

Thin wrapper running the three PV-input builders in order. owner_ledger (step 01) and the upstream derived
kernel/REPOS/mobility products must already exist.
"""
from wlpr.io import paths
from wlpr.pv import alpha_rooftop, obs_stock_kernel, wfh_spatial_weight

if __name__ == "__main__":
    paths.assert_not_source_tree()
    obs_stock_kernel.main()
    alpha_rooftop.main()
    wfh_spatial_weight.main()
