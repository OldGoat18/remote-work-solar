"""Step 12 — R3 spatial layer (thin wrapper): per-mesh matching fields + reallocation counterfactual +
benefit-distribution indicators. Each substep streams baseline_p0 mesh-day-first (red-line compliant)."""
from wlpr.io import paths
from wlpr.social import benefit_distribution, reallocation, spatial_matching_r3

if __name__ == "__main__":
    paths.assert_not_source_tree()
    spatial_matching_r3.main()
    reallocation.main()
    benefit_distribution.main()
