"""Step 09 — the A-caliber net-load / relief / SCR / shape index (S1-S10, BTM-free; thin wrapper)."""
from wlpr.io import paths
from wlpr.scenario import indices_s1s10

if __name__ == "__main__":
    paths.assert_not_source_tree()
    indices_s1s10.main()
