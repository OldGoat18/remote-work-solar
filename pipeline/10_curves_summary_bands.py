"""Step 10 — validated-window scenario curves + mesh-grain summary + partial-ID increment bands (thin wrapper).

Runs (in dependency order): window load curves, the mesh-grain net-load/reverse summary, the W1
forward-driver conformal/fold uncertainty scorecards, and the Layer-B partial-identification relief
increment band (which reads the W1 split-conformal q80 + fold coefficient stability).
"""
from wlpr.io import paths
from wlpr.scenario import increment_bands, mesh_summary, window_curves
from wlpr.validation import w1_conformal_fold

if __name__ == "__main__":
    paths.assert_not_source_tree()
    window_curves.main()
    mesh_summary.main()
    w1_conformal_fold.build_w1_conformal_fold()   # produces the conformal-coverage + fold-stability scorecards
    increment_bands.main()
