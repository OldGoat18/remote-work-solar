"""Step 03 — survey-anchored cross-pref WFH re-allocation (thin wrapper)."""
from wlpr.baseline import survey_anchor
from wlpr.io import paths

if __name__ == "__main__":
    paths.assert_not_source_tree()
    survey_anchor.main()
