"""wlpr — remote-work load, rooftop-PV integration, and storage models.

This code-only distribution requires separately supplied inputs. It does not
include datasets, experiment outputs, figures, or manuscript documents.

Core model layers:
  baseline/   LOAD-FIT decomposition of observed LV load -> baseline_p0
  wfh_state/  CELL-NATIVE a_dwfh WFH-increment engine (survey-anchored, S1-S10)
  scenario/   net = L - G_pv at mesh x date x slot (the formal grain red line)

"""

__version__ = "0.1.0"
