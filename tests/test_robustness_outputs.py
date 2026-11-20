from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_behavior_robustness_outputs_are_consistent_when_built():
    p = ROOT / "outputs/scorecards/behavior_wfh_scale_sensitivity.csv"
    if not p.exists():
        return
    x = pd.read_csv(p)
    s1 = x[x.scale == 1.0].iloc[0]
    assert abs(s1.actual_match - 0.0383) < 0.002
    assert s1.free_placement_ceiling > s1.actual_match
    assert s1.magnitude_wall > 0


def test_storage_equivalence_has_upper_and_lower_policies_when_built():
    p = ROOT / "outputs/scorecards/storage_equivalence.csv"
    if not p.exists():
        return
    x = pd.read_csv(p).set_index("policy")
    assert {"perfect_foresight_upper", "greedy_lower"}.issubset(x.index)
    assert (x.captured_surplus_GWh >= 35.0).all()
    assert x.loc["greedy_lower", "storage_energy_GWh"] >= x.loc["perfect_foresight_upper", "storage_energy_GWh"]
