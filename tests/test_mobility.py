"""Step 14 mobility SI-validation gates. Import/interface gates run always; the data-run gate runs only when
the two derived diurnal-presence parquets + their derived deps are staged (skipif), so the suite stays green
pre-staging and proves the validation chain post-staging."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
MOB_M = DERIVED / "mobility" / "mesh_diurnal_presence_m.parquet"
MOB_L = DERIVED / "mobility" / "mesh_diurnal_presence_l.parquet"
R3 = DERIVED / "social" / "r3_spatial_matching_mesh.parquet"
SW = DERIVED / "pv" / "wfh_spatial_weight.parquet"
WFHCURVE = DERIVED / "scenario_window" / "scenario_window_curves_month_slot.parquet"


def test_mobility_modules_import_and_expose_entrypoints():
    from wlpr.mobility import extract_presence, wfh_validation
    assert callable(extract_presence.main)
    assert callable(extract_presence.extract)
    assert callable(extract_presence.extract_spatial_inputs)
    assert callable(extract_presence.build_spatial_inputs)
    assert callable(wfh_validation.build)
    assert callable(wfh_validation.run_gates)
    assert callable(wfh_validation.main)


def test_mobility_tar_registered_as_raw():
    from wlpr.io import paths
    idx = paths._inputs_index()
    assert "mobility_lm00020_tar" in idx
    tier, rel = idx["mobility_lm00020_tar"]
    assert tier == "raw"


def test_mobility_is_validation_only_never_feeds_model():
    """RED LINE: presence is a validation source only; magnitude is NON-IDENTIFIED, never claimed."""
    src = (ROOT / "src/wlpr/mobility/wfh_validation.py").read_text(encoding="utf-8")
    assert "magnitude_identified" in src and "False" in src
    assert "NON-IDENTIFIED" in src


def test_mobility_gate_thresholds_preserved():
    """Verbatim gate thresholds: population>=0.85, timing>=0.70, placement>=0.20."""
    src = (ROOT / "src/wlpr/mobility/wfh_validation.py").read_text(encoding="utf-8")
    assert 'r["population_corr"] >= 0.85' in src
    assert 'r["timing_corr_day"] >= 0.70' in src
    assert '>= 0.20' in src


def test_no_hardcoded_tar_path_in_extract():
    src = (ROOT / "src/wlpr/mobility/extract_presence.py").read_text(encoding="utf-8")
    assert "E:\\" not in src and "E:/" not in src
    assert 'input_path("mobility_lm00020_tar")' in src


def test_spatial_windows_match_model_contract():
    from wlpr.mobility import extract_presence
    assert extract_presence.MIDDAY_SLOTS == set(range(20, 31))
    assert extract_presence.NIGHT_SLOTS == {0, 1, 2, 3, 4, 5, 46, 47}


_STAGED = MOB_M.exists() and MOB_L.exists() and R3.exists() and SW.exists() and WFHCURVE.exists()


@pytest.mark.skipif(not _STAGED, reason="derived mobility-validation inputs not staged yet")
def test_mobility_validation_gates_pass():
    from wlpr.mobility import wfh_validation
    r = wfh_validation.build()
    assert r["population_corr"] >= 0.85
    assert r["timing_corr_day"] >= 0.70
    assert min(r["placement_corr_m"], r["placement_corr_l"]) >= 0.20
    assert r["magnitude_identified"] is False
