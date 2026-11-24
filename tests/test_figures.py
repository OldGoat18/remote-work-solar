"""Figure-cluster gates. Import/interface gates run always; data-run gates run only when the derived
products are staged (skipif), so the suite stays green pre-staging and proves the builders post-staging.

The three RESULT figures port build_load_reshaping_figure.py / build_netload_pv_figure.py /
build_spatial_matching_figure.py. Each builder owns its scientific gates (S1->S10 family present, S10
trough > S1, evening-peak invariant, spatial-mismatch finding) inside run_gates()."""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / _asset_location('asset_0002') / _asset_location('asset_0088')


def test_figure_modules_import_and_expose_entrypoints():
    from wlpr.figures import load_reshaping, netload_pv, spatial_matching
    for mod in (load_reshaping, netload_pv, spatial_matching):
        assert callable(mod.build)
        assert callable(mod.run_gates)
        assert callable(mod.main)


def test_figures_route_paths_through_helpers_no_hardcoding():
    """Figure builders must use the shared, privately configured output identity."""
    drive_lit = re.compile(r"['\"][A-Za-z]:[\\/]")  # quoted drive-letter absolute path (CI gate pattern)
    for name in ("load_reshaping.py", "netload_pv.py", "spatial_matching.py"):
        src = (ROOT / _asset_location('asset_0163') / name).read_text(encoding="utf-8")
        assert "derived_path(" in src
        calls = [node for node in ast.walk(ast.parse(src))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "out_path" and node.args]
        assert calls, name
        identifiers = []
        for call in calls:
            binding = call.args[0]
            assert isinstance(binding, ast.Call) and isinstance(binding.func, ast.Name), name
            assert binding.func.id == "_asset_location" and len(binding.args) == 1, name
            assert isinstance(binding.args[0], ast.Constant), name
            identifiers.append(binding.args[0].value)
        assert "asset_0059" in identifiers, name
        assert not drive_lit.search(src), f"{name}: hard-coded drive-letter path"


def test_load_reshaping_uses_frozen_twin_and_keeps_F_unit_fix():
    """Fig 1 reads the frozen daily_lv_twin via input_path and keeps the verbatim F=2.0 energy->power fix."""
    src = (ROOT / "src/wlpr/figures/load_reshaping.py").read_text(encoding="utf-8")
    assert 'input_path("lv_twin_pred_pref_v4")' in src
    assert "F = 2.0" in src


def test_figures_exclude_userstate_btm():
    for name in ("load_reshaping.py", "netload_pv.py", "spatial_matching.py"):
        src = (ROOT / _asset_location('asset_0163') / name).read_text(encoding="utf-8")
        for forbidden in ["future_userstate", "theta_marg", "kappa"]:
            assert forbidden not in src, f"{name}: {forbidden}"


# ---- data-gated run gates (skip until the derived products are staged) ----
_CURVES = DERIVED / _asset_location('asset_0080') / "scenario_window_curves_month_slot.parquet"
_NL = DERIVED / _asset_location('asset_0063') / "netload_system_date_slot.parquet"
_MESH = DERIVED / _asset_location('asset_0056') / "r3_spatial_matching_mesh.parquet"


@pytest.mark.skipif(not _CURVES.exists(), reason="scenario_window_curves not staged")
def test_load_reshaping_builds_and_passes_gates():
    from wlpr.figures import load_reshaping
    w, order = load_reshaping.build()
    assert not load_reshaping.run_gates(w, order)


@pytest.mark.skipif(not _NL.exists(), reason="netload_system_date_slot not staged")
def test_netload_pv_builds_and_passes_gates():
    from wlpr.figures import netload_pv
    wc, wf, order, scr, rel, pw, nd = netload_pv.build()
    assert not netload_pv.run_gates(wc, wf, order, scr, rel, pw, nd)


@pytest.mark.skipif(not _MESH.exists(), reason="r3_spatial_matching_mesh not staged")
def test_spatial_matching_builds_and_passes_gates():
    from wlpr.figures import spatial_matching
    f, sc = spatial_matching.build()
    assert not spatial_matching.run_gates(f, sc)
