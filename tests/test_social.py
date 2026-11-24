"""R3 social-layer gates. Import/interface + helper-math gates run always; the data-run gate runs only when
the upstream baseline_p0 ledger is built (skipif), so the suite stays green pre-staging and proves the chain
post-staging."""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_social_modules_import_and_expose_entrypoints():
    from wlpr.social import benefit_distribution, reallocation, spatial_matching_r3
    for mod in (spatial_matching_r3, reallocation, benefit_distribution):
        assert callable(mod.main)
        assert callable(mod.run_gates)
    assert callable(spatial_matching_r3.build_fields)
    assert callable(spatial_matching_r3.scorecard)
    assert callable(reallocation.build)
    assert callable(benefit_distribution.build)


def test_mesh_date_first_surplus_and_midday_window():
    """Red line: surplus (nonlinear max) is taken per mesh x date x slot BEFORE summing, and the midday
    window matches C1 (slots 20..30). Guard the load-bearing SQL/constants verbatim."""
    src = (ROOT / "src/wlpr/social/spatial_matching_r3.py").read_text(encoding="utf-8")
    assert "MIDDAY = (20, 30)" in src
    assert "greatest(G_fut-L,0)" in src               # surplus = max(G_fut - L, 0) at the slot grain
    baseline = next(node.value for node in ast.walk(ast.parse(src))
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "BASE" for target in node.targets))
    assert isinstance(baseline, ast.Call) and isinstance(baseline.func, ast.Name)
    assert baseline.func.id == "derived_path" and len(baseline.args) == 2
    for argument, identifier in zip(baseline.args, ("asset_0052", "asset_0053")):
        assert isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name)
        assert argument.func.id == "_asset_location" and len(argument.args) == 1
        assert isinstance(argument.args[0], ast.Constant) and argument.args[0].value == identifier
    re_src = (ROOT / "src/wlpr/social/reallocation.py").read_text(encoding="utf-8")
    assert "least(dwfh, greatest(G_fut - L_no, 0))" in re_src   # per-slot actual absorption before pooling
    assert 'df[["B", "S"]].min(axis=1)' in re_src              # A_spatial = min(budget, surplus) per slot


def test_no_excluded_pv_userstate_pulled_in():
    """The social layer reads the metered ledger + alpha/kernel; it must not import the BTM user-state twin."""
    for name in ("spatial_matching_r3", "reallocation", "benefit_distribution"):
        src = (ROOT / f"src/wlpr/social/{name}.py").read_text(encoding="utf-8")
        for forbidden in ["eta", "theta_marg", "future_userstate", "least(eta", "kappa"]:
            assert forbidden not in src, f"{name}: {forbidden}"


def test_munic_bridge_registered():
    from wlpr.io import paths
    assert "repos_mesh_munic_bpg" in paths._inputs_index()


def test_benefit_distribution_helper_math():
    """Lock the inequality primitives on known inputs (numerics unchanged from source)."""
    from wlpr.social import benefit_distribution as bd
    # perfectly equal -> Gini 0; one-takes-all -> Gini -> 1 - 1/n
    assert abs(bd._gini(np.ones(100))) < 1e-9
    onehot = np.zeros(100); onehot[0] = 1.0
    assert bd._gini(onehot) == pytest.approx(1 - 1.0 / 100, abs=1e-9)
    # dissimilarity: identical -> 0, disjoint -> 1
    a = np.array([1.0, 0.0, 1.0]); b = np.array([1.0, 0.0, 1.0]); c = np.array([0.0, 1.0, 0.0])
    assert bd._dissimilarity(a, b) == pytest.approx(0.0, abs=1e-12)
    assert bd._dissimilarity(a, c) == pytest.approx(1.0, abs=1e-12)
    # grid index parses a 9-char JIS mesh into integer (iy, ix)
    iy, ix = bd._grid_index("533945092")
    assert isinstance(iy, int) and isinstance(ix, int)


_LEDGER = (ROOT / _asset_location('asset_0173')).exists()


@pytest.mark.skipif(not _LEDGER, reason="baseline_p0 ledger not built yet (run steps 02->04 first)")
def test_r3_spatial_matching_runs_and_gates():
    from wlpr.social import spatial_matching_r3 as r3
    f = r3.build_fields()
    sc = r3.scorecard(f)
    s1 = sc[sc.arm == "S1"].iloc[0]
    # realized relief cross-checks the 26.2 GWh non-owner grid relief; demand surplus ~688.9 GWh
    assert abs(s1.realized_relief_GWh - 26.2) < 1.5
    assert abs(s1.demand_surplus_GWh - 688.9) < 5.0
    # the spatial mismatch: co-locates with generation (>0.5) but anti-locates with surplus (<0)
    assert s1.colocation_wfh_generation > 0.5 and s1.colocation_wfh_surplus_active < 0
    r3.run_gates(f, sc)
