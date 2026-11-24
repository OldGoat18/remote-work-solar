"""Cross-cluster integration gates — the seams between independently-ported clusters must line up.
These catch the class of bug where module A writes to path X but module B reads from path Y."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import wlpr
SRC = Path(wlpr.__file__).resolve().parent


SCORECARD_BINDING = 'asset_0058'
PRODUCT_BINDING = 'asset_0107'


def _read(p):
    return (SRC / p).read_text(encoding="utf-8")


def _uses_binding(source, function, identifier):
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        first = node.args[0]
        if name == function and isinstance(first, ast.Call) and isinstance(first.func, ast.Name):
            if first.func.id == "_asset_location" and first.args and isinstance(first.args[0], ast.Constant):
                if first.args[0].value == identifier:
                    return True
    return False


def test_scorecards_read_and_written_from_same_root():
    """Producers and consumers share the same configured output identity."""
    for fig in ["figures/netload_pv.py", "figures/spatial_matching.py"]:
        assert not _uses_binding(_read(fig), "derived_path", SCORECARD_BINDING), fig
        assert _uses_binding(_read(fig), "out_path", SCORECARD_BINDING), fig
    for prod in ["scenario/storage_physics.py", "social/spatial_matching_r3.py",
                 "social/reallocation.py", "social/benefit_distribution.py"]:
        assert _uses_binding(_read(prod), "out_path", SCORECARD_BINDING), prod


def test_canonical_path_independent_of_excluded_products():
    """The live a_dwfh path (step 06/07) and the net-load core must not HARD-require products of the excluded
    BTM chain — those reads must be guarded by os.path.exists."""
    mat = _read("wfh_state/materialize.py")
    # every CUR_ADWFH read is guarded by an os.path.exists(CUR_ADWFH) branch
    assert mat.count("os.path.exists(CUR_ADWFH)") >= 2, "CUR_ADWFH reads must be guarded"
    ms = _read("scenario/mesh_summary.py")
    assert "os.path.exists(MESHSUM)" in ms, "userstate_mesh_summary read must be guarded"


def test_grid_component_product_read_from_derived():
    """Component-product consumers must use the producer's configured derived identity."""
    for consumer in ["baseline/survey_anchor.py", "baseline/ledger_p0.py",
                     "validation/decompose.py", "figures/main_validation.py"]:
        txt = _read(consumer)
        if "grid_component_daily_product" in txt:
            assert _uses_binding(txt, "derived_path", PRODUCT_BINDING), consumer


def test_no_derived_path_scorecards_anywhere():
    """Scorecards must use their configured output bindings."""
    bad = []
    for p in SRC.rglob("*.py"):
        if _uses_binding(p.read_text(encoding="utf-8", errors="ignore"), "derived_path", SCORECARD_BINDING):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_emit_adwfh_is_btm_free():
    txt = _read("wfh_state/emit_adwfh.py")
    # only docstring mentions of the excluded tokens are allowed; no executable BTM call
    for tok in ["build_future_userstate_ledger", "eta_shrunk", "KAPPA_SELF", "theta_marg"]:
        assert tok not in txt, tok
