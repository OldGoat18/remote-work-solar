"""Cross-cluster integration gates — the seams between independently-ported clusters must line up.
These catch the class of bug where module A writes to path X but module B reads from path Y."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/wlpr"


def _read(p):
    return (SRC / p).read_text(encoding="utf-8")


def test_scorecards_read_and_written_from_same_root():
    """Figures read scorecards from out_path('scorecards', ...); producers must write them there too
    (the derived_path('scorecards') mismatch that broke Fig2/Fig3 must stay fixed)."""
    for fig in ["figures/netload_pv.py", "figures/spatial_matching.py"]:
        assert 'derived_path("scorecards"' not in _read(fig), fig
        assert 'out_path("scorecards"' in _read(fig), fig
    for prod in ["scenario/storage_physics.py", "social/spatial_matching_r3.py",
                 "social/reallocation.py", "social/benefit_distribution.py"]:
        assert 'out_path("scorecards"' in _read(prod), prod


def test_canonical_path_independent_of_excluded_products():
    """The live a_dwfh path (step 06/07) and the net-load core must not HARD-require products of the excluded
    BTM chain — those reads must be guarded by os.path.exists."""
    mat = _read("wfh_state/materialize.py")
    # every CUR_ADWFH read is guarded by an os.path.exists(CUR_ADWFH) branch
    assert mat.count("os.path.exists(CUR_ADWFH)") >= 2, "CUR_ADWFH reads must be guarded"
    ms = _read("scenario/mesh_summary.py")
    assert "os.path.exists(MESHSUM)" in ms, "userstate_mesh_summary read must be guarded"


def test_grid_component_product_read_from_derived():
    """Producers write the component product to derived_path('products'); every consumer must read it there
    (not out_path), or it would silently get an empty/missing path."""
    for consumer in ["baseline/survey_anchor.py", "baseline/ledger_p0.py",
                     "validation/decompose.py", "figures/main_validation.py"]:
        txt = _read(consumer)
        if "grid_component_daily_product" in txt:
            assert 'derived_path("products"' in txt, consumer


def test_no_derived_path_scorecards_anywhere():
    """No module may read/write scorecards under data/derived (scorecards live under outputs/)."""
    bad = []
    for p in SRC.rglob("*.py"):
        if 'derived_path("scorecards"' in p.read_text(encoding="utf-8", errors="ignore"):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_emit_adwfh_is_btm_free():
    txt = _read("wfh_state/emit_adwfh.py")
    # only docstring mentions of the excluded tokens are allowed; no executable BTM call
    for tok in ["build_future_userstate_ledger", "eta_shrunk", "KAPPA_SELF", "theta_marg"]:
        assert tok not in txt, tok
