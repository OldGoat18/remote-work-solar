"""PV cluster gates. Import/interface gates run always; data-run gates run only when the required raw/derived
inputs are staged (skipif), so the suite stays green pre-staging and proves the chain post-staging."""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_pv_modules_import_and_expose_entrypoints():
    from wlpr.pv import owner_ledger, obs_stock_kernel, alpha_rooftop, wfh_spatial_weight
    for mod in (owner_ledger, obs_stock_kernel, alpha_rooftop, wfh_spatial_weight):
        assert callable(mod.main)
        assert callable(mod.run_gates)
    assert callable(owner_ledger.build_owner_ledger)
    assert callable(obs_stock_kernel.build_obs_stock_kernel)
    assert callable(alpha_rooftop.build_alpha_rooftop)
    assert callable(wfh_spatial_weight.build_wfh_spatial_weight)


def test_alpha_central_is_15pct():
    from wlpr.pv import alpha_rooftop
    assert abs(alpha_rooftop.CENTRAL - 0.15) < 1e-12
    assert alpha_rooftop.ALPHAS == [0.05, 0.15, 0.30, 0.50, 1.00]


def test_btm_identity_math_present_in_owner_ledger():
    """Source-of-truth BTM identities must be the literal math: gen=self+exp, load=self+imp, net=imp-exp."""
    src = (ROOT / "src/wlpr/pv/owner_ledger.py").read_text(encoding="utf-8")
    assert "selfcons+exp pv_gen_kwh" in src
    assert "selfcons+imp gross_btm_load_kwh" in src
    assert "imp-exp net_import_kwh" in src


def test_pv_excludes_btm_userstate_twin():
    """The PV cluster keeps the metered ledger but must NOT pull in the excluded BTM/future user-state twin."""
    for fn in ["owner_ledger.py", "obs_stock_kernel.py", "alpha_rooftop.py", "wfh_spatial_weight.py"]:
        src = (ROOT / _asset_location('asset_0175') / fn).read_text(encoding="utf-8")
        for forbidden in ["eta_calibration", "theta_marg", "future_userstate", "kappa", "least("]:
            assert forbidden not in src, f"{fn}:{forbidden}"


def test_no_hardcoded_drive_paths_in_pv():
    import re
    pat = re.compile(r"[A-Za-z]:[\\/]")
    for fn in ["owner_ledger.py", "obs_stock_kernel.py", "alpha_rooftop.py", "wfh_spatial_weight.py"]:
        src = (ROOT / _asset_location('asset_0175') / fn).read_text(encoding="utf-8")
        assert not pat.search(src), f"hardcoded drive path in {fn}"


def test_pv_inputs_registered():
    from wlpr.io import paths
    idx = paths._inputs_index()
    assert "btm_selfconsumption_zip" in idx
    assert "mesh_denominator_panel" in idx
    assert "demographic_mesh_panel" in idx


# ---- data-gated run gates ----
_RAW_ZIP = (ROOT / _asset_location('asset_0168'))
_KERNEL_SRC = (ROOT / _asset_location('asset_0169'))
_REPOS = (ROOT / _asset_location('asset_0170'))
_CAP = (ROOT / _asset_location('asset_0171'))
_MOB = (ROOT / _asset_location('asset_0172'))
_LED = (ROOT / _asset_location('asset_0173'))
_DEMO = (ROOT / _asset_location('asset_0174'))


@pytest.mark.skipif(not _RAW_ZIP.exists(), reason="restricted BTM self-consumption zip not staged")
def test_owner_ledger_btm_identities():
    from wlpr.pv import owner_ledger
    owner_ledger.build_owner_ledger(test=True)
    res = owner_ledger.run_gates()
    assert res["gen_gap"] == 0 and res["load_gap"] == 0 and res["net_gap"] == 0


@pytest.mark.skipif(not _KERNEL_SRC.exists(), reason="upstream r6_w6 kernel not staged")
def test_obs_stock_kernel_yield_sane():
    from wlpr.pv import obs_stock_kernel
    obs_stock_kernel.build_obs_stock_kernel()
    res = obs_stock_kernel.run_gates()
    assert res["yield_nonneg"] and res["rows"] > 0


@pytest.mark.skipif(not (_REPOS.exists() and _CAP.exists() and _DEMO.exists()),
                    reason="REPOS potential / btm capacity / demographic panel not staged")
def test_alpha_rooftop_central_15_and_floor():
    from wlpr.pv import alpha_rooftop
    out = alpha_rooftop.run_gates()
    assert abs(out["central_alpha"] - 0.15) < 1e-9
    assert out["central_future_GW"] >= out["cur_GW"] - 1e-9


@pytest.mark.skipif(not (_MOB.exists() and _LED.exists() and _DEMO.exists()),
                    reason="mobility presence / baseline ledger / demographic panel not staged")
def test_wfh_spatial_weight_nonneg():
    from wlpr.pv import wfh_spatial_weight
    out = wfh_spatial_weight.run_gates()
    assert out["weight_nonneg"]
