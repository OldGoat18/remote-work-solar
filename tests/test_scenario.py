"""Scenario-cluster gates. Import/interface gates run always; data-run gates run only when the derived
baseline_p0 ledger (+ a_dwfh / pv kernels) is staged (skipif), so the suite stays green pre-staging and proves
the mesh × date × slot net-load + scenario layer post-staging.

The RED LINE these gates protect: non-linear surplus / reverse / storage are computed mesh-day-FIRST (never on a
rep-day), and the scenario index is BTM-FREE (no owner_BTM / future_BTM / kappa column)."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_scenario_modules_import_and_expose_entrypoints():
    from wlpr.scenario import (increment_bands, indices_s1s10, mesh_summary,
                               netload_mesh_date, storage_physics, window_curves)
    # every module exposes main()
    for mod in (netload_mesh_date, indices_s1s10, mesh_summary, window_curves,
                storage_physics, increment_bands):
        assert callable(mod.main), mod.__name__
    # gates are exposed either as run_gates() or as an inline gate() helper (per the porting rule)
    for mod in (netload_mesh_date, window_curves, storage_physics, increment_bands):
        assert callable(mod.run_gates), mod.__name__
    for mod in (indices_s1s10, mesh_summary):
        assert callable(mod.gate), mod.__name__


def test_index_is_btm_free():
    """The ported net-load index must EMIT no BTM / kappa column (the BTM twin is excluded). Checked on the
    actual idx_rows dict keys (docstrings legitimately mention the dropped BTM), not raw substrings."""
    import inspect

    from wlpr.scenario import indices_s1s10
    body = inspect.getsource(indices_s1s10.main)
    # split off any comment text so the negative mentions in prose don't trip the check
    code = "\n".join(line.split("#", 1)[0] for line in body.splitlines())
    # these must never be EMITTED as index columns (they survive only inside the legacy-guard gate labels,
    # so check that no dict-key assignment carries them)
    for forbidden in ["owner_BTM", "adopter_BTM", "future_BTM", "kappa_self", "selfcons_gain"]:
        assert f"{forbidden}=" not in code.replace(" ", ""), forbidden
    for keep in ["wfh_energy_multiplier", "pv_selfcons_rate", "nonowner_relief_GWh"]:
        assert keep in code, keep


def test_increment_bands_drops_btm_channels():
    from wlpr.scenario import increment_bands
    assert increment_bands.CHANNELS == ["nonowner_relief_GWh"]
    # the Layer-A validation-band / figure / verify surface is dropped on port
    for dropped in ("layer_a", "fig_v1_v2", "fig_b", "verify"):
        assert not hasattr(increment_bands, dropped), dropped


def test_redline_mesh_day_first_surplus():
    """Surplus / reverse must be summed at mesh grain BEFORE the system sum (window-fn over partition mesh,date)."""
    import inspect

    from wlpr.scenario import netload_mesh_date as nl
    recon = inspect.getsource(nl._prim_sql) + inspect.getsource(nl._recon_sql)
    assert "partition by mesh,date" in recon
    assert "greatest(G_cur-L,0) surp_cur" in recon and "greatest(G_fut-L,0) surp_fut" in recon
    # the net-load builder reads only the C1 inputs; scenario_aggregate must NOT be a read path
    assert "scenario_aggregate" not in inspect.getsource(nl.build) + inspect.getsource(nl._sys_sql)
    # storage physics builds strictly on the C1 net-load layer
    sp = (ROOT / "src/wlpr/scenario/storage_physics.py").read_text(encoding="utf-8")
    assert "netload_mesh_date" in sp


def test_no_hardcoded_drive_literals():
    import re
    for f in (ROOT / "src/wlpr/scenario").glob("*.py"):
        txt = f.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:[\\/]", txt), f"hardcoded drive literal in {f.name}"


_BASE = ROOT / "data/derived/ledger/baseline_p0"
_ADW = ROOT / "data/derived/wfh_state/a_dwfh_mesh_mon_slot.parquet"
_STAGED = _BASE.exists() and _ADW.exists()


@pytest.mark.skipif(not _STAGED, reason="derived baseline_p0 + a_dwfh not built yet (run steps 04 + wfh_state)")
def test_netload_redline_and_columns():
    from wlpr.scenario import netload_mesh_date as nl
    sysdf, primdf = nl.build()
    shape = nl.rebase_shape(sysdf)
    nl.run_gates(sysdf, primdf, shape)
    need = {"scenario_id", "pv_scenario", "date", "slot", "net_load_MW", "local_surplus_MWh", "n_surplus_mesh"}
    assert need <= set(sysdf.columns)
    assert sysdf.slot.nunique() == 48
    assert sysdf.groupby(["scenario_id", "pv_scenario"]).date.nunique().min() >= 150


_IDX = ROOT / "data/derived/scenario_s1s10/scenario_indices_s1s10.csv"


@pytest.mark.skipif(not _IDX.exists(), reason="scenario index not built yet (run step 09)")
def test_index_scr_and_relief_closure():
    import pandas as pd
    idx = pd.read_csv(_IDX)
    assert not any(("_BTM" in c) or ("kappa" in c) for c in idx.columns)
    assert idx.pv_selfcons_rate.between(0, 1).all()
    s1f = idx[(idx.scenario_id == "S1") & (idx.pv_scenario == "future_pv_ai")].iloc[0]
    s10f = idx[(idx.scenario_id == "S10") & (idx.pv_scenario == "future_pv_ai")].iloc[0]
    assert s10f.pv_selfcons_rate > s1f.pv_selfcons_rate > 0
    assert float((idx.delta_mesh_reverse_GWh_vs_no_wfh + idx.nonowner_relief_GWh).abs().max()) < 1e-6
