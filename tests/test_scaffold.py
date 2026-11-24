"""Foundational governance gates (Phase 1). Hold for the lifetime of the package."""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_and_meta():
    import wlpr
    from wlpr.io import paths, manifest
    assert wlpr.__version__
    assert paths.run_from() in ("frozen", "raw")
    assert set(paths.TIERS) == {"raw", "frozen", "derived"}
    # integrity gate: registered inputs may be UNSTAGED (missing is fine — 00 reports them); the package is
    # only unhealthy if a STAGED file's sha256 MISMATCHES the manifest (corruption).
    assert manifest.verify()["mismatched"] == []


def test_sha256_dir_fingerprint(tmp_path):
    """Directory-asset fingerprint (used for pv_checkpoints): deterministic, content-free, size-sensitive."""
    from wlpr.io import manifest
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"0123456789")
    (tmp_path / "b.bin").write_bytes(b"abc")
    h1, n1 = manifest.sha256_dir(str(tmp_path))
    assert len(h1) == 64 and n1 == 13                         # 10 + 3 bytes
    assert manifest.sha256_dir(str(tmp_path))[0] == h1        # deterministic / stable order
    (tmp_path / "b.bin").write_bytes(b"abcd")                 # size change -> fingerprint changes
    assert manifest.sha256_dir(str(tmp_path))[0] != h1


def test_source_guard_blocks_writing_into_source(monkeypatch):
    from wlpr.io import paths
    paths.assert_not_source_tree()  # must not raise (we are in the new package)
    protected = os.path.join(_asset_location('asset_0122'), _asset_location('asset_0123'))
    monkeypatch.setattr(paths, "ROOT", protected)
    with pytest.raises(AssertionError):
        paths.assert_not_source_tree()


def test_configs_load():
    from wlpr.io import paths
    rc, mc = paths.run_config(), paths.model_config()
    assert rc["prefs"] == ["11", "12", "13", "14"]
    assert rc["s1_adwfh_window_total_mwh"] == 12534.9
    assert mc["pv"]["alpha_central"] == 0.15
    assert mc["unit_powers"]["device_kw"] == 0.090


def test_no_hardcoded_external_paths_in_src():
    """Published source must not contain machine-specific location literals."""
    bad = []
    pat = re.compile(r"['\"][A-Za-z]:[\\/]")  # drive-letter absolute path literal
    for p in (ROOT / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pat.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, "Use private bindings instead of machine-specific locations:\n" + "\n".join(bad)


EXCLUDED = ["build_p1bu", "future_userstate", "build_annual_a", "build_wfh_v2_pv_bridge",
            "bayes_network", "param_inference", "mesh_date_ledger", "pv_enduse_cross"]


def test_no_excluded_modules_present():
    """Excluded subsystems must remain outside the published implementation."""
    present = [m for m in EXCLUDED for _ in (ROOT / "src").rglob(f"*{m}*.py")]
    assert not present, f"excluded modules leaked into src: {present}"
