"""Single source of truth for paths in the wlpr reproduction package.

Three runtime data tiers (inputs are supplied separately from this repository):
  data/raw/      external raw sources
  data/frozen/   frozen input snapshots
  data/derived/  regenerated products or separately supplied derived snapshots

Every read resolves through configs/inputs.yaml (logical-id -> {tier, path}); every write resolves
under data/derived/ or outputs/. Source modules resolve paths through these helpers
(checked in tests/test_scaffold.py).
"""
from __future__ import annotations

import os

import yaml

# package root = three levels up from src/wlpr/io/paths.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIGS = os.path.join(ROOT, "configs")
DATA = os.path.join(ROOT, "data")
TIERS = {"raw": os.path.join(DATA, "raw"),
         "frozen": os.path.join(DATA, "frozen"),
         "derived": os.path.join(DATA, "derived")}

_CACHE: dict = {}


def _load_yaml(name: str):
    key = f"yaml::{name}"
    if key not in _CACHE:
        with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
            _CACHE[key] = yaml.safe_load(f)
    return _CACHE[key]


def _inputs_index() -> dict:
    """Flatten configs/inputs.yaml {group: {id: {tier, path}}} -> {id: (tier, relpath)}."""
    if "inputs" not in _CACHE:
        cfg = _load_yaml("inputs.yaml")
        idx = {}
        for group, entries in (cfg or {}).items():
            if not isinstance(entries, dict):
                continue
            for iid, spec in entries.items():
                if isinstance(spec, dict):
                    idx[iid] = (spec.get("tier", "frozen"), spec["path"])
                else:  # bare string -> frozen tier by default
                    idx[iid] = ("frozen", spec)
        _CACHE["inputs"] = idx
    return _CACHE["inputs"]


def input_path(input_id: str, must_exist: bool = True) -> str:
    """Absolute path to a registered input by logical id. Asserts it stays under its data tier."""
    idx = _inputs_index()
    if input_id not in idx:
        raise KeyError(f"unknown input_id {input_id!r}; known: {sorted(idx)[:8]}...")
    tier, rel = idx[input_id]
    base = TIERS[tier]
    p = os.path.normpath(os.path.join(base, rel))
    assert p.startswith(os.path.normpath(base)), f"input {input_id} escapes data/{tier}: {p}"
    if must_exist and not os.path.exists(p):
        raise FileNotFoundError(f"{tier} input missing: {p} (stage raw / run 00_verify_inputs)")
    return p.replace("\\", "/")


def derived_path(*parts: str) -> str:
    """Absolute path under data/derived/, creating the parent dir. The ONLY write root for products."""
    p = os.path.join(TIERS["derived"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p.replace("\\", "/")


def out_path(*parts: str) -> str:
    """Absolute path under outputs/, creating the parent dir (figures, scorecards, logs, release)."""
    p = os.path.join(ROOT, "outputs", *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p.replace("\\", "/")


def assert_not_source_tree() -> None:
    """Hard guard: refuse to run if ROOT is the original source project (prevents mutating it)."""
    bad = os.path.normpath(os.path.join("远程办公论文", "bottom_up_microsim_product"))
    assert bad not in os.path.normpath(ROOT), f"refusing to write inside the source project: {ROOT}"


def model_config() -> dict:
    return _load_yaml("model.yaml")


def run_config() -> dict:
    return _load_yaml("run.yaml")


def run_from() -> str:
    """'frozen' (read derived snapshot, fast) or 'raw' (regenerate derived from raw+frozen)."""
    return os.environ.get("WLPR_RUN_FROM", (run_config() or {}).get("run_from", "frozen"))
