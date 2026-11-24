"""Resolve logical input and output bindings supplied privately by the operator.

Raw, frozen and derived denote provenance tiers, not published storage locations.
The public input catalogue contains binding identifiers only.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os

import yaml

# Resolve the installed source root independently of research asset locations.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIGS = os.path.join(ROOT, "configs")
DATA = os.path.join(ROOT, _asset_location('asset_0002'))
TIERS = {"raw": os.path.join(DATA, _asset_location('asset_0120')),
         "frozen": os.path.join(DATA, _asset_location('asset_0121')),
         "derived": os.path.join(DATA, _asset_location('asset_0088'))}

_CACHE: dict = {}


def _load_yaml(name: str):
    key = f"yaml::{name}"
    if key not in _CACHE:
        with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
            _CACHE[key] = yaml.safe_load(f)
    return _CACHE[key]


def _inputs_index() -> dict:
    """Resolve the logical input catalogue using private asset bindings."""
    if "inputs" not in _CACHE:
        cfg = _load_yaml("inputs.yaml")
        idx = {}
        for group, entries in (cfg or {}).items():
            if not isinstance(entries, dict):
                continue
            for iid, spec in entries.items():
                if isinstance(spec, dict):
                    idx[iid] = (spec.get("tier", "frozen"), _asset_location(spec["asset"]))
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
    assert os.path.commonpath([p, os.path.normpath(base)]) == os.path.normpath(base), f"input {input_id} escapes its configured tier"
    if must_exist and not os.path.exists(p):
        raise FileNotFoundError(f"Required input {input_id} is not staged for tier {tier}.")
    return p.replace("\\", "/")


def derived_path(*parts: str) -> str:
    """Resolve a derived product and create its parent within the configured product tier."""
    p = os.path.join(TIERS["derived"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p.replace("\\", "/")


def out_path(*parts: str) -> str:
    """Resolve a configured output and prepare its container."""
    p = os.path.join(ROOT, _asset_location('asset_0087'), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p.replace("\\", "/")


def assert_not_source_tree() -> None:
    """Hard guard: refuse to run if ROOT is the original source project (prevents mutating it)."""
    bad = os.path.normpath(os.path.join(_asset_location('asset_0122'), _asset_location('asset_0123')))
    assert bad not in os.path.normpath(ROOT), f"refusing to write inside the source project: {ROOT}"


def model_config() -> dict:
    return _load_yaml("model.yaml")


def run_config() -> dict:
    return _load_yaml("run.yaml")


def run_from() -> str:
    """'frozen' (read derived snapshot, fast) or 'raw' (regenerate derived from raw+frozen)."""
    return os.environ.get("WLPR_RUN_FROM", (run_config() or {}).get("run_from", "frozen"))
