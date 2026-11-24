"""Resolve private asset bindings supplied by the person running an analysis.

Published source contains identifiers only. No research location is inferred,
downloaded or supplied as a fallback.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path


class AssetConfigurationError(RuntimeError):
    """A private asset binding is missing or invalid."""


@lru_cache(maxsize=8)
def _read_bindings(configuration: str) -> dict:
    try:
        document = json.loads(Path(configuration).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise AssetConfigurationError(
            "Cannot read the private asset configuration selected by WLPR_ASSET_CONFIG."
        ) from None
    if not isinstance(document, dict) or not isinstance(document.get("bindings"), dict):
        raise AssetConfigurationError("Private asset configuration requires a bindings object.")
    return document["bindings"]


def asset(identifier: str) -> str:
    """Return one explicitly configured string without revealing it in errors."""
    configuration = os.environ.get("WLPR_ASSET_CONFIG", "")
    if not configuration:
        raise AssetConfigurationError(
            "Set WLPR_ASSET_CONFIG to a private JSON configuration before using research assets."
        )
    value = _read_bindings(configuration).get(identifier)
    if not isinstance(value, str) or not value:
        raise AssetConfigurationError(f"Supply a nonempty private binding for {identifier}.")
    return value
