"""Private configuration must be explicit and must not leak through errors."""
import json

import pytest

from wlpr.io.bindings import AssetConfigurationError, asset, _read_bindings


@pytest.fixture(autouse=True)
def reset_binding_cache():
    _read_bindings.cache_clear()
    yield
    _read_bindings.cache_clear()


def test_missing_configuration_has_no_fallback(monkeypatch):
    monkeypatch.delenv("WLPR_ASSET_CONFIG", raising=False)
    with pytest.raises(AssetConfigurationError, match="WLPR_ASSET_CONFIG"):
        asset("fixture")


def test_explicit_binding_round_trip(tmp_path, monkeypatch):
    value = str(tmp_path / "fixture")
    profile = tmp_path / "bindings.json"
    profile.write_text(json.dumps({"bindings": {"fixture": value}}))
    monkeypatch.setenv("WLPR_ASSET_CONFIG", str(profile))
    assert asset("fixture") == value


@pytest.mark.parametrize("value", [None, 7, [], {}, ""])
def test_missing_or_invalid_value_is_rejected(tmp_path, monkeypatch, value):
    profile = tmp_path / "bindings.json"
    profile.write_text(json.dumps({"bindings": {"fixture": value}}))
    monkeypatch.setenv("WLPR_ASSET_CONFIG", str(profile))
    with pytest.raises(AssetConfigurationError, match="fixture") as error:
        asset("fixture")
    assert str(profile) not in str(error.value)


def test_unreadable_profile_does_not_echo_location(tmp_path, monkeypatch):
    profile = tmp_path / "not-created.json"
    monkeypatch.setenv("WLPR_ASSET_CONFIG", str(profile))
    with pytest.raises(AssetConfigurationError) as error:
        asset("fixture")
    assert str(profile) not in str(error.value)


@pytest.mark.parametrize("content", ["broken JSON", "[]", '{"bindings": []}'])
def test_invalid_profile_rejected_without_echo(tmp_path, monkeypatch, content):
    profile = tmp_path / "bindings.json"
    profile.write_text(content)
    monkeypatch.setenv("WLPR_ASSET_CONFIG", str(profile))
    with pytest.raises(AssetConfigurationError) as error:
        asset("fixture")
    assert str(profile) not in str(error.value)
