"""Standard Statistics conversion contracts."""

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path

import pytest

from wlpr.io.stdstats import _outer_members, _sum_day


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / _asset_location('asset_0182')


@pytest.mark.skipif(not SOURCE.exists(), reason="restricted Standard Statistics ZIP is not staged")
def test_stdstats_collapses_repeated_totals_and_emits_48_slots(tmp_path):
    import zipfile

    from wlpr.io.stdstats import _extract_member

    with zipfile.ZipFile(SOURCE) as outer:
        member = _outer_members(outer)["13"]
        nested_path = tmp_path / "13.zip"
        _extract_member(outer, member, nested_path)
    with zipfile.ZipFile(nested_path) as nested:
        day = next(name for name in nested.namelist()
                   if "202310" in name and name.endswith("_01.csv.zip"))
    rows = _sum_day(nested_path, day)
    assert len(rows) == 48
    assert {date for date, _hour, _value in rows} == {"20231001"}
    assert {hour for _date, hour, _value in rows} == {f"{slot:02d}" for slot in range(1, 49)}
    assert sum(value for _date, _hour, value in rows) == pytest.approx(78_072_718.47)
