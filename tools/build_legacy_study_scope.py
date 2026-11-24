#!/usr/bin/env python3
"""Rebuild the legacy 34,327-mesh study universe from Standard Statistics.

The released validation artifact establishes the expected positive-load mesh
set.  The selection rule itself is reconstructed from the raw source:

1. retain meshes for which the published aggregate-use field is observed at
   least once in the study window (do not backfill it from contract classes);
2. for meshes present in two prefectural archives, assign the mesh to the
   prefecture with the larger window-total aggregate use.

This yields 34,307 positive-load meshes plus 20 aggregate-zero meshes.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import argparse
import io
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from wlpr.io.paths import ROOT, derived_path, input_path, out_path


PREFS = ("11", "12", "13", "14")
WINDOW = ("202310", "202311", "202312", "202401", "202402")
DATE_COL = "対象年月（日）"
SLOT_COL = "時刻コード"
MESH_COL = "メッシュコード"
AGG_USE_COL = "集計値使用電力量"
READ_COLUMNS = [DATE_COL, SLOT_COL, MESH_COL, AGG_USE_COL]
MEMBER_RE = re.compile(
    'WL11-(11|12|13|14)-(202310|202311|202312|202401|202402)-.*\\.csv\\.zip$'
)

OUT = Path(derived_path(_asset_location('asset_0073'), "study_mesh_universe.parquet"))
CACHE = Path(derived_path(_asset_location('asset_0073'), _asset_location('asset_0196')))
SCORECARD = Path(out_path(_asset_location('asset_0058'), "legacy_scope_rebuild.csv"))
ZERO_LIST = Path(out_path(_asset_location('asset_0058'), "legacy_scope_zero_load_meshes.csv"))
RELEASE_ZIP = Path(ROOT) / "WFH_LV_PV_reproduction_code.zip"
RELEASE_GRID = _asset_location('asset_0128')


def _outer_member(archive: zipfile.ZipFile, pref: str) -> str:
    return next(
        name for name in archive.namelist()
        if f"/{pref}_" in name and "/01_" in name and name.endswith(".zip")
    )


def _scan_pref(source: Path, pref: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    with tempfile.TemporaryDirectory(prefix=f"wlpr-legacy-scope-{pref}-") as tmp:
        nested_path = Path(tmp) / f"{pref}.zip"
        with zipfile.ZipFile(source) as outer:
            member = _outer_member(outer, pref)
            with outer.open(member) as src, nested_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)

        with zipfile.ZipFile(nested_path) as nested:
            members = sorted(
                name for name in nested.namelist()
                if (match := MEMBER_RE.search(name)) and match.group(1) == pref
            )
            if len(members) != 152:
                raise RuntimeError(f"legacy scope expected 152 daily files for {pref}, got {len(members)}")
            for index, member in enumerate(members, 1):
                with zipfile.ZipFile(io.BytesIO(nested.read(member))) as daily:
                    csv_name = next(name for name in daily.namelist() if name.lower().endswith(".csv"))
                    with daily.open(csv_name) as raw:
                        table = pacsv.read_csv(
                            raw,
                            read_options=pacsv.ReadOptions(block_size=8 << 20, use_threads=False),
                            convert_options=pacsv.ConvertOptions(
                                include_columns=READ_COLUMNS,
                                column_types={
                                    DATE_COL: pa.string(), SLOT_COL: pa.string(),
                                    MESH_COL: pa.string(), AGG_USE_COL: pa.float64(),
                                },
                                null_values=["", "N/A"],
                            ),
                        )
                valid = pc.and_kleene(
                    pc.is_valid(table[AGG_USE_COL]),
                    pc.match_substring_regex(table[MESH_COL], r"^\d{9}$"),
                )
                grouped = (table.filter(pc.fill_null(valid, False))
                           .group_by([DATE_COL, SLOT_COL, MESH_COL])
                           .aggregate([(AGG_USE_COL, "max")]))
                values = grouped.to_pydict()
                for mesh, use in zip(values[MESH_COL], values[f"{AGG_USE_COL}_max"]):
                    totals[mesh][0] += 1
                    totals[mesh][1] += float(use or 0.0)
                if index % 30 == 0 or index == len(members):
                    print(f"[legacy-scope] pref={pref} {index}/{len(members)} meshes={len(totals):,}", flush=True)

    frame = pd.DataFrame(
        [(mesh, pref, int(value[0]), value[1]) for mesh, value in totals.items()],
        columns=["mesh", "pref", "aggregate_rows", "aggregate_use_kwh"],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def _released_positive_meshes() -> pd.DataFrame | None:
    if not RELEASE_ZIP.exists():
        return None
    with zipfile.ZipFile(RELEASE_ZIP) as archive:
        if RELEASE_GRID not in archive.namelist():
            return None
        frame = pd.read_parquet(io.BytesIO(archive.read(RELEASE_GRID)), columns=["g", "pref"])
    return (frame.rename(columns={"g": "mesh"})
            .assign(mesh=lambda x: x.mesh.astype(str), pref=lambda x: x.pref.astype(str))
            .drop_duplicates("mesh"))


def build(cache_dir: Path = CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(input_path("stdstats_zip"))
    pairs = pd.concat(
        [_scan_pref(source, pref, cache_dir / f"{_asset_location('asset_0016')}{pref}.parquet") for pref in PREFS],
        ignore_index=True,
    )
    pair_counts = pairs.groupby("mesh").size()
    boundary_meshes = int((pair_counts > 1).sum())

    # Largest aggregate use exactly reproduces all released cross-pref assignments.
    scope = (pairs.sort_values(
        ["mesh", "aggregate_use_kwh", "pref"], ascending=[True, False, True]
    ).drop_duplicates("mesh", keep="first").sort_values("mesh").reset_index(drop=True))
    scope["load_status"] = scope.aggregate_use_kwh.gt(0).map(
        {True: "positive_aggregate_load", False: "zero_aggregate_load"}
    )
    scope["selection_rule"] = "aggregate_use_observed; dominant_pref_by_window_aggregate_use"

    positive = int(scope.aggregate_use_kwh.gt(0).sum())
    zero = int(scope.aggregate_use_kwh.eq(0).sum())
    selected_rows = int(scope.aggregate_rows.sum())
    if (len(scope), positive, zero, boundary_meshes) != (34327, 34307, 20, 241):
        raise RuntimeError(
            "legacy scope gate failed: "
            f"meshes={len(scope)} positive={positive} zero={zero} boundary={boundary_meshes}"
        )

    released = _released_positive_meshes()
    released_positive_match = None
    released_pref_match = None
    if released is not None:
        rebuilt_positive = scope[scope.aggregate_use_kwh > 0][["mesh", "pref"]]
        joined = rebuilt_positive.merge(released, on="mesh", suffixes=("_rebuilt", "_released"))
        released_positive_match = len(joined)
        released_pref_match = int((joined.pref_rebuilt == joined.pref_released).sum())
        if (len(released), released_positive_match, released_pref_match) != (34307, 34307, 34307):
            raise RuntimeError(
                "released mesh gate failed: "
                f"released={len(released)} ids={released_positive_match} prefs={released_pref_match}"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SCORECARD.parent.mkdir(parents=True, exist_ok=True)
    scope.to_parquet(OUT, index=False)
    scorecard = pd.DataFrame([
        {"metric": "study_meshes", "value": len(scope)},
        {"metric": "positive_aggregate_load_meshes", "value": positive},
        {"metric": "zero_aggregate_load_meshes", "value": zero},
        {"metric": "cross_pref_meshes_before_assignment", "value": boundary_meshes},
        {"metric": "selected_aggregate_rows", "value": selected_rows},
        {"metric": "released_positive_id_matches", "value": released_positive_match},
        {"metric": "released_positive_pref_matches", "value": released_pref_match},
    ])
    scorecard.to_csv(SCORECARD, index=False)
    scope.loc[scope.aggregate_use_kwh == 0, [
        "mesh", "pref", "aggregate_rows", "aggregate_use_kwh", "selection_rule"
    ]].to_csv(ZERO_LIST, index=False)
    return scope, scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args()
    scope, scorecard = build(args.cache_dir)
    print(scorecard.to_string(index=False))
    print(f"[legacy-scope] wrote {OUT} ({len(scope):,} meshes)")


if __name__ == "__main__":
    main()
