"""Rebuild the missing closure surface from staged Standard Statistics grid data.

This is a transparent reconstruction, not a claim of byte identity with the missing frozen snapshot.
The raw source supplies mesh x date x 48-slot observed LV energy. The retained load-twin design supplies
a prefecture/date/slot prediction. The reconstructed closure preserves that prefecture prediction by
applying the observed mesh share:

    oshare(g,d,s) = obs(g,d,s) / sum_g obs(g,d,s)
    pv(g,d,s)     = pred_pref(d,s)
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import csv
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from wlpr.io import manifest, paths
from wlpr.io.stdstats import PREFS, WINDOW_YMS, _fit_pref_twin, _outer_members, _read_mesh_day


OUT_ROOT = Path(paths.TIERS["frozen"]) / _asset_location('asset_0117')
CLOSURE = OUT_ROOT / "_share_tmp.parquet"
TWIN = OUT_ROOT / "daily_lv_twin_pred_pref_v4.parquet"
PART_ROOT = Path(paths.TIERS["derived"]) / _asset_location('asset_0118')


def _rename_date_column(path: Path) -> None:
    """Migrate an already-written rebuild partition to the project's canonical d column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if "d" in parquet.schema_arrow.names:
        return
    if "date" not in parquet.schema_arrow.names:
        raise ValueError(f"closure partition has neither d nor date: {path}")
    temp_path = path.with_suffix(".parquet.part")
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=250_000):
            columns = batch.to_pydict()
            columns["d"] = columns.pop("date")
            if writer is None:
                schema = parquet.schema_arrow
                date_index = schema.get_field_index("date")
                fields = [field for index, field in enumerate(schema)
                          if index != date_index]
                fields.insert(date_index, pa.field("d", schema.field("date").type))
                writer = pq.ParquetWriter(temp_path, pa.schema(fields), compression="zstd")
            writer.write_table(pa.Table.from_pydict(columns, schema=writer.schema))
    finally:
        if writer is not None:
            writer.close()
    os.replace(temp_path, path)


def _nested_archives(source: Path) -> dict[str, Path]:
    temp = Path(tempfile.mkdtemp(prefix="wlpr_closure_nested_"))
    result: dict[str, Path] = {}
    with zipfile.ZipFile(source) as outer:
        members = _outer_members(outer)
        for pref in PREFS:
            destination = temp / f"{pref}.zip"
            with outer.open(members[pref]) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 << 20)
            result[pref] = destination
    return result


def _month_members(nested: zipfile.ZipFile, pref: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    pattern = re.compile(f"WL11-{pref}-(\\d{{6}}){_asset_location('asset_0116')}$")
    for name in nested.namelist():
        match = pattern.search(name)
        if match and match.group(1) in WINDOW_YMS:
            result.setdefault(match.group(1), []).append(name)
    return {ym: sorted(names) for ym, names in result.items()}


def _rebuild_month(nested_path: Path, pref: str, ym: str, members: list[str], twin: pd.DataFrame,
                   force: bool) -> Path:
    part_path = PART_ROOT / f"{_asset_location('asset_0016')}{pref}" / f"{_asset_location('asset_0070')}{ym}" / "part.parquet"
    if part_path.exists() and not force:
        print(f"[closure] skip {part_path.relative_to(paths.ROOT)}", flush=True)
        return part_path
    if not members:
        raise ValueError(f"no grid files found for {pref}-{ym}")

    # Each member is one spatial partition for a whole month. Aggregate each member before
    # concatenating so the in-memory frame contains mesh/date/slot rows, not class-level rows.
    chunks: list[pd.DataFrame] = []
    with zipfile.ZipFile(nested_path) as nested:
        for index, member in enumerate(members, 1):
            chunks.append(_read_mesh_day(nested_path, member))
            if index % 5 == 0 or index == len(members):
                print(f"[closure] {pref}-{ym}: {index}/{len(members)} spatial files", flush=True)
    obs = pd.concat(chunks, ignore_index=True)
    del chunks
    obs = obs.groupby(["d", "g", "slot"], as_index=False, sort=False).obs.sum()
    obs["pref"] = pref

    pref_twin = twin[twin.pref == pref].rename(columns={"d": "date"})
    obs = obs.rename(columns={"d": "date"})
    obs = obs.merge(pref_twin[["date", "slot", "pred_mwh"]], on=["date", "slot"], how="left")
    missing = int(obs.pred_mwh.isna().sum())
    if missing:
        sample = obs.loc[obs.pred_mwh.isna(), "date"].drop_duplicates().head(5).tolist()
        raise ValueError(f"twin missing {missing:,} mesh rows for {pref}-{ym}; dates={sample}")
    totals = (obs.groupby(["date", "slot"], as_index=False, sort=False).obs.sum()
              .rename(columns={"obs": "obs_pref"}))
    obs = obs.merge(totals, on=["date", "slot"], how="left")
    positive = obs.obs_pref > 0
    obs["oshare"] = 0.0
    obs.loc[positive, "oshare"] = obs.loc[positive, "obs"] / obs.loc[positive, "obs_pref"]
    obs["pv"] = obs["pred_mwh"]
    obs["source_version"] = "closure_surface_rebuild_v1"
    out = obs[["g", "pref", "date", "slot", "obs", "pv", "oshare", "source_version"]]
    out = out.rename(columns={"date": "d"})
    part_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = part_path.with_suffix(".parquet.part")
    out.to_parquet(temp_path, index=False, compression="zstd")
    os.replace(temp_path, part_path)
    print(f"[closure] wrote {part_path.relative_to(paths.ROOT)} rows={len(out):,}", flush=True)
    return part_path


def _combine_parts(parts: list[Path]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temp_path = CLOSURE.with_suffix(".parquet.part")
    writer = None
    try:
        for path in sorted(parts):
            parquet = pq.ParquetFile(path)
            if writer is None:
                schema = parquet.schema_arrow
                if "date" in schema.names and "d" not in schema.names:
                    date_index = schema.get_field_index("date")
                    fields = [field for index, field in enumerate(schema)
                              if index != date_index]
                    fields.insert(date_index, pa.field("d", schema.field("date").type))
                    schema = pa.schema(fields)
                writer = pq.ParquetWriter(
                    temp_path,
                    schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            for batch in parquet.iter_batches(batch_size=250_000):
                columns = batch.to_pydict()
                if "date" in columns and "d" not in columns:
                    columns["d"] = columns.pop("date")
                writer.write_table(pa.Table.from_pydict(columns, schema=writer.schema))
    finally:
        if writer is not None:
            writer.close()
    os.replace(temp_path, CLOSURE)


def build(force: bool = False) -> dict:
    """Build the pref twin and the partitioned mesh closure surface."""
    source = Path(paths.input_path("stdstats_zip"))
    load_path = Path(paths.input_path("lv_load_hourly"))
    if not load_path.exists():
        raise FileNotFoundError(f"missing aggregate LV load: {load_path}; run pipeline/00_rebuild_stdstats.py")

    load = pd.read_parquet(load_path)
    twin = _fit_pref_twin(load)
    twin["source_version"] = "standard_statistics_grid_rebuild_v1"
    TWIN.parent.mkdir(parents=True, exist_ok=True)
    twin_temp = TWIN.with_suffix(".parquet.part")
    twin.to_parquet(twin_temp, index=False, compression="zstd")
    os.replace(twin_temp, TWIN)

    nested = _nested_archives(source)
    try:
        parts: list[Path] = []
        for pref in PREFS:
            with zipfile.ZipFile(nested[pref]) as z:
                month_map = _month_members(z, pref)
            for ym in WINDOW_YMS:
                parts.append(_rebuild_month(nested[pref], pref, ym, month_map.get(ym, []), twin, force))
        _combine_parts(parts)
    finally:
        if nested:
            shutil.rmtree(next(iter(nested.values())).parent, ignore_errors=True)

    _refresh_manifest()
    return {"twin_rows": len(twin), "closure_bytes": CLOSURE.stat().st_size}


def _refresh_manifest() -> None:
    import pyarrow.parquet as pq

    rows = manifest.load()
    updates = {
        "closure_surface_share": (
            CLOSURE,
            "standard_statistics_grid_rebuild_v1;oshare=mesh_obs/pref_obs;pv=pref_twin_prediction",
        ),
        "lv_twin_pred_pref_v4": (
            TWIN,
            "standard_statistics_grid_rebuild_v1;pref_date_slot_NNLS_twin",
        ),
    }
    for row in rows:
        if row["input_id"] not in updates:
            continue
        product, provenance = updates[row["input_id"]]
        old_hash = row.get("sha256", "")
        legacy = row.get("provenance", "")
        if old_hash and "legacy_sha256=" not in legacy:
            legacy = (legacy + ";" if legacy else "") + f"legacy_sha256={old_hash}"
        if ";standard_statistics_grid_rebuild_v1" in legacy:
            legacy = legacy.split(";standard_statistics_grid_rebuild_v1", 1)[0]
        row.update({
            "sha256": manifest.sha256_file(str(product)),
            "byte_count": str(product.stat().st_size),
            "row_count": str(pq.ParquetFile(product).metadata.num_rows),
            "materialize": "restricted_raw_rebuild",
            "evidence_level": "restricted_raw_reconstructed",
            "provenance": (legacy + ";" if legacy else "") + provenance,
        })
    with Path(manifest.MANIFEST).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest.COLUMNS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in manifest.COLUMNS} for row in rows])


def main() -> None:
    result = build(force="--force" in __import__("sys").argv)
    print(f"[closure] rebuilt twin rows={result['twin_rows']:,}; closure={result['closure_bytes']:,} bytes")


if __name__ == "__main__":
    main()
