"""Rebuild the project LV-load input from the restricted Standard Statistics ZIP."""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import csv
import datetime
import hashlib
import io
import os
import re
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from scipy.optimize import nnls

from wlpr.io import manifest, paths


PREFS = ("11", "12", "13", "14")
DATE_RE = re.compile('WL\\d{2}-(\\d{2})-(\\d{6})-.*_(\\d{2})\\.csv\\.zip$')
SOURCE_COLUMNS = [
    "対象年月（日）", "時刻コード", "メッシュコード", "契約電力分類_使用電力量",
    "集計値使用電力量",
]
WINDOW_YMS = ("202310", "202311", "202312", "202401", "202402")
STN = {"11": "埼玉熊谷", "12": "千葉", "13": "東京", "14": "神奈川横浜"}
from wlpr.io.activity_calendar import EXCLUDED_DATES
HOL = EXCLUDED_DATES


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _outer_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members = {}
    for name in archive.namelist():
        for pref in PREFS:
            if f"/{pref}_" in name and "/01_" in name and name.endswith(".zip"):
                members[pref] = name
    missing = sorted(set(PREFS) - set(members))
    if missing:
        raise ValueError(f"standard-statistics ZIP lacks forward-flow LV archives for {missing}")
    return members


def _sum_day(outer_path: Path, member: str) -> list[tuple[str, str, float]]:
    with zipfile.ZipFile(outer_path) as outer:
        payload = outer.read(member)
    with zipfile.ZipFile(io.BytesIO(payload)) as daily:
        csv_name = next((name for name in daily.namelist() if name.lower().endswith(".csv")), None)
        if csv_name is None:
            raise ValueError(f"daily archive has no CSV: {member}")
        with daily.open(csv_name) as source:
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(block_size=8 << 20, use_threads=False),
                convert_options=pacsv.ConvertOptions(
                    include_columns=SOURCE_COLUMNS,
                    column_types={
                        "対象年月（日）": pa.string(),
                        "時刻コード": pa.string(),
                        "メッシュコード": pa.string(),
                        "契約電力分類_使用電力量": pa.float64(),
                        "集計値使用電力量": pa.float64(),
                    },
                    null_values=["", "N/A"],
                ),
            )
            valid = pc.and_kleene(
                pc.and_kleene(pc.not_equal(table["対象年月（日）"], ""),
                              pc.not_equal(table["時刻コード"], "")),
                pc.not_equal(table["メッシュコード"], ""),
            )
            table = table.filter(valid)
            mesh_slot = table.group_by([
                "対象年月（日）", "時刻コード", "メッシュコード"
            ]).aggregate([
                ("集計値使用電力量", "max"),
                ("契約電力分類_使用電力量", "sum"),
            ])
            # The aggregate total is repeated on every contract-class row. Collapse it with max;
            # if privacy suppression hides that total, retain any visible class-level components.
            value = pc.coalesce(
                mesh_slot["集計値使用電力量_max"],
                mesh_slot["契約電力分類_使用電力量_sum"],
            )
            mesh_slot = mesh_slot.append_column("kwh", value)
            totals = mesh_slot.group_by(["対象年月（日）", "時刻コード"]).aggregate([
                ("kwh", "sum")
            ]).to_pydict()
    return [(date, hour, float(value or 0.0)) for date, hour, value in zip(
        totals["対象年月（日）"], totals["時刻コード"], totals["kwh_sum"]
    )]


def _read_mesh_day(outer_path: Path, member: str, date: str | None = None) -> pd.DataFrame:
    """Read one monthly spatial archive and retain the mesh/date grain."""
    with zipfile.ZipFile(outer_path) as outer:
        payload = outer.read(member)
    with zipfile.ZipFile(io.BytesIO(payload)) as daily:
        csv_name = next((name for name in daily.namelist() if name.lower().endswith(".csv")), None)
        if csv_name is None:
            raise ValueError(f"daily archive has no CSV: {member}")
        with daily.open(csv_name) as source:
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(block_size=8 << 20, use_threads=False),
                convert_options=pacsv.ConvertOptions(
                    include_columns=SOURCE_COLUMNS,
                    column_types={
                        "対象年月（日）": pa.string(),
                        "時刻コード": pa.string(),
                        "メッシュコード": pa.string(),
                        "契約電力分類_使用電力量": pa.float64(),
                        "集計値使用電力量": pa.float64(),
                    },
                    null_values=["", "N/A"],
                ),
            )
            valid = pc.and_kleene(
                pc.and_kleene(pc.not_equal(table["対象年月（日）"], ""),
                              pc.not_equal(table["時刻コード"], "")),
                pc.not_equal(table["メッシュコード"], ""),
            )
            table = table.filter(valid)
            if date is not None:
                table = table.filter(pc.equal(table["対象年月（日）"], str(date)))
            mesh_slot = table.group_by([
                "対象年月（日）", "時刻コード", "メッシュコード"
            ]).aggregate([
                ("集計値使用電力量", "max"),
                ("契約電力分類_使用電力量", "sum"),
            ])
            value = pc.coalesce(
                mesh_slot["集計値使用電力量_max"],
                mesh_slot["契約電力分類_使用電力量_sum"],
            )
            mesh_slot = mesh_slot.append_column("kwh", value)
            frame = mesh_slot.select([
                "対象年月（日）", "時刻コード", "メッシュコード", "kwh"
            ]).to_pandas()

    frame = frame.rename(columns={"対象年月（日）": "d", "時刻コード": "hour", "メッシュコード": "g"})
    frame["g"] = frame.g.astype(str).str.strip()
    frame = frame[frame.g.str.fullmatch(r"\d{9}")].copy()
    frame["slot"] = pd.to_numeric(frame.hour, errors="coerce") - 1
    frame["obs"] = pd.to_numeric(frame.kwh, errors="coerce").fillna(0.0) / 1000.0
    frame = frame[frame.slot.between(0, 47)][["d", "g", "slot", "obs"]]
    frame["d"] = frame.d.astype(str).str.strip()
    return frame.sort_values(["d", "g", "slot"]).reset_index(drop=True)


def _fit_pref_twin(load: pd.DataFrame) -> pd.DataFrame:
    """Fit the retained aggregate load-twin design without using the missing closure surface.

    This is the load-fit design already used by component_product.py: weather-driven HVAC/cooling,
    a daytime WFH basis, and non-thermal intraday bumps, with leave-one-month-out NNLS and pref-level
    ridge pooling. It reconstructs the pref/date/slot interface needed by the closure builder.
    """
    cfg = paths._load_yaml("model.yaml")
    hdd_base = float(cfg["weather"]["hdd_base_c"])
    cdd_base = float(cfg["weather"]["cdd_base_c"])
    own_weight = {str(k): float(v) for k, v in cfg["weather"]["pop_weight_own_station"].items()}
    tau = float(cfg["agent_prior"]["pooling_tau"])
    n_bumps = int(cfg["agent_prior"]["env_bumps"])

    den = pd.read_parquet(paths.input_path("mesh_denominator_panel"),
                          columns=["pref", "conditioned_dwelling", "person", "wfh_eligible"])
    den["pref"] = den.pref.astype(str)
    totals = den.groupby("pref")[["conditioned_dwelling", "person", "wfh_eligible"]].sum()
    norm = totals.mean()
    pref_den = {
        pref: {
            "HVAC": totals.loc[pref, "conditioned_dwelling"] / norm.conditioned_dwelling,
            "person": totals.loc[pref, "person"] / norm.person,
            "wfh": totals.loc[pref, "wfh_eligible"] / norm.wfh_eligible,
        }
        for pref in STN
    }

    weather = pd.read_parquet(paths.input_path("weather_amedas"),
                              columns=["timestamp", "station", "temp"])
    weather["timestamp"] = pd.to_datetime(weather.timestamp) + pd.Timedelta(hours=9)
    weather["d"] = weather.timestamp.dt.strftime("%Y%m%d")
    weather["hr"] = weather.timestamp.dt.hour
    weather["temp"] = pd.to_numeric(weather.temp, errors="coerce")
    hourly = weather.groupby(["station", "d", "hr"], as_index=False).temp.mean()
    station_slot: dict[tuple[str, str], np.ndarray] = {}
    for (station, date), frame in hourly.groupby(["station", "d"]):
        values = frame.set_index("hr").temp.reindex(range(24)).interpolate().bfill().ffill().values
        station_slot[(station, date)] = np.interp(np.arange(48) * 0.5, np.arange(24), values, period=24)

    slot_temp: dict[tuple[str, str], np.ndarray] = {}
    for pref, station in STN.items():
        for (candidate_station, date), values in station_slot.items():
            if candidate_station != station or ("東京", date) not in station_slot:
                continue
            slot_temp[(pref, date)] = (own_weight[pref] * values +
                                       (1.0 - own_weight[pref]) * station_slot[("東京", date)])

    load = load.copy()
    load["pref"] = load.pref_code.astype(str)
    load["d"] = load.date_str.astype(str)
    load["slot"] = pd.to_numeric(load.hour_str, errors="coerce") - 1
    load["mwh"] = pd.to_numeric(load.kwh, errors="coerce").fillna(0.0) / 1000.0
    observed = {(pref, date): frame.sort_values("slot").mwh.to_numpy(float)
                for (pref, date), frame in load.groupby(["pref", "d"])
                if len(frame) == 48 and frame.slot.nunique() == 48}

    hod = np.arange(48) / 2.0
    gauss = lambda center, width: np.exp(-0.5 * ((hod - center) / width) ** 2)
    active = ((hod >= 9) & (hod <= 17)).astype(float) * gauss(13, 3.0)
    evening = 1.0 / (1.0 + np.exp(-(hod - 16)))
    bumps = [gauss(center, 1.25) for center in np.linspace(0.5, 23.5, n_bumps)]
    env_names = [f"env{k:02d}" for k in range(n_bumps)]
    names = ["HVAC", "HVAC_eve", "COOL", "wfh", *env_names]

    def shapes(pref: str, date: str) -> dict[str, np.ndarray]:
        temperature = slot_temp[(pref, date)]
        hdd = np.clip(hdd_base - temperature, 0, None)
        cdd = np.clip(temperature - cdd_base, 0, None)
        result = {"HVAC": hdd, "HVAC_eve": hdd * evening, "COOL": cdd, "wfh": active}
        result.update({name: bump for name, bump in zip(env_names, bumps)})
        return result

    def xrow(pref: str, date: str, nonworkday: bool) -> np.ndarray:
        sh = shapes(pref, date)
        columns = []
        for name in names:
            channel = "HVAC" if name in ("HVAC", "HVAC_eve", "COOL") else (
                "wfh" if name == "wfh" else "person")
            values = sh[name].copy()
            if nonworkday and name == "wfh":
                values = np.zeros(48)
            columns.append(pref_den[pref][channel] * values)
        return np.column_stack(columns)

    def nonworkday(date: str) -> bool:
        dt = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8]))
        return dt.weekday() >= 5 or date in HOL

    def collect(months: tuple[str, ...], is_nonworkday: bool, pref: str | None = None):
        xs, ys = [], []
        prefs = [pref] if pref else list(STN)
        for candidate_pref in prefs:
            for (p, date), values in observed.items():
                if p != candidate_pref or date[:6] not in months or nonworkday(date) != is_nonworkday:
                    continue
                xs.append(xrow(candidate_pref, date, is_nonworkday)); ys.append(values)
        return (np.vstack(xs), np.concatenate(ys)) if xs else (None, None)

    rows = []
    for held_out in WINDOW_YMS:
        train_months = tuple(month for month in WINDOW_YMS if month != held_out)
        for is_nonworkday in (False, True):
            x_train, y_train = collect(train_months, is_nonworkday)
            if x_train is None:
                continue
            global_beta = nnls(x_train, y_train)[0]
            for pref in STN:
                x_pref, y_pref = collect(train_months, is_nonworkday, pref)
                beta = (nnls(np.vstack([x_pref, tau * np.eye(len(names))]),
                             np.concatenate([y_pref, tau * global_beta]))[0]
                        if x_pref is not None else global_beta)
                for (candidate_pref, date), values in observed.items():
                    if candidate_pref != pref or date[:6] != held_out or nonworkday(date) != is_nonworkday:
                        continue
                    prediction = xrow(pref, date, is_nonworkday) @ beta
                    rows.extend({"pref": pref, "d": date, "slot": slot,
                                 "obs_mwh": float(values[slot]), "pred_mwh": float(prediction[slot])}
                                for slot in range(48))

    twin = pd.DataFrame(rows).sort_values(["pref", "d", "slot"]).reset_index(drop=True)
    expected = len(observed) * 48
    if len(twin) != expected:
        raise AssertionError(f"pref twin row count changed: expected {expected}, got {len(twin)}")
    twin["pred_mwh"] = twin.pred_mwh.clip(lower=0.0)
    return twin


def _extract_member(source: zipfile.ZipFile, member: str, destination: Path) -> None:
    with source.open(member) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)


def build_lv_load(workers: int = 4, force: bool = False) -> tuple[Path, pd.DataFrame, dict]:
    source_path = Path(paths.input_path("stdstats_zip"))
    destination = Path(paths.input_path("lv_load_hourly", must_exist=False))
    indexed = {row["input_id"]: row for row in manifest.load()}
    source_size = int(indexed["stdstats_zip"].get("byte_count") or 0)
    product_hash = indexed["lv_load_hourly"].get("sha256", "")
    if (not force and destination.exists() and source_path.stat().st_size == source_size
            and product_hash and _sha256(destination) == product_hash):
        load = pd.read_parquet(destination)
        check = _validate_released_midday(load)
        print(f"[stdstats] verified existing {destination.relative_to(paths.ROOT)} ({len(load):,} rows)")
        return destination, load, check

    rows = []
    with zipfile.ZipFile(source_path) as source:
        members = _outer_members(source)
        with tempfile.TemporaryDirectory(prefix="wlpr_stdstats_") as temp:
            for pref in PREFS:
                nested_path = Path(temp) / f"{pref}.zip"
                print(f"[stdstats] extracting pref {pref} LV archive", flush=True)
                _extract_member(source, members[pref], nested_path)
                with zipfile.ZipFile(nested_path) as nested:
                    day_members = sorted(
                        name for name in nested.namelist()
                        if DATE_RE.search(name) and DATE_RE.search(name).group(1) == pref
                    )
                print(f"[stdstats] pref {pref}: parsing {len(day_members)} daily archives", flush=True)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for index, day_rows in enumerate(
                        pool.map(lambda name: _sum_day(nested_path, name), day_members), 1
                    ):
                        rows.extend((pref, date, hour, value, "順", "低圧")
                                    for date, hour, value in day_rows)
                        if index % 20 == 0 or index == len(day_members):
                            print(f"[stdstats] pref {pref}: {index}/{len(day_members)} days", flush=True)

    load = pd.DataFrame(rows, columns=[
        "pref_code", "date_str", "hour_str", "kwh", "flow", "voltage"
    ]).sort_values(["pref_code", "date_str", "hour_str"]).reset_index(drop=True)
    counts = load.groupby(["pref_code", "date_str"]).hour_str.nunique()
    if counts.empty or counts.min() != 48 or counts.max() != 48:
        raise AssertionError(f"standard-statistics day coverage is not 48 slots: {counts.describe().to_dict()}")
    if load.duplicated(["pref_code", "date_str", "hour_str"]).any():
        raise AssertionError("standard-statistics output contains duplicate prefecture-day-slot rows")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    load.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)

    check = _validate_released_midday(load)
    _refresh_manifest(destination, len(load), source_path)
    print(f"[stdstats] wrote {len(load):,} pref-day-slot rows -> {destination.relative_to(paths.ROOT)}")
    return destination, load, check


def _validate_released_midday(load: pd.DataFrame) -> dict:
    reference_path = Path(paths.ROOT) / _asset_location('asset_0130')
    if not reference_path.exists():
        return {"midday_rows": 0, "midday_exact": 0, "midday_max_abs_mwh": None,
                "midday_max_rel": None}
    selected = load[load.hour_str.astype(int).sub(1).between(19, 30)].copy()
    rebuilt = (selected.groupby(["pref_code", "date_str"], as_index=False).kwh.sum()
               .rename(columns={"pref_code": "pref", "date_str": "d", "kwh": "rebuilt"}))
    rebuilt["rebuilt"] /= 1000.0
    reference = pd.read_csv(reference_path, dtype={"pref": str, "d": str})[
        ["pref", "d", "midday_obs"]
    ]
    joined = reference.merge(rebuilt, on=["pref", "d"], how="left")
    difference = (joined.midday_obs - joined.rebuilt).abs()
    check = {
        "midday_rows": len(joined),
        "midday_exact": int((difference < 1e-8).sum()),
        "midday_max_abs_mwh": float(difference.max()),
        "midday_max_rel": float((difference / joined.midday_obs).max()),
    }
    if joined.rebuilt.isna().any() or check["midday_max_rel"] > 0.005:
        raise AssertionError(f"rebuilt LV load does not reproduce released midday observations: {check}")
    print(f"[stdstats] released midday check: {check['midday_exact']}/{check['midday_rows']} exact, "
          f"max rel {check['midday_max_rel']:.3%}")
    return check


def _refresh_manifest(product: Path, row_count: int, source: Path) -> None:
    rows = manifest.load()
    target = next(row for row in rows if row["input_id"] == "lv_load_hourly")
    old_hash = target.get("sha256", "")
    provenance = target.get("provenance", "")
    if old_hash and "legacy_sha256=" not in provenance:
        provenance = (provenance + ";" if provenance else "") + f"legacy_sha256={old_hash}"
    target.update({
        "sha256": _sha256(product),
        "byte_count": str(product.stat().st_size),
        "row_count": str(row_count),
        "materialize": "restricted_raw_aggregate",
        "evidence_level": "restricted_reconstructed",
        "provenance": (provenance + ";" if provenance else "")
                      + f"source={source.relative_to(paths.ROOT)};mesh_slot_total_with_visible_class_fallback",
    })
    with Path(manifest.MANIFEST).open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=manifest.COLUMNS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in manifest.COLUMNS} for row in rows])
    print('[stdstats] refreshed lv_load_hourly in configured asset')


def main() -> None:
    destination, _load, check = build_lv_load(force="--force" in sys.argv)
    score = pd.DataFrame([{"source": "stdstats", "rows": len(_load), **check}])
    score.to_csv(paths.out_path(_asset_location('asset_0058'), "stdstats_input_rebuild.csv"), index=False)
    print(f"[stdstats] rebuild scorecard -> {Path(paths.out_path('scorecards', 'stdstats_input_rebuild.csv')).relative_to(paths.ROOT)}")


if __name__ == "__main__":
    paths.assert_not_source_tree()
    main()
