"""Step 14a — extract FULL 30-min diurnal presence curves per study mesh (weekday vs weekend) from the RAW
LM00020 mobility tar. Ports `_maintainer/builders/extract_mesh_diurnal_presence.py`.

BOTH series: data_m (current-location, activity-weighted) and data_l (population/residence-anchored).

RED LINE: ex_volume = estimated staying-population PRESENCE, NEVER kWh / NEVER WFH-headcount-truth. This is a
SPATIAL + TEMPORAL validation source only; the production WFH weight stays residence × occupation
(use_mobility=False).

PROVENANCE (one-time, heavy): streams the raw `mobility/LM00020.tar` (298 GB, nationwide, 500 m mesh,
columns time,mesh,ex_volume, 288 snapshots/day = every 5 min). Keeps only the 48 half-hour snapshots, filters
to the 4-pref study meshes on the fly, accumulates per (mesh, daytype, slot). Window 2023-12-08..2024-02-29
(winter study window). data_m total swings 42M(00h)->61M(12h) = activity-inflated; data_l ~34M flat =
population-scale (CV 0.145).

RUN-ONCE / IDEMPOTENT: skips a series if its output parquet already exists (delete it or pass --force).
Each series is a ~20-35 min streaming pass; the tiny (~20 MB) frozen parquets are the deliverable consumed
by wfh_validation.py.

Outputs: derived_path("mobility", "mesh_diurnal_presence_{m,l}.parquet")
         (mesh, daytype[wd/we], slot[0-47], presence, n).
"""
from __future__ import annotations

import datetime
import csv
import gzip
import io
import os
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd

from wlpr.io import manifest
from wlpr.io.paths import derived_path, input_path, out_path

TIMES = {f"{h:02d}{mm:02d}" for h in range(24) for mm in (0, 30)}
MIDDAY_SLOTS = set(range(20, 31))
NIGHT_SLOTS = {0, 1, 2, 3, 4, 5, 46, 47}
SPATIAL_TIMES = {f"{slot // 2:02d}{30 if slot % 2 else 0:02d}"
                 for slot in MIDDAY_SLOTS | NIGHT_SLOTS}
HOL = {"20231229", "20231230", "20231231", "20240101", "20240102", "20240103",
       "20240108", "20240211", "20240212", "20240223"}   # JP New-Year + holidays -> weekend bucket


def _out(series: str) -> str:
    return derived_path("mobility", f"mesh_diurnal_presence_{series}.parquet")


def _r3_path() -> str:
    return derived_path("social", "r3_spatial_matching_mesh.parquet")


def _spatial_out() -> str:
    return derived_path("pv", "mesh_mobility_presence.parquet")


def _daily_out() -> str:
    return input_path("mobility_presence_panel", must_exist=False)


def _daytype(date: str) -> str:
    if date in HOL:
        return "we"
    return "we" if datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8])).weekday() >= 5 else "wd"


def _slot(hhmm: str) -> int:
    return int(hhmm[:2]) * 2 + (1 if hhmm[2:] == "30" else 0)


def extract(series: str, study: set, tar_path: str) -> pd.DataFrame:
    pat = re.compile(rf"data_{series}_(\d{{8}})(\d{{4}})\.csv\.gz$")
    acc: dict = {}
    tf = tarfile.open(tar_path, "r:")
    nfile = 0
    for ti in tf:
        if not ti.isfile():
            continue
        mt = pat.search(ti.name)
        if not mt:
            continue
        date, hhmm = mt.group(1), mt.group(2)
        if date > "20240229":
            break
        if date < "20231208" or hhmm not in TIMES:
            continue
        dt, sl = _daytype(date), _slot(hhmm)
        raw = gzip.GzipFile(fileobj=io.BytesIO(tf.extractfile(ti).read())).read().decode("utf-8", "replace")
        for ln in raw.splitlines()[1:]:
            c1 = ln.find(",")
            if c1 < 0:
                continue
            c2 = ln.find(",", c1 + 1)
            if c2 < 0:
                continue
            mesh = ln[c1 + 1:c2]
            if mesh not in study:
                continue
            try:
                v = float(ln[c2 + 1:])
            except ValueError:
                continue
            k = (mesh, dt, sl)
            a = acc.get(k)
            if a is None:
                acc[k] = [v, 1]
            else:
                a[0] += v
                a[1] += 1
        nfile += 1
        if nfile % 400 == 0:
            print(f"  ...{nfile} snapshots (last {date} {hhmm}), keys {len(acc):,}", flush=True)
    tf.close()
    print(f"  processed {nfile} half-hour snapshots -> {len(acc):,} cells", flush=True)
    return pd.DataFrame([dict(mesh=m, daytype=dt, slot=sl, presence=s / n, n=n)
                         for (m, dt, sl), (s, n) in acc.items()])


def _study_mesh_pref() -> tuple[dict[str, str], str]:
    """Resolve the load-bearing study mesh universe from the best available local product."""
    r3 = Path(_r3_path())
    if r3.exists():
        frame = pd.read_parquet(r3, columns=["mesh", "pref"])
        source = "r3_spatial_matching_mesh"
    else:
        grid = Path(out_path("scorecards", "decomp_grid_nmae.parquet"))
        if grid.exists():
            frame = pd.read_parquet(grid, columns=["g", "pref"]).rename(columns={"g": "mesh"})
            source = "decomp_grid_nmae"
        else:
            demo = Path(input_path("demographic_mesh_panel"))
            frame = pd.read_parquet(demo, columns=["mesh", "pref"])
            source = "demographic_mesh_panel_fallback"
    frame["mesh"] = frame.mesh.astype("int64").astype(str)
    frame["pref"] = frame.pref.astype(str).str.zfill(2)
    frame = frame[frame.pref.isin(["11", "12", "13", "14"])].drop_duplicates("mesh")
    return dict(zip(frame.mesh, frame.pref)), source


def extract_spatial_inputs(study_pref: dict[str, str], tar_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the Step-05 mesh windows and frozen daily-pref panel in one LM00020 pass.

    The spatial prior uses data_m because it measures current-location presence and therefore
    contains the weekday commute signal. Only the 19 half-hour midday/night snapshots are read.
    """
    pat = re.compile(r"data_m_(\d{8})(\d{4})\.csv\.gz$")
    study_bytes = {mesh.encode("ascii"): (mesh, pref) for mesh, pref in study_pref.items()}
    # Per mesh: wd_mid sum/n, wd_night sum/n, we_mid sum/n, we_night sum/n.
    mesh_acc: dict[str, list[float]] = {}
    # Per pref/date/window: sum of snapshot totals / number of snapshots.
    daily_acc: dict[tuple[str, str, str], list[float]] = {}
    nfile = 0

    with tarfile.open(tar_path, "r:") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            mt = pat.search(ti.name)
            if not mt:
                continue
            date, hhmm = mt.group(1), mt.group(2)
            if date > "20240229":
                break
            if date < "20231208" or hhmm not in SPATIAL_TIMES:
                continue

            daytype = _daytype(date)
            window = "mid" if _slot(hhmm) in MIDDAY_SLOTS else "night"
            column = {("wd", "mid"): 0, ("wd", "night"): 2,
                      ("we", "mid"): 4, ("we", "night"): 6}[(daytype, window)]
            pref_totals = {pref: 0.0 for pref in ("11", "12", "13", "14")}

            member = tf.extractfile(ti)
            if member is None:
                continue
            with gzip.GzipFile(fileobj=member) as raw:
                next(raw, None)
                for line in raw:
                    c1 = line.find(b",")
                    if c1 < 0:
                        continue
                    c2 = line.find(b",", c1 + 1)
                    if c2 < 0:
                        continue
                    hit = study_bytes.get(line[c1 + 1:c2])
                    if hit is None:
                        continue
                    mesh, pref = hit
                    try:
                        value = float(line[c2 + 1:])
                    except ValueError:
                        continue
                    acc = mesh_acc.setdefault(mesh, [0.0] * 8)
                    acc[column] += value
                    acc[column + 1] += 1
                    pref_totals[pref] += value

            for pref, total in pref_totals.items():
                acc = daily_acc.setdefault((pref, date, window), [0.0, 0.0])
                acc[0] += total
                acc[1] += 1
            nfile += 1
            if nfile % 250 == 0:
                print(f"  ...{nfile} selected snapshots (last {date} {hhmm}), "
                      f"meshes {len(mesh_acc):,}", flush=True)

    rows = []
    for mesh, values in mesh_acc.items():
        means = [values[i] / values[i + 1] if values[i + 1] else float("nan")
                 for i in range(0, 8, 2)]
        rows.append({"mesh": mesh, "pref": study_pref[mesh],
                     "wd_mid": means[0], "wd_night": means[1],
                     "we_mid": means[2], "we_night": means[3],
                     "source_version": "LM00020_data_m_20231208_20240229_rebuild_v1"})
    spatial = pd.DataFrame(rows).sort_values(["pref", "mesh"]).reset_index(drop=True)

    daily_rows = []
    for (pref, date, window), (total, count) in daily_acc.items():
        daily_rows.append({"pref": pref, "date": date, "window": window,
                           "presence": total / count if count else float("nan")})
    daily = pd.DataFrame(daily_rows).pivot(index=["pref", "date"], columns="window", values="presence").reset_index()
    daily.columns.name = None
    daily = daily.rename(columns={"mid": "midday_presence", "night": "night_presence"})
    daily["source_version"] = "LM00020_data_m_20231208_20240229_rebuild_v1"
    daily = daily.sort_values(["pref", "date"]).reset_index(drop=True)
    print(f"  processed {nfile} selected snapshots -> {len(spatial):,} meshes / {len(daily):,} pref-days")
    return spatial, daily


def _atomic_write(frame: pd.DataFrame, destination: str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.unlink(missing_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False, compression="zstd")
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _refresh_daily_manifest(row_count: int) -> None:
    rows = manifest.load()
    target = next(row for row in rows if row["input_id"] == "mobility_presence_panel")
    old_hash = target.get("sha256", "")
    provenance = target.get("provenance", "")
    if old_hash and "legacy_sha256=" not in provenance:
        provenance = (provenance + ";" if provenance else "") + f"legacy_sha256={old_hash}"
    product = Path(_daily_out())
    target.update({
        "sha256": manifest.sha256_file(str(product)), "byte_count": str(product.stat().st_size),
        "row_count": str(row_count), "materialize": "restricted_raw_rebuild",
        "evidence_level": "restricted_raw_reconstructed",
        "provenance": (provenance + ";" if provenance else "") +
                      "source=data/raw/mobility/LM00020.tar;series=data_m;window=20231208..20240229",
    })
    with Path(manifest.MANIFEST).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest.COLUMNS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in manifest.COLUMNS} for row in rows])


def build_spatial_inputs(force: bool = False) -> tuple[str, str]:
    spatial_out, daily_out = _spatial_out(), _daily_out()
    if not force and os.path.exists(spatial_out) and os.path.exists(daily_out):
        print(f"[skip] {spatial_out} and {daily_out} exist (pass --force to rebuild)")
        return spatial_out, daily_out
    study_pref, source = _study_mesh_pref()
    print(f"[spatial] study meshes: {len(study_pref):,} ({source})")
    spatial, daily = extract_spatial_inputs(study_pref, input_path("mobility_lm00020_tar"))
    _atomic_write(spatial, spatial_out)
    _atomic_write(daily, daily_out)
    _refresh_daily_manifest(len(daily))
    print(f"[saved] {spatial_out} ({len(spatial):,} rows)")
    print(f"[saved] {daily_out} ({len(daily):,} rows; manifest refreshed)")
    return spatial_out, daily_out


def main():
    force = "--force" in sys.argv
    if "--spatial-only" in sys.argv:
        build_spatial_inputs(force=force)
        print("extract_mesh_mobility_inputs: DONE")
        return
    tar_path = input_path("mobility_lm00020_tar")
    study = set(pd.read_parquet(_r3_path()).mesh.astype("int64").astype(str))
    print(f"study meshes: {len(study)}")
    for series in ("m", "l"):
        out = _out(series)
        if os.path.exists(out) and not force:
            print(f"[skip] {os.path.basename(out)} exists (pass --force to rebuild)")
            continue
        print(f"[extract] data_{series}_ ...")
        df = extract(series, study, tar_path)
        df.to_parquet(out, index=False)
        print(f"[saved] {out}  ({len(df):,} rows; meshes {df.mesh.nunique():,})")
    print("extract_mesh_diurnal_presence: DONE")


if __name__ == "__main__":
    main()
