"""Rebuild public census, weather and REPOS inputs from pinned downloads.

The historical reproduction package shipped processed Parquet snapshots but omitted
their public-source ingestion code. This module restores that lineage. Downloads are
SHA256-pinned and writes are atomic, so an interrupted or changed source cannot replace
a valid input.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wlpr.io import manifest, paths


CONFIG = Path(paths.CONFIGS) / "public_sources.yaml"
PREFS = {"11", "12", "13", "14"}
METEOSTAT_COLUMNS = [
    "date", "hour", "temp", "dwpt", "rhum", "prcp", "snow",
    "wdir", "wspd", "wpgt", "pres", "tsun", "coco",
]


def _config() -> dict:
    with CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _raw_path(input_id: str) -> Path:
    tier, rel = paths._inputs_index()[input_id]
    if tier != "raw":
        raise ValueError(f"public source {input_id} must be registered in the raw tier")
    return Path(paths.TIERS[tier]) / rel


def _download(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(destination) == expected_sha256:
        print(f"[public] verified existing {destination.relative_to(paths.ROOT)}")
        return destination

    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "wlpr-public-input-rebuilder/1.0"})
    print(f"[public] download {url}")
    with urllib.request.urlopen(request, timeout=120) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
    got = _sha256(tmp)
    if got != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"source hash changed for {url}: expected {expected_sha256}, got {got}")
    os.replace(tmp, destination)
    print(f"[public] saved {destination.relative_to(paths.ROOT)} ({destination.stat().st_size:,} bytes)")
    return destination


def _public_specs(cfg: dict) -> list[dict]:
    specs = list(cfg["census"]["prefectures"].values())
    for table in cfg.get("normalized_census", {}).get("tables", {}).values():
        specs.extend(table["prefectures"].values())
    specs.extend(cfg["weather"]["stations"].values())
    specs.append(cfg["repos"])
    return specs


def fetch_sources(cfg: dict) -> dict[str, Path]:
    fetched: dict[str, Path] = {}
    for spec in _public_specs(cfg):
        iid = spec["input_id"]
        fetched[iid] = _download(spec["url"], _raw_path(iid), spec["sha256"])
    return fetched


def _atomic_parquet(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.unlink(missing_ok=True)
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, destination)


def _atomic_csv(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, destination)


def _read_mesh_table(table: str, specs: dict, fetched: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for pref, spec in specs.items():
        archive = fetched[spec["input_id"]]
        with zipfile.ZipFile(archive) as zf:
            member = next(name for name in zf.namelist() if name.lower().endswith(".txt"))
            with zf.open(member) as raw:
                reader = csv.DictReader(line.decode("cp932", errors="replace") for line in raw)
                rows = []
                for row in reader:
                    mesh = (row.get("KEY_CODE") or "").strip()
                    if len(mesh) != 9 or not mesh.isdigit():
                        continue
                    out = {"mesh": mesh, "pref": str(pref)}
                    out.update({key: value for key, value in row.items() if key and key.startswith(table)})
                    rows.append(out)
        frames.append(pd.DataFrame(rows))

    frame = pd.concat(frames, ignore_index=True)
    value_cols = [col for col in frame if col.startswith(table)]
    for col in value_cols:
        frame[col] = pd.to_numeric(frame[col].replace({"*": None, "-": None, "X": None}), errors="coerce")
    priority = f"{table}001"
    frame["_priority"] = frame[priority].fillna(-1) if priority in frame else -1
    return (frame.sort_values(["mesh", "_priority", "pref"], ascending=[True, False, True])
            .drop_duplicates("mesh", keep="first").drop(columns="_priority").reset_index(drop=True))


INDUSTRY_FIELDS = {
    "industry_A_agriculture_forestry": "T001193016",
    "industry_B_fisheries": "T001193019",
    "industry_C_mining": "T001193025",
    "industry_D_construction": "T001193028",
    "industry_E_manufacturing": "T001193031",
    "industry_F_utilities": "T001193037",
    "industry_G_information_communications": "T001193040",
    "industry_H_transport_postal": "T001193043",
    "industry_I_wholesale_retail": "T001193046",
    "industry_J_finance_insurance": "T001193049",
    "industry_K_real_estate_leasing": "T001193052",
    "industry_L_professional_technical": "T001193055",
    "industry_M_accommodation_food": "T001193058",
    "industry_N_lifestyle_entertainment": "T001193061",
    "industry_O_education": "T001193064",
    "industry_P_medical_welfare": "T001193067",
    "industry_Q_cooperative_services": "T001193070",
    "industry_R_other_services": "T001193073",
    "industry_S_government": "T001193076",
}

OCCUPATION_FIELDS = {
    "occupation_A_managerial": "T001193079",
    "occupation_B_professional_technical": "T001193082",
    "occupation_C_clerical": "T001193085",
    "occupation_D_sales": "T001193088",
    "occupation_E_service": "T001193091",
    "occupation_F_security": "T001193094",
    "occupation_G_agriculture_fishery": "T001193097",
    "occupation_H_production_process": "T001193100",
    "occupation_I_transport_machine": "T001193103",
    "occupation_J_construction_mining": "T001193106",
    "occupation_K_carrying_cleaning_packaging": "T001193109",
}


def build_normalized_census(cfg: dict, fetched: dict[str, Path], housing_path: Path) -> list[tuple[str, Path, int]]:
    table_cfg = cfg["normalized_census"]["tables"]
    t101 = _read_mesh_table("T001101", table_cfg["T001101"]["prefectures"], fetched)
    t108 = _read_mesh_table("T001108", table_cfg["T001108"]["prefectures"], fetched)
    t193 = _read_mesh_table("T001193", table_cfg["T001193"]["prefectures"], fetched)
    t194 = pd.read_parquet(housing_path)
    t194 = t194.rename(columns={"pref_code": "pref"})

    keep101 = ["mesh", "pref", "T001101001", "T001101019", "T001101035"]
    keep108 = ["mesh", "T001108001", "T001108034", "T001108084", "T001108085", "T001108088"]
    keep193 = ["mesh", "T001193001", "T001193004", "T001193010"] + list(INDUSTRY_FIELDS.values()) + list(OCCUPATION_FIELDS.values())
    keep194 = ["mesh", "T001194001", "T001194007", "T001194008", "T001194009", "T001194020"]
    merged = (t101[keep101].merge(t108[keep108], on="mesh", how="left")
              .merge(t193[keep193], on="mesh", how="left")
              .merge(t194[keep194], on="mesh", how="left"))

    pop = merged["T001101001"].fillna(merged["T001194020"]).fillna(0.0).clip(lower=0.0)
    households = merged["T001194001"].fillna(merged["T001101035"]).fillna(0.0).clip(lower=0.0)
    housing_people = merged["T001194020"].fillna(pop).clip(lower=0.0)
    employees = merged["T001108001"].fillna(merged["T001193004"]).fillna(0.0).clip(lower=0.0)
    elderly = merged["T001101019"].fillna(0.0).clip(lower=0.0)
    office_occupations = merged[list(OCCUPATION_FIELDS.values())[:4]].fillna(0.0).sum(axis=1).clip(lower=0.0)
    typed_dwellings = merged[["T001194007", "T001194008", "T001194009"]].fillna(0.0).sum(axis=1)
    conditioned = typed_dwellings.where(typed_dwellings > 0, households)

    ratio = lambda num, den: np.divide(num, den, out=np.zeros(len(merged), dtype=float), where=den > 0)
    demographic = pd.DataFrame({
        "mesh": merged.mesh.astype(str), "pref": merged.pref.astype(str), "pop": pop,
        "mean_hh_size": ratio(housing_people.to_numpy(float), households.to_numpy(float)),
        "wage_share": np.clip(ratio(employees.to_numpy(float), pop.to_numpy(float)), 0.0, 1.0),
        "eld_pop_share": np.clip(ratio(elderly.to_numpy(float), pop.to_numpy(float)), 0.0, 1.0),
        "source_version": "estat_2020_T001101_T001108_T001193_T001194_rebuild_v1",
    })
    denominator = pd.DataFrame({
        "mesh": merged.mesh.astype(str), "pref": merged.pref.astype(str),
        "conditioned_dwelling": conditioned.astype(float), "person": pop.astype(float),
        "wfh_eligible": office_occupations.astype(float),
        "source_version": "estat_2020_office_occupation_eligibility_rebuild_v1",
    })

    normalized = Path(paths.TIERS["frozen"]) / "normalized_inputs"
    demo_path = normalized / "step11h_0_demographic_mesh_panel.parquet"
    den_path = normalized / "step11g_1_mesh_denominator_panel.parquet"
    _atomic_parquet(demographic, demo_path)
    _atomic_parquet(denominator, den_path)

    scenario_dir = Path(paths.TIERS["derived"]) / "wfh_scenarios"
    selected193 = pd.DataFrame({
        "mesh": t193.mesh.astype(str),
        "labor_force_total": t193["T001193001"].fillna(0.0),
        "non_labor_force_total": t193["T001193010"].fillna(0.0),
    })
    for name, source in {**INDUSTRY_FIELDS, **OCCUPATION_FIELDS}.items():
        selected193[name] = t193[source].fillna(0.0)
    selected108 = pd.DataFrame({
        "mesh_code": t108.mesh.astype(str),
        "non_ag_worker_households": t108["T001108084"].fillna(0.0),
        "non_worker_households": t108["T001108085"].fillna(0.0),
        "resident_student_total": t108["T001108088"].fillna(0.0),
    })
    _atomic_csv(selected193, scenario_dir / "t001193_mesh_industry_occupation_selected.csv")
    _atomic_csv(selected108, scenario_dir / "t001108_mesh_employment_commute_selected.csv")
    print(f"[census] normalized demographic/denominator panels: {len(demographic):,} meshes")
    return [("demographic_mesh_panel", demo_path, len(demographic)),
            ("mesh_denominator_panel", den_path, len(denominator))]


def build_survey_anchor() -> tuple[str, Path, int]:
    rates = paths.model_config()["survey_anchor"]["pref_rates"]
    frame = pd.DataFrame([{"pref": str(pref), "rate_total": float(rate),
                           "provenance": "recovered_from_released_wfh_anchor_scorecard"}
                          for pref, rate in rates.items()])
    destination = Path(paths.TIERS["frozen"]) / "normalized_inputs/wfh_survey_anchored_pref.parquet"
    _atomic_parquet(frame, destination)
    return "wfh_survey_pref_rates", destination, len(frame)


def build_census(cfg: dict, fetched: dict[str, Path]) -> tuple[Path, int, dict]:
    frames = []
    table = cfg["census"]["table"]
    for pref, spec in cfg["census"]["prefectures"].items():
        archive = fetched[spec["input_id"]]
        member = f"tbl{table}H{pref}.txt"
        with zipfile.ZipFile(archive) as zf, zf.open(member) as raw:
            # Headers are ASCII and labels are CP932. Keep only data fields required by the model.
            text = (line.decode("cp932", errors="replace") for line in raw)
            reader = csv.DictReader(text)
            rows = []
            for row in reader:
                mesh = (row.get("KEY_CODE") or "").strip()
                if len(mesh) != 9 or not mesh.isdigit():
                    continue
                out = {"mesh": mesh, "pref_code": pref}
                for key, value in row.items():
                    if key and key.startswith(table):
                        out[key] = value
                rows.append(out)
        frame = pd.DataFrame(rows)
        frames.append(frame)
        print(f"[census] pref {pref}: {len(frame):,} mesh rows")

    census = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in census if c.startswith(table)]
    for col in value_cols:
        census[col] = pd.to_numeric(census[col].replace({"*": None, "-": None, "X": None}), errors="coerce")

    # A border mesh can occur in multiple prefecture extracts. The electrical project assigns each
    # mesh to one prefecture, so retain the record with the largest residential household count.
    census["_hh"] = census[f"{table}001"].fillna(-1)
    census = (census.sort_values(["mesh", "_hh", "pref_code"], ascending=[True, False, True])
              .drop_duplicates("mesh", keep="first").drop(columns="_hh").reset_index(drop=True))
    census = census[["mesh", "pref_code"] + sorted(value_cols)]

    destination = Path(paths.TIERS["frozen"]) / "raw_sources/estat_r2_census_500m_T001194_project_mesh_join_check.parquet"
    _atomic_parquet(census, destination)

    check = {"project_meshes": 0, "matched": 0, "pref_agreement": None}
    reference = Path(paths.ROOT) / "outputs/scorecards/decomp_grid_nmae.parquet"
    if reference.exists():
        ref = pd.read_parquet(reference, columns=["g", "pref"]).drop_duplicates("g")
        ref["g"] = ref.g.astype(str); ref["pref"] = ref.pref.astype(str)
        joined = ref.merge(census[["mesh", "pref_code"]], left_on="g", right_on="mesh", how="left")
        hit = joined.pref_code.notna()
        check = {
            "project_meshes": len(ref),
            "matched": int(hit.sum()),
            "pref_agreement": float((joined.loc[hit, "pref"] == joined.loc[hit, "pref_code"]).mean()),
        }
        print(f"[census] project reference: {check['matched']:,}/{check['project_meshes']:,} have housing rows; "
              f"pref agreement {check['pref_agreement']:.3f}")
    print(f"[census] wrote {len(census):,} unique meshes -> {destination.relative_to(paths.ROOT)}")
    return destination, len(census), check


def build_weather(cfg: dict, fetched: dict[str, Path]) -> tuple[Path, int, dict]:
    start = pd.Timestamp(cfg["weather"]["start"])
    end_exclusive = pd.Timestamp(cfg["weather"]["end"]) + pd.Timedelta(days=1)
    frames = []
    for station_id, spec in cfg["weather"]["stations"].items():
        source = fetched[spec["input_id"]]
        frame = pd.read_csv(source, compression="gzip", header=None, names=METEOSTAT_COLUMNS)
        hour = frame["hour"].astype(str).str.zfill(2)
        frame.insert(0, "timestamp", pd.to_datetime(frame["date"] + " " + hour + ":00:00", errors="coerce"))
        frame.insert(1, "station", spec["station"])
        frame.insert(2, "station_id", station_id)
        frame = frame[(frame.timestamp >= start) & (frame.timestamp < end_exclusive)].copy()
        frame = frame.drop(columns=["date", "hour"])
        for col in METEOSTAT_COLUMNS[2:]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames.append(frame)
        print(f"[weather] {spec['station']} ({station_id}): {len(frame):,} UTC hours")

    weather = pd.concat(frames, ignore_index=True).sort_values(["station", "timestamp"]).reset_index(drop=True)
    destination = Path(paths.TIERS["frozen"]) / "raw_sources/amedas_47prefs_hourly.parquet"
    _atomic_parquet(weather, destination)

    check = {"hdd_rows": 0, "hdd_exact": 0, "hdd_max_abs": None}
    reference = Path(paths.ROOT) / "outputs/scorecards/midday_diag_perday.csv"
    if reference.exists():
        pref_by_station = {"埼玉熊谷": "11", "千葉": "12", "東京": "13", "神奈川横浜": "14"}
        daily = weather[["timestamp", "station", "temp"]].copy()
        daily["d"] = (daily.timestamp + pd.Timedelta(hours=9)).dt.strftime("%Y%m%d")
        daily = daily.groupby(["station", "d"], as_index=False).temp.mean()
        daily["pref"] = daily.station.map(pref_by_station)
        daily["hdd_rebuilt"] = (18.0 - daily.temp).clip(lower=0.0)
        old = pd.read_csv(reference, dtype={"pref": str, "d": str})[["pref", "d", "hdd"]]
        joined = old.merge(daily[["pref", "d", "hdd_rebuilt"]], on=["pref", "d"], how="left")
        diff = (joined.hdd - joined.hdd_rebuilt).abs()
        check = {
            "hdd_rows": len(joined),
            "hdd_exact": int((diff < 1e-12).sum()),
            "hdd_max_abs": float(diff.max()),
        }
        if joined.hdd_rebuilt.isna().any() or check["hdd_max_abs"] > 1e-10:
            raise AssertionError(f"rebuilt weather does not reproduce the released HDD: {check}")
        print(f"[weather] released HDD check: {check['hdd_exact']}/{check['hdd_rows']} exact, "
              f"max abs {check['hdd_max_abs']:.3g}")
    print(f"[weather] wrote {len(weather):,} rows -> {destination.relative_to(paths.ROOT)}")
    return destination, len(weather), check


def latlon_to_mesh500(lat: float, lon: float) -> str:
    """Convert a geographic point to the Japanese 9-digit half (500 m) mesh code."""
    y = lat * 1.5
    y1 = int(np.floor(y)); y = (y - y1) * 8
    y2 = int(np.floor(y)); y = (y - y2) * 10
    y3 = int(np.floor(y)); y4 = int(np.floor((y - y3) * 2))

    x = lon - 100.0
    x1 = int(np.floor(x)); x = (x - x1) * 8
    x2 = int(np.floor(x)); x = (x - x2) * 10
    x3 = int(np.floor(x)); x4 = int(np.floor((x - x3) * 2))
    quadrant = y4 * 2 + x4 + 1
    return f"{y1:02d}{x1:02d}{y2}{x2}{y3}{x3}{quadrant}"


def mesh500_centroid(mesh: str) -> tuple[float, float]:
    s = str(mesh).zfill(9)
    y1, x1 = int(s[:2]), int(s[2:4])
    y2, x2, y3, x3, q = map(int, s[4:])
    y4, x4 = (q - 1) // 2, (q - 1) % 2
    lat = (y1 + y2 / 8 + y3 / 80 + y4 / 160 + 1 / 320) / 1.5
    lon = 100 + x1 + x2 / 8 + x3 / 80 + x4 / 160 + 1 / 320
    return lat, lon


def build_repos(cfg: dict, fetched: dict[str, Path]) -> tuple[Path, Path, int, dict]:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("REPOS conversion requires pyshp>=2.3; install the project environment") from exc

    archive = fetched[cfg["repos"]["input_id"]]
    with tempfile.TemporaryDirectory(prefix="wlpr_repos_") as tmp:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if Path(name).suffix.lower() in {".shp", ".shx", ".dbf", ".cpg", ".prj"}:
                    zf.extract(name, tmp)
        shp = Path(tmp) / "R3_pv_grid500.shp"
        reader = shapefile.Reader(str(shp), encoding="utf-8")
        fields = [field[0] for field in reader.fields[1:]]
        positions = {name: fields.index(name) for name in ["MUNIC_CODE", "PV_B999_IC", "PV_B999_PG"]}

        selected = []
        for index, record in enumerate(reader.iterRecords()):
            munic = str(record[positions["MUNIC_CODE"]]).strip()
            if munic[:2] not in PREFS:
                continue
            selected.append((index, munic, float(record[positions["PV_B999_IC"]] or 0.0),
                             float(record[positions["PV_B999_PG"]] or 0.0)))

        rows = []
        for count, (index, munic, b_ic, b_pg) in enumerate(selected, 1):
            bbox = reader.shape(index).bbox
            lon, lat = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            mesh = latlon_to_mesh500(lat, lon)
            rows.append((mesh, munic, b_ic, b_pg))
            if count % 10000 == 0:
                print(f"[repos] converted {count:,}/{len(selected):,} selected polygons")
        reader.close()

    repos = pd.DataFrame(rows, columns=["mesh", "munic", "B_IC", "B_PG"])
    repos = repos.sort_values(["mesh", "munic"]).reset_index(drop=True)
    capacity_gw = float(repos.B_IC.sum() / 1e6)
    expected = float(cfg["repos"]["four_pref_building_capacity_gw"])
    if abs(capacity_gw - expected) > 5e-6:
        raise AssertionError(f"REPOS four-pref capacity changed: expected {expected:.6f}, got {capacity_gw:.6f}")

    for mesh in repos.mesh.head(100):
        if latlon_to_mesh500(*mesh500_centroid(mesh)) != mesh:
            raise AssertionError(f"JIS mesh conversion does not round-trip: {mesh}")

    potential = Path(paths.TIERS["derived"]) / "pv/repos_mesh500_potential.parquet"
    bridge = Path(paths.TIERS["frozen"]) / "pv/repos_mesh_munic_bpg.parquet"
    _atomic_parquet(repos, potential)
    _atomic_parquet(repos[["mesh", "munic", "B_PG"]], bridge)
    check = {
        "records": len(repos),
        "unique_meshes": int(repos.mesh.nunique()),
        "capacity_gw": capacity_gw,
    }
    print(f"[repos] four-pref B_IC {capacity_gw:.6f} GW across {len(repos):,} records / "
          f"{check['unique_meshes']:,} JIS meshes")
    print(f"[repos] wrote {potential.relative_to(paths.ROOT)} and {bridge.relative_to(paths.ROOT)}")
    return potential, bridge, len(repos), check


def _refresh_manifest(cfg: dict, products: list[tuple[str, Path, int]]) -> None:
    rows = manifest.load()
    indexed = {row["input_id"]: row for row in rows}

    for spec in _public_specs(cfg):
        iid = spec["input_id"]
        tier, rel = paths._inputs_index()[iid]
        path = Path(paths.TIERS[tier]) / rel
        indexed[iid] = {
            "input_id": iid, "tier": tier, "relative_path": rel,
            "sha256": spec["sha256"], "byte_count": str(path.stat().st_size), "row_count": "",
            "materialize": "download", "schema_path": "", "evidence_level": "public_raw",
            "allowed_use": "rebuild", "forbidden_use": "", "provenance": spec["url"],
        }

    for iid, product, row_count in products:
        old = indexed[iid]
        old_hash = old.get("sha256", "")
        provenance = old.get("provenance", "")
        if old_hash and "legacy_sha256=" not in provenance:
            provenance = (provenance + ";" if provenance else "") + f"legacy_sha256={old_hash}"
        recovered = iid == "wfh_survey_pref_rates"
        old.update({
            "sha256": _sha256(product), "byte_count": str(product.stat().st_size),
            "row_count": str(row_count),
            "materialize": "released_scorecard_recovery" if recovered else "public_rebuild",
            "evidence_level": "released_parameter_recovery" if recovered else "public_reconstructed",
            "provenance": provenance,
        })

    ordered_ids = [row["input_id"] for row in rows]
    ordered_ids += sorted(set(indexed) - set(ordered_ids))
    with Path(manifest.MANIFEST).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest.COLUMNS)
        writer.writeheader()
        writer.writerows([{key: indexed[iid].get(key, "") for key in manifest.COLUMNS} for iid in ordered_ids])
    print(f"[public] refreshed hashes for rebuilt inputs in {Path(manifest.MANIFEST).relative_to(paths.ROOT)}")


def main() -> None:
    cfg = _config()
    fetched = fetch_sources(cfg)
    census_path, census_rows, census_check = build_census(cfg, fetched)
    weather_path, weather_rows, weather_check = build_weather(cfg, fetched)
    _repos_path, bridge_path, repos_rows, repos_check = build_repos(cfg, fetched)
    normalized_products = build_normalized_census(cfg, fetched, census_path)
    survey_product = build_survey_anchor()
    _refresh_manifest(cfg, [
        ("census_housing_T001194", census_path, census_rows),
        ("weather_amedas", weather_path, weather_rows),
        ("repos_mesh_munic_bpg", bridge_path, repos_rows),
        *normalized_products,
        survey_product,
    ])

    score = pd.DataFrame([
        {"source": "census", "rows": census_rows, **census_check},
        {"source": "weather", "rows": weather_rows, **weather_check},
        {"source": "repos", "rows": repos_rows, **repos_check},
    ])
    destination = Path(paths.out_path("scorecards", "public_input_rebuild.csv"))
    score.to_csv(destination, index=False)
    print(f"[public] rebuild scorecard -> {destination.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    paths.assert_not_source_tree()
    main()
