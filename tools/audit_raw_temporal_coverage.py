#!/usr/bin/env python3
"""Audit nested delivered archives, independently of documentation/derived windows.

No extraction to disk. Every Standard Statistics inner member is inventoried;
every LV-forward daily CSV is opened and its first data record checked. For
other standard products one daily CSV per month is opened. Every owner monthly
CSV is opened and its leading 00_all series read through the first mesh record.
This proves aggregate temporal coverage, NOT full mesh completeness/CRC health.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures/extra/v4.7/nc_completion/raw_coverage"
STD = ROOT / "data/raw/stdstats/standard_statistics.zip"
BTM = ROOT / "data/raw/btm/Hitotsubashi_SelfConsumption_LV.zip"


def stamp(p):
    st = p.stat()
    return {"path": str(p.resolve()), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def standard():
    members, samples, content = [], [], []
    with zipfile.ZipFile(STD) as outer:
        for info in outer.infolist():
            match = re.search(r"/(\d{2})_[^/]+/(\d{2})_[^/]+\.zip$", info.filename)
            if not match:
                continue
            pref, product = match.groups()
            print(f"Standard pref={pref} product={product}: reading nested archive", flush=True)
            # At most one (up to 2.4 GB) nested payload resides in memory.
            payload = io.BytesIO(outer.read(info.filename))
            with zipfile.ZipFile(payload) as inner:
                seen_months = set()
                for item in inner.infolist():
                    m = re.search(r"-(\d{6})-.*_(\d{2})\.csv\.zip$", item.filename)
                    if not m:
                        if not item.is_dir():
                            month_match = re.search(r"-(\d{6})-.*\.csv(?:\.zip)?$", item.filename)
                            ym = month_match.group(1) if month_match else ""
                            members.append(dict(pref=pref, product=product, archive=info.filename,
                                                member=item.filename, date="", month=ym,
                                                recognized=bool(month_match), grain="monthly",
                                                nested_size_bytes=item.file_size, nested_crc32=item.CRC))
                            if month_match:
                                def scan(raw):
                                    header = next(csv.reader([raw.readline().decode('utf-8-sig')]))
                                    date_column = next(c for c in header if c.startswith('対象年月'))
                                    reader = pacsv.open_csv(raw,
                                        read_options=pacsv.ReadOptions(block_size=4 << 20, use_threads=False, column_names=header),
                                        convert_options=pacsv.ConvertOptions(
                                            include_columns=[date_column],
                                            column_types={date_column:pa.string()}))
                                    total = 0
                                    actual_dates = set()
                                    for batch in reader:
                                        total += batch.num_rows
                                        actual_dates.update(pc.unique(batch.column(0)).to_pylist())
                                    for actual in sorted(v for v in actual_dates if v is not None):
                                        content.append(dict(pref=pref, product=product, member=item.filename,
                                                            filename_month=ym, actual_record_date=actual,
                                                            file_total_rows=total,
                                                            date_column=date_column,
                                                            scope="full CSV date column scanned; CRC checked at EOF"))
                                if item.filename.endswith('.zip'):
                                    with zipfile.ZipFile(io.BytesIO(inner.read(item.filename))) as daily:
                                        cn = next(n for n in daily.namelist() if n.endswith('.csv'))
                                        with daily.open(cn) as raw:
                                            scan(raw)
                                else:
                                    with inner.open(item.filename) as raw:
                                        scan(raw)
                        continue
                    ym, day = m.groups()
                    date = ym + day
                    members.append(dict(pref=pref, product=product, archive=info.filename,
                                        member=item.filename, date=date, month=ym, recognized=True,
                                        grain="daily",
                                        nested_size_bytes=item.file_size, nested_crc32=item.CRC))
                    if product != "01" and ym in seen_months:
                        continue
                    seen_months.add(ym)
                    with zipfile.ZipFile(io.BytesIO(inner.read(item.filename))) as daily:
                        names = [n for n in daily.namelist() if n.lower().endswith('.csv')]
                        if len(names) != 1:
                            raise ValueError(f"Unexpected CSV membership: {names}")
                        with daily.open(names[0]) as raw:
                            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                            record = next(reader)
                            actual = record["対象年月（日）"]
                            samples.append(dict(pref=pref, product=product, member=item.filename,
                                                expected_date=date, actual_record_date=actual,
                                                date_matches=actual == date,
                                                slot=record.get("時刻コード"),
                                                mesh=record.get("メッシュコード"),
                                                schema=json.dumps(reader.fieldnames, ensure_ascii=False),
                                                scope="first_data_record; not full daily mesh completeness"))
            payload.close()
            pd.DataFrame(members).to_csv(OUT / "standard_members.csv", index=False)
            pd.DataFrame(samples).to_csv(OUT / "standard_record_checks.csv", index=False)
            pd.DataFrame(content).to_csv(OUT / "standard_monthly_actual_dates.csv", index=False)
    m = pd.DataFrame(members)
    rows = []
    for (pref, product, month), x in m[m.recognized].groupby(["pref", "product", "month"]):
        expected = pd.Period(month[:4]+"-"+month[4:], freq="M").days_in_month
        dates = {v for v in x.date if v}
        missing = [f"{month}{day:02}" for day in range(1, expected+1) if f"{month}{day:02}" not in dates]
        rows.append(dict(pref=pref, product=product, month=month, days=len(dates),
                         expected_days=expected, missing_days=";".join(missing),
                         duplicate_days=len(x)-len(dates) if len(dates) else 0,
                         grain="daily" if len(dates) else "monthly_file; see actual_dates report"))
    pd.DataFrame(rows).to_csv(OUT / "standard_month_coverage.csv", index=False)


def owners():
    records, summary = [], []
    with zipfile.ZipFile(BTM) as outer:
        for info in outer.infolist():
            match = re.search(r"solar_low-(\d{2})-(\d{6})\.zip$", info.filename)
            if not match:
                continue
            pref, ym = match.groups()
            with zipfile.ZipFile(io.BytesIO(outer.read(info.filename))) as inner:
                name = next(n for n in inner.namelist() if n.endswith('.csv'))
                with inner.open(name) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                    timestamps = []
                    first_mesh = None
                    for record in reader:
                        if record["メッシュ"] != "00_all":
                            first_mesh = record["メッシュ"]
                            break
                        ts = record["年月日時分"]
                        timestamps.append(ts)
                        records.append(dict(pref=pref, month=ym, timestamp=ts,
                                            self_consumption_kWh=record["自家消費量"],
                                            import_kWh=record["電力量_供給"],
                                            export_kWh=record["電力量_受電"],
                                            installed_capacity_kW=record["発電設備容量"],
                                            contracts=record["契約口数"]))
            times = pd.to_datetime(pd.Series(timestamps), format="%Y%m%d %H:%M")
            start = pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01")
            expected = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="30min", inclusive="left")
            missing = expected.difference(pd.DatetimeIndex(times))
            unexpected = pd.DatetimeIndex(times).difference(expected)
            summary.append(dict(pref=pref, month=ym, aggregate_records=len(times),
                                unique_days=times.dt.date.nunique(), expected_slots=len(expected),
                                missing_slots=len(missing), unexpected_slots=len(unexpected),
                                duplicate_slots=int(times.duplicated().sum()),
                                actual_min=str(times.min()), actual_max=str(times.max()),
                                first_mesh_after_aggregate=first_mesh,
                                archive=info.filename, outer_member_crc32=info.CRC,
                                scope="all leading 00_all records; individual meshes not fully scanned"))
            print(f"Owner pref={pref} month={ym}: {len(times)} actual aggregate slots; missing={len(missing)}", flush=True)
            pd.DataFrame(summary).to_csv(OUT / "owner_month_coverage.csv", index=False)
    pd.DataFrame(records).to_csv(OUT / "owner_aggregate_records.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {"standard": stamp(STD), "owners": stamp(BTM)}
    standard()
    owners()
    s = pd.read_csv(OUT / "standard_month_coverage.csv", dtype={"pref":str,"product":str,"month":str})
    b = pd.read_csv(OUT / "owner_month_coverage.csv", dtype={"pref":str,"month":str})
    checks = pd.read_csv(OUT / "standard_record_checks.csv")
    result = {"sources":sources,
              "standard_months_by_product":s.groupby('product')['month'].unique().apply(list).to_dict(),
              "standard_lv_forward_months_by_pref":s[s['product'].eq('01')].groupby('pref')['month'].unique().apply(list).to_dict(),
              "owner_months_by_pref":b.groupby('pref')['month'].unique().apply(list).to_dict(),
              "standard_daily_record_checks":len(checks),
              "standard_date_mismatch_count":int((~checks.date_matches).sum()),
              "owner_missing_aggregate_slots":int(b.missing_slots.sum()),
              "owner_unexpected_aggregate_slots":int(b.unexpected_slots.sum()),
              "owner_duplicate_aggregate_slots":int(b.duplicate_slots.sum()),
              "standard_non_daily_date_scan":"standard_monthly_actual_dates.csv; full date-column scan rather than filenames",
              "limitation":"Archive membership and specified actual CSV records verified; full mesh/day/slot completeness and full CSV CRC are not claimed."}
    (OUT / 'coverage_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
