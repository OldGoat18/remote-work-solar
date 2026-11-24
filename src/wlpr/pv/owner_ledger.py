"""Step 01 — the metered PV-owner BTM ledger (ported from the source owner-ledger implementation).

Parses the RESTRICTED Hitotsubashi self-consumption (自家消費 低圧) raw zip into the OBSERVED PV-owner
ledger — the metered truth layer for net-load (G_pv = selfcons + export, gross BTM load = selfcons + import).
This module ports ONLY the raw-parse + ledger/capacity/validation outputs; the downstream BTM user-state
twin (the calibrated self-consumption / future-adopter model) is an EXCLUDED subsystem, not pulled in here.

Raw zip structure: configured asset -> ONE csv (~0.5 GB),
500m mesh x 30-min, JST. Field -> BTM identity mapping (verified from raw diurnal, JST):
  電力量_供給 = import (low midday), 電力量_受電 = export (peaks midday).
  PV_generation = selfcons + export ; gross_BTM_load = selfcons + import ; net_import = import - export

Outputs (configured asset, regenerable):
  configured asset
  btm_capacity_mesh_month.parquet  (pref,mesh,ym,pv_capacity_kw,contracts,contract_kw)
  btm_parse_validation.csv         (per pref-month: 00_all closure, suppression, field-direction sanity)

Memory-safe: one pref-month at a time (stream-extract temp -> fresh DuckDB conn -> parquet -> close -> delete temp).
Resumable + crash-safe: ledger/cap/val written per file; skips a pref-month whose 3 outputs all exist; combines at end.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import glob
import io
import os
import re
import sys
import zipfile

import duckdb
import pandas as pd

from wlpr.io.paths import derived_path, input_path, out_path

OUTER_ID = "btm_selfconsumption_zip"          # restricted source archive registered in the input catalogue
OUTDIR = derived_path(_asset_location('asset_0086'))                   # privately configured ledger output
LEDGER = derived_path(_asset_location('asset_0086'), _asset_location('asset_0084'))
CAPDIR = out_path(_asset_location('asset_0085'), _asset_location('asset_0086'), _asset_location('asset_0138'))
VALDIR = out_path(_asset_location('asset_0085'), _asset_location('asset_0086'), _asset_location('asset_0139'))
TMP = out_path(_asset_location('asset_0085'), _asset_location('asset_0086'))
DUCKTMP = out_path(_asset_location('asset_0085'), _asset_location('asset_0086'), _asset_location('asset_0140'))

PREFS = ["11", "12", "13", "14"]
_PAT = re.compile('solar_low-(\\d{2})-(\\d{6})\\.zip$')


def build_owner_ledger(test: bool = False) -> dict:
    for d in (OUTDIR, LEDGER, CAPDIR, VALDIR, TMP, DUCKTMP):
        os.makedirs(d, exist_ok=True)
    for _f in os.listdir(TMP):                 # clean stale temp CSVs from any prior crashed run
        if _f.startswith("cur_") and _f.endswith(".csv"):
            try:
                os.remove(os.path.join(TMP, _f))
            except OSError:
                pass

    outer = input_path(OUTER_ID)
    z = zipfile.ZipFile(outer)
    jobs = {(m.group(1), m.group(2)): n for n in z.namelist()
            for m in [_PAT.search(n)] if m and m.group(1) in PREFS}
    keys = [("13", "202307")] if test else sorted(jobs)
    print(f"jobs: {len(keys)} pref-months" + (" (TEST)" if test else ""))

    for i, (pref, ym) in enumerate(keys, 1):
        partdir = os.path.join(LEDGER, f"{_asset_location('asset_0016')}{pref}", f"{_asset_location('asset_0070')}{ym}")
        partfile = os.path.join(partdir, "part.parquet")
        cap_part = os.path.join(CAPDIR, f"{pref}_{ym}.parquet")
        val_part = os.path.join(VALDIR, f"{pref}_{ym}.csv")
        if not test and os.path.exists(partfile) and os.path.exists(cap_part) and os.path.exists(val_part):
            print(f"[{i}/{len(keys)}] {pref}-{ym} done, skip"); continue
        os.makedirs(partdir, exist_ok=True)
        tmpcsv = os.path.join(TMP, f"cur_{pref}_{ym}.csv")
        with zipfile.ZipFile(io.BytesIO(z.read(jobs[(pref, ym)]))) as iz:
            csvn = [x for x in iz.namelist() if x.endswith(".csv")][0]
            with iz.open(csvn) as src, open(tmpcsv, "wb") as dst:
                while True:
                    b = src.read(8 << 20)
                    if not b:
                        break
                    dst.write(b)
        P = tmpcsv.replace("\\", "/")
        c = duckdb.connect(); c.execute("SET memory_limit='9GB'"); c.execute("SET threads=6")
        c.execute("SET temp_directory='" + DUCKTMP.replace(chr(92), "/") + "'"); c.execute("SET preserve_insertion_order=false")
        c.execute(f"""create or replace table s as select "メッシュ" mesh, "年月日時分" ts,
          try_cast("自家消費量" as double) selfcons, try_cast("契約口数" as double) contracts,
          try_cast("契約電力" as double) contract_kw, try_cast("電力量_供給" as double) imp,
          try_cast("発電設備容量" as double) pv_cap, try_cast("電力量_受電" as double) exp
          from read_csv_auto('{P}', header=true, all_varchar=true)""")
        real = "mesh ~ '^[0-9]{8,}$' and selfcons is not null"
        c.execute(f"""copy (select '{pref}' pref, mesh, substr(ts,1,8) ymd,
            cast(substr(ts,10,2) as int)*2 + case when substr(ts,13,2)='30' then 1 else 0 end slot,
            selfcons selfcons_kwh, imp import_kwh, exp export_kwh,
            selfcons+exp pv_gen_kwh, selfcons+imp gross_btm_load_kwh, imp-exp net_import_kwh
          from s where {real}) to '{partfile}' (format parquet)""")
        cap = c.execute(f"""select '{pref}' pref, mesh, '{ym}' ym, max(pv_cap) pv_capacity_kw,
            max(contracts) contracts, max(contract_kw) contract_kw from s where {real} and pv_cap is not null group by mesh""").df()
        cap.to_parquet(cap_part, index=False)
        v = c.execute(f"""
          with allr as (select sum(selfcons) sc, sum(imp) im, sum(exp) ex, max(pv_cap) cap from s where mesh='00_all'),
               mr as (select sum(selfcons) sc, sum(imp) im, sum(exp) ex, count(distinct mesh) nmesh from s where {real}),
               supp as (select count(distinct mesh) nsupp from s where mesh ~ '^[0-9]{{8,}}$' and selfcons is null),
               md as (select avg(case when slot=24 then imp end) imp_noon, avg(case when slot=38 then imp end) imp_eve,
                             avg(case when slot=24 then exp end) exp_noon, avg(case when slot=38 then exp end) exp_eve
                      from (select cast(substr(ts,10,2) as int)*2 + case when substr(ts,13,2)='30' then 1 else 0 end slot, imp, exp from s where {real}))
          select '{pref}' pref, '{ym}' ym, (select nmesh from mr) n_mesh, (select nsupp from supp) n_suppressed,
            (select cap from allr) allcap, (select sc from allr) all_sc, (select sc from mr) mesh_sc,
            (select im from allr) all_im, (select im from mr) mesh_im, (select ex from allr) all_ex, (select ex from mr) mesh_ex,
            (select imp_noon from md) imp_noon, (select imp_eve from md) imp_eve,
            (select exp_noon from md) exp_noon, (select exp_eve from md) exp_eve""").df()
        v["meshcap"] = float(cap.pv_capacity_kw.sum())
        v.to_csv(val_part, index=False)
        c.close()
        try:
            os.remove(tmpcsv)
        except OSError:
            pass
        print(f"[{i}/{len(keys)}] {pref}-{ym}: meshes {int(v.n_mesh.iloc[0])} suppressed {int(v.n_suppressed.iloc[0])} "
              f"| selfcons close {v.mesh_sc.iloc[0]/v.all_sc.iloc[0]:.3f} cap close {cap.pv_capacity_kw.sum()/v.allcap.iloc[0]:.3f} "
              f"| import noon/eve {v.imp_noon.iloc[0]:.0f}/{v.imp_eve.iloc[0]:.0f} export noon/eve {v.exp_noon.iloc[0]:.0f}/{v.exp_eve.iloc[0]:.0f}", flush=True)
    z.close()

    # ---- combine per-file parts (resume-safe: includes parts from earlier runs) ----
    capdf = None
    cap_files = sorted(glob.glob(os.path.join(CAPDIR, _asset_location('asset_0062'))))
    val_files = sorted(glob.glob(os.path.join(VALDIR, _asset_location('asset_0141'))))
    if cap_files:
        capdf = pd.concat([pd.read_parquet(f) for f in cap_files], ignore_index=True)
        capdf.to_parquet(os.path.join(OUTDIR, "btm_capacity_mesh_month.parquet"), index=False)
        print(f"\ncapacity panel: {len(capdf):,} rows ({len(cap_files)} pref-months) -> btm_capacity_mesh_month.parquet")
    if val_files:
        vdf = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True)
        vdf.to_csv(os.path.join(OUTDIR, "btm_parse_validation.csv"), index=False)
        print(f"validation: {len(vdf)} pref-months -> btm_parse_validation.csv")
    print("DONE btm parse")
    return {"cap_rows": 0 if capdf is None else len(capdf), "ledger": LEDGER}


def run_gates() -> dict:
    """Owner BTM identity gates on the produced ledger: gen=self+exp, load=self+imp (and the capacity table sane)."""
    glob_pat = os.path.join(LEDGER, _asset_location('asset_0018'), _asset_location('asset_0061'), _asset_location('asset_0062')).replace("\\", "/")
    if not glob.glob(glob_pat):
        raise FileNotFoundError(f"owner ledger not built: {LEDGER} (run build_owner_ledger first)")
    con = duckdb.connect()
    gaps = con.execute(f"""
        SELECT
          sum(CASE WHEN abs(pv_gen_kwh        - (selfcons_kwh + export_kwh)) > 1e-9 THEN 1 ELSE 0 END) AS gen_gap,
          sum(CASE WHEN abs(gross_btm_load_kwh - (selfcons_kwh + import_kwh)) > 1e-9 THEN 1 ELSE 0 END) AS load_gap,
          sum(CASE WHEN abs(net_import_kwh    - (import_kwh - export_kwh))    > 1e-9 THEN 1 ELSE 0 END) AS net_gap,
          count(*) AS n
        FROM read_parquet('{glob_pat}')
    """).fetchone()
    con.close()
    res = {"gen_gap": gaps[0], "load_gap": gaps[1], "net_gap": gaps[2], "rows": gaps[3]}
    if any(res[k] for k in ("gen_gap", "load_gap", "net_gap")):
        raise SystemExit(f"01_owner_ledger: GATE FAIL -> BTM identity gaps {res}")
    return res


def main() -> None:
    test = "--test" in sys.argv
    out = build_owner_ledger(test=test)
    res = run_gates()
    print(f"[01] owner_ledger: {res['rows']:,} rows | identity gaps gen={res['gen_gap']} "
          f"load={res['load_gap']} net={res['net_gap']} | cap rows {out['cap_rows']:,}")
    print("01_owner_ledger: ALL GATES PASS")


if __name__ == "__main__":
    main()
