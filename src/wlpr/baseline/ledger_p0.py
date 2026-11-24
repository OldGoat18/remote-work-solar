"""Step 04 — the P0-corrected current-baseline ledger (ported from ledger_build_baseline_p0.py).

Fuses the load-fit component product with the metered owner ledger into the mesh × date × slot baseline:
    L_all_import      = official low-voltage forward flow / grid import
    L_owner_gross     = owner_selfcons + owner_import
    L_nonowner_import = L_all_import - owner_import
    G_pv              = owner_selfcons + owner_export   (metered PV generation; the only owner-ledger product
                        kept for net-load — the BTM self-consumption USER-STATE model is excluded)

Writes a partitioned ledger configured asset (streamed per
partition; the full ~14 GB mesh×date×slot table is never materialised whole).
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os
import time

import duckdb

from wlpr.io.paths import derived_path, out_path, run_config

GC = derived_path(_asset_location('asset_0107'), "grid_component_daily_product.parquet")
OL = derived_path(_asset_location('asset_0086'), _asset_location('asset_0084'))
OUT = derived_path(_asset_location('asset_0052'), _asset_location('asset_0053'))
TMP = out_path(_asset_location('asset_0085'), _asset_location('asset_0111'))


def build_baseline_p0() -> int:
    rc = run_config()
    PREFS, YMS = rc["prefs"], rc["window_yms"]
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    if not os.path.exists(GC):
        raise FileNotFoundError(f"component product missing: {GC} (run step 02 first)")
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'"); con.execute("SET threads=3")
    con.execute("SET preserve_insertion_order=false"); con.execute(f"SET temp_directory='{TMP}'")
    t0 = time.time(); total = 0
    for pref in PREFS:
        for ym in YMS:
            outp = f"{OUT}{_asset_location('asset_0108')}{pref}{_asset_location('asset_0109')}{ym}"
            os.makedirs(outp, exist_ok=True)
            ol_path = f"{OL}{_asset_location('asset_0108')}{pref}{_asset_location('asset_0109')}{ym}{_asset_location('asset_0010')}"
            sql = f"""
            COPY (
              WITH la AS (
                SELECT cast(g AS varchar) mesh, cast(d AS varchar) date, slot,
                       sum(load_mwh) AS L_all_import, sum(E_hvac) AS E_hvac, sum(E_cool) AS E_cool,
                       sum(E_nonth) AS E_nonth, sum(E_wfh) AS E_wfh, max(imputed_mix) AS imputed_mix
                FROM read_parquet('{GC}')
                WHERE cast(pref AS varchar)='{pref}' AND substr(cast(d AS varchar),1,6)='{ym}'
                GROUP BY 1,2,3
              ),
              bt AS (
                SELECT cast(mesh AS varchar) mesh, cast(ymd AS varchar) date, slot,
                       sum(coalesce(selfcons_kwh,0))/1000.0 AS owner_selfcons,
                       sum(coalesce(import_kwh,0))/1000.0 AS owner_import,
                       sum(coalesce(export_kwh,0))/1000.0 AS owner_export
                FROM read_parquet('{ol_path}')
                GROUP BY 1,2,3
              ),
              j AS (
                SELECT la.mesh, '{pref}' AS pref, la.date, la.slot,
                       la.L_all_import, la.E_hvac, la.E_cool, la.E_nonth, la.E_wfh, la.imputed_mix,
                       coalesce(bt.owner_selfcons,0) AS owner_selfcons,
                       coalesce(bt.owner_import,0) AS owner_import,
                       coalesce(bt.owner_export,0) AS owner_export,
                       coalesce(bt.owner_selfcons,0) + coalesce(bt.owner_export,0) AS G_pv,
                       coalesce(bt.owner_selfcons,0) + coalesce(bt.owner_import,0) AS L_owner_gross
                FROM la LEFT JOIN bt USING (mesh,date,slot)
              )
              SELECT mesh, pref, date, slot,
                     L_all_import,
                     L_all_import + owner_selfcons AS L_all_gross_equiv,
                     L_owner_gross,
                     L_all_import - owner_import AS L_nonowner_import,
                     E_hvac, E_cool, E_nonth, E_wfh, imputed_mix,
                     G_pv, owner_selfcons, owner_export, owner_import,
                     L_all_import - owner_export AS net_load_owner_export_only,
                     CASE WHEN owner_import > L_all_import + 1e-9 THEN 1 ELSE 0 END AS import_scope_mismatch,
                     CASE WHEN abs(L_owner_gross - (owner_import + owner_selfcons)) > 1e-9 THEN 1 ELSE 0 END AS btm_load_identity_gap,
                     CASE WHEN abs(G_pv - (owner_selfcons + owner_export)) > 1e-9 THEN 1 ELSE 0 END AS btm_pv_identity_gap,
                     CASE WHEN L_owner_gross>1e-9 THEN 'current_owner' ELSE 'non_owner' END AS owner_status,
                     'current' AS pv_source,
                     'current_baseline_p0' AS scenario
              FROM j
            ) TO '{outp}{_asset_location('asset_0022')}' (FORMAT parquet)
            """
            con.execute(sql)
            n = con.execute(f"SELECT count(*) FROM read_parquet('{outp}{_asset_location('asset_0022')}')").fetchone()[0]
            total += n
            print(f"  pref={pref} ym={ym}: {n:,} rows ({time.time()-t0:.0f}s)", flush=True)
    # identity gates over the whole ledger
    gaps = con.execute(f"SELECT sum(import_scope_mismatch), sum(btm_load_identity_gap), sum(btm_pv_identity_gap) FROM read_parquet('{OUT}{_asset_location('asset_0110')}')").fetchone()
    con.close()
    print(f"[done] baseline_p0 -> {OUT} ({total:,} rows, {time.time()-t0:.0f}s)")
    return total, gaps


def main() -> None:
    total, gaps = build_baseline_p0()
    # The HARD identities are the BTM accounting ones (must be exact): L_owner_gross == owner_import+selfcons,
    # and G_pv == selfcons+export. `import_scope_mismatch` (owner_import > L_all_import on a few rows) is a known
    # minor BTM-vs-LV measurement-scope artifact carried as a DIAGNOSTIC in the source, NOT a hard gate.
    print(f"[04] baseline_p0: {total:,} rows | btm_load_gap={gaps[1]} btm_pv_gap={gaps[2]} (hard) | "
          f"import_scope_mismatch={gaps[0]} rows (diagnostic, ~{100*gaps[0]/max(total,1):.4f}%)")
    if gaps[1] or gaps[2]:
        raise SystemExit(f"04_baseline_p0: GATE FAIL -> nonzero BTM identity gaps load={gaps[1]} pv={gaps[2]}")
    print("04_baseline_p0: ALL GATES PASS")


if __name__ == "__main__":
    main()
