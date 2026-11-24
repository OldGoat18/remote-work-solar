"""Step 05a — observed-stock per-kW yield kernel (ported from the source observed-stock model).

Packages the observed-stock per-kW yield kernel (R6-W6) as the future-PV mainline kernel. Interface-compatible
with the scenario builders (column sim_ykw_kwh_per_kw_30min on pref,m,slot). Semantics: this kernel is the
OBSERVED current-stock yield shape (selfcons+export per installed kW, BTM ledger, Oct2023-Feb2024 window
months); assumption = future rooftop systems behave like the current rooftop stock (strictly better than the
all-3D-surface PV100 envelope for FUTURE-INSTALLED scenarios). PV100 stays available as the technical-potential
sensitivity kernel (excluded here).

Reads the upstream derived kernel configured asset (built by the kernel-fit
stage, not yet ported); writes configured asset
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import os

import duckdb
import pandas as pd

from wlpr.io.paths import derived_path

SRC = derived_path(_asset_location('asset_0054'), "r6_w6_obs_stock_kernel.parquet")
DST = derived_path(_asset_location('asset_0054'), "obs_stock_yield_kernel.parquet")
LEDGER = derived_path(_asset_location('asset_0086'), _asset_location('asset_0084'))
CAPACITY = derived_path(_asset_location('asset_0086'), "btm_capacity_mesh_month.parquet")
WINDOW_YMS = ["202310", "202311", "202312", "202401", "202402"]


def build_upstream_kernel() -> pd.DataFrame:
    """Rebuild the omitted R6-W6 source from the metered BTM ledger.

    obs_ykw is mean generation in a half-hour slot per installed kW and day:
    sum(generation_kwh) / (sum(capacity_kw) * observed_days).
    """
    ledger_glob = os.path.join(LEDGER, _asset_location('asset_0018'), _asset_location('asset_0061'), _asset_location('asset_0062')).replace("\\", "/")
    if not os.path.exists(CAPACITY) or not os.path.isdir(LEDGER):
        raise FileNotFoundError("BTM ledger/capacity missing; run pipeline/01_owner_ledger.py first")

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=6")
    yms = ",".join(f"'{ym}'" for ym in WINDOW_YMS)
    kernel = con.execute(f"""
        WITH generation AS (
          SELECT cast(pref AS varchar) pref, substr(cast(ymd AS varchar),1,6) ym, slot,
                 sum(coalesce(pv_gen_kwh,0)) gen_kwh,
                 count(distinct cast(ymd AS varchar)) ndays
          FROM read_parquet('{ledger_glob}')
          WHERE substr(cast(ymd AS varchar),1,6) IN ({yms})
          GROUP BY 1,2,3
        ), capacity AS (
          SELECT cast(pref AS varchar) pref, cast(ym AS varchar) ym,
                 sum(coalesce(pv_capacity_kw,0)) cap_kw
          FROM read_parquet('{CAPACITY.replace(chr(92), "/")}')
          WHERE cast(ym AS varchar) IN ({yms})
          GROUP BY 1,2
        )
        SELECT g.pref, cast(substr(g.ym,5,2) AS integer) m, g.slot,
               CASE WHEN c.cap_kw>0 AND g.ndays>0
                    THEN g.gen_kwh/(c.cap_kw*g.ndays) ELSE 0 END obs_ykw
        FROM generation g JOIN capacity c USING(pref,ym)
        ORDER BY pref,m,slot
    """).df()
    con.close()
    assert len(kernel) == 4 * 5 * 48, f"unexpected kernel rows: {len(kernel)}"
    assert kernel.groupby(["pref", "m"]).size().eq(48).all()
    assert kernel.obs_ykw.ge(0).all()
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    kernel.to_parquet(SRC, index=False)
    print(f"[rebuilt] {SRC}: {len(kernel)} rows from metered BTM ledger")
    return kernel


def build_obs_stock_kernel() -> dict:
    if not os.path.exists(SRC):
        build_upstream_kernel()
    k = pd.read_parquet(SRC)
    k = k.rename(columns={"obs_ykw": "sim_ykw_kwh_per_kw_30min"})
    k["kernel_source"] = "observed_stock_btm_ledger_selfcons_plus_export_per_kw"
    k["evidence_label"] = "observed_yield_shape_not_technical_potential"
    _gates(k)
    k.to_parquet(DST, index=False)
    print(f"[saved] {DST}: {len(k)} rows, months {sorted(k.m.unique())}")
    return {"rows": len(k), "months": sorted(k.m.unique())}


def _gates(k: pd.DataFrame) -> None:
    assert set(k.columns) >= {"pref", "m", "slot", "sim_ykw_kwh_per_kw_30min"}
    assert k.groupby(["pref", "m"]).size().eq(48).all(), "kernel must have 48 slots per pref-month"
    assert k["sim_ykw_kwh_per_kw_30min"].ge(0).all()


def run_gates() -> dict:
    import os
    if not os.path.exists(DST):
        raise FileNotFoundError(f"kernel not built: {DST} (run build_obs_stock_kernel first)")
    k = pd.read_parquet(DST)
    _gates(k)
    return {"rows": len(k), "slots_per_pref_month_48": True,
            "yield_nonneg": bool(k["sim_ykw_kwh_per_kw_30min"].ge(0).all())}


def main() -> None:
    out = build_obs_stock_kernel()
    run_gates()
    print(f"[05a] obs_stock_yield_kernel: {out['rows']} rows | months {out['months']}")
    print("05a_obs_stock_kernel: ALL GATES PASS")


if __name__ == "__main__":
    main()
