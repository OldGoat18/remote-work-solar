"""Step 03 — survey-anchored cross-pref WFH re-allocation (ported from source closure/survey_anchor.py).

The Employment Status Survey participation rate is a PREF-LEVEL signal: under the per-pref-normalised WFH
channel it cancels within pref, so its content is the CROSS-PREF WFH split. Re-weight each pref's WFH energy by
m_p = surv_p · ΣW_p / Σ(W_p·surv_p) (preserves TOTAL WFH energy, bounded by the load fit) and absorb the
difference into non-thermal, so every (g,d,slot) load_mwh is byte-unchanged. This anchored product feeds the
validation/decomposition (NMAE); the baseline_p0 ledger reads the un-anchored component product directly.
"""
from __future__ import annotations

import os

import duckdb
import pandas as pd

from wlpr.io.paths import derived_path, input_path, out_path

PREFS = ["11", "12", "13", "14"]


def apply_survey_anchor() -> dict:
    src = derived_path("products", "grid_component_daily_product.parquet")
    if not os.path.exists(src):
        raise FileNotFoundError(f"component product missing: {src} (run step 02 first)")
    rates = pd.read_parquet(input_path("wfh_survey_pref_rates"))
    surv = {str(r.pref): float(r.rate_total) for _, r in rates.iterrows() if str(r.pref) in PREFS}

    tmp = out_path("_tmp", "_duck"); os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(); con.execute("SET memory_limit='10GB'"); con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{tmp}'"); con.execute("SET preserve_insertion_order=false")

    W = {r[0]: r[1] for r in con.execute(f"select pref, sum(E_wfh) from '{src}' group by 1").fetchall()}
    prefs = [p for p in PREFS if p in W and p in surv]
    num = sum(W[p] for p in prefs); den = sum(W[p] * surv[p] for p in prefs)
    mP = {p: surv[p] * num / den for p in prefs}
    before = {p: con.execute(f"select sum(E_wfh)/sum(load_mwh)*100 from '{src}' where pref='{p}'").fetchone()[0] for p in prefs}

    mcase = "case " + " ".join(f"when pref='{p}' then {mP[p]:.10f}" for p in prefs) + " else 1.0 end"
    out = derived_path("products", "wfh_survey_anchored_component_product.parquet")
    con.execute(f"""COPY (
      select g, pref, d, slot, imputed_mix, load_mwh, E_hvac, E_cool,
             E_nonth + (E_wfh - E_wfh*({mcase})) as E_nonth,
             E_wfh*({mcase}) as E_wfh
      from '{src}'
    ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")

    chk = con.execute(f"select sum(load_mwh), sum(E_hvac+E_cool+E_nonth+E_wfh) from '{out}'").fetchone()
    rowmax = con.execute(f"select max(abs(load_mwh-(E_hvac+E_cool+E_nonth+E_wfh))) from '{out}'").fetchone()[0]
    nneg = con.execute(f"select count(*) from '{out}' where E_wfh<0 or E_nonth<0").fetchone()[0]
    after = {p: con.execute(f"select sum(E_wfh)/sum(load_mwh)*100 from '{out}' where pref='{p}'").fetchone()[0] for p in prefs}

    sc = pd.DataFrame(
        [("multiplier_m_" + p, mP[p]) for p in prefs] +
        [("wfh_share_before_" + p, before[p]) for p in prefs] +
        [("wfh_share_after_" + p, after[p]) for p in prefs] +
        [("survey_rate_" + p, surv[p]) for p in prefs] +
        [("agg_reldiff", abs(chk[0] - chk[1]) / chk[0]), ("per_row_closure_max", rowmax),
         ("negative_component_rows", nneg)],
        columns=["metric", "value"])
    sc.to_csv(out_path("scorecards", "wfh_anchor_scorecard.csv"), index=False)
    return dict(mP=mP, before=before, after=after, reldiff=abs(chk[0] - chk[1]) / chk[0],
                rowmax=float(rowmax), nneg=int(nneg))


def main() -> None:
    r = apply_survey_anchor()
    print(f"[03] survey anchor multipliers {{ {', '.join(f'{p}:{m:.3f}' for p,m in r['mP'].items())} }} | "
          f"reldiff={r['reldiff']:.2e} rowmax={r['rowmax']:.2e} nneg={r['nneg']}")
    fails = []
    if r["reldiff"] >= 1e-9:
        fails.append(f"closure not preserved (reldiff={r['reldiff']:.2e})")
    if r["rowmax"] >= 1e-6:
        fails.append(f"per-row closure broken (rowmax={r['rowmax']:.2e})")
    if r["nneg"] != 0:
        fails.append(f"negative components ({r['nneg']})")
    if fails:
        raise SystemExit("03_survey_anchor: GATE FAILS -> " + "; ".join(fails))
    print("03_survey_anchor: ALL GATES PASS")


if __name__ == "__main__":
    main()
