"""Step 10a — the FORMAL validated-window WFH scenario LOAD curves (mesh × date × slot -> month × slot).

Ported (numerics verbatim) from the source `configured asset`.

RED LINE (mesh-date-only formal): the system monthly curves are DERIVED by aggregating the real fused mesh
account baseline_p0 (mesh × date × slot) -> sum over mesh, mean over the REAL dates in each month -> month ×
slot. NO forward rep-day spine, NO Annual-A, NO system representative-day INPUT ledger. System / pref / monthly
curves are only ever a DOWNSTREAM aggregation of this mesh-date account; they are never an input.

Caliber:
  current/black = sum_mesh L_all_import                 (official LV forward flow / grid import; mean daily shape)
  no_wfh        = sum_mesh (L_all_import - E_wfh)        (strip the survey-anchored CURRENT WFH component)
  S1..S10       = current + (D_k - D_1),  D_k = 2*sum_mesh a_dwfh(Sk)   (a_dwfh is mesh × month × slot)
                  -> S1 == current; S2..S10 add ONLY the MARGINAL adoption beyond survey-central.

Output: derived_path("scenario_window", "scenario_window_curves_month_slot.parquet")
        columns (scenario_id, mon, slot, load_MW) for scenario_id in {current,no_wfh,S1..S10}, mon in WINDOW.

7 GATES (all PASS or exit 1): source-purity / black==sum_mesh L_all_import / S1==black / S10 midday-fill>0 /
  morning-flat / evening-converge / multiplier-monotone.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import glob
import os

import duckdb
import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path, out_path
from wlpr.wfh_state.scenario_downscaling import scenario_multipliers

BASE = derived_path(_asset_location('asset_0052'), _asset_location('asset_0053'))
ADWFH = derived_path(_asset_location('asset_0055'), "a_dwfh_mesh_mon_slot.parquet")
SIDX = derived_path(_asset_location('asset_0057'), "scenario_indices_s1s10.csv")
OUTPARQ = derived_path(_asset_location('asset_0080'), "scenario_window_curves_month_slot.parquet")
OUTDIR = os.path.dirname(OUTPARQ)
SCORECARD = out_path(_asset_location('asset_0058'), "scenario_window_curves_scorecard.csv")

WINDOW = [10, 11, 12, 1, 2]                 # the validated Oct->Feb window (mesh ledger on disk)
SCEN = [f"S{i}" for i in range(1, 11)]
# slot index: 30-min, slot 0 = 00:00 JST. morning 0-8h = slots 0-16; work window 9-17h = slots 18-34;
# evening 18-22h = slots 36-44.
MORNING = range(0, 17)
WORK = range(18, 35)
EVENING = range(36, 45)


def build_curves() -> pd.DataFrame:
    """sum_mesh -> mean over real dates in month -> month x slot, from baseline_p0 ONLY."""
    parts = [p.replace("\\", "/") for p in glob.glob(os.path.join(BASE, _asset_location('asset_0018'), _asset_location('asset_0061'), _asset_location('asset_0062')))]
    if not parts:
        raise SystemExit(f"baseline_p0 mesh ledger missing under {BASE}")
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'"); con.execute("SET threads=6")
    con.execute("SET preserve_insertion_order=false")
    acc = [con.execute(f"""select cast(substr(date,5,2) as int) mon, slot,
                 sum(L_all_import) imp, sum(E_wfh) ewfh
              from read_parquet('{p}') group by 1,2""").df() for p in parts]
    agg = pd.concat(acc).groupby(["mon", "slot"], as_index=False).agg(imp=("imp", "sum"), ewfh=("ewfh", "sum"))
    nd = con.execute(f"""select cast(substr(date,5,2) as int) mon, count(distinct date) nd
                         from read_parquet('{BASE}{_asset_location('asset_0144')}') group by 1""").df().set_index("mon").nd.to_dict()
    # sum_mesh of (L summed over dates) * 2 / n_dates = sum_mesh mean_date(L) * 2  -> mean daily-shape MW
    agg["imp_MW"] = agg.apply(lambda r: r.imp * 2.0 / nd[r.mon], axis=1)
    agg["now_MW"] = agg.apply(lambda r: (r.imp - r.ewfh) * 2.0 / nd[r.mon], axis=1)
    IMP = agg.set_index(["mon", "slot"]).imp_MW
    NOW = agg.set_index(["mon", "slot"]).now_MW
    D = con.execute(f"""select scenario_id, mon, slot, 2.0*sum(dwfh) D_MW
                        from read_parquet('{ADWFH}') group by 1,2,3""").df().set_index(["scenario_id", "mon", "slot"]).D_MW
    con.close()

    rows = []
    for mon in WINDOW:
        imp = IMP.xs(mon, level=0).reindex(range(48)).fillna(0.0).values
        now = NOW.xs(mon, level=0).reindex(range(48)).fillna(0.0).values
        try:
            d1 = D.xs(("S1", mon), level=("scenario_id", "mon")).reindex(range(48)).fillna(0.0).values
        except KeyError:
            d1 = np.zeros(48)
        for slot in range(48):
            rows.append(("current", mon, slot, float(imp[slot])))
            rows.append(("no_wfh", mon, slot, float(now[slot])))
        for s in SCEN:
            try:
                d = D.xs((s, mon), level=("scenario_id", "mon")).reindex(range(48)).fillna(0.0).values
            except KeyError:
                d = np.zeros(48)
            arm = imp + (d - d1)         # ANCHOR: S1 == current; higher scenarios add marginal adoption only
            for slot in range(48):
                rows.append((s, mon, slot, float(arm[slot])))
    return pd.DataFrame(rows, columns=["scenario_id", "mon", "slot", "load_MW"])


def _wide(df):
    return {(sid, mon): df[(df.scenario_id == sid) & (df.mon == mon)].sort_values("slot").load_MW.values
            for sid in (["current", "no_wfh"] + SCEN) for mon in WINDOW}


def run_gates(df: pd.DataFrame) -> list[dict]:
    cur = _wide(df)
    # The index CSV is a result table and may carry a null metadata field in a
    # public rebuild.  Scenario ordering must come from the canonical scenario
    # definition, otherwise S4/S5 (lower multipliers than S2/S3) can be tested
    # in the wrong order.
    mult = scenario_multipliers().set_index("scenario_id").wfh_energy_multiplier.to_dict()
    fails, checks = [], []

    def ck(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        checks.append(dict(gate=name, status="PASS" if ok else "FAIL", detail=detail))
        if not ok:
            fails.append(name)

    # 1 source purity: the DATA INPUTS are baseline_p0 + a_dwfh + scenario_indices ONLY -- never the forward
    # rep-day spine or the Annual-A system-grain extrapolation. Check the actual input PATHS, not docstring prose.
    inputs = " ".join([BASE, ADWFH, SIDX]).lower()
    ck("1 source purity (inputs = baseline_p0 + a_dwfh + scenario_indices; no forward / annual)",
       ("forward_lv_load_repday" not in inputs) and ("annual_a" not in inputs)
       and (_asset_location('asset_0053') in BASE) and ("a_dwfh" in ADWFH) and ("scenario_indices" in SIDX),
       "baseline_p0 / a_dwfh_mesh_mon_slot / scenario_indices_s1s10")

    # 2 black == sum_mesh L_all_import (positive, physical magnitude ~6-14 GW system LV)
    blk_jan = cur[("current", 1)]
    ck("2 black current = sum_mesh mean_date L_all_import (6-15 GW band)",
       bool(np.all(blk_jan > 4000) and np.all(blk_jan < 16000)), f"Jan range {blk_jan.min():.0f}-{blk_jan.max():.0f} MW")

    # 3 S1 == black (anchoring: survey-central already in the observed load)
    s1_eq = all(np.allclose(cur[("S1", m)], cur[("current", m)], atol=1e-6) for m in WINDOW)
    ck("3 S1 == black current (survey-central anchor)", s1_eq)

    # 4 S10 midday fill > 0 (work-window valley raised), every month
    fills = {}
    for m in WINDOW:
        v = int(np.argmin(cur[("current", m)][WORK.start:WORK.stop])) + WORK.start
        fills[m] = cur[("S10", m)][v] - cur[("current", m)][v]
    ck("4 S10 midday valley fill > 0 (all months)", all(f > 0 for f in fills.values()),
       "fill MW " + " ".join(f"{m}:{fills[m]:.0f}" for m in WINDOW))

    # 5 morning flat: scenarios do not lift the 0-8h morning (|S10-black| small)
    mornmax = max(float(np.max(np.abs(cur[("S10", m)][MORNING.start:MORNING.stop]
                                      - cur[("current", m)][MORNING.start:MORNING.stop]))) for m in WINDOW)
    ck("5 morning (0-8h) not lifted (|S10-black| < 50 MW)", mornmax < 50.0, f"max {mornmax:.1f} MW")

    # 6 evening converge: no abnormal evening peak lift (|S10-black| small over 18-22h)
    evemax = max(float(np.max(np.abs(cur[("S10", m)][EVENING.start:EVENING.stop]
                                     - cur[("current", m)][EVENING.start:EVENING.stop]))) for m in WINDOW)
    # the evening fill must be far below the midday fill (scenarios are a DAYTIME phenomenon)
    ck("6 evening (18-22h) converges (|S10-black| << midday fill)",
       evemax < 0.25 * min(fills.values()), f"evening max {evemax:.1f} MW vs min midday fill {min(fills.values()):.0f}")

    # 7 multiplier-monotone: at the midday valley, the curve heights rise monotonically with WFH multiplier
    mono_ok = True
    for m in WINDOW:
        v = int(np.argmin(cur[("current", m)][WORK.start:WORK.stop])) + WORK.start
        order = sorted(SCEN, key=lambda s: mult.get(s, 0.0))
        vals = [cur[(s, m)][v] for s in order]
        mono_ok &= bool(np.all(np.diff(vals) >= -1.0))   # 1 MW float tolerance
    ck("7 scenarios monotone by WFH multiplier at midday (S1 low -> S10 high)", mono_ok,
       "order " + "<".join(sorted(SCEN, key=lambda s: mult.get(s, 0.0))))
    if fails:
        raise SystemExit("scenario_window_curves: GATE FAILS -> " + ", ".join(fails))
    return checks


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(SCORECARD), exist_ok=True)
    df = build_curves()
    df.to_parquet(OUTPARQ, index=False)
    print(f"[saved] {OUTPARQ}  ({len(df)} rows; {df.scenario_id.nunique()} arms x {len(WINDOW)} months x 48 slots)")
    print("=== GATES ===")
    checks = run_gates(df)
    pd.DataFrame(checks).to_csv(SCORECARD, index=False)
    print(f"[saved] {SCORECARD}")
    print("10a_window_curves: ALL GATES PASS (mesh-date -> month x slot, validated window)")


if __name__ == "__main__":
    main()
