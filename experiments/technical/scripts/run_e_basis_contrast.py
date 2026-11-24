"""Approved P85–96: fixed-design, paired leave-one-month-out WFH-basis contrast.
Fit routine adapted from configured asset::_fit_pref_twin.
Saves raw aggregate predictions BEFORE the mesh component allocation/closure.
No new parameter scan, significance test, baseline overwrite or test suite.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location
from pathlib import Path
import datetime
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls
REV=(next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()) / _asset_location('asset_0011'))
ROOT=next(p for p in REV.parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(ROOT/'src'))
from wlpr.io import paths
from wlpr.io.stdstats import STN,HOL,WINDOW_YMS
D=REV/_asset_location('asset_0002')

def fit_model(load: pd.DataFrame, include_wfh: bool):
    """Same retained basis, pooling and folds; optionally omit the WFH column."""
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
    names = ["HVAC", "HVAC_eve", "COOL"] + (["wfh"] if include_wfh else []) + env_names

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

    rows = []; coefficients = []
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
                coefficients.extend(dict(held_out_month=held_out, pref=pref,
                    day_class='nonworkday' if is_nonworkday else 'eligible',
                    basis=name, coefficient=float(value), global_coefficient=float(global_beta[j]),
                    training_records=len(y_pref) if y_pref is not None else 0)
                    for j,(name,value) in enumerate(zip(names,beta)))
                for (candidate_pref, date), values in observed.items():
                    if candidate_pref != pref or date[:6] != held_out or nonworkday(date) != is_nonworkday:
                        continue
                    design = xrow(pref, date, is_nonworkday)
                    prediction = design @ beta
                    work_prediction = design[:, names.index('wfh')]*beta[names.index('wfh')] if include_wfh else np.zeros(48)
                    rows.extend({"pref": pref, "d": date, "slot": slot,
                                 "obs_mwh": float(values[slot]), "pred_mwh": float(prediction[slot]),
                                 "wfh_basis_mwh": float(work_prediction[slot]),
                                 "held_out_month": held_out,
                                 "day_class": 'nonworkday' if is_nonworkday else 'eligible'}
                                for slot in range(48))

    twin = pd.DataFrame(rows).sort_values(["pref", "d", "slot"]).reset_index(drop=True)
    expected = len(observed) * 48
    if len(twin) != expected:
        raise AssertionError(f"pref twin row count changed: expected {expected}, got {len(twin)}")
    return twin, pd.DataFrame(coefficients)

def metrics(z):
    y=z.obs_mwh.to_numpy();e=z.pred_mwh.to_numpy()-y
    return dict(records=len(z),dates=int(z.d.nunique()),observed_MWh=float(y.sum()),
        predicted_MWh=float(z.pred_mwh.sum()),MAE_MWh=float(np.abs(e).mean()),
        RMSE_MWh=float(np.sqrt(np.mean(e*e))),bias_MWh=float(e.mean()),
        WAPE_pct=float(100*np.abs(e).sum()/y.sum()),wfh_basis_MWh=float(z.wfh_basis_mwh.sum()))

def main():
    load=pd.read_parquet(paths.input_path('lv_load_hourly'))
    load=load[(load.voltage=='低圧')&(load.flow=='順')&load.pref_code.astype(str).isin(STN)].copy()
    all_predictions=[];all_coeff=[]
    for model,flag in [('registered_basis',True),('without_wfh_basis',False)]:
        pred,coef=fit_model(load,flag)
        pred['model']=model;coef['model']=model
        all_predictions.append(pred);all_coeff.append(coef)
        print(json.dumps(dict(model=model,**metrics(pred))),flush=True)
    pred=pd.concat(all_predictions,ignore_index=True)
    coef=pd.concat(all_coeff,ignore_index=True)
    pred.to_parquet(D/'e_basis_holdout_predictions.parquet',index=False)
    coef.to_csv(D/'e_basis_coefficients.csv',index=False)
    summaries=[]
    windows={'all_intervals':pred,'eligible_dates':pred[pred.day_class.eq('eligible')],
       'eligible_work_slots':pred[pred.day_class.eq('eligible')&pred.slot.between(18,34)],
       'nonworkday':pred[pred.day_class.eq('nonworkday')]}
    for window,f in windows.items():
        for model,z in f.groupby('model'):
            summaries.append(dict(model=model,scope='all_prefs_all_months',pref='all',held_out_month='all',window=window,**metrics(z)))
        for (model,pref,month),z in f.groupby(['model','pref','held_out_month']):
            summaries.append(dict(model=model,scope='pref_month',pref=pref,held_out_month=month,window=window,**metrics(z)))
    summary=pd.DataFrame(summaries)
    summary.to_csv(D/'e_basis_holdout_summary.csv',index=False)
    keys=['pref','held_out_month','window','scope']
    orig=summary[summary.model.eq('registered_basis')].drop(columns='model')
    no=summary[summary.model.eq('without_wfh_basis')].drop(columns='model')
    pair=orig.merge(no,on=keys,suffixes=('_registered','_without'))
    for m in ['MAE_MWh','RMSE_MWh','bias_MWh','WAPE_pct']:
        pair[m+'_without_minus_registered']=pair[m+'_without']-pair[m+'_registered']
    pair.to_csv(D/'e_basis_holdout_paired.csv',index=False)
    identity=pred.pivot(index=['pref','d','slot'],columns='model',values='pred_mwh')
    report=dict(date='2026-11-14',status='calculated_raw_heldout_predictions',
      source_function='src/wlpr/io/stdstats.py::_fit_pref_twin',
      source_model='src/wlpr/baseline/component_product.py',
      input_records=len(load),folds=list(WINDOW_YMS),prefs=list(STN),
      units='MWh per half-hour; errors are raw aggregate errors before mesh allocation',
      configuration=dict(pooling_tau=.2,env_bumps=13,models=['registered_basis','without_wfh_basis'],
           work_basis='09:00–17:00 slot starts, Gaussian centre 13:00 width 3h; zero on nonworkdays'),
      aggregate_summary=summary[summary.scope.eq('all_prefs_all_months')].to_dict('records'),
      workday_wfh_coefficients=coef[(coef.model.eq('registered_basis'))&(coef.basis.eq('wfh'))&(coef.day_class.eq('eligible'))].to_dict('records'),
      max_paired_prediction_difference_MWh=float(np.abs(identity.registered_basis-identity.without_wfh_basis).max()),
      baseline_changed=False,mesh_product_regenerated=False,tests_run=False,safety_review_run=False)
    (REV/_asset_location('asset_0043')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(summary[summary.scope.eq('all_prefs_all_months')].to_string(index=False),flush=True)
if __name__=='__main__':main()
