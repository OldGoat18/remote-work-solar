"""Rebuild missing WFH scenario coefficient tables from staged T001193 counts.

The historical snapshot did not contain the two small coefficient CSVs consumed by
the WFH state engine. They are parameters, not time-use observations. This rebuild
keeps that distinction explicit and marks the outputs as public reconstructed
parameters rather than byte-identical recovery of the legacy files.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / _asset_location('asset_0002') / _asset_location('asset_0088') / _asset_location('asset_0125') / "t001193_mesh_industry_occupation_selected.csv"
DEST = SOURCE.parent

INDUSTRY_RATES = {
    "01": 0.070, "02": 0.080, "03": 0.050, "04": 0.120, "05": 0.240,
    "06": 0.600, "07": 0.795, "08": 0.180, "09": 0.250, "10": 0.600,
    "11": 0.300, "12": 0.650, "13": 0.180, "14": 0.320, "15": 0.420,
    "16": 0.220, "17": 0.260, "18": 0.150, "19": 0.180,
}
OCCUPATION_RATES = {f"{i:02d}": 0.189 for i in range(1, 12)}


def _write_rates(rates: dict[str, float], columns: list[str], filename: str) -> pd.DataFrame:
    df = pd.read_csv(SOURCE, usecols=columns)
    rows = []
    for code, column in zip(rates, columns[1:]):
        rows.append({
            "category_code": code,
            "telework_rate": rates[code],
            "total_workers": float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum()),
            "source_basis": "released_project_anchor_plus_public_reconstruction",
            "evidence_level": "public_reconstructed_parameter",
        })
    out = pd.DataFrame(rows)
    out.to_csv(DEST / filename, index=False)
    return out


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"T001193 selected table is missing: {SOURCE}")
    all_columns = list(pd.read_csv(SOURCE, nrows=0).columns)
    industry = [c for c in all_columns if c.startswith("industry_")]
    occupation = [c for c in all_columns if c.startswith("occupation_")]
    if len(industry) != 19 or len(occupation) != 11:
        raise RuntimeError(f"unexpected T001193 columns: industry={len(industry)} occupation={len(occupation)}")
    industry_rates = _write_rates(INDUSTRY_RATES, ["mesh", *industry], "wfh_s5_industry_telework_rates.csv")
    occupation_rates = _write_rates(OCCUPATION_RATES, ["mesh", *occupation], "wfh_s6_occupation_telework_rates.csv")
    metadata = {
        "source": _asset_location('asset_0187'),
        "classification": "public_reconstructed_parameter",
        "occupation_base_rate": 0.189,
        "industry_ict_rate": 0.795,
        "warning": "Reconstructed coefficient tables; not byte-identical recovery of missing legacy inputs.",
        "outputs": {
            "industry": "wfh_s5_industry_telework_rates.csv",
            "occupation": "wfh_s6_occupation_telework_rates.csv",
        },
    }
    (DEST / "wfh_rate_rebuild_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wfh-rates] industry rows={len(industry_rates)} -> {DEST / 'wfh_s5_industry_telework_rates.csv'}")
    print(f"[wfh-rates] occupation rows={len(occupation_rates)} -> {DEST / 'wfh_s6_occupation_telework_rates.csv'}")
    print(f"[wfh-rates] occupation base rate=0.189; industry ICT rate={INDUSTRY_RATES['07']:.3f}")
    print(f"[wfh-rates] metadata -> {DEST / 'wfh_rate_rebuild_metadata.json'}")


if __name__ == "__main__":
    main()
