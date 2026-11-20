"""Figure manifest — the official release figure set, each mapped to its builder and the claim it supports.
15_validation_suite writes outputs/figures/FIGURE_MANIFEST.csv at the end of the run; a verify step requires every
figure listed here to exist (png + pdf). This turns "a figure happens to be on disk" into "the named release figure
set is complete and current".

Ported verbatim (numerics unchanged) from the source `figures/manifest.py`.
"""
import os, csv
from wlpr.io.paths import out_path

# (stem, title, builder module, claim / evidence role)
FIGURES = [
    ("fig1_main_validation", "Main daily-LV validation (system aggregate + component physics)",
     "figures/main_validation.py", "primary accuracy + component coherence (C1, C4)"),
    ("fig2_calibration_ablation", "Calibration ablation — the residual is a floor, not a black-box cal gap",
     "figures/ablation.py", "floor proof, no post-hoc deployment (C2)"),
    ("fig3_grid_validation", "Per-grid spatial validation + low-exposure risk band",
     "figures/grid_validation.py", "spatial fidelity + honest risk band (C2, C6)"),
    ("fig4_annual_evidence_tiers", "Component-based annual generation by evidence tier (validated / interpolation / scenario)",
     "figures/annual.py", "annual scenario, tiered + summer scenario-only (C11)"),
    ("figS_mobility_validation", "Raw 5-min mobility — independent occupancy validation + floor robustness",
     "figures/mobility_validation.py", "second-source occupancy validation, floor robust (C12)"),
    ("model_qa_dashboard", "Model QA dashboard — supervisor target line, PASS / FLOOR buckets",
     "validation/qa_dashboard.py", "target-line QA across every claim"),
]


def write_figure_manifest():
    figdir = os.path.dirname(out_path("figures", "_"))
    rows = []
    for stem, title, builder, role in FIGURES:
        png = os.path.join(figdir, stem + ".png"); pdf = os.path.join(figdir, stem + ".pdf")
        rows.append(dict(stem=stem, title=title, builder=builder, evidence_role=role,
                         png_exists=os.path.exists(png), pdf_exists=os.path.exists(pdf),
                         png_bytes=os.path.getsize(png) if os.path.exists(png) else 0))
    p = out_path("figures", "FIGURE_MANIFEST.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return p, rows


def main():
    p, rows = write_figure_manifest()
    miss = [r["stem"] for r in rows if not (r["png_exists"] and r["pdf_exists"])]
    print(f"[15] figure manifest: {len(rows)} figures, missing files: {miss or 'none'} -> {os.path.basename(p)}")
    return p, rows


if __name__ == "__main__":
    main()
