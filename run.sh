#!/usr/bin/env bash
# Single entry point (POSIX) for the wlpr reproduction package. Mirrors run.ps1.
# Usage: ./run.sh [env|public|stdstats|verify|frozen|raw|figures|test|all]
set -euo pipefail
MODE="${1:-frozen}"
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

run_step() { echo "==> $1"; python "$1"; }

BUILD=(
  pipeline/00_verify_inputs.py pipeline/01_owner_ledger.py pipeline/02_component_product.py
  pipeline/03_survey_anchor.py pipeline/04_baseline_p0.py pipeline/05_pv_inputs.py
  pipeline/06_adwfh_components.py pipeline/07_adwfh_emit.py pipeline/08_netload.py pipeline/09_indices.py
  pipeline/10_curves_summary_bands.py pipeline/11_storage_physics.py pipeline/12_r3.py
)
PUBLIC=(pipeline/00_rebuild_public_inputs.py)
STDSTATS=(pipeline/00_rebuild_stdstats.py)
FIGURES=(pipeline/13_figures.py pipeline/14_mobility.py pipeline/15_validation_suite.py)

case "$MODE" in
  env)     echo "conda env create -f environment.yml ; conda activate WFH" ;;
  public)  export WLPR_RUN_FROM=raw; for s in "${PUBLIC[@]}"; do run_step "$s"; done ;;
  stdstats) export WLPR_RUN_FROM=raw; for s in "${STDSTATS[@]}"; do run_step "$s"; done ;;
  verify)  export WLPR_RUN_FROM=frozen; run_step pipeline/00_verify_inputs.py ;;
  test)    python -m pytest tests/ -q ;;
  figures) export WLPR_RUN_FROM=frozen; for s in "${FIGURES[@]}"; do [ -f "$s" ] && run_step "$s"; done ;;
  frozen)  export WLPR_RUN_FROM=frozen; for s in "${BUILD[@]}" "${FIGURES[@]}"; do [ -f "$s" ] && run_step "$s"; done ;;
  raw|all) export WLPR_RUN_FROM=raw;    for s in "${PUBLIC[@]}" "${STDSTATS[@]}" "${BUILD[@]}" "${FIGURES[@]}"; do [ -f "$s" ] && run_step "$s"; done ;;
  *) echo "unknown mode '$MODE' (env|public|stdstats|verify|frozen|raw|figures|test|all)" >&2; exit 2 ;;
esac
echo "run.sh ($MODE): DONE"
