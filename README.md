# Remote Work and Solar Integration

Model code, algorithms and experiment scripts for studying how remote work
changes residential electricity demand, rooftop photovoltaic utilization and
storage requirements in the Tokyo metropolitan region.

The repository contains source code and configuration files. Research datasets,
manuscripts, figures and generated experiment results are supplied separately or
generated locally; they are not included in this repository.

## Models and analyses

- Residential load construction from work participation, occupancy, device use,
  household services and weather-dependent heating and cooling.
- Rooftop photovoltaic generation and half-hourly supply–demand accounting.
- Spatial allocation of remote-work activity and comparison of local absorption,
  regional net utilization and pooled upper bounds.
- Storage dispatch and capacity comparisons under specified electricity-service
  targets.
- Sensitivity, validation, scheduling and equal-budget policy experiments.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/wlpr/` | Core load, photovoltaic, scenario, storage, spatial and validation modules |
| `pipeline/` | Ordered baseline workflow and additional experiment scripts |
| `tools/` | Input preparation, validation and supplementary analysis utilities |
| `results/v2/technical_audit/revisions/` | Supplemental and technical analysis scripts; generated results are excluded |
| `configs/` | Input registry, public source locations, model parameters and run settings |
| `tests/` | Unit checks and checks that require separately staged inputs or results |
| `run.sh`, `run.ps1` | Workflow entry points for POSIX shells and PowerShell |

## Installation

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
Alternatively, create the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate WFH
```

## Inputs and execution

Input locations are defined in `configs/inputs.yaml`. The workflow uses
`data/raw/` for external source files, `data/frozen/` for prepared inputs and
`data/derived/` for computed products or separately staged snapshots. Generated
figures and summaries are written under `outputs/` and the relevant analysis
directories.

Public input sources are listed in `configs/public_sources.yaml`. Some workflows
also require separately obtained mobility, electricity and prepared study inputs.
Installing the package alone does not supply these inputs or the input manifest
at `data/frozen/manifests/input_manifest.csv`.

Run commands from the repository root after preparing the required inputs:

| Command | Purpose |
| --- | --- |
| `./run.sh public` | Download and prepare the configured public inputs |
| `./run.sh stdstats` | Process the separately staged Standard Statistics archive |
| `./run.sh verify` | Check files recorded in the locally supplied input manifest |
| `./run.sh frozen` | Run the baseline workflow using prepared inputs and snapshots |
| `./run.sh raw` | Run public-input preparation and the baseline rebuild workflow |
| `./run.sh figures` | Build figures and validation summaries from prepared products |
| `./run.sh test` | Run the included checks; some require data or generated products |

PowerShell uses the corresponding `./run.ps1 -Mode <mode>` commands. Additional
experiments in `pipeline/`, `tools/` and `results/v2/technical_audit/revisions/`
run separately and require their upstream inputs and products. For example,
`run_technical.py` reads the prepared validation table
`results/v2/technical_audit/data/stage04/stage4_holdout_summary.csv`.

The publication file list in `.gitignore` limits tracked content to the selected
source files, configuration files and this README.
