# Research experiments

The baseline workflow and additional experiments have separate prerequisites.
Running the baseline does not execute every analysis in the project.

## Analysis groups

- Spatial allocation, participation, scheduling and policy comparisons.
- Common-sample alignment, residual propagation and calendar checks.
- Behavioral robustness and joint behavior–storage comparisons.
- Seasonal diagnostics, annual events and storage validation.
- Holdout validation, spatial uncertainty and sensitivity experiments.

## Required preparation

Supply the selected analysis's private asset bindings and stage its inputs before
execution. Some analyses require seasonal ledgers, municipality membership,
geographic holdout summaries or outputs from earlier analyses. The presence of
source code does not mean these inputs are distributed with the publication.

Some modules perform setup at import time. Use a disposable workspace for checks
and do not run a full experiment merely to verify that an import succeeds.

Both scientific data definitions remain valid. Record the chosen definition,
input identity and alignment rules. Do not expose private locations in shared
manifests, error reports or documentation.
