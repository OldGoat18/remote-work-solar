# Remote Work and Solar Integration

Model code, algorithms and experiment scripts for studying how remote work
changes residential electricity demand, rooftop photovoltaic utilization and
storage requirements in the Tokyo metropolitan region.

This publication contains research code and logical configuration. Research
inputs, manuscripts, figures and generated results are supplied separately.
No private storage layout or machine-specific location is distributed.

## Scientific scope

- Residential load construction from work participation, occupancy, device use,
  household services and weather-dependent heating and cooling.
- Rooftop photovoltaic generation and half-hourly supply–demand accounting.
- Spatial allocation and comparison of local absorption, regional net
  utilization and pooled upper bounds.
- Storage dispatch and capacity comparisons under specified service targets.
- Sensitivity, validation, scheduling and equal-budget policy experiments.

## Data definitions

The original-package snapshot and reconstructed definitions are both valid
research assets. Their input coverage and processing rules differ. Choose by
scientific purpose, record provenance and align samples before comparison.
Neither definition should be dismissed because of a legacy label.

## Setup and execution

Use Python 3.11 or later with the declared dependencies. Prepare a private
asset-binding configuration and select it with the WLPR_ASSET_CONFIG environment
variable. CONFIGURATION.md describes the contract without disclosing locations.
Installing the code does not provide research inputs or their integrity records.

The shell launchers support public-input preparation, source conversion,
verification, baseline reconstruction, figures and tests. Additional analyses
have separate prerequisites and are not all run by the baseline launcher.

Scientific observation dates, model specifications, random seeds and required
software metadata are retained for reproducibility. Private bindings, source
manifests and generated products must not be included in publication commits.
