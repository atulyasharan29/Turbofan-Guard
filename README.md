# Turbofan Sensor-FDI-Bench

Turbofan Sensor-FDI-Bench: A Synthetic Dataset for Sensor Fault Detection & Isolation under Degradation and Operating Variability


## Reference

Aytunc Yildirim, Martin Bolemant, and Marvin Nöthen,
9th European Conference of the Prognostics and Health Management Society, 2026, Oslo.


## License and Reuse

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Users may copy, share, adapt, and build upon the dataset, provided that appropriate credit is given to the dataset record and the associated publication.

See the LICENSE file for details or visit:
https://creativecommons.org/licenses/by/4.0/

## Generation and Scope

This repository contains synthetic aircraft-engine cruise operating-point snapshot data.

The published material includes only generated dataset outputs and accompanying metadata.

The component health-index trajectories used during data generation are derived from open literature and are injected into the synthetic data generation to represent slow performance deterioration.

This dataset is intended as a research benchmark for the development, evaluation, and comparison of methods for sensor fault detection, sensor fault isolation, and sensor-signal denoising under degradation and operating variability.

The dataset is not intended to represent validated real-engine fault behaviour and is not a substitute for validated operational or maintenance data.

## Citation Requirement

If you use this dataset in academic work, publications, software, benchmarks, derived datasets, or other public outputs, please cite:

1. the Zenodo dataset record DOI, and
2. the associated publication:

Aytunc Yildirim, Martin Bolemant, and Marvin Nöthen,
Turbofan Sensor-FDI-Bench: A Synthetic Dataset for Sensor Fault Detection & Isolation under Degradation and Operating Variability,
9th European Conference of the Prognostics and Health Management Society, 2026, Oslo.

## General Dataset Structure

This folder contains four dataset suites for supervised denoising and sensor-fault studies on synthetic turbofan data.

Each suite folder contains:
- parquet/train.parquet
- parquet/val.parquet
- parquet/test.parquet
- metadata/suite_metadata.json
- metadata/engine_manifest.csv
- metadata/family_manifest.csv (DS03 and DS04 only)


## Common Settings Across All Dataset Suites

The following scenario flags are set commonly as such in all dataset suites:
- deterioration             : enabled
- deterioration variation   : enabled
- component scatter         : enabled
- production scatter        : enabled
- component fault           : disabled
- sensor random noise       : enabled
- sensor peak random noise  : enabled


## Sampling Convention

Each row in a split file corresponds to one engine cruise snapshot at one flight.


## Core Columns

Each split file is a flat table with:
- suite_id
- engine_id
- fault_id
- flight_cycle
- operating conditions: XM, ALT, DTISA, EPR
- health indices
- observed sensor channels: <sensor>_obs
- clean target channels: <sensor>_truth

## Dataset Suite Overview


### DS01
- Status: present
- Conditions: CR - fixed operating condition
- Fault Modes: random measurement noise + peak random measurement noise
- Scenario flag: 011110_1100
- Purpose: baseline denoising under fixed operating conditions
- Cycles per engine: 200
- Total engines: 200
- Engine IDs: 20270001 to 20270200



### DS02
- Status: present
- Conditions: CR - variable operating conditions
- Fault Modes: random measurement noise + peak random measurement noise
- Scenario flag: 111110_1100
- Purpose: denoising under operating-condition variation
- Cycles per engine: 200
- Total engines: 200
- Engine IDs: 20271001 to 20271200



### DS03
- Status: present
- Conditions: CR - variable operating conditions
- Fault Modes: random measurement noise + peak random measurement noise + single structured sensor faults
- Scenario flags:
  - drift families -> 111110_1110
  - step/bias families -> 111110_1101
- Purpose: denoising with one structured fault per trajectory
- Fault families:
  - 14 drift fault families
  - 14 step fault families
- Engines per family: 20
- Cycles per engine: 200
- Total engines: 560
- Engine IDs: 20272001 to 20272560



### DS04
- Status: present
- Conditions: CR - variable operating conditions
- Fault Modes: random measurement noise + peak random measurement noise + multi-structured-fault families
- Scenario flags:
  - drift-only families -> 111110_1110
  - step-only families -> 111110_1101
  - drift+step families -> 111110_1111
- Purpose: denoising with combinations of structured sensor faults
- DS04 family design:
  - Double-fault families:
    - 12 step+step families
    - 12 drift+drift families
    - 24 step+drift families
    - total double-fault families = 48
    - engines per double-fault family = 12
  - Triple-fault families:
    - 4 step+step+step families
    - 4 drift+drift+drift families
    - 4 step+drift+drift families
    - 4 step+step+drift families
    - total triple-fault families = 16
    - engines per triple-fault family = 8
- Total DS04 families: 64
- Cycles per engine: 200
- Total engines: 704
- Engine IDs: 20273001 to 20273704


## Disclaimer

This dataset is provided on an "as is" basis, without warranties of any kind, whether express or implied.

The dataset is intended as a synthetic benchmark resource for research and method comparison. It was not developed, calibrated, or validated for direct deployment in real-world operational, maintenance, certification, or safety-critical decision processes.

Any conclusions, models, or algorithms developed using this dataset require separate validation on representative real-world data before any practical use.

The dataset authors and publishers assume no responsibility or liability for any direct or indirect consequences arising from the use, misuse, interpretation, or downstream application of this dataset or of methods developed using it.

## Notes

- Splits are engine-disjoint.
- DS03 and DS04 families are split independently so every family is represented in train, validation, and test.
- Detailed family definitions are stored in metadata/family_manifest.csv.
- Detailed per-engine mapping is stored in metadata/engine_manifest.csv.
- DS03 and DS04 fault IDs follow the scenarioFlagString-based policy and combination encoding implemented in the generator.
