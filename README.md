# TurbofanGuard

**TurbofanGuard** is a deep learning system for aircraft jet engines designed to detect faulty sensors, isolate which sensors are broken, and reconstruct clean physical readings in real time under flight variability and engine degradation.

```text
+-----------------------+      +-----------------------------------------+      +----------------------------------------+
| 1. Telemetry Stream   | ---> | 2. TurbofanGuard Core                   | ---> | 3. Operational Outputs                 |
| • 4 Flight Conditions |      | • Neural Backbone (57k params)          |      | • Clean Signals (14 channels)          |
| • 14 Raw Sensors      |      | • Virtual Sensor (Reconstruction Head)  |      | • Zero-Noise-Spike Fault Alarms        |
+-----------------------+      | • Diagnostic AI (FDI Isolation Head)    |      | • Exact Broken Sensor Identified       |
                               +-----------------------------------------+      +----------------------------------------+
```

## Quick Links to Documentation

All technical documentation is organized in simple, easy-to-read guides with GitHub math blocks:

* **[Documentation Hub](file:///Users/atulyasharan/Documents/TurbofanGuard/docs/README.md)**: Main landing page, glossary of terms, and progress summary.
* **[Dataset & Ingestion Guide](file:///Users/atulyasharan/Documents/TurbofanGuard/docs/dataset_and_pipeline.md)**: Overview of suites DS01–DS04, 18 input features, 14 clean targets, strict data leakage isolation, and the raw Parquet reader.
* **[Feature Scaling & Normalization](file:///Users/atulyasharan/Documents/TurbofanGuard/docs/data_scaling.md)**: Why neural networks need scaling, StandardScaler, zero-leakage training fit, solving the DS01 zero-variance issue, and inverse transforms.
* **[Neural Backbone Architecture](file:///Users/atulyasharan/Documents/TurbofanGuard/docs/backbone_architecture.md)**: Deep dive into the 57,664-parameter backbone, multi-scale 1D convolutions (k=3, k=5), temporal pooling, snapshot fallback, and 64D latent state.
* **[Machine Learning Strategy](file:///Users/atulyasharan/Documents/TurbofanGuard/docs/ml_strategy.md)**: The physics of analytical redundancy, Step 1 Autoencoder vs. Step 2 Dual-Head network, adaptive thresholding (two-factor authentication for alarms), and evaluation metrics.
* **[Developer & Testing Guide](file:///Users/atulyasharan/Documents/TurbofanGuard/docs/developer_guide.md)**: How to run automated test scripts (`uv run python scripts/...`), manage JSON configs, and build next-stage modules.

---

## Benchmark Dataset Reference

TurbofanGuard is built and evaluated on the **Turbofan Sensor-FDI-Bench** dataset:

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
