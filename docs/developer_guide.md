# TurbofanGuard: Developer & Testing Guide

This guide provides practical instructions on how the codebase is organized, how to run automated verification tests, how configuration files work, and how to build upon the existing modules.

---

## 1. Project Directory Structure

```text
TurbofanGuard/
├── configs/                          # Hyperparameter configuration JSON files
│   ├── backbone_config.json          # Architecture settings for the shared backbone
│   └── dual_head_config.json         # Architecture settings for Step 2 Dual-Head network
├── docs/                             # Full documentation suite
│   ├── README.md                     # Documentation hub and glossary
│   ├── dataset_and_pipeline.md       # Dataset guide, 18D inputs, 14D targets, data audit
│   ├── data_scaling.md               # Normalization, StandardScaler, zero-leakage, JSON
│   ├── backbone_architecture.md      # Multi-scale 1D CNN, temporal pooling, latent state
│   ├── ml_strategy.md                # Analytical redundancy, Step 1 vs Step 2, metrics
│   └── developer_guide.md            # This testing and developer guide
├── scripts/                          # Automated verification and audit test scripts
│   ├── audit_data_integrity.py       # Scans all splits for nulls, disjointness, continuity
│   ├── test_reader.py                # Tests Parquet loading and column partitioning
│   ├── test_scaler.py                # Tests zero-leakage scaling and inverse transform
│   ├── test_backbone.py              # Tests neural backbone forward pass, gradients, and MPS
│   ├── test_baseline_ae.py           # Tests Step 1 baseline autoencoder
│   ├── evaluate_baseline_fdi.py      # Evaluates Step 1 baseline on DS03
│   ├── test_dual_head.py             # Tests Step 2 dual-head architecture and adaptive logic
│   └── evaluate_dual_head_fdi.py     # Evaluates Step 2 and benchmarks against Step 1
├── src/                              # Core Python library
│   ├── data/
│   │   ├── reader.py                 # Raw Parquet ingestion and column categorization
│   │   ├── scaler.py                 # TurbofanScaler (StandardScaler wrapper with whitelists)
│   │   └── dataset.py                # TurbofanDataset (sliding windows, masking, multi-hot labels)
│   ├── models/
│   │   ├── backbone.py               # TurbofanBackbone (PyTorch neural network)
│   │   ├── baseline_ae.py            # TurbofanBaselineAE (Step 1 virtual sensor)
│   │   └── dual_head_fdi.py          # TurbofanDualHeadFDI (Step 2 physics + diagnostic AI)
│   ├── training/
│   │   ├── losses.py                 # MultiTaskFDILoss (MSE + Weighted Multi-Label BCE)
│   │   ├── train_baseline.py         # Step 1 baseline training pipeline
│   │   └── train_dual_head.py        # Step 2 multi-task joint training pipeline
│   └── utils/
│       └── config.py                 # BackboneConfig & DualHeadConfig dataclasses
├── DS01/                             # Benchmark Suite 1: Fixed cruise baseline
├── DS02/                             # Benchmark Suite 2: Variable flight envelope
├── DS03/                             # Benchmark Suite 3: Single structured sensor faults
├── DS04/                             # Benchmark Suite 4: Concurrent multi-sensor faults
├── pyproject.toml                    # Dependencies and project metadata
└── README.md                         # Top-level repository overview
```

---

## 2. Environment Setup & Dependencies

The project uses `uv` for ultra-fast, reproducible Python environment management.

### Key Dependencies
* **`torch`**: Deep learning framework used for `TurbofanBackbone` and multi-task optimization.
* **`pandas` & `pyarrow`**: High-performance columnar Parquet dataset reading.
* **`scikit-learn`**: Production-grade `StandardScaler` foundation.
* **`numpy`**: Fast array math and inverse transformation verification.

### Running Commands with `uv`
You do not need to manually activate virtual environments. Simply prefix any Python command with `uv run`:

```bash
uv run python <script_path>
```

---

## 3. Running Automated Tests & Audits

We maintain four standalone verification scripts in the `scripts/` directory. Each script verifies a specific layer of the system.

### 1. Ingestion Pipeline Test (`scripts/test_reader.py`)
Verifies that raw Parquet files load properly for all suites (`DS01` to `DS04`), that manifests are found, and that columns are partitioned correctly without missing features.

```bash
uv run python scripts/test_reader.py
```

* **What it checks**:
  * Loads `train.parquet`, `val.parquet`, and `test.parquet`.
  * Verifies that the 4 flight conditions, 14 observed sensors, 14 clean targets, and 10 health indices are separated cleanly.
  * Checks that metadata dictionaries and manifests load without errors.

### 2. Normalization & Scaler Test (`scripts/test_scaler.py`)
Verifies that `TurbofanScaler` scales inputs and targets strictly using training statistics with zero data leakage.

```bash
uv run python scripts/test_scaler.py
```

* **What it checks**:
  * Fits exclusively on `train.parquet`.
  * Verifies transformed arrays have correct shapes `(N, 18)` and `(N, 14)`.
  * Verifies mean is $0.0$ and non-constant feature variance is $1.0$.
  * Handles constant conditions in `DS01` without division-by-zero errors.
  * Confirms round-trip inverse transformation accuracy (relative error $< 10^{-7}$).
  * Verifies saving to and loading from JSON.

### 3. Data Integrity & Leakage Audit (`scripts/audit_data_integrity.py`)
Performs a deep diagnostic scan across the entire dataset.

```bash
uv run python scripts/audit_data_integrity.py
```

* **What it checks**:
  * **Engine Disjointness**: Confirms that zero engine serial numbers overlap between train, val, and test splits.
  * **Missing Values**: Confirms that nulls, NaNs, and infinities are exactly zero across all columns.
  * **Cycle Continuity**: Confirms that every engine flies sequentially from cycle 1 to 200 without gaps.
  * **Target Isolation**: Verifies that internal simulator health indices (`DETA*`, `CW*`) are not leaked into model inputs.

### 4. Neural Backbone Test (`scripts/test_backbone.py`)
Verifies the PyTorch `TurbofanBackbone` architecture across all operating modes and hardware devices.

```bash
uv run python scripts/test_backbone.py
```

* **What it checks**:
  * Parameter count is exactly 57,664.
  * Successfully processes 3D temporal sequence windows with variable lengths ($W = 10, 16, 24, 30$).
  * Successfully processes single-engine streaming windows `(1, 16, 18)`.
  * Successfully processes 2D single-snapshot inputs `(batch_size, 18)` via the fallback projection.
  * Verifies backward gradient flow across all convolutional and dense layers.
  * Verifies hardware acceleration on Apple Silicon GPU (`MPS`) or CPU.

### 5. Step 1 Baseline Autoencoder Test (`scripts/test_baseline_ae.py`)
Verifies `TurbofanBaselineAE` forward geometry, parameter count (< 100k), and static threshold residual FDI.

```bash
uv run python scripts/test_baseline_ae.py
```

### 6. Step 1 Baseline Evaluation on DS03 (`scripts/evaluate_baseline_fdi.py`)
Evaluates the trained virtual sensor on single-fault flights (`DS03` test set) using static residual thresholding.

```bash
uv run python scripts/evaluate_baseline_fdi.py
```

### 7. Step 2 Dual-Head Architecture Test (`scripts/test_dual_head.py`)
Verifies `TurbofanDualHeadFDI` forward pass, `MultiTaskFDILoss` computation, simultaneous two-head gradient flow, and Two-Factor Authentication logic.

```bash
uv run python scripts/test_dual_head.py
```

### 8. Step 2 Dual-Head FDI Evaluation (`scripts/evaluate_dual_head_fdi.py`)
Evaluates the dual-head multi-task model on `DS03` and `DS04` test sets, printing comparative benchmark metrics directly against Step 1.

```bash
uv run python scripts/evaluate_dual_head_fdi.py --suite DS03 --split test
```

---

## 4. Configuration Management

Hyperparameters are decoupled from model code using typed Python dataclasses and portable JSON files.

### Configuration Files
* **Backbone Configuration (`configs/backbone_config.json`)**:
  Defines input dimensions, temporal sequence length $W$, multi-scale 1D CNN channels, kernel sizes, and latent bottleneck dimension (64D).
* **Dual-Head Multi-Task Configuration (`configs/dual_head_config.json`)**:
  Defines Head 1 and Head 2 layer dimensions, joint loss weighting ($\lambda$), positive fault class weighting (`pos_weight`), and adaptive threshold boundaries ($\tau_{\text{high}} = 4.5\sigma$, $\tau_{\text{low}} = 2.0\sigma$).

### Programmatic Usage in Python
```python
from src.utils.config import BackboneConfig, DualHeadConfig
from src.models.dual_head_fdi import TurbofanDualHeadFDI

# 1. Load configuration from JSON
config = DualHeadConfig.from_json("configs/dual_head_config.json")

# 2. Modify hyperparameters safely with automatic validation
experiment_config = config.copy_with(pos_weight=8.0, lambda_fdi=0.6)

# 3. Instantiate model with config
model = TurbofanDualHeadFDI(config=experiment_config)
```

If an engineer accidentally inputs an invalid hyperparameter (e.g. `tau_low >= tau_high` or `dropout = 1.5`), validation immediately raises a descriptive `ValueError`.

---

## 5. Architectural Progression & Status

1. **PyTorch Dataset & DataLoader (`src/data/dataset.py`)** - [COMPLETE]
   * Sliding temporal sequence window slicing across 200-cycle trajectories.
   * Training-time random channel masking augmentation.
   * Dynamic multi-hot ground truth matrix ($\mathbf{m}_t$) supporting `DS01` through `DS04`.

2. **Step 1 Baseline Autoencoder (`src/models/baseline_ae.py`)** - [COMPLETE]
   * Virtual sensor reconstruction decoder attached to shared backbone.
   * Trained on healthy flights (`DS02`), achieving 0.21% physical MAPE.
   * Benchmarked static residual thresholding on single-fault `DS03`.

3. **Step 2 Dual-Head Multi-Task Network (`src/models/dual_head_fdi.py`)** - [COMPLETE]
   * Parallel Head 1 (Physics Reconstruction) and Head 2 (Diagnostic Classifier).
   * Joint multi-task loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{FDI}}$).
   * Two-Factor Authentication with adaptive thresholding ($\tau_{\text{adaptive}}$).
   * Comparative evaluation against Step 1 baseline.

