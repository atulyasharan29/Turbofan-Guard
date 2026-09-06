# TurbofanGuard: Developer & Testing Guide

This guide provides practical instructions on how the codebase is organized, how to run automated verification tests, how configuration files work, and how to build upon the existing modules.

---

## 1. Project Directory Structure

```text
TurbofanGuard/
├── configs/                          # Hyperparameter configuration JSON files
│   └── backbone_config.json          # Architecture settings for the shared backbone
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
│   └── test_backbone.py              # Tests neural backbone forward pass, gradients, and MPS
├── src/                              # Core Python library
│   ├── data/
│   │   ├── reader.py                 # Raw Parquet ingestion and column categorization
│   │   └── scaler.py                 # TurbofanScaler (StandardScaler wrapper with whitelists)
│   ├── models/
│   │   └── backbone.py               # TurbofanBackbone (PyTorch neural network)
│   └── utils/
│       └── config.py                 # BackboneConfig dataclass and JSON serialization
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
  * Verifies that full backward gradient flow reaches every convolutional filter and dense weight matrix.
  * Verifies hardware acceleration on Apple Silicon GPU (`MPS`) or CPU.

---

## 4. Configuration Management

Hyperparameters are decoupled from model code using typed Python dataclasses and portable JSON files.

### Configuration File (`configs/backbone_config.json`)
```json
{
  "input_dim": 18,
  "window_length": 16,
  "conv_channels": [32, 64],
  "kernel_sizes": [3, 5],
  "dense_hidden_dims": [128, 64],
  "latent_dim": 64,
  "dropout": 0.1,
  "activation": "gelu"
}
```

### Programmatic Usage in Python
```python
from src.utils.config import BackboneConfig
from src.models.backbone import TurbofanBackbone

# 1. Load configuration from JSON
config = BackboneConfig.from_json("configs/backbone_config.json")

# 2. Modify a hyperparameter safely (with automatic validation)
experiment_config = config.copy_with(latent_dim=128, dropout=0.2)

# 3. Instantiate model with config
model = TurbofanBackbone(config=experiment_config)
```

If an engineer accidentally inputs an invalid hyperparameter (e.g. `dropout = 1.5` or `activation = "invalid"`), `BackboneConfig.validate()` raises a clear `ValueError` immediately, preventing silent failures during long training runs.

---

## 5. Next Steps for Contributors

The foundational data ingestion, scaling, and backbone architecture are complete and 100% verified. The immediate next milestones are:

1. **PyTorch Dataset & DataLoader (`src/data/dataset.py`)**:
   * Build a PyTorch `Dataset` that slices continuous 200-cycle engine trajectories into sliding windows of length $W$.
   * Include random channel masking augmentation on input sensors during training.
   * Dynamically build the multi-hot isolation ground truth ($\mathbf{m}_t$) by joining Parquet flags with `engine_manifest.csv`.

2. **Step 1 Baseline Autoencoder (`src/models/baseline_ae.py`)**:
   * Attach a reconstruction decoder head to the backbone.
   * Train on nominal flights (`DS02`).
   * Evaluate static residual thresholding on single-fault `DS03` flights.

3. **Step 2 Dual-Head Multi-Task Network (`src/models/dual_head_fdi.py`)**:
   * Attach both Head 1 (Reconstruction) and Head 2 (Diagnostic Classifier).
   * Train with the joint loss ($\mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{FDI}}$).
   * Implement adaptive thresholding and benchmark on multi-fault `DS04` flights.
