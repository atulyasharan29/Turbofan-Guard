# TurbofanGuard: Data Scaling & Normalization Pipeline

This document explains why feature scaling is critical for neural networks, how the `TurbofanScaler` pipeline works, how we solved the zero-variance challenge in fixed-condition flights, and how normalized numbers are converted back into physical engineering units.

---

## 1. Why Do Neural Networks Need Scaling?

In an aircraft jet engine, physical sensors measure completely different physical quantities with vastly different numerical scales:

| Sensor / Condition | Physical Quantity | Typical Raw Values |
| :--- | :--- | :--- |
| **`PS0`** (Ambient Pressure) | Pascals ($\text{Pa}$) | $\approx 23,000 \text{ to } 101,325 \text{ Pa}$ |
| **`NH`** (Core Spool Speed) | Rotational Speed ($\text{RPM}$) | $\approx 8,500 \text{ to } 9,500 \text{ RPM}$ |
| **`T030`** (Compressor Exit Temp) | Temperature ($\text{K}$) | $\approx 650 \text{ to } 750 \text{ Kelvin}$ |
| **`WFE`** (Fuel Flow) | Mass Flow ($\text{kg/s}$) | $\approx 0.3 \text{ to } 1.5 \text{ kg/s}$ |
| **`XM`** (Airspeed) | Mach Number (Ratio) | $\approx 0.75 \text{ to } 0.82$ |

```text
WITHOUT SCALING (THE RAW DATA PROBLEM):
  • Pressure: 100,000 Pa (huge magnitude) ──> Gradients explode on pressure!
  • Mach Speed: 0.78 (tiny magnitude)     ──> Mach number is completely ignored!

WITH TURBOFANSCALER (THE SOLUTION):
  • Scaled Pressure: Mean = 0.0, Std = 1.0 (between -2.0 and +2.0)
  • Scaled Mach:     Mean = 0.0, Std = 1.0 (between -2.0 and +2.0)
  ──> Balanced, stable gradient learning where every physical sensor has an equal voice!
```

### What Happens Without Scaling?
If you feed raw numbers directly into a neural network:
1. **Gradient Domination**: The pressure measurement ($100,000$) is more than $100,000$ times larger than the Mach number ($0.78$). During backpropagation, the gradients for large numbers dominate the network, causing weights to oscillate or explode.
2. **Ignored Features**: Subtle changes in Mach number or temperature get drowned out by large pressure numbers, making it impossible for the network to detect small sensor drifts.

### The Solution: Standardization (Z-Score Normalization)
We transform every feature so that its average (mean) is centered at $0.0$, and its typical spread (standard deviation) is scaled to $1.0$:

```math
z = \frac{x - \mu}{\sigma}
```

* **In Plain English**: For each reading, we subtract the historical average ($\mu$), which centers the numbers at zero. Then we divide by the standard deviation ($\sigma$), which shrinks or expands the numbers so that most values comfortably lie between $-2.0$ and $+2.0$.
* **Variables**:
  * $z$: The scaled, normalized value fed into the neural network.
  * $x$: The original raw physical measurement (e.g., $9,200 \text{ RPM}$ or $710 \text{ K}$).
  * $\mu$ (mu): The mean (average value) of that feature across the training dataset.
  * $\sigma$ (sigma): The standard deviation (typical spread/variance) of that feature across the training dataset.

---

## 2. Explicit Feature Whitelists

A common source of bugs in data science is slicing columns by raw numerical indices (like `df.iloc[:, 4:22]`). If someone reorders the table columns or adds a metadata column, the entire machine learning model silently ingests the wrong features.

In TurbofanGuard, `src/data/scaler.py` uses **explicit, named whitelists**:

```python
INPUT_FEATURES = [
    # 4 Flight Operating Conditions
    "XM", "ALT", "DTISA", "EPR",
    # 14 Observed Sensor Telemetry Channels
    "NH_obs", "NL_obs", "WFE_obs", "PS0_obs", "P2_obs", "P023_obs", "P030_obs",
    "P044_obs", "P050_obs", "P134_obs", "T2_obs", "T023_obs", "T030_obs", "T050_obs",
]

TARGET_FEATURES = [
    # 14 Clean Ground-Truth Sensors
    "NH_truth", "NL_truth", "WFE_truth", "PS0_truth", "P2_truth", "P023_truth", "P030_truth",
    "P044_truth", "P050_truth", "P134_truth", "T2_truth", "T023_truth", "T030_truth", "T050_truth",
]
```

* **Why this matters**:
  * `INPUT_FEATURES` strictly selects the 18 columns that physical aircraft systems actually see.
  * `TARGET_FEATURES` strictly selects the 14 clean target columns that the network learns to reconstruct.
  * Internal simulator health indices (`DETA*`, `CW*`) and identifiers (`engine_id`, `flight_cycle`) can **never** accidentally leak into the model inputs.

---

## 3. Strict Zero-Leakage Fitting Rule

A fundamental rule of rigorous machine learning is:

> [!IMPORTANT]
> **Never fit a scaler on test or validation data!**
> The mean ($\mu$) and standard deviation ($\sigma$) must be computed **strictly on the training split**.

```text
Training Split Data ──> scaler.fit() ──> Learns μ_train and σ_train
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       ▼                                                 ▼                               ▼
df_train[X] ──> transform() ──> X_train         df_val[X] ──> transform() ──> X_val      df_test[X] ──> transform() ──> X_test
```

### Why is this essential?
In the real world, a model deployed on an airplane in 2027 will see new flights it has never encountered before. It cannot calculate the future mean of those upcoming flights in advance!
* If a scaler is fitted on the entire dataset (combining train, validation, and test), the model peeks at future statistical distributions. This is called **statistical leakage** (or data snooping).
* In TurbofanGuard, `TurbofanScaler.fit()` is called exclusively on `df_train`. When transforming validation or test sets, it applies the exact training statistics $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.

---

## 4. The "Constant Feature" Problem in DS01

In dataset suite `DS01`, every single flight cycle for all 200 engines is conducted under identical cruise flight conditions:
* Altitude: exactly $10,668 \text{ m}$ ($35,000 \text{ ft}$)
* Mach: exactly $0.78$
* Temperature deviation: exactly $0.0 \text{ K}$
* Throttle (EPR): exactly $1.8118$

Because these four flight condition values never change in `DS01`, their standard deviation is exactly zero:
```math
\sigma = 0
```

### The Division-by-Zero Risk
If you apply a naive formula:
```math
z = \frac{x - \mu}{\sigma} = \frac{x - \mu}{0} \implies \text{Division by Zero! (NaN / Inf Crash)}
```

### How TurbofanScaler Handles It
`TurbofanScaler` wraps scikit-learn's optimized `StandardScaler`. When scikit-learn detects that a feature has zero variance ($\sigma = 0$), it automatically sets the scale factor to $1.0$:

```math
\text{If } \sigma = 0 \implies \text{scale} = 1.0, \quad z = \frac{x - \mu}{1.0} = \frac{10668 - 10668}{1.0} = 0.0
```

* **The Result**: The constant operating condition is cleanly centered at $0.0$. It causes zero mathematical errors, injects zero infinite values, and allows the neural network to train smoothly without crashing.

---

## 5. Converting Back to Physical Units: Inverse Transformation

While the neural network thinks and computes in normalized numbers (centered around $0.0$), flight controllers and maintenance engineers do not speak in normalized standard deviations. An engineer needs to know:
* *"Is the compressor exit temperature $695 \text{ Kelvin}$ or $740 \text{ Kelvin}$?"*
* *"Is the spool speed $9,100 \text{ RPM}$ or $9,600 \text{ RPM}$?"*

`TurbofanScaler` provides exact mathematical inversion:

```math
x_{\text{physical}} = z \cdot \sigma + \mu
```

* **In Plain English**: To convert a normalized network prediction back to real physical engineering units, we simply reverse the math: multiply by the training standard deviation ($\sigma$) and add back the training mean ($\mu$).

### Precision Verification
In our automated test script (`scripts/test_scaler.py`), we performed round-trip verification:
```text
Raw Physical Telemetry ──> Scaled Array ──> Inverse Transformed Telemetry
```
* **Validation Result**: The maximum relative reconstruction error between original raw telemetry and inverse-transformed telemetry is less than $10^{-7}$ (one ten-millionth of a percent). The round-trip is mathematically exact within floating-point precision.

---

## 6. Portability: Saving and Loading Scalers via JSON

To deploy TurbofanGuard in production or share models with other researchers, you cannot rely on Python-specific pickle files, which can break across Python versions and present security risks.

`TurbofanScaler` serializes its parameters directly to clean, human-readable JSON:

```json
{
  "scaler_x": {
    "mean": [0.78, 10668.0, 0.0, 1.8118, ...],
    "scale": [0.012, 450.2, 5.1, 0.082, ...]
  },
  "scaler_y": {
    "mean": [9120.4, 3450.1, 0.89, ...],
    "scale": [145.2, 52.8, 0.04, ...]
  }
}
```

* **`scaler.save("checkpoints/scaler_DS02.json")`**: Saves the fitted parameters to disk.
* **`scaler = TurbofanScaler.load("checkpoints/scaler_DS02.json")`**: Instantly reloads the exact scaler parameters without needing access to the original raw training dataset.
