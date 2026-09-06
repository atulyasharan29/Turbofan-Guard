# TurbofanGuard: Complete Machine Learning Architecture Strategy

This document provides a comprehensive explanation of the machine learning architecture strategy behind **TurbofanGuard**. It covers the physical foundations of jet engine sensing, mathematical formulations, fault injection mechanisms, the two-step architectural progression, loss functions, and evaluation metrics.

---

## 1. Executive Summary & The Three Operational Goals

When deployed on an aircraft engine's live telemetry stream during flight, TurbofanGuard continuously solves three simultaneous operational objectives:

```text
+-------------------------------------------------------------+
| Telemetry Stream (18D: 4 Flight Conditions + 14 Raw Sensors)|
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|              TurbofanGuard Multi-Task Engine                |
+-------------------------------------------------------------+
        │                           │                       │
        ▼                           ▼                       ▼
+-----------------------+   +-------------------+   +--------------------+
| Goal 1: Detection     |   | Goal 2: Isolation |   | Goal 3: Virtual    |
| 'Is anything broken?' |   | 'Which sensor?'   |   | Sensing (Clean ŷ)  |
| Healthy (0)/Fault (1) |   | e.g. T030, NL     |   | True K, Pa, RPM    |
+-----------------------+   +-------------------+   +--------------------+
```

### Goal 1: Fault Detection (Is Something Broken?)
Determines whether any sensor on the engine is currently experiencing a failure:

```math
\text{State}_t \in \{0, 1\} \quad (0 = \text{Healthy}, \; 1 = \text{Faulty})
```

* **In Plain English**: At every flight cycle, the system makes a binary decision: is the overall sensing system operating normally (state = 0), or has at least one sensor suffered an electrical or mechanical breakdown (state = 1)?
* **Variables**:
  * `State_t`: Binary health status of the engine sensing system at flight cycle t.
  * `0`: Normal / healthy state (all sensors operating normally within expected noise).
  * `1`: Active fault state (one or more sensors are malfunctioning).

### Goal 2: Fault Isolation (Which Sensor is Broken?)
Pinpoints precisely which of the 14 sensors is malfunctioning:

```math
\mathbf{m}_t \in \{0, 1\}^{14}
```

* **In Plain English**: A binary checklist of 14 switches, one for each sensor channel. If sensor 13 (compressor exit temperature T030) has a broken thermocouple, switch 13 flips to 1, while all other healthy switches remain 0.
* **Variables**:
  * `m_t`: Multi-hot binary fault indicator vector for the 14 sensors at flight cycle t.
  * `m_{t, i} = 1`: Sensor channel i is actively faulty at flight cycle t.
  * `m_{t, i} = 0`: Sensor channel i is operating normally at flight cycle t.

### Goal 3: Signal Reconstruction (Virtual Sensing)
Estimates the clean, uncorrupted thermodynamic truth to replace damaged telemetry:

```math
\hat{\mathbf{y}}_t \in \mathbb{R}^{14}
```

* **In Plain English**: An on-wing software sensor that calculates what all 14 engine instruments *should* read under true physics, with all electrical noise and fault offsets removed.
* **Variables**:
  * `ŷ_t` ("y-hat"): The 14-dimensional vector of clean reconstructed sensor estimates at flight cycle t.
  * `R^14`: 14 continuous physical numbers in real engineering units (Kelvin, Pascals, RPM, kg/s).

---

## 2. Mathematical Formulation of Telemetry and Faults

### 2.1 Input Feature Space (18 Features)
At each discrete flight cycle t (from cycle 1 to 200), the engine telemetry delivers an 18-dimensional vector:

```math
\mathbf{x}_t = \begin{bmatrix} \mathbf{c}_t \\ \mathbf{x}_t^{\text{sensor}} \end{bmatrix} \in \mathbb{R}^{18}
```

* **In Plain English**: The input vector is formed by stacking the 4 environmental flight conditions on top of the 14 raw sensor telemetry readings.
* **Variables**:
  * `x_t`: Full 18-dimensional telemetry input vector at flight cycle t.
  * `c_t`: 4-dimensional vector of flight operating conditions.
  * `x_t^sensor`: 14-dimensional vector of observed sensor telemetry.

#### 1. Operating Conditions (4 Features)
```math
\mathbf{c}_t = \begin{bmatrix} \text{ALT}_t \\ \text{XM}_t \\ \text{DTISA}_t \\ \text{EPR}_t \end{bmatrix} \in \mathbb{R}^4
```

* **In Plain English**: These 4 numbers inform the model where the aircraft is flying, how fast it is moving, how warm or cold the ambient air is, and how much thrust the pilot has demanded.
* **Variables**:
  * `ALT_t`: Flight altitude in meters (baseline cruise is 10,668 m, or 35,000 ft).
  * `XM_t`: Flight Mach number (dimensionless speed relative to sound, baseline is 0.78).
  * `DTISA_t`: Delta ISA ambient temperature deviation in Kelvin.
  * `EPR_t`: Engine Pressure Ratio (primary throttle command, baseline is 1.8118).

#### 2. Observed Sensor Telemetry (14 Channels)
```math
\mathbf{x}_t^{\text{sensor}} = \mathbf{y}_t^* + \boldsymbol{\epsilon}_t + \mathbf{f}_t \in \mathbb{R}^{14}
```

* **In Plain English**: What a physical sensor actually records equals the true underlying engine physics, plus measurement noise, plus any error caused by an active sensor breakdown.
* **Variables**:
  * `x_t^sensor`: 14 observed sensor channels recorded in telemetry at cycle t.
  * `y_t^*`: True, latent thermodynamic engine state (the clean signal if no noise or faults existed).
  * `ϵ_t`: Total measurement noise vector added by electronics and sensor instrumentation.
  * `f_t`: Fault injection vector representing physical or electrical failure on one or more sensors (equals zero when healthy).

---

### 2.2 Telemetry Noise Model
In real aircraft, telemetry is corrupted by a composite noise process:

```math
\boldsymbol{\epsilon}_t = \boldsymbol{\eta}_t + \mathbf{p}_t
```

* **In Plain English**: Sensor noise is composed of two independent phenomena: constant background Gaussian noise present on every reading, plus occasional high-magnitude spikes.
* **Variables**:
  * `ϵ_t`: Composite noise vector affecting all 14 sensors at cycle t.
  * `η_t`: High-frequency Gaussian measurement noise.
  * `p_t`: Sparse peak noise vector (intermittent large electrical spikes occurring on approximately 1% of samples).

Where background Gaussian measurement noise is defined as:

```math
\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_t)
```

* **In Plain English**: Background noise follows a bell curve centered at zero error, with individual standard deviations for each sensor that fluctuate randomly between 1.0x and 3.0x across different flights.
* **Variables**:
  * `η_t`: Gaussian noise vector at cycle t.
  * `N`: Normal (Gaussian) probability distribution.
  * `0`: Mean vector of 14 zeros (noise does not create a permanent bias).
  * `Σ_t`: Covariance matrix containing the individual variance for each sensor channel.

---

### 2.3 Sensor Fault Injection Mechanisms
When an engine is operating normally before a fault begins:

```math
\mathbf{f}_t = \mathbf{0} \quad (t < t_{\text{start}})
```

* **In Plain English**: Before a fault starts, the fault vector is zero on all 14 channels—the engine sensors are undamaged.
* **Variables**:
  * `f_t`: 14-dimensional sensor fault vector at cycle t.
  * `t`: Current flight cycle counter (from cycle 1 to 200).
  * `t_start`: Flight cycle at which a sensor fault begins.

When a sensor `i` experiences a failure starting at flight cycle `t_start`, the error `f_t[i]` follows one of four realistic mathematical patterns:

#### Pattern A: Linear Drift
```math
\mathbf{f}_t[i] = k_i \cdot (t - t_{\text{start}}) \quad (t \ge t_{\text{start}})
```

* **In Plain English**: Sensor i gradually drifts away from true physics at a steady rate of k_i engineering units per flight cycle.
* **Variables**:
  * `f_t[i]`: Fault error added to sensor i at cycle t.
  * `k_i`: Drift slope (rate of error added per flight cycle; can be positive or negative).
  * `t - t_start`: Number of flight cycles elapsed since the fault began.

#### Pattern B: Exponential / Non-Linear Drift
```math
\mathbf{f}_t[i] = \text{sign}_i \cdot \alpha_i \cdot (t - t_{\text{start}})^{\beta_i} \quad (t \ge t_{\text{start}}, \; \beta_i \in [2.0, 5.0])
```

* **In Plain English**: Sensor i drifts away slowly at first, but accelerates non-linearly over time according to a power-law exponent between 2.0 and 5.0.
* **Variables**:
  * `sign_i`: Direction of the drift (+1 for positive upward drift, -1 for negative downward drift).
  * `α_i`: Drift intensity scaling coefficient.
  * `β_i`: Non-linear growth exponent (between 2.0 and 5.0).

#### Pattern C: Abrupt Step / Bias
```math
\mathbf{f}_t[i] = b_i \cdot \mathbb{I}(t \ge t_{\text{start}})
```

* **In Plain English**: At cycle t_start, sensor i instantaneously jumps by a fixed constant bias b_i and stays offset permanently.
* **Variables**:
  * `b_i`: Fixed bias magnitude (e.g. +15 Kelvin or -20 kPa).
  * `I(...)`: Mathematical indicator function, which equals 1 when t >= t_start, and 0 otherwise.

#### Pattern D: Rapid-Growth Step
```math
\mathbf{f}_t[i] = b_i \cdot \min\left(1, \; \frac{t - t_{\text{start}}}{\Delta t_{\text{growth}}}\right) \quad (\Delta t_{\text{growth}} \in [2, 6])
```

* **In Plain English**: The fault ramps up linearly from zero to full bias b_i over a brief window of 2 to 6 flights, and then remains permanently clamped at b_i.
* **Variables**:
  * `Δt_growth`: Ramp-up duration in flight cycles (takes 2 to 6 cycles to reach full error).
  * `min(1, ...)`: Clamps the multiplier at 1.0 once the ramp period completes so the bias stops growing.

---

## 3. The Foundational Principle: "Analytical Redundancy"

To understand why machine learning can diagnose jet engines, consider how an aircraft engine works:

> **The Jet Engine is a Connected Physical Machine**:
> An aircraft turbofan is governed by strict laws of thermodynamics (conservation of mass, momentum, and energy). No sensor operates in isolation.
> When the pilot pushes the throttle forward (Engine Pressure Ratio `EPR` increases):
> * Fuel flow (`WFE`) **must** increase to deliver chemical energy.
> * High-pressure spool speed (`NH`) and low-pressure fan speed (`NL`) **must** accelerate.
> * Pressures along the gas path (`P023`, `P030`, `P044`, `P050`) **must** rise in exact aerodynamic ratios.
> * Gas temperatures (`T023`, `T030`, `T050`) **must** climb predictably.

```text
TRADITIONAL APPROACH: HARDWARE REDUNDANCY
  Sensor A + Sensor B (duplicate) + Sensor C (triplicate) ──> Majority Voting
  • Heavy structural weight, extra wiring harnesses, higher maintenance costs.

MODERN APPROACH: ANALYTICAL REDUNDANCY (TurbofanGuard)
  Single Physical Sensors ──> Digital Twin (Understands thermodynamic coupling)
  • If 13 sensors & conditions agree on 691 K, but T030 reads 740 K ──> FAULT ISOLATED!
  • Zero duplicate hardware needed.
```

### Hardware Redundancy vs. Analytical Redundancy
* **Hardware Redundancy**: In older aircraft, engineers installed two or three duplicate sensors on every pipe. If Sensor A read 700 K and Sensor B read 700 K, but Sensor C read 750 K, a voting computer dropped Sensor C. However, adding physical sensors adds structural weight, cabling, cost, and additional failure points.
* **Analytical Redundancy**: Instead of extra physical hardware, we use **software intelligence**. Because all 14 sensors are physically connected through the thermodynamic cycle, 13 healthy sensors can mathematically verify the 14th sensor!
  * If altitude, airspeed, throttle, and 13 sensors all indicate standard cruise at 691 K, but sensor `T030` suddenly reports 740 K, the laws of thermodynamics are violated.
  * The engine cannot physically produce 740 K without other pressures and speeds changing. Therefore, **the engine is fine, but sensor `T030` is broken**.

---

## 4. The Two-Step Architectural Strategy

Rather than building an opaque, black-box model that tries to do everything in one unexplainable step, TurbofanGuard follows a structured two-step roadmap:

```text
Step 1: Baseline Denoising Autoencoder / Virtual Sensor (Physics Residual FDI)
   └── Concept: "Virtual Sensing via Analytical Redundancy"
   └── Training: Self-supervised on healthy flights only (DS01 & DS02)
   └── Supervision: Clean targets y* (Zero fault labels needed during training!)
   └── Decision: Static residual thresholding (r_i > tau_i)
   └── Strength: Highly interpretable, zero-fault training, physics-grounded

Step 2: Dual-Head Multi-Task Network (Physics + Diagnostic AI)
   └── Concept: "Two-Factor Authentication for Sensor Alarms"
   └── Training: End-to-end multi-task on healthy + faulted data (DS02, DS03, DS04)
   └── Head 1 (Reconstruction): Estimates clean continuous signals ŷ_t ∈ ℝ¹⁴
   └── Head 2 (Diagnostic FDI): Outputs multi-label fault probabilities p_t ∈ [0, 1]¹⁴
   └── Decision: Adaptive dynamic thresholding fusing physical residuals with AI confidence
   └── Strength: Rejects noise spikes, near-zero detection latency on slow drifts
```

---

## 5. Step 1: Baseline Denoising Autoencoder (Physics-Informed Residual FDI)

Step 1 trains a virtual sensor on **nominal (healthy) flights only** (`DS01` and `DS02`). It requires **zero fault labels** during training.

```text
1. COMPRESSION (ENCODER):
   Observed Telemetry x_t (18D) ──> Encoder Layers + Masking ──> Latent Bottleneck z_t (64D)

2. RECONSTRUCTION (DECODER):
   Latent Bottleneck z_t (64D)  ──> Decoder Layers           ──> Physics Prediction ŷ_t (14D)

3. RESIDUAL COMPARISON:
   Observed x_t (14 sensors) - Predicted ŷ_t (14 sensors)   ──> Residual r_i = |x_i - ŷ_i| / σ_i

4. DECISION LOGIC:
   • If r_i > Threshold τ_i: FAULT on Sensor i! (Drop bad reading, replace with ŷ_i)
   • If r_i ≤ Threshold τ_i: Healthy Sensor (Normal operation)
```

### 1. The Information Bottleneck & Channel Masking
The network compresses the 18 telemetry features through a narrow latent bottleneck (`z_t`).
* **Channel Masking Augmentation**: If sensor `T030` suffers a massive +50 K jump during flight, we do not want that bad number to corrupt the latent state `z_t` and distort the predictions for the other 13 sensors.
* To prevent this, the encoder is trained with **random channel masking**: during training, 1 or 2 sensor channels are randomly zeroed out. This forces the network to learn how to predict all 14 sensors from flight conditions and the remaining unmasked sensors.

### 2. Training Objective: Reconstruction Loss
Trained on healthy flights to minimize the difference between estimated signals and clean targets:

```math
\mathcal{L}_{\text{recon}} = \frac{1}{14} \sum_{i=1}^{14} (\hat{y}_{t, i} - y_{t, i}^*)^2
```

* **In Plain English**: The model measures the average squared difference between its virtual sensor predictions and the true physical values across all 14 channels, updating its weights to make this error as small as possible.
* **Variables**:
  * `L_recon`: Reconstruction Mean Squared Error (MSE).
  * `ŷ_{t, i}`: Model predicted clean value for sensor i.
  * `y_{t, i}^*`: True ground-truth clean value for sensor i.
  * `14`: Number of sensor channels.

### 3. Real-Time Inference: Normalized Physical Residuals
During flight, the model continuously compares what the physical sensor reports against what the virtual sensor predicts:

```math
r_{t, i} = \frac{|x_{t, i}^{\text{sensor}} - \hat{y}_{t, i}|}{\sigma_{i, \text{nominal}}}
```

* **In Plain English**: The residual measures how many standard deviations of normal noise the sensor reading is away from what the physics model says it should be.
* **Variables**:
  * `r_{t, i}`: Normalized residual for sensor i at flight cycle t (a dimensionless score).
  * `x_{t, i}^sensor`: Raw physical measurement reported by the sensor.
  * `ŷ_{t, i}`: Virtual sensor prediction.
  * `σ_{i, nominal}`: Expected normal noise standard deviation for that sensor.

### 4. Decision Rule (Static Thresholding)
* **Healthy Sensor**: Telemetry matches virtual prediction closely:
```math
x_{t, i} \approx \hat{y}_{t, i} \implies r_{t, i} < \tau_i \quad (\text{typically } \tau_i \approx 3.5 \text{ to } 4.5)
```
* **Faulty Sensor**: Telemetry diverges from physics:
```math
r_{t, i} > \tau_i \implies \text{Alarm! Sensor } i \text{ is faulty!}
```
* **Signal Healing**: Once flagged, the flight computer discards the damaged physical reading and adopts the clean virtual prediction:
```math
x_{t, i}^{\text{validated}} \leftarrow \hat{y}_{t, i}
```

---

## 6. Step 2: Dual-Head Multi-Task Architecture (Physics + Diagnostic AI)

While Step 1 is effective, relying solely on static residual thresholds faces practical limitations:
1. **Noise Spikes Cause False Alarms**: A single random electrical spike can momentarily breach a static threshold of 4.0 sigma, triggering a false alarm.
2. **Detection Latency on Slow Drifts**: A subtle drift growing by only 0.1 sigma per flight cycle can take 35 to 40 flights before it finally crosses a high static threshold.
3. **Multi-Fault Ambiguity (DS04)**: When 2 or 3 sensors break at the same time, multiple residuals spike, requiring an intelligent classifier to disentangle them.

Step 2 resolves these limitations with a **Dual-Head Multi-Task Network**:

```text
1. SHARED NEURAL BACKBONE (THE ENGINE BRAIN):
   Input Sequence Window (W × 18) ──> TurbofanBackbone ──> Shared Latent State z_t (64D)
                                                                     │
                                    ┌────────────────────────────────┴──────────────────────────────┐
                                    ▼                                                               ▼
2. DUAL HEADS:          Head 1: Signal Reconstruction                                   Head 2: Diagnostic FDI
                        (Dense Layers)                                                  (Dense Layers + Sigmoid)
                                    │                                                               │
                                    ▼                                                               ▼
                        Clean Signals ŷ_t ∈ ℝ¹⁴                                         Fault Probabilities p_t ∈ [0, 1]¹⁴
                        (Kelvin, Pascals, RPM)                                          (e.g., T030: 98.2%, NL: 1.1%)
                                    │                                                               │
                                    └────────────────────────────────┬──────────────────────────────┘
                                                                     ▼
3. ADAPTIVE FUSION:                                  Compute Residuals: r_i = |x_i - ŷ_i| / σ_i
("Two-Factor Authentication")                                        │
                                                     Check Trigger: r_i > τ_adaptive(p_i) ?
                                                                     │
                                            ┌────────────────────────┴────────────────────────┐
                                            ▼                                                 ▼
                                     [YES: CONFIRMED ALARM]                         [NO: SUPPRESSED]
                                     • Zero false alarms on spikes                  Transient electrical spike
                                     • Low latency on subtle drifts                 ignored safely
                                     • Replace faulty sensor with ŷ_i
```

### Head 1: Continuous Signal Reconstruction
```math
\hat{\mathbf{y}}_t = g_{\text{recon}}(\mathbf{z}_t) \in \mathbb{R}^{14}
```
* **In Plain English**: A dense regression network that translates the shared latent state into continuous physical sensor measurements.

### Head 2: Multi-Label Fault Classification
```math
\mathbf{p}_t = \sigma\left( g_{\text{diag}}(\mathbf{z}_t) \right) \in [0, 1]^{14}
```
* **In Plain English**: A dense classification network followed by a Sigmoid function (sigma) that outputs 14 independent probabilities between 0% and 100%, indicating the AI's confidence that each sensor has suffered a fault.

---

## 7. Adaptive Thresholding: "Two-Factor Authentication" for Alarms

Instead of relying on a rigid, static threshold, TurbofanGuard introduces **Adaptive Residual Thresholding**:

```math
\tau_{\text{adaptive}}(p_{t, i}) = \tau_{\text{high}} - (\tau_{\text{high}} - \tau_{\text{low}}) \cdot p_{t, i}
```

* **In Plain English**: The threshold required to trigger an alarm automatically adapts based on AI confidence:
  * When the AI sees no pattern (probability near 0%), the threshold stays very high (tau_high = 4.5 sigma).
  * When the AI recognizes a multi-cycle drift trend (probability approaching 100%), the threshold smoothly drops to a sensitive level (tau_low = 2.0 sigma).
* **Variables**:
  * `tau_adaptive`: Dynamic threshold for sensor i at cycle t.
  * `tau_high`: High conservative threshold (e.g., 4.5 sigma).
  * `tau_low`: Low sensitive threshold (e.g., 2.0 sigma).
  * `p_{t, i}`: AI diagnostic probability for sensor i.

```math
\text{Trigger Alarm on Sensor } i \iff r_{t, i} > \tau_{\text{adaptive}}(p_{t, i})
```

```text
Scenario A: An electrical noise spike hits a healthy sensor
   • Physical Residual spikes momentarily: r_i = 3.8
   • AI Diagnostic Head inspects sequence window: No drift trend detected! p_i = 0.05
   • Dynamic Threshold stays high: tau_adaptive = 4.5 - (4.5 - 2.0) * 0.05 = 4.38
   • Decision: 3.8 < 4.38 ──> ALARM SUPPRESSED! (Zero false alarms on noise!)

Scenario B: A subtle drift begins developing on sensor T030
   • Flight 1 to 10: Drift slowly climbs: r_i = 2.2
   • AI Diagnostic Head spots multi-cycle trend: p_i = 0.90
   • Dynamic Threshold relaxes: tau_adaptive = 4.5 - (4.5 - 2.0) * 0.90 = 2.25
   • By Flight 11: r_i reaches 2.3 > 2.25 ──> CONFIRMED ALARM!
   • Result: Caught 25 flights earlier than a static 4.5 threshold!
```

---

## 8. Joint Multi-Task Loss Formulation

During training of Step 2, the neural network learns to optimize both tasks at once using a combined loss function:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{FDI}}
```

* **In Plain English**: The overall training loss adds the physics reconstruction error to the fault classification error, balanced by a weighting knob lambda.
* **Variables**:
  * `L_total`: Total multi-task loss minimized by the optimizer.
  * `L_recon`: Reconstruction loss (penalizes inaccurate virtual sensor values).
  * `L_FDI`: Classification loss (penalizes missed faults or false alarms).
  * `lambda`: Hyperparameter balancing the two loss scales (typically lambda between 0.1 and 1.0).

### The Classification Loss
Because flights with sensor faults are rare compared to healthy flights (class imbalance), we use **Weighted Multi-Label Binary Cross-Entropy**:

```math
\mathcal{L}_{\text{FDI}} = -\frac{1}{14} \sum_{i=1}^{14} \left[ w_{\text{pos}} \cdot m_{t, i} \log(p_{t, i}) + (1 - m_{t, i}) \log(1 - p_{t, i}) \right]
```

* **In Plain English**: Standard cross-entropy, but missing a real fault is penalized w_pos times more severely than a false alarm.
* **Variables**:
  * `m_{t, i}`: Ground-truth label for sensor i (1 if broken, 0 if healthy).
  * `p_{t, i}`: Predicted fault probability for sensor i.
  * `w_pos`: Positive class weight multiplier (typically 5.0 to 10.0).
  * `log`: Natural logarithm.

---

## 9. Performance Evaluation Metrics

To rigorously evaluate TurbofanGuard against aerospace industry standards, we use six quantitative metrics:

### 1. Denoising Accuracy (RMSE & MAPE)
* **Root Mean Squared Error (RMSE)**:
```math
\text{RMSE}_i = \sqrt{\frac{1}{N} \sum_{t=1}^N (\hat{y}_{t, i} - y_{t, i}^*)^2}
```
Measures average reconstruction error in real physical units (Kelvin, Pascals, RPM).

* **Mean Absolute Percentage Error (MAPE)**:
```math
\text{MAPE}_i = \frac{100\%}{N} \sum_{t=1}^N \left| \frac{\hat{y}_{t, i} - y_{t, i}^*}{y_{t, i}^*} \right|
```
Measures virtual sensor error as a clean, intuitive percentage (e.g. 0.3% error).

### 2. Detection Reliability (FAR & TPR)
* **False Alarm Rate (FAR)**:
```math
\text{FAR} = \frac{\text{False Alarms Triggered on Healthy Flights}}{\text{Total Healthy Flight Cycles Evaluated}} \quad (\text{Target: } < 1.0\%)
```
In commercial aviation, false alarms must stay strictly below 1.0% to avoid ground delays and unnecessary part replacements.

* **True Positive Rate (TPR / Recall)**:
```math
\text{TPR} = \frac{\text{Successfully Detected Fault Cycles}}{\text{Total Active Fault Cycles}} \quad (\text{Target: } > 95\%)
```
Ensures that at least 95% of active sensor faults are successfully caught.

### 3. Detection Latency
```math
\Delta t_{\text{det}} = t_{\text{alarm}} - t_{\text{fault\_start}}
```
Measures how many flight cycles elapse between the moment a sensor begins failing and the moment the alarm triggers (lower is better).

### 4. Multi-Sensor Accuracy (Hamming Loss & Subset Accuracy)
* **Exact Match Ratio (Subset Accuracy)**:
```math
\text{Subset Accuracy} = \frac{1}{N} \sum_{t=1}^N \mathbb{I}(\hat{\mathbf{m}}_t = \mathbf{m}_t)
```
The percentage of flight cycles where all 14 sensors are simultaneously diagnosed 100% correctly.

* **Hamming Loss**:
```math
\text{Hamming Loss} = \frac{1}{14 \cdot N} \sum_{t=1}^N \sum_{i=1}^{14} \mathbb{I}(\hat{m}_{t, i} \ne m_{t, i})
```
Evaluates average per-sensor error on DS04. A Hamming loss of 0.02 means that 98% of all individual sensor diagnostic flags were correct.
