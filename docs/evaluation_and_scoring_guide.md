# TurbofanGuard: Evaluation & Scoring Guide

This guide explains how **TurbofanGuard** is evaluated, how every quantitative metric is calculated, and what each score means in real-world aviation operations.

Every metric is explained in **plain, simple language**, accompanied by **real-world analogies**, **math blocks**, and **concrete examples** from our benchmark evaluations on dataset suites `DS03` (single sensor faults) and `DS04` (concurrent multi-sensor faults).

---

## 1. Executive Summary: The Evaluation Scorecard

Before diving into individual metrics, here is the complete benchmark scorecard comparing **Step 1 (Static Baseline)** against **Step 2 (Dual-Head Multi-Task Network)** on the `DS03` test set (15,540 flight windows across 554 engines):

| Operational Goal | Metric Name | Step 1 (Baseline Static) | Step 2 (Dual-Head Adaptive) | Aerospace Target | What the Result Means in Practice |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Virtual Sensing** | **Physical MAPE** | 0.21% | **0.18%** | $< 0.50\%$ | Reconstructed sensor signals are accurate to within 0.18% of true physics. |
| **Fleet Safety** | **Engine Mission TPR** | 94.05% | **98.81%** | $> 95.0\%$ | **83 out of 84 failing engines** were caught during flight. |
| **Avionics Health** | **Operational Latched TPR** | 71.42% | **82.11%** | $> 80.0\%$ | Flight software protected 82.11% of all active fault cycles. |
| **Cycle Sensitivity**| **Point-by-Point TPR** | 28.64% | **34.80%** | N/A | Instantaneous per-cycle detection during the early incubation phase. |
| **Airline Economics**| **False Alarm Rate (FAR)** | 5.44% | **3.89%** | $< 1.0\%$ | 28.5% fewer false alarms thanks to Two-Factor Authentication. |
| **Fault Pinpointing**| **Isolation Accuracy** | 85.49% | **92.86%** | $> 90.0\%$ | When an alarm sounds, the system identifies the exact broken sensor 93% of the time. |
| **Detection Speed** | **Average Latency** | 16.0 cycles | **20.4 cycles** | $< 25$ cycles | Caught slow drifts within ~20 flights, before mechanical damage occurred. |
| **Multi-Sensor** | **Hamming Loss** | N/A | **0.0361** | $< 0.05$ | Across all 14 sensors, diagnostic switches were correct 96.4% of the time. |

---

## 2. Signal Reconstruction Metrics (Virtual Sensing Accuracy)

### What is Virtual Sensing?
A **Virtual Sensor** (or Digital Twin) is an on-wing software algorithm that calculates what each of the 14 engine sensors *should* read under true laws of thermodynamics, even if the physical sensor has failed or is noisy.

### Metric 1: Mean Absolute Percentage Error (MAPE)
MAPE measures how far off the virtual sensor's predictions are from the true physical signal, expressed as a clean percentage.

* **In Plain English**: If a sensor's true temperature is 691.0 Kelvin and the virtual sensor predicts 692.0 Kelvin, the prediction is off by 1.0 Kelvin, which is an error of about 0.14%.

```math
\text{MAPE}_i = \frac{100\%}{N} \sum_{t=1}^N \left| \frac{\hat{y}_{t, i} - y_{t, i}^*}{y_{t, i}^*} \right|
```

* **Formula Breakdown**:
  * `N`: Total number of flight cycles evaluated.
  * `ŷ_{t, i}` ("y-hat"): The virtual sensor's predicted reading for sensor channel `i` at cycle `t`.
  * `y_{t, i}^*` ("y-star"): The true, clean physical measurement.
  * `|...|`: Absolute value (treats positive and negative errors equally).

* **Concrete Example from Step 2 Benchmark**:
  On the `DS03` test set, our model achieved an overall physical MAPE of **0.18%**.
  * **High-Pressure Spool Speed (`NH`)**: True = 9,850 RPM | Predicted = 9,857 RPM $\to$ **0.075% MAPE**
  * **Fan Spool Speed (`NL`)**: True = 2,420 RPM | Predicted = 2,422 RPM $\to$ **0.084% MAPE**
  * **Inlet Temperature (`T2`)**: True = 288.1 K | Predicted = 288.4 K $\to$ **0.107% MAPE**

### Metric 2: Root Mean Squared Error (RMSE)
RMSE measures the average prediction error in **real physical units** (Kelvin, Pascals, RPM).

```math
\text{RMSE}_i = \sqrt{\frac{1}{N} \sum_{t=1}^N (\hat{y}_{t, i} - y_{t, i}^*)^2}
```

* **Why Square the Errors?**: Squaring penalizes large blunders much more heavily than small errors. An error of 4 degrees counts 16 times worse than an error of 1 degree.
* **Step 2 Result**: Scaled RMSE was **0.0795**, meaning typical prediction error is less than 0.08 standard deviations.

---

## 3. The Three Ways to Score Fault Detection (TPR / Recall)

One of the most important concepts in condition monitoring is understanding how **True Positive Rate (TPR)** (also called **Recall**) is defined. In our benchmark, TPR can be viewed from three distinct operational perspectives:

```text
+-----------------------------------------------------------------------------------+
| THREE EVALUATION PERSPECTIVES FOR FAULT DETECTION                                 |
+-----------------------------------------------------------------------------------+
| 1. Engine Mission-Level TPR: 98.81% (83 of 84 failing engines caught!)            |
|    "Did the system notify maintenance that this airplane had a failing sensor?"   |
|                                                                                   |
| 2. Operational Latched TPR: 82.11% (Flight software protection mode)              |
|    "Once the fault emerged, did the alarm stay active and protect the flight?"    |
|                                                                                   |
| 3. Instantaneous Point-by-Point TPR: 34.80% (Second-by-second test)               |
|    "Did the alarm fire on every single flight cycle, including early micro-drift?"|
+-----------------------------------------------------------------------------------+
```

---

### Perspective A: Engine Mission-Level Detection Rate (98.81%)

* **In Plain English**: This answers the primary question asked by airline fleet managers and pilots: *"If an engine suffers a failing sensor, did TurbofanGuard catch it before the flight ended?"*

```math
\text{Engine Mission TPR} = \frac{\text{Number of Faulted Engines Successfully Alarmed}}{\text{Total Number of Faulted Engines}} \times 100\%
```

* **Real-World Result**:
  * In the `DS03` test set, exactly **84 engines** suffered sensor failures.
  * TurbofanGuard triggered confirmed alarms on **83 out of the 84 engines**.
  * **Score = 98.81% Success Rate**.

---

### Perspective B: Operational Latched Alarm TPR (82.11%)

* **In Plain English**: In real aircraft flight computers (such as Full Authority Digital Engine Control, or FADEC), alarms are **latched with persistence**.
  * When an alarm confirms that Sensor 13 (Compressor Exit Temperature `T030`) has broken, the flight computer **latches the warning switch ON for the rest of the flight**.
  * The flight computer does not turn the warning off just because background noise momentarily dipped downward for one cycle.
  * Instead, it switches to the virtual sensor (`ŷ_t`) permanently until ground maintenance replaces the broken thermocouple.

```math
\text{Latched Operational TPR} = \frac{\text{Flight Cycles Protected After Alarm Inception}}{\text{Total Active Fault Flight Cycles}} \times 100\%
```

* **Real-World Result**:
  * Out of 10,707 active fault flight cycles, **8,791 cycles** were fully protected under the latched alarm.
  * **Score = 82.11% Operational TPR**.
  * The remaining ~17.9% of cycles occurred during the initial incubation period before the drift physically emerged from the noise floor.

---

### Perspective C: Instantaneous Point-by-Point TPR (34.80%)

* **In Plain English**: This is a strict, memory-less mathematical test that checks every single flight cycle in isolation: *"Did the instantaneous sensor reading breach the alarm line at this exact second?"*

```math
\text{Point-by-Point TPR} = \frac{\text{Total Individual Cycles with an Alarm Firing}}{\text{Total Flight Cycles Labeled as Fault-On}} \times 100\%
```

#### Why is Point-by-Point TPR 34.80%? (The Slow Pinhole Leak Analogy)

To understand why this number is 34.80%, consider a **slow pinhole puncture in a car tire**:

```text
Tire Pressure over 60 minutes:
32.0 PSI ──> 31.95 PSI ──> 31.90 PSI ──────> 28.0 PSI ───────────> 15.0 PSI (Flat)
[Minute 0]   [Minute 5]    [Minute 10]        [Minute 25]           [Minute 60]
│                                            │
└── Pinhole puncture starts here!             └── Car dashboard beeps: "Low Pressure!"
    (Error is 0.05 PSI - hidden by road bumps)     (Leak is now distinct from bumps)
```

1. **Minutes 1 to 20 (The Incipient Phase)**:
   * The tire is technically leaking, but the pressure has only dropped by `0.05 PSI`.
   * Normal road bumps and temperature fluctuations cause pressure changes of `0.5 PSI` (10 times larger than the leak!).
   * No sensor in the world can tell if a `0.05 PSI` drop is a puncture or just cold road asphalt.
2. **Minute 25 (The Detection Point)**:
   * The tire pressure drops below `28.0 PSI`. The car dashboard alerts: **"Low Tire Pressure!"**
   * The driver safely exits the highway. **The system worked perfectly!**
3. **Why the Math Test Gives a 35% Grade**:
   * A strict robot grades every single minute:
     * Minute 1: Leaking? Yes. Alarm? No. $\to$ **FAIL**
     * Minute 2: Leaking? Yes. Alarm? No. $\to$ **FAIL**
     * ...
     * Minute 20: Leaking? Yes. Alarm? No. $\to$ **FAIL**
   * In dataset `DS03`, sensor faults are **slow linear drifts**:
     ```math
     f_t[i] = k_i \cdot (t - t_{\text{start}})
     ```
   * During the first 15 to 20 flights after a drift starts, the drift error is $+0.05\text{ K}$, while background sensor noise is $\pm 1.5\text{ K}$.
   * The AI prudently waits until the drift grows larger than the noise floor (average latency: 20.4 flights) before sounding the alarm.
   * But because the dataset labels all those early incubation cycles as `fault_on = 1`, they are counted as "missed detections" in the formula, deflating the score to 34.80%.

---

## 4. False Alarm Rate (FAR - Airline Economics)

### What is a False Alarm?
A **False Alarm** occurs when the detection algorithm sounds an alarm on an engine where **all sensors are 100% healthy**.

### Why are False Alarms Dangerous in Aviation?
In commercial airline operations, false alarms are catastrophic:
* A false sensor alarm can force a pilot to perform an emergency air turn-back or divert to an unscheduled airport.
* It can cause ground crews to delay a flight, pull the aircraft out of service, and replace perfectly good multi-thousand-dollar components.
* For this reason, the aerospace industry standard mandates that **FAR must remain strictly below 1.0%**.

```math
\text{FAR} = \frac{\text{Total False Alarms Triggered on Healthy Flight Cycles}}{\text{Total Healthy Flight Cycles Evaluated}} \times 100\%
```

* **Step 1 Baseline Static Threshold**:
  * Had a static alarm tripwire set at $3.5\sigma$.
  * **Result**: **5.44% FAR** (263 false alarms on 4,833 healthy cycles). Random electrical noise spikes momentarily breached the 3.5 line, triggering false alarms.
* **Step 2 Dual-Head Adaptive Threshold**:
  * Uses **Two-Factor Authentication**:
    1. Did the reading violate physics? (Residual check)
    2. Does the AI diagnostic radar recognize a multi-flight drift pattern? (AI probability check)
  * When a single electrical spike occurs on a healthy flight, the AI probability remains near 0.0, keeping the threshold high at $4.5\sigma$.
  * **Result**: **3.89% FAR** on `DS03` and **3.62% FAR** on `DS04` — a **28.5% reduction in false alarms**.

---

## 5. Fault Isolation Accuracy (Pinpointing the Broken Part)

### What is Fault Isolation?
Once an alarm confirms that *something* is broken on the engine, **Fault Isolation** determines *which specific sensor* is malfunctioning out of the 14 instruments.

* **In Plain English**: If sensor 13 (Compressor Exit Temperature `T030`) has a broken thermocouple, does the algorithm correctly flag sensor 13, or does it accidentally blame sensor 2 (`NL` fan speed)?

```math
\text{Isolation Accuracy} = \frac{\text{Correctly Identified Broken Sensor Channels}}{\text{Total Active Alarms Triggered}} \times 100\%
```

* **Step 1 Baseline Result**: 85.49% isolation accuracy.
* **Step 2 Dual-Head Result**: **92.86% isolation accuracy** (3,460 out of 3,726 detected fault cycles correctly isolated the exact failing channel).

---

## 6. Multi-Sensor Diagnostics (DS04 Stress Test)

In dataset suite `DS04`, the system is stress-tested on **concurrent multi-sensor failures** (where 2 or 3 sensors break simultaneously during the same flight).

### Metric 1: Multi-Label Hamming Loss
Hamming Loss measures the fraction of individual sensor decisions that were wrong across all 14 channels.

```math
\text{Hamming Loss} = \frac{1}{14 \cdot N} \sum_{t=1}^N \sum_{i=1}^{14} \mathbb{I}(\hat{m}_{t, i} \ne m_{t, i})
```

* **In Plain English**: Think of the 14 sensors as 14 light switches (0 = healthy, 1 = broken).
  * If sensor 3 is broken and sensor 7 is broken, the true pattern is 12 zeros and 2 ones.
  * If the model predicts sensor 3 is broken and sensor 8 is broken, it got 12 healthy switches right, 1 broken switch right, but made 2 mistakes (missed sensor 7, falsely blamed sensor 8).
  * Its score is $2 / 14 = 0.14$.
* **Step 2 Result**:
  * `DS03` (single faults): Hamming loss = **0.0361** (96.4% per-sensor accuracy).
  * `DS04` (concurrent multi-faults): Hamming loss = **0.0803** (92.0% per-sensor accuracy).

### Metric 2: Exact Match Ratio (Subset Accuracy)
The percentage of flight cycles where all 14 sensor switches were diagnosed **100% perfectly with zero mistakes**.

```math
\text{Exact Match Ratio} = \frac{1}{N} \sum_{t=1}^N \mathbb{I}(\hat{\mathbf{m}}_t = \mathbf{m}_t) \times 100\%
```

* **Step 2 Result**: **51.36%** on `DS03` and **30.15%** on `DS04`.

---

## 7. Detection Latency (Speed of Warning)

### What is Detection Latency?
Detection Latency ($\Delta t_{\text{det}}$) measures how many flight cycles elapse between the moment a sensor begins failing ($t_{\text{start}}$) and the moment the flight computer confirms the alarm ($t_{\text{alarm}}$).

```math
\Delta t_{\text{det}} = t_{\text{alarm}} - t_{\text{start}}
```

* **In Plain English**: If an oil pressure sensor begins drifting at Flight 50, and the computer confirms the alarm at Flight 70, the detection latency is:
  $$\Delta t_{\text{det}} = 70 - 50 = 20\text{ flights}$$
* **Why Latency Matters**:
  * In commercial aviation, engines fly 200-cycle inspection intervals.
  * Detecting a subtle drift in ~20 flights gives maintenance engineers **over 100 flights of advance notice** to schedule a replacement during routine overnight hangar stops, completely avoiding emergency flight cancellations.
* **Step 2 Result**: Average detection latency is **20.4 flight cycles** on `DS03` and **18.7 flight cycles** on `DS04`.
