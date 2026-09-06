# TurbofanGuard: Neural Backbone Architecture

This document explains the deep learning architecture of the **Shared Neural Backbone** (`TurbofanBackbone`), the mathematical principles behind its design, and why each layer was chosen.

---

## 1. What is the Shared Backbone? (The Engine Brain)

In modern multi-task deep learning, instead of training several independent models that each try to understand the engine from scratch, we build a **single shared backbone** (an "Engine Brain").

```text
Telemetry Input (B, 16, 18)
          │
          ▼
TurbofanBackbone (57,664 params — 'The Engine Brain')
• Multi-scale 1D Convolutions (k=3, k=5)
• Dual Temporal Pooling (Endpoint + Mean)
• Residual Dense Layers + LayerNorm
          │
          ▼
Shared Latent State z_t ∈ ℝ⁶⁴ (Pure, compressed 64D engine state)
     ┌────┴──────────────────────────┐
     ▼                               ▼
Head 1: Virtual Sensor          Head 2: Diagnostic AI
(Reconstructs clean sensors)    (Classifies broken sensors)
```

* **What it does**: The backbone takes the 18 telemetry features (4 flight conditions + 14 raw sensor channels), filters out measurement noise, understands how the engine has changed over recent flights, and distills everything into a compact **64-dimensional latent state** ($\mathbf{z}_t$).
* **Why it is shared**: Both the Virtual Sensor (reconstructing clean signals) and the Diagnostic AI (detecting broken sensors) rely on the same fundamental question: *"What is the true physical state of this jet engine right now?"* By sharing the backbone, both tasks help each other learn faster and produce more consistent answers.

---

## 2. Parameter Efficiency: Built for Real Aircraft Avionics

In machine learning research, it is tempting to use gigantic models with tens of millions of parameters. However, in real aviation:
* Aircraft avionics computers have strict limits on electrical power, heat dissipation, and memory.
* Algorithms must make decisions in milliseconds during flight.

`TurbofanBackbone` was specifically engineered to be **lightweight, fast, and mathematically efficient**:
* **Total Parameters**: Exactly **57,664 parameters** (less than $0.25 \text{ MB}$ of memory).
* **Speed**: Runs in fractions of a millisecond per flight cycle on standard CPUs and onboard embedded GPUs.
* **Tested**: Verified with full gradient backpropagation on both standard CPU and Apple Silicon GPU (`MPS`).

---

## 3. Multi-Scale 1D Temporal Convolutions

When inspecting sensor telemetry over a sequence of flight cycles (for example, a window of the last $W = 16$ flights), a single sensor reading does not tell the whole story.

```text
Flight Cycle:    1    2    3    4    5    6    7    8   ...   16
Telemetry:     [691, 692, 690, 715, 691, 693, 695, 698, ... 710]
                             ▲                       ▲
                        Fast Spike              Slow Drift
                       (Single Cycle)         (Gradual Climb)
```

Sensor anomalies come in two very different flavors:
1. **High-Frequency Spikes**: An electrical surge causes a sensor to jump for just 1 cycle and then immediately return to normal.
2. **Low-Frequency Drifts**: A damaged sensor slowly creeps upward by 1 degree every cycle over 30 flights.

To capture both patterns simultaneously, `TurbofanBackbone` uses **Multi-Scale 1D Temporal Convolutions** with two parallel branches:

```text
Input Window (B, 18, W)
     ├──> Branch 1: Fast Kernel (k=3, 3-cycle receptive field) ──> Detects rapid spikes
     └──> Branch 2: Slow Kernel (k=5, 5-cycle receptive field) ──> Detects multi-cycle drifts
               │
               ▼
     Concatenate Features (64 Channels)
               │
               ▼
     Stage 2 Convolution (k=3, 64 Channels) + GroupNorm + GELU
```

### The Mathematics of 1D Temporal Convolutions
A 1D convolution slides a small learnable filter across the timeline of flight cycles:

```math
\mathbf{h}[t] = \sum_{k=-K}^{K} \mathbf{W}[k] \cdot \mathbf{x}[t + k] + \mathbf{b}
```

* **In Plain English**: To understand what is happening at cycle $t$, the convolution looks at a small window of neighboring cycles ($t-K$ to $t+K$), multiplies them by learned weights, and adds a bias.
* **Variables**:
  * $\mathbf{h}[t]$: Filtered feature representation at flight cycle $t$.
  * $\mathbf{x}[t+k]$: Raw telemetry at neighboring flight cycle $t+k$.
  * $\mathbf{W}[k]$: Learnable convolutional weight matrix.
  * $\mathbf{b}$: Learnable bias vector.
  * $K$: Half-width of the filter (determines the kernel size).

* **Branch 1 (Kernel Size 3)**: Has a compact receptive field of 3 cycles. It acts like a fast-reacting high-pass filter, catching instantaneous steps and single-cycle spikes.
* **Branch 2 (Kernel Size 5)**: Has a broader receptive field of 5 cycles. It acts like a trend detector, recognizing subtle multi-cycle slopes.
* **Stage 2 (GroupNorm + GELU)**: Merges both perspectives into 64 high-level temporal channels.

---

## 4. Temporal Pooling: Combining Instantaneous State with Historical Baseline

Once the convolutions have scanned across the window of flight cycles, how should the model summarize what it learned?

A common mistake is using only a global average across the window. But if you only average the window, you blur out what is happening on the very latest flight!

`TurbofanBackbone` solves this with a **Dual Temporal Aggregation strategy**:

```text
Temporal Feature Map (B, 64, W)
     ├──> 1. Endpoint State (Latest Flight): c2[:, :, -1] (64 channels)  ──> "What is engine doing now?"
     └──> 2. Temporal Mean Pool:            c2.mean(dim=-1) (64 channels) ──> "What was baseline average?"
               │
               ▼
     Concatenate Features: (64 + 64 = 128 Channels)
```

```math
\mathbf{f}_{\text{temporal}} = \begin{bmatrix} \mathbf{c}_{2}[:, :, -1] \\ \frac{1}{W} \sum_{w=1}^W \mathbf{c}_{2}[:, :, w] \end{bmatrix} \in \mathbb{R}^{128}
```

* **In Plain English**: We take the exact feature slice from the very last flight cycle in the window (the instantaneous state) and glue it to the average of all cycles across the window (the historical baseline).
* **Variables**:
  * $\mathbf{f}_{\text{temporal}}$: Combined 128-dimensional temporal feature vector.
  * $\mathbf{c}_{2}[:, :, -1]$: Feature vector of the final (most recent) flight cycle in the window.
  * $\frac{1}{W} \sum_{w=1}^W \mathbf{c}_{2}[:, :, w]$: Mean feature vector averaged across all $W$ flight cycles.
  * $W$: Sequence window length (default: 16 cycles).

* **Why this is powerful**: To detect a subtle drift, an engineer asks: *"How much does today's reading differ from the engine's average over the last two weeks?"* This dual-pooling design gives the neural network both pieces of information side-by-side.

---

## 5. Snapshot Fallback Pathway: Supporting Single-Cycle Inference

While processing a sequence window ($W = 16$) is ideal, there are situations where a user or ground system only has a **single flight snapshot** available ($W = 1$, shape `[batch_size, 18]`).

If a model only supported 3D temporal windows, it would crash when handed a single snapshot.

`TurbofanBackbone` includes an **automatic snapshot fallback pathway**:
* When an input tensor has 3 dimensions `(batch_size, window_length, 18)`, it passes through the multi-scale temporal convolutions.
* When an input tensor has 2 dimensions `(batch_size, 18)`, it automatically routes through `self.snapshot_proj`:

```python
self.snapshot_proj = nn.Sequential(
    nn.Linear(18, 128),
    nn.LayerNorm(128),
    nn.GELU(),
)
```

This ensures that the model can be used seamlessly for both historical sequence analysis and instantaneous single-snapshot health checks without code changes or crashes.

---

## 6. Thermodynamic Cross-Feature Block (Dense Residual Layers)

After temporal aggregation, the model has 128 intermediate features. Now, it must model the **cross-sensor physics** (how temperature relates to pressure, and how fuel flow relates to spool speed).

This is performed by two dense residual blocks:

```math
\mathbf{h}_1 = \text{Dropout}\left( \text{GELU}\left( \text{LayerNorm}(\mathbf{W}_1 \mathbf{f} + \mathbf{b}_1) \right) \right) + \mathbf{W}_{\text{skip}, 1} \mathbf{f}
```

```math
\mathbf{h}_2 = \text{Dropout}\left( \text{GELU}\left( \text{LayerNorm}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \right) \right) + \mathbf{h}_1
```

* **In Plain English**: Each block applies a dense matrix transformation, standardizes the activations with Layer Normalization, applies a smooth non-linear curve (GELU), and adds a **skip connection** (adding the input directly to the output).
* **Variables**:
  * $\mathbf{f}$: Input feature vector entering the block.
  * $\mathbf{h}_1, \mathbf{h}_2$: Intermediate hidden representations.
  * $\mathbf{W}_1, \mathbf{W}_2$: Trainable weight matrices.
  * $\mathbf{b}_1, \mathbf{b}_2$: Trainable bias vectors.
  * $\mathbf{W}_{\text{skip}, 1}$: Linear projection matching dimensions for the first skip connection.
  * $\text{LayerNorm}$: Normalizes activations across features, preventing training instability.
  * $\text{GELU}$: Gaussian Error Linear Unit, a smooth activation function superior to standard ReLU.
  * $\text{Dropout}$: Randomly zeros 10% of activations during training to prevent overfitting.

### Why Use Residual Skip Connections?
In traditional feed-forward networks, information can get degraded or lost as it passes through multiple layers. Skip connections provide an "information highway" that allows raw physical signals to bypass layers directly, preventing vanishing gradients and ensuring stable, rapid training.

---

## 7. The Shared Latent State ($\mathbf{z}_t \in \mathbb{R}^{64}$)

The final layer of the backbone is a linear projection and Layer Normalization that outputs the shared latent state:

```math
\mathbf{z}_t = \text{LayerNorm}(\mathbf{W}_{\text{latent}} \mathbf{h}_2 + \mathbf{b}_{\text{latent}}) \in \mathbb{R}^{64}
```

* **In Plain English**: The backbone takes all the complex temporal dynamics and cross-sensor relationships and projects them into a compact vector of 64 clean numbers.
* **Variables**:
  * $\mathbf{z}_t$: The 64-dimensional latent state vector for flight cycle $t$.
  * $\mathbf{W}_{\text{latent}}$: Projection matrix mapping from hidden dimension (64) to latent dimension (64).
  * $\mathbf{b}_{\text{latent}}$: Bias vector.

### Why 64 Dimensions? (The Information Funnel)
In jet engine thermodynamics at cruise, the engine's behavior is dictated by approximately **4 to 6 primary degrees of freedom** (altitude, speed, ambient temperature, throttle setting, and component wear).
* Compressing 18 sensor channels over 16 flight cycles ($18 \times 16 = 288$ raw numbers) down to 64 clean latent dimensions acts as an **information bottleneck**.
* High-frequency electrical noise cannot easily squeeze through this bottleneck, leaving only the pure, true physical state of the engine.

---

## 8. Hyperparameter Configuration

The backbone is fully configured via `src/utils/config.py` and `configs/backbone_config.json`:

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

Every parameter is automatically validated when loaded, guaranteeing that invalid dimensions or unsupported activation functions are caught immediately before training begins.
