# QCNN architecture decision

Status: accepted for functionality-first implementation
Date: 2026-08-26

## Selected pipeline

```text
28x28 grayscale ultrasound image
  -> deterministic adaptive average pooling (2x2)
  -> row-major four-feature vector
  -> scale [0, 1] to [-pi, pi]
  -> four RY angle encodings
  -> shared local SU(4) convolution
  -> three-parameter unitary pooling, 4 -> 2
  -> shared local SU(4) convolution
  -> three-parameter unitary pooling, 2 -> 1
  -> final Pauli-Z expectation
  -> trainable affine logit
  -> malignant probability
```

Eight-qubit runs use a 2x4 pooled grid and one additional convolution/pooling stage.

## Exact trainable blocks

Each stage owns one shared 15-parameter convolution tensor. Within a stage, the same tensor is applied first to disjoint adjacent pairs and then to shifted/ring-neighbour pairs. The two-qubit unitary is:

1. `U3` on each wire (six parameters);
2. `IsingXX`, `IsingYY`, and `IsingZZ` (three parameters);
3. another `U3` on each wire (six parameters).

Each source/sink pooling pair shares three parameters and applies `CRZ`, an X-conjugated `CRX`, and `CRY`, all controlled by the source and targeting the sink. The source wire is retired after the block. Pooling is unitary; it does not claim physical qubit reset or measurement-based nonlinearity.

The four-qubit model therefore has `2 * (15 + 3) = 36` quantum parameters plus affine scale/bias. The eight-qubit model has `3 * (15 + 3) = 54` quantum parameters plus scale/bias.

## Optimization defaults

- Backend: `lightning.qubit`
- Shots: analytic (`null`/`None`)
- Differentiation: adjoint with Torch; portable backprop fallback only when the configured backend requires it
- Optimizer: Adam
- QCNN learning rate: `0.01`
- Weight decay: `0.0001`
- Batch size: `16`
- Parameter initialization: normal standard deviation `0.05`
- Loss: training-fold positive-weighted BCE with logits
- Checkpoint: minimum validation BCE
- Tuning rank: mean validation AUROC across the configured seeds

The shallow, locally shared design limits expressivity and simulator cost deliberately. Gradient norm and variance are recorded every epoch to expose flat or unstable optimization.

## Comparisons and ablations

- Spatial pooling is primary because it preserves a simple image-local ordering.
- Train-fitted flatten-plus-PCA is an ablation and must never see validation/test data during fitting.
- WDBC uses fold-local standardization/PCA; its PCA component order is not treated as literal image locality.
- Matched classical baselines receive the exact QCNN feature matrix and indices.
- Native CNN and ImageNet models receive original images, so their comparison answers practical image-classification performance rather than controlled representation performance.

## Claims boundary

An improvement over reduced-feature baselines is at most evidence about this encoding/model at this small scale. It is not a computational quantum advantage. Beating a single classical seed, using fewer parameters, or running with finite shots does not establish advantage. The software is not clinically validated.
