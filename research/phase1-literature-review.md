# Phase 1 Literature Review: QML for Disease Classification

## Review scope and evidence standard

This review asks a narrower question than whether quantum computing is promising in
general: what can a small, reproducible SIH prototype honestly learn from quantum
kernels and shallow variational classifiers on public biomedical tabular data?

The 15-source corpus in `literature-matrix.csv` combines peer-reviewed foundational
work, a systematic digital-health review, recent biomedical benchmarks, fair-
benchmarking research, and official PennyLane/Qiskit implementation documentation.
Each source is separated into author claims, demonstrated evidence, our
interpretation, and whether quantum advantage is established. A source is rated:

- **Strong** when its claims match rigorous theory or a transparent systematic or
  controlled design with appropriate limitations.
- **Moderate** when it is valuable but application-specific, small, recently
  preprinted, weakly repeated, or mainly instructional.
- **Weak for efficacy** when it is an API/tutorial or a one-split toy demonstration;
  this does not imply the source is technically poor.

## Quantum kernels

A data-encoding circuit prepares a quantum feature state

\[
|\phi(x)\rangle = U_\phi(x)|0\rangle.
\]

The fidelity kernel used here is

\[
k(x,x') = |\langle\phi(x')|\phi(x)\rangle|^2.
\]

The [PennyLane kernel tutorial](https://pennylane.ai/demos/tutorial_kernel_based_training)
computes this overlap by applying `AngleEmbedding(x)`, applying the adjoint encoding
for `x'`, and measuring the probability of returning to the all-zero state. This uses
one qubit per input feature: each selected scalar controls a single-qubit rotation.
For `M` training points, a naive Gram matrix requires `M^2` kernel calls, while a
symmetric implementation needs `M(M+1)/2` unique calls. Prediction adds up to
`M_test * M` calls. The resulting matrix is supplied to scikit-learn's SVC as a
precomputed kernel; SVC optimization remains classical.

Kernel training is convex in the SVM coefficients, but that does not make every
quantum kernel informative. [Thanasilp et al.](https://doi.org/10.1038/s41467-024-49287-w)
show that kernels can concentrate so that different inputs become nearly
indistinguishable, including through expressive embeddings and noise. This project
therefore measures diagonal error, symmetry, minimum eigenvalue and negative
eigenvalues rather than assuming that an exponentially large Hilbert space helps.

`AngleEmbedding` provides the faithful tutorial baseline but is a product-state map.
The controlled secondary map is `IQPEmbedding`, inspired by the nonlinear feature
spaces in [Havlíček et al.](https://doi.org/10.1038/s41586-019-0980-2). It is tested as
a hypothesis, not described as classically hard on our small natural datasets.

## Variational quantum classifiers

A VQC adds trainable operations and measures an observable:

\[
f_\theta(x)=\langle 0|U_\phi(x)^\dagger W(\theta)^\dagger Z_0
W(\theta)U_\phi(x)|0\rangle.
\]

Phase 1 uses Y-angle embedding, one or two shallow strongly-entangling layers, a
Pauli-Z output, binary cross-entropy, and a trainable bias. The architecture is kept
shallow because [McClean et al.](https://doi.org/10.1038/s41467-018-07090-4) show that
broad random-circuit regimes can develop exponentially vanishing gradients. Every
initialization is preserved, including a failure to lower the loss.

The parameter-shift rule estimates each partial derivative with shifted circuit
executions. For `P` trainable parameters, `B` examples per batch, `S` optimizer steps
and a two-term shift rule, the leading training-call estimate is approximately
`2 * P * B * S`, plus forward evaluations and predictions. Simulator-only
backpropagation can reduce this count but would obscure hardware-compatible cost, so
the primary VQC deliberately uses parameter shift.

Kernel and variational training solve different optimization problems. The kernel
method pays approximately quadratic cost in sample count; the parameter-shift VQC
pays heavily in parameter count, batch size and steps. The PennyLane tutorial's
practical conclusion is conditional: kernels can require fewer quantum executions
when the VQC parameter count is not much smaller than the number of training samples.
It is not a universal claim that kernels scale better.

## Data encoding and feature reduction

[Schuld, Sweke and Meyer](https://doi.org/10.1103/PhysRevA.103.032430) relate data
encoding to the Fourier frequencies a variational model can represent. Re-uploading
can enrich a function class, but expressivity is not accuracy, generalization,
trainability, or advantage. [Huang et al.](https://doi.org/10.1038/s41467-021-22539-9)
also show that classical learners can exploit training data to erase apparent
separations for many quantum models.

The input bottleneck is immediate for biomedical data. WDBC has 30 features, while
Phase 1 circuits use at most eight qubits. PCA can preserve high-variance directions
but obscures which measurements matter. Mutual information is more interpretable but
univariate and may miss interactions. Both can inflate results if fitted before a
split. Consequently, imputation, scaling, PCA/selection and final angle scaling are
fitted inside every training fold, and matched classical models receive the identical
reduced representation. Full-feature classical models quantify information discarded
for circuit compatibility.

## Biomedical QML evidence

The strongest high-level finding is conservative. The 2025
[systematic review of QML for digital health](https://doi.org/10.1038/s41746-025-01597-z)
screened 4,915 records. It found 169 eligible studies before excluding 123 for
insufficient rigor; only 16 considered realistic noisy simulation or hardware. It
found no consistent performance trend supporting empirical quantum utility.

The August 2026 [oncology benchmark](https://arxiv.org/abs/2608.11373) compares QML
and AutoML-optimized classical models on tabular, omics and spatial datasets,
including WDBC, and reports no evidence of quantum advantage. Although it is a recent
preprint, it is directly relevant: weak or unmatched classical baselines can make
small QML experiments look more favorable than they are.

[Verdone et al.](https://doi.org/10.1016/j.bspc.2025.109185) report up to 90.98%
accuracy for an autoencoder-plus-QNN on Cleveland Heart Disease. The result motivates
our cross-dataset experiment, but it does not isolate the quantum component: learned
classical compression, selected baselines, a small dataset and lack of external
validation limit the claim. Our experiment deliberately uses transparent PCA and
mutual-information alternatives and reports sensitivity, specificity, MCC, AUROC and
AUPRC in addition to accuracy.

The 2026 [MedMNIST hardware benchmark](https://doi.org/10.1038/s41598-026-35605-3)
is important for a different reason. It trains shallow models on classical simulator/
GPU infrastructure, freezes them, and runs inference on IBM hardware with device-
aware circuits and mitigation. Some tasks degrade materially under hardware noise,
and mitigation recovers only part of the loss. It is a useful Phase 2 precedent for
small frozen inference, not justification for real-hardware training or an image
pipeline in Phase 1.

## Generalization, noise and hardware

[Caro et al.](https://doi.org/10.1038/s41467-022-32550-3) bound QML generalization in
terms of trainable gates and sample size. The result explains why small parameterized
models need not require exponentially many examples, but a bound does not prove that
a QML model outperforms a classical model on classical biomedical data.

Finite shots add sampling variance. Depolarizing noise can reduce self-similarity,
distort pairwise kernel values and make the measured Gram matrix indefinite. Noise
also changes the kernel on which an SVM is trained, so the controlled Phase 1 noise
study retrains the SVM per kernel condition and labels that choice. Simulator wall
time cannot be interpreted as QPU speed, because state-vector simulation, queueing,
transpilation, shots, mitigation and data loading have different costs.

## Evidence for and against quantum utility

Evidence **for continued research** includes a mathematically coherent quantum-
feature-space framework, executable small circuits, convex use of measured kernels,
hardware demonstrations of frozen inference, and possible inductive biases worth
testing. Evidence **against a current advantage claim** is stronger for this MVP:
small reduced datasets are classically easy; input loading is not free; kernel cost
is quadratic; VQC gradients are expensive; noise degrades results; and systematic
benchmarks often favor classical baselines.

The large controlled study by
[Bowles, Ahmed and Schuld](https://arxiv.org/abs/2403.07059) tests 12 QML models on
160 generated binary datasets and finds that out-of-box classical models usually
outperform them. Removing entanglement is often neutral or beneficial. This is a
direct warning against selecting only a favorable quantum circuit or assuming
entanglement is valuable without an ablation.

## Implications for the SIH platform

Phase 1 should present QML as an auditable experimental layer connected to—but
scientifically distinct from—Quantum Helix Lab's deterministic genomic search and
mutation analysis. A defensible demo shows:

1. exactly which public dataset, label and fold were used;
2. identical preprocessing for matched classical and QML models;
3. full-feature classical context;
4. sensitivity, specificity and uncertainty, not accuracy alone;
5. qubits, depth, two-qubit gates, parameters, shots and circuit executions;
6. ideal-versus-noisy degradation and every failed run; and
7. a plain statement that no quantum advantage was demonstrated unless the complete
   evidence genuinely changes that conclusion.

The most realistic Phase 2 candidate is expected to be a small quantum-kernel or
shallow VQC demonstration with frozen preprocessing and classical comparators. The
value for SIH is transparency, hybrid-system engineering and scientific judgment—not
forcing QML to win.

