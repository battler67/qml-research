I want you to design and implement a new, self-contained research experiment studying Quantum Convolutional Neural Networks (QCNNs) for breast-cancer detection. If scientifically and computationally appropriate, also investigate a Quantum Recurrent Neural Network (QRNN). Do not implement a Quantum Vision Transformer (QViT).

This must be a separate experiment from my existing QSVM/VQC experiments. You may reuse stable preprocessing, evaluation, logging and plotting utilities, but do not break or silently modify existing experiments.

## Mandatory research phase

Before writing or modifying implementation code:

1. Inspect the entire repository structure, its instructions, existing experiments, dependencies and coding conventions.
2. Locate and carefully read every relevant document in the `research-papers-docs/` folder.
3. Extract the following from each relevant paper:
   - Research question
   - Dataset and number of samples
   - Input representation
   - Number of qubits
   - Data-encoding method
   - QCNN/QRNN architecture
   - Quantum convolution and pooling operations
   - Circuit depth and trainable parameter count
   - Optimizer, loss, batch size and epochs
   - Simulator or quantum hardware
   - Noise model and number of shots
   - Classical baselines
   - Evaluation methodology
   - Reported results
   - Limitations

4. Research additional authoritative sources if something important is missing. Prioritize original papers, official MedMNIST documentation, official dataset pages, and official PennyLane/Qiskit documentation. Do not rely on unverified blogs for technical conclusions.
5. Cite every external source with a working link.
6. Create a research summary and architecture decision document before implementation.
7. Do not blindly reproduce a paper’s architecture. Evaluate whether it is appropriate for our datasets, computing resources and experimental goals.

Do not claim quantum advantage unless the experiments genuinely support it. Distinguish clearly between:

- Quantum advantage
- Competitive quantum performance
- Parameter efficiency
- Proof-of-concept feasibility
- Results caused by small sample sizes or preprocessing

## Experimental tracks

Design the project around the following tracks.

### Track A: QCNN on tabular breast-cancer data

Use the Wisconsin Diagnostic Breast Cancer dataset, preferably the official UCI or scikit-learn version.

Build a leakage-free pipeline containing:

1. Data loading and validation
2. Missing-value analysis
3. Duplicate detection
4. Class-distribution analysis
5. Stratified train/validation/test splitting
6. Feature scaling fitted only on training data
7. PCA or another justified dimensionality-reduction technique fitted only on training data
8. Quantum feature encoding
9. QCNN training
10. Classical baseline training
11. Evaluation and artifact generation

Start with a computationally realistic configuration such as 4 PCA features and 4 qubits. Investigate an 8-feature/8-qubit configuration only if it is practical.

Design a genuine QCNN architecture with repeated quantum convolution and pooling stages, for example:

- 8 qubits → 4 qubits → 2 qubits → 1 measured qubit
- Or 4 qubits → 2 qubits → 1 measured qubit

Do not simply rename a generic variational circuit as a QCNN. Explain why each two-qubit convolution block, pooling operation, entangling pattern and measurement is used.

### Track B: QCNN on MedMNIST breast imaging

Investigate whether MedMNIST, particularly BreastMNIST or the most relevant officially available breast-cancer subset, can be used responsibly for this experiment.

First verify and document:

- Dataset name and medical task
- Image modality
- Original image dimensions
- Number of classes
- Official train, validation and test sizes
- Class distributions
- Licensing and citation requirements
- Whether the dataset is suitable for breast-cancer classification

Preserve the official MedMNIST splits unless there is a strong documented reason not to.

A full image cannot be carelessly placed into a small quantum circuit. Compare feasible input-reduction strategies such as:

- Resizing or pooling
- Patch-level summary features
- PCA
- Classical feature extraction followed by a quantum classifier
- Amplitude encoding, only if state preparation and interpretability are discussed honestly
- Angle encoding with a small, justified number of features
- Data re-uploading

Any dimensionality-reduction or learned feature-extraction stage must be fitted only on the training split. Avoid train/test leakage.

Implement at least one realistic QCNN imaging pipeline if the dataset can be loaded and the architecture is computationally feasible. If downloading MedMNIST is temporarily impossible, still implement the loader and configuration cleanly, document the blocker, and ensure the tabular QCNN experiment remains runnable.

### Track C: QRNN feasibility experiment

Investigate QRNNs, but do not force a QRNN onto static images without a defensible sequential representation.

A QRNN may be implemented only if you can justify a meaningful sequence, such as:

- A sequence of image patches
- Ordered feature groups
- A medically or spatially meaningful sequence derived from the input

Do not flatten an image into an arbitrary sequence merely to call the model recurrent.

If QRNN implementation is justified:

- Keep it as an optional, separate experiment.
- Document the recurrent state, encoding method, number of qubits, sequence length, parameter sharing and measurement strategy.
- Compare it with an appropriate classical RNN or small recurrent baseline.
- Keep the default configuration runnable on 8 GB RAM.

If it is not scientifically justified or computationally practical, document the reasons and provide a concrete future implementation plan instead of adding misleading code.

## Classical baselines

Use fair classical baselines appropriate to each dataset.

For WDBC, include at least:

- Logistic regression
- Linear SVM
- RBF-SVM
- Random forest
- A small classical MLP

For MedMNIST, include at least:

- A simple logistic/MLP baseline using the same reduced features as the QCNN
- A small CNN trained on the original images, if computationally feasible

Clearly separate two types of comparison:

1. A controlled comparison using exactly the same reduced input features
2. A practical classical image baseline using the original image representation

Do not present the second comparison as perfectly capacity-matched.

## Hardware-aware architecture design

My current laptop has 8 GB system RAM. Design the default configuration so it can run locally without exhausting memory.

Explicitly analyze:

- Statevector memory scaling as \(O(2^q)\)
- Maximum practical number of qubits
- Circuit depth
- Number of trainable parameters
- Gradient-computation cost
- Parameter-shift versus simulator backpropagation or adjoint differentiation
- Full-batch versus mini-batch training
- Number of circuit evaluations per epoch
- Shot-based versus analytic simulation
- Expected effects of shot noise
- Effects of realistic quantum noise
- Runtime scaling with samples, qubits, layers, epochs and folds
- Barren plateaus
- Trainability
- Hardware connectivity and transpilation
- Two-qubit gate count
- NISQ limitations
- Feasibility of eventual real-hardware execution

Use exact analytic statevector simulation by default where appropriate. Shot-based and noisy simulations should be optional configurations, not the default development mode.

Provide estimated resource usage before launching expensive runs. Add safeguards that warn or require explicit confirmation when the requested configuration is likely to be excessive.

## Configurable experiment sizes

The code must work on both my 8 GB laptop and another machine with more RAM.

Do not hard-code sample counts, split sizes, qubit counts or training hyperparameters. Provide a central YAML/TOML configuration and CLI overrides for at least:

- Dataset
- Random seed
- Training sample limit
- Validation sample limit
- Test sample limit
- Number of samples per class
- Stratified-sampling toggle
- Number of PCA/reduced features
- Number of qubits
- Circuit layers/stages
- Batch size
- Epochs
- Learning rate
- Optimizer
- Number of shots
- Simulator/backend
- Gradient method
- Noise-model toggle
- Number of cross-validation folds
- Number of repeated seeds
- Number of data-loader workers
- Output directory
- Checkpoint frequency
- Early-stopping patience

Provide predefined profiles such as:

- `smoke`: very small run for validating the pipeline
- `laptop_8gb`: safe default for my laptop
- `full_dataset`: uses all available samples
- `high_memory`: larger samples/qubits for a stronger machine
- `noisy_simulation`: shot-based/noisy evaluation
- `real_hardware_ready`: conservative circuit configuration

Sample limits must preserve class balance whenever possible. Log both the requested and actual numbers of samples.

## Reproducibility and experiment tracking

Every run must record:

- Timestamp and experiment ID
- Git commit if available
- Complete configuration
- Random seeds
- Package versions
- Python version
- Operating system
- CPU, RAM and GPU information
- Dataset version
- Dataset split indices or reproducible split seed
- Number of raw and retained samples
- Training, validation and test counts
- Per-class counts for every split
- Number of original and reduced features
- Qubit count
- Circuit depth
- Trainable parameter count
- Gate counts, especially two-qubit gates
- Number of shots
- Gradient method
- Training duration
- Inference duration
- Peak memory if measurable
- Best epoch and stopping reason
- Final metrics

Cache safely reusable preprocessing outputs and quantum computations where appropriate. Cache keys must incorporate the dataset, split, preprocessing, circuit and relevant configuration so incompatible results are never reused.

## Metrics and statistical evaluation

For binary breast-cancer detection, report at least:

- Accuracy
- Balanced accuracy
- Precision
- Recall/sensitivity
- Specificity
- F1 score
- ROC-AUC
- PR-AUC or average precision
- Matthews correlation coefficient
- Confusion matrix
- False positives
- False negatives

Because false negatives are important in disease detection, highlight sensitivity and the false-negative count. If prediction scores are available, allow threshold analysis instead of relying only on the default 0.5 threshold.

Where runtime permits, use repeated stratified evaluation and report:

- Mean
- Standard deviation
- Per-run values
- Confidence intervals where justified

For MedMNIST, use the official evaluation conventions in addition to the above metrics.

## Required visual outputs

Generate and save publication-quality figures for each applicable model:

- Dataset class-distribution chart
- Example MedMNIST images with labels
- PCA explained-variance plot
- Training and validation loss curves
- Training and validation metric curves
- ROC curves
- Precision–recall curves
- Confusion matrices
- Per-model metric comparison chart
- Runtime comparison
- Peak-memory comparison if available
- Circuit diagram
- QCNN architecture diagram
- Gate-count/resource summary
- Prediction-confidence distribution
- Sample-size versus performance plot when multiple sizes are tested
- Qubit count or circuit depth versus runtime/performance when tested

Use readable labels, legends, units and high-resolution output. Save both individual run plots and summary plots.

## Documentation requirements

Create clear documentation containing:

1. Research-paper review
2. Dataset descriptions and citations
3. Preprocessing pipeline
4. Data-leakage prevention measures
5. QCNN architecture
6. Optional QRNN design and justification
7. Mathematical explanation of encoding, convolution, pooling, measurement and loss
8. Circuit diagrams
9. Hardware and simulator limitations
10. Complexity and resource analysis
11. Configuration guide
12. Commands for every experiment profile
13. Result-table schema
14. Interpretation guidelines
15. Known limitations
16. Reproduction instructions
17. Guidance for moving the project to a higher-RAM computer
18. Guidance for future real-quantum-hardware experiments

Include a table comparing the architectures extracted from the papers with the architecture ultimately selected for this project. Explain every significant deviation.

## Suggested project separation

Use a clear experiment directory such as:

`experiments/qcnn_breast_cancer/`

A reasonable structure might include:

- `configs/`
- `data/` or dataset-loader modules
- `models/`
- `quantum/`
- `classical/`
- `training/`
- `evaluation/`
- `visualization/`
- `scripts/`
- `tests/`
- `results/`
- `docs/`

Adapt this to the existing repository conventions instead of duplicating infrastructure unnecessarily.

Do not commit downloaded datasets, huge caches, credentials, backend tokens or generated binary artifacts unless the repository explicitly expects them.

## Testing and quality requirements

Add tests for:

- Dataset loading
- Official and custom split handling
- Stratified subsampling
- No preprocessing leakage
- Configuration validation
- Circuit output shape
- Model forward pass
- One optimization step
- Metrics
- Result serialization
- Cache-key correctness
- Smoke-profile execution

Use a small smoke test in CI or local verification. Do not run the most expensive full experiment automatically.

## Execution order

Follow this order:

1. Inspect the repository and instructions.
2. Read `research-papers-docs/`.
3. Write the research summary.
4. Produce the architecture and resource plan.
5. Estimate feasibility on 8 GB RAM.
6. Inspect existing reusable pipeline code.
7. Implement configuration and data pipelines.
8. Implement classical baselines.
9. Implement the smallest QCNN.
10. Verify it using the smoke profile.
11. Add WDBC full-dataset support.
12. Add MedMNIST support.
13. Evaluate QRNN feasibility and implement it only if justified.
14. Add metrics, visualizations and result aggregation.
15. Run safe validation experiments.
16. Complete documentation and reproduction instructions.

Do not begin with a large training run. Start with unit tests and a tiny end-to-end smoke experiment.

## Final handoff

At completion, give me:

- A concise summary of what was implemented
- Architecture decisions and their research basis
- Exact files created or modified
- Exact commands for smoke, 8 GB laptop and high-memory runs
- Dataset and split statistics from completed runs
- Metrics and locations of generated plots
- Runtime and memory observations
- Tests performed and their results
- Anything that could not be completed
- Scientific and engineering limitations
- Recommended next experiments

Be careful and research-driven. Ask me only if a decision is genuinely blocked by missing information. Otherwise, make conservative, documented choices and continue. Do not implement QViT anywhere in this experiment.

## Summary and research decision

- First checkpoint the existing Phase 1 project on qml-research-phase1 after the currently passing 18 tests and a secret/large-file scan.
  Then create feature/qcnn-breast-cancer. Create specs/qcnn-breast-cancer.md before implementation and record decisions, commits,
  experiments, and verification there. No push is planned because no remote is currently configured.

- Build a self-contained experiments/qcnn_breast_cancer/ experiment without breaking the existing QSVM/VQC runner.
- Primary task: BreastMNIST malignant-versus-normal/benign classification from 28×28 grayscale breast-ultrasound images. It has 546/78/156
  official train/validation/test samples; malignant will be remapped to disease-positive label 1. The dataset is CC BY 4.0 and explicitly
  not intended for clinical use. Official MedMNIST metadata (https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py), MedMNIST v2
  paper (https://doi.org/10.1038/s41597-022-01721-8).

- Secondary control: WDBC diagnostic breast-mass classification using its 569 samples and 30 FNA-derived features. This tests the same
  QCNN under a naturally low-dimensional reduction but is not an image-screening or early-detection task. UCI WDBC
  (https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic).

- Before model code, write research/qcnn_literature_review.md with the requested per-paper extraction matrix. The review will distinguish:
  - Cong-style inverse-MERA QCNN with repeated convolution and pooling—the selected definition. Foundational QCNN paper
    (https://arxiv.org/abs/1810.03787).

  - The supplied Medical-MNIST paper, which predicts imaging modalities rather than diseases.
  - HQCNN’s classical CNN plus four-qubit VQC head. HQCNN preprint (https://arxiv.org/abs/2509.14277).
  - Digital-analog quanvolution as a non-trainable patch-feature extractor, not a hierarchical QCNN. Physical Review Research paper
    (https://doi.org/10.1103/PhysRevResearch.6.L042060).

  - The supplied QCNN/QRNN/QViT comparison as robustness and architecture evidence, while excluding QViT as requested. QNN comparison
    (https://arxiv.org/abs/2604.26110).

  - Kernel concentration as justification for shallow, local circuits, gradient monitoring, and conservative quantum claims. Nature
    Communications paper (https://doi.org/10.1038/s41467-024-49287-w).

- QRNN will receive a documented feasibility assessment only. Static ultrasound images have no natural temporal order; an arbitrary patch
  sequence would add assumptions without answering the QCNN goal.

## Architecture and implementation phases

1. Research and data foundation
   - Add pinned optional dependencies: torch==2.13.0, torchvision==0.28.0, and medmnist==3.0.2; retain PennyLane 0.45.1. Document CPU
     installation and separate official CUDA installation instructions.

   - Download BreastMNIST through the official API, verify its published checksum, preserve official splits, store provenance and
     selected-index manifests, and reject unexpected label/schema changes.

   - Provide image-input inference that accepts a PNG/JPEG, applies the saved preprocessing pipeline, and returns malignant probability,
     thresholded class, model/run identifier, and a research-only warning.

2. Functional genuine QCNN
   - Default four-qubit image representation: adaptive-average-pool each 28×28 image to an ordered 2×2 grid, flatten row-major, scale
     intensities to [-π, π], and apply one RY angle per qubit.

   - Apply two hierarchy stages: 4→2→1. Each stage uses a shared 15-parameter SU(4)-style convolution block—local U3 rotations plus
     IsingXX, IsingYY, and IsingZZ entanglers—over adjacent and shifted pairs.

   - Pool using a three-parameter controlled-rotation unitary that transfers information from source to sink; retired source wires are
     excluded from later stages. This avoids simulator-specific mid-circuit measurement while retaining hierarchical quantum pooling.

   - Measure Pauli-Z on the final active qubit and pass it through a trainable affine calibration to produce a binary logit. Use weighted
     binary cross-entropy, Adam, analytic lightning.qubit, shots=None, and adjoint differentiation.

   - Expected trainable size: 36 quantum parameters plus two output-calibration parameters for four qubits; the optional 8→4→2→1 model
     has 54 quantum plus two calibration parameters.

   - Add two controlled ablations: train-fitted PCA-to-four features for BreastMNIST and PCA-to-four features for WDBC. Eight-qubit
     spatial pooling to 2×4 is reserved for high_memory.

3. Functionality-first classical benchmarks
   - Matched reduced-feature models receive exactly the QCNN features and splits: logistic regression, linear SVM, RBF SVM, random
     forest, and a small four-input MLP.

   - Native-image benchmark: a compact three-block CNN trained from scratch on 28×28 grayscale images.
   - ImageNet benchmarks:
     - ResNet-18 with explicit IMAGENET1K_V1 weights, first as a frozen linear probe and later with layer4 plus classifier fine-tuned.
     - MobileNetV3-Small with explicit IMAGENET1K_V1 weights, likewise frozen then partially fine-tuned.
     - Randomly initialized ResNet-18 as the transfer-learning control.

   - Grayscale images are repeated to three channels and processed with each pinned weight’s official 224×224 transform. Literature
     results remain in a separate reference table; only locally reproduced runs appear in the main leaderboard. TorchVision weight API
     (https://docs.pytorch.org/vision/stable/models.html).

4. Validation, tuning, and robustness
   - Complete smoke and fixed-default benchmarks before hyperparameter search.
   - Tune only against training/validation data using deterministic seeded random search. Select checkpoints by validation BCE and rank
     configurations by mean validation AUROC across three seeds.

   - Lock the chosen configuration before evaluating the official test set.
   - Evaluate the best ideal QCNN checkpoint under finite shots and depolarizing noise; do not retrain under noise initially. Record
     circuit depth, gate counts, two-qubit gates, executions, memory, timing, gradient norms, and gradient variance using PennyLane
     resource inspection.

   - The real-hardware profile prepares a provider-neutral circuit/resource bundle only. No paid job or external hardware submission
     occurs without separate authorization.

## Public interfaces, configuration, and reproducibility

- Add qml-research qcnn commands:
  - data prepare|verify
  - research verify
  - run --profile <name> [--set key=value ...]
  - tune --profile <name>
  - evaluate --run-id <id>
  - predict --image <path> --checkpoint <path>
  - report

- Keep all hyperparameters in experiments/qcnn_breast_cancer/configs/base.yaml plus profile overlays. No sample count, split, qubit count,
  optimizer setting, backend, or output path may be hard-coded.

- Config validation covers dataset/version, split sizes, reducer, image dimensions, qubits, encoding, convolution/pooling blocks,
  optimizer, learning rates, weight decay, class weighting, batches, epochs, early stopping, seeds, shots, noise, backend, precision,
  workers, pretrained mode, cache/output paths, tuning space, and expensive-run confirmation.

- Profiles:
  - smoke: 32/16/32 BreastMNIST samples, one seed, four qubits, QCNN two epochs, image CNN one epoch, no weight download.
  - laptop_8gb: 256/all/all, seeds [17,42,73], four qubits, QCNN 30 epochs with patience 6, image models 40 epochs.
  - full_dataset: 546/78/156, seeds [7,17,42,73,101], four qubits, QCNN 60 epochs with patience 10, all reproduced baselines.
  - high_memory: full data, eight-qubit QCNN, five seeds, 100-epoch ceiling, larger classical batches, partial ImageNet fine-tuning.
  - noisy_simulation: best locked four-qubit checkpoint, shots [128,512,1024,4096], depolarizing probabilities [0,0.001,0.005,0.01].
  - real_hardware_ready: no provider by default, 1,024 shots, export/resource analysis only.

- Any run projected above 25,000 quantum forwards, two hours, or requiring eight qubits/noisy sweeps must stop after printing its estimate
  unless --confirm-expensive is supplied.

- Each run stores the resolved config, dependency and hardware metadata, Git commit/dirty state, dataset checksum, split indices, seed,
  predictions, metrics, threshold, checkpoint, training history, quantum resources, timing, peak memory, and completion status. Cache keys
  include every input affecting results, and interrupted runs resume without mixing configurations.

## Benchmark and statistical protocol

- BreastMNIST always preserves its official train/validation/test partitions. Configured sample sizes produce deterministic stratified
  subsets; all models share identical index manifests.

- WDBC uses repeated stratified five-fold outer evaluation with an inner stratified validation split fitted only from the outer training
  fold. Scaling and PCA remain fold-local.

- Primary metrics: AUROC and balanced accuracy. Also report accuracy, precision, malignant sensitivity, specificity, F1, AUPRC, MCC,
  confusion matrix, false positives, false negatives, training/prediction time, and model/resource size.

- Report both threshold 0.5 and a Youden-J threshold selected on validation only, with sensitivity-specificity and calibration plots.
- Produce mean, standard deviation, and 95% confidence intervals across seeds; perform paired stratified bootstrap comparisons against the
  best matched-feature and best image baseline, correcting multiple comparisons.

- Sample-efficiency curves use shared training sizes [32,64,128,256,546].
- “Quantum advantage” may be written only if the locked QCNN beats the relevant classical comparator on preregistered metrics and the
  paired 95% confidence interval excludes zero. Simulator execution, reduced parameter count, or one favorable seed is not evidence of
  computational advantage. Benchmarking evidence suggests classical models often remain stronger on these small simulated tasks. QML
  benchmarking study (https://arxiv.org/abs/2403.07059).

## Tests and acceptance gates

- Preserve all 18 existing Phase 1 tests.
- Add tests for dataset checksum/splits, malignant-positive remapping, train-only preprocessing, shared sample manifests, config/profile
  validation, CLI overrides, four- and eight-qubit topology/parameter counts, active-wire pooling, finite gradients, deterministic seeds,
  metrics and thresholds, cache invalidation, resume behavior, pretrained/offline handling, and expensive-run guards.

- The smoke profile must complete on CPU within the stated 8 GB target, produce a checkpoint and predictions, and regenerate circuit,
  sample/class, training, ROC/PR, confusion-matrix, threshold, resource, and comparison plots.

- Run quality gates in order: research-document completeness, import/config tests, focused QCNN tests, smoke execution, full pytest, Ruff,
  fixed-default laptop benchmark, tuning, locked full benchmark, then noise evaluation.

- Final handoff documents installation for CPU and GPU laptops, every profile command, expected runtime/resource ranges, GitHub cloning
  and cache locations, how to alter training/testing sizes and hyperparameters, reproduced-versus-literature result separation, known
  limitations, and the absence of clinical validation or proven quantum speedup.
