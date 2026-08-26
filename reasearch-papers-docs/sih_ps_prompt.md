I am beginning Phase 1 of an SIH project titled:

“Hybrid Quantum Machine Learning Platform for Early Disease Detection.”

I already have an existing platform called Quantum Helix Lab:

https://quantum-helix-lab.vercel.app/

The existing platform focuses on quantum genomic analysis. It includes capabilities such as:

- NCBI genomic-data integration
- DNA sequence retrieval and upload
- Quantum sequence search using Grover/QGSA-style circuits
- Mutation detection using quantum circuits
- FRQI-based experimental representations
- Simulator and real-quantum-hardware execution
- Noise simulation
- Qubit and hardware-resource estimation
- Report generation

These capabilities use quantum circuits, but they are not yet a complete Quantum Machine Learning pipeline because they do not train a model from labelled biomedical samples.

For the new problem statement, I eventually want to extend Quantum Helix Lab into a hybrid platform containing two connected layers:

1. Deterministic quantum genomic analysis for identifying known mutations and sequence patterns.
2. Predictive QML models that learn disease-related patterns from labelled biomedical datasets.

Do not redesign or integrate the complete web platform during this phase. Phase 1 is exclusively for:

- Researching the current state of QML in disease detection
- Understanding accepted experimental standards
- Reproducing trusted QML examples
- Running fair experiments on standard datasets
- Comparing QML models with classical baselines
- Identifying which model and dataset should be used for the MVP

## Primary objective

Create a reproducible Phase 1 QML research workspace that answers these questions:

1. How do quantum-kernel methods work in practice?
2. How do variational quantum classifiers work in practice?
3. How do these methods perform on small standard biomedical datasets?
4. How sensitive are they to feature count, circuit design, dataset size and noise?
5. How do they compare with strong classical models under identical conditions?
6. Which QML method is realistic enough to use in our SIH MVP?
7. Which claims can we defend before judges?
8. Which claims, especially “quantum advantage,” are not supported by our evidence?

This phase must produce both a literature-review report and runnable experiments. Do not produce only theoretical notes.

## Starting reference

Begin with this PennyLane tutorial:

https://pennylane.ai/demos/tutorial_kernel_based_training

Read it carefully and document:

- How the quantum feature map encodes input data
- How the quantum kernel is calculated from state overlap
- How the quantum Gram matrix is constructed
- How the quantum kernel is supplied to scikit-learn’s SVC
- Why the tutorial uses one qubit per input feature
- How kernel-based training differs from variational training
- Why parameter-shift differentiation requires many circuit executions
- How the number of circuit evaluations scales for kernel and variational methods
- Under what conditions the tutorial suggests quantum kernels may be more practical

Important: the PennyLane example uses a small binary subset of Iris because model performance is not its focus. Do not treat its perfect accuracy as evidence of quantum advantage.

Also inspect its preprocessing carefully. In our experiments, avoid data leakage by fitting scalers, PCA and feature selectors only on training data inside each cross-validation fold.

## Literature-review task

Find and review approximately 12–15 high-quality sources.

Prioritize:

1. Peer-reviewed research papers
2. Systematic reviews
3. Papers that compare QML with classical ML
4. Papers using biomedical or genomic datasets
5. Papers testing noise or real quantum hardware
6. Official PennyLane and Qiskit documentation
7. Papers discussing limitations, scalability, trainability or quantum advantage

Avoid relying on Medium posts, marketing pages or low-quality articles when a primary source is available.

Include at least these starting references:

- PennyLane kernel-based training tutorial:
  https://pennylane.ai/demos/tutorial_kernel_based_training

- Qiskit quantum-kernel tutorial:
  https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html

- Qiskit VQC documentation:
  https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html

- Systematic review of QML for digital health:
  https://www.nature.com/articles/s41746-025-01597-z

- Benchmarking quantum and classical ML on oncological data:
  https://arxiv.org/html/2608.11373v1

- MedMNIST benchmarking on real quantum hardware:
  https://arxiv.org/pdf/2502.13056

Find additional foundational or relevant papers on:

- Quantum feature maps
- Quantum kernels
- QSVC
- Variational quantum classifiers
- Data encoding
- Barren plateaus
- Kernel concentration
- QML generalization
- Medical or genomic QML
- Fair quantum-versus-classical benchmarking
- Noise effects on QML

For every source, record the following in a literature matrix:

| Field               | Required information                                          |
| ------------------- | ------------------------------------------------------------- |
| Citation            | Title, authors, year, venue and link                          |
| Research question   | What problem was studied?                                     |
| Dataset             | Name, size, features and target                               |
| Medical task        | Screening, diagnosis, prognosis or variant classification     |
| Preprocessing       | Scaling, PCA, feature selection or image reduction            |
| QML model           | Quantum kernel, QSVC, VQC, QNN, QCNN, etc.                    |
| Encoding            | Angle, amplitude, basis, IQP, ZZ or another encoding          |
| Circuit             | Qubits, layers, depth and entanglement pattern                |
| Backend             | Ideal simulator, noisy simulator or real hardware             |
| Classical baselines | Models used for comparison                                    |
| Validation          | Holdout, cross-validation, repeated CV or external validation |
| Metrics             | Accuracy, sensitivity, specificity, F1, AUROC, etc.           |
| Main result         | What did the authors find?                                    |
| Claimed advantage   | Accuracy, efficiency, generalization or parameter count       |
| Evidence quality    | Strong, moderate or weak, with justification                  |
| Limitations         | Small data, leakage, weak baseline, simulator-only, etc.      |
| Reproducibility     | Whether code and data are available                           |

Separate the following clearly:

- What the authors claim
- What the experiments actually demonstrate
- Our interpretation
- Whether quantum advantage is established

Do not describe a model as having quantum advantage merely because it achieves slightly higher accuracy on one split.

## Dataset plan

Run experiments in increasing order of difficulty.

### Dataset A: Binary Iris sanity check

Purpose:

- Reproduce and understand the PennyLane tutorial
- Verify the quantum-kernel implementation
- Verify the variational-classifier implementation
- Count circuit executions
- Confirm that the research environment works

Use only the first two Iris classes, matching the tutorial.

Do not use Iris as evidence for medical usefulness.

### Dataset B: Wisconsin Diagnostic Breast Cancer

Use scikit-learn’s `load_breast_cancer` or the official UCI dataset.

Properties:

- 569 samples
- 30 numerical features
- Binary benign/malignant target

Purpose:

- Primary biomedical benchmark
- Study feature reduction
- Compare quantum and classical models
- Measure sensitivity and specificity

Document that this is a diagnostic classification dataset based on features from breast-mass samples. It is not longitudinal or pre-symptomatic early-detection data.

### Dataset C: UCI Heart Disease

Use the official UCI Heart Disease dataset or a clearly documented equivalent.

Purpose:

- Test whether the pipeline generalizes to another biomedical dataset
- Handle mixed numerical and categorical features
- Study performance on a smaller dataset
- Compare feature-selection strategies

Convert the target into presence versus absence of heart disease only if this is clearly documented.

### Optional Dataset D: Parkinson’s voice measurements

Only implement this after the first three datasets work.

If repeated recordings belong to the same person, use patient-grouped splitting. Never allow recordings from the same patient to appear in both training and testing.

### Do not begin MedMNIST or large genomics experiments yet

Medical images and high-dimensional gene-expression datasets will be considered after the small tabular benchmark is stable. Do not spend Phase 1 resources building a large CNN or full genomic pipeline.

## Models to implement

### Classical baselines

Implement at least:

1. Logistic Regression
2. Linear SVM
3. RBF-kernel SVM
4. Random Forest

XGBoost may be added if it is already available and does not complicate the environment.

### Quantum Model 1: Quantum Kernel SVM

Implement a quantum kernel using PennyLane.

Start with:

- `AngleEmbedding`
- One qubit per selected feature
- State-overlap or adjoint-based kernel evaluation
- scikit-learn `SVC` with a precomputed or callable quantum kernel

Check that:

- `K(x, x)` is approximately 1
- The kernel matrix is symmetric
- The kernel matrix is numerically positive semidefinite or any numerical violation is reported
- Train and test kernel matrices have correct dimensions
- Circuit executions are counted

After the basic version works, test one additional encoding such as:

- IQPEmbedding, or
- A data re-uploading feature map

Do not add several feature maps before establishing a correct baseline.

### Quantum Model 2: Variational Quantum Classifier

Implement a VQC with:

- Angle embedding
- One or two shallow trainable layers initially
- A hardware-compatible entangling pattern
- Binary output from a Pauli-Z expectation or probability
- A clearly defined loss function
- Adam, COBYLA or another documented optimizer
- Training-loss history
- Fixed random seeds

Start with a shallow model. Increase the depth only through a controlled experiment.

Track:

- Number of trainable parameters
- Training iterations
- Circuit executions
- Training time
- Convergence behaviour
- Validation performance
- Variance across random initializations

## Feature configurations

Quantum experiments must not use all 30 WDBC features directly at first.

Evaluate matched feature counts:

- 2 features
- 4 features
- 6 features
- 8 features

Use at least two reduction approaches:

1. PCA
2. Mutual-information or another interpretable feature selector

All preprocessing must happen inside the training fold.

For every quantum experiment, run a matched classical experiment using exactly the same:

- Samples
- Train/test folds
- Selected features
- Scaling
- Random seed

Also run a full-feature classical baseline. This shows how much performance is lost because of quantum-compatible feature reduction.

## Experimental protocol

Use a reproducible evaluation strategy.

Preferred default:

- Stratified 5-fold cross-validation
- Seeds: 42, 123 and 2026
- Report mean and standard deviation
- Use exactly the same folds for classical and quantum models

Because quantum kernels scale approximately quadratically with training-set size, start progressively with:

- 50 samples
- 100 samples
- 200 samples
- Full dataset if runtime remains reasonable

Use stratified sampling for these subsets.

If full repeated cross-validation is too expensive for a QML model:

- Do not silently reduce the experiment
- Record the limitation
- Use a smaller, fixed stratified subset
- Keep the classical comparison on that same subset
- Explain why the reduced experiment was necessary

Never compare a quantum model on 100 samples with a classical result trained on a different 500-sample split.

## Required metrics

For every biomedical experiment, calculate:

- Accuracy
- Balanced accuracy
- Precision
- Recall/sensitivity
- Specificity
- F1 score
- Matthews correlation coefficient
- AUROC
- AUPRC
- Confusion matrix
- Training time
- Prediction time

Where possible, calculate bootstrap or cross-validation confidence intervals.

For QML models, additionally report:

- Number of qubits
- Circuit depth
- Total gate count
- Two-qubit gate count
- Number of trainable parameters
- Number of shots, if finite-shot mode is used
- Number of quantum-circuit executions
- Simulator/backend name

For screening-oriented interpretation, explain why false negatives and sensitivity matter. Do not optimize only for accuracy.

## Noise experiment

After the ideal-simulator experiments work, run one controlled noise study.

Test at least:

- Ideal simulation
- Finite-shot simulation
- A simple depolarizing-noise configuration

Use the same trained setup or clearly explain if retraining is performed.

Record how noise changes:

- Kernel values
- Kernel-matrix quality
- Accuracy
- Sensitivity
- Specificity
- AUROC

Do not attempt real-hardware training in Phase 1. Prepare the code so that a small frozen inference experiment can later run on IBM hardware.

## Fairness and scientific-integrity rules

Follow these rules strictly:

- Do not claim quantum advantage from one result.
- Do not cherry-pick the best seed.
- Do not tune on the test set.
- Do not fit PCA, feature selection, imputation or scaling before splitting.
- Do not compare models using different data without clearly labelling the difference.
- Do not report only accuracy.
- Do not call WDBC “pre-symptomatic detection.”
- Do not hide failed or non-convergent QML runs.
- Do not interpret simulator execution time as a real-QPU speedup.
- Do not confuse fewer model parameters with lower total computation.
- Clearly distinguish ideal simulator, noisy simulator and real hardware.
- Preserve raw results so that every graph can be regenerated.
- Use public, non-identifiable datasets only.

If a paper makes a strong quantum-advantage claim, examine whether it used:

- A strong classical baseline
- Identical preprocessing
- Adequate sample size
- Multiple seeds
- Cross-validation
- Noise simulation
- Real hardware
- Statistical uncertainty

## Repository and implementation requirements

Before making changes:

1. Inspect the repository structure.
2. Read any existing `AGENTS.md`, README and dependency files.
3. Identify the current framework and Python environment.
4. Preserve all existing Quantum Helix Lab functionality.
5. Do not overwrite unrelated files.

Create an isolated research area following existing repository conventions. If no relevant convention exists, use:

```text
qml-research/
├── README.md
├── requirements.txt or pyproject.toml
├── configs/
├── data/
│   └── README.md
├── notebooks/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── classical/
│   ├── quantum/
│   ├── evaluation/
│   └── visualization/
├── experiments/
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
├── research/
│   ├── literature-matrix.csv
│   ├── phase1-literature-review.md
│   └── research-gaps.md
└── tests/
```

Prefer reusable Python scripts and modules over placing all logic in notebooks. Notebooks may be used for explanation and visualization, but core experiments must be runnable from scripts.

Use configuration files for:

- Dataset
- Feature-selection method
- Feature count
- Model
- Random seed
- Number of folds
- Circuit layers
- Shots
- Backend
- Noise settings

Save results in CSV or JSON. Do not hardcode reported results into the report.

## Required visualizations

Generate:

1. Classical-versus-quantum metric comparison
2. Confusion matrices
3. ROC curves
4. Precision-recall curves
5. VQC training-loss curve
6. Quantum kernel Gram-matrix heatmap
7. Performance versus number of selected features
8. Performance versus dataset size
9. Circuit evaluations versus dataset size
10. Ideal-versus-noisy performance
11. Circuit diagram for each QML architecture

Each plot must include:

- Dataset
- Model
- Seed or aggregation method
- Feature count
- Backend
- Clear axis labels

## Required written deliverables

### 1. `phase1-literature-review.md`

Include:

- QML concepts required for this project
- Summary of quantum kernels
- Summary of VQCs
- Biomedical QML landscape
- Evidence for and against quantum utility
- Hardware and noise limitations
- Data-encoding bottleneck
- Research gaps
- Implications for our SIH platform
- Proper citations and working links

### 2. `literature-matrix.csv`

One row per source using the fields defined above.

### 3. `experiment-protocol.md`

Document:

- Research questions
- Hypotheses
- Datasets
- Preprocessing
- Models
- Hyperparameters
- Validation procedure
- Metrics
- Runtime limitations
- Rules for determining whether one model is meaningfully better

### 4. `experiment-results.md`

Include:

- Complete result tables
- Mean and standard deviation
- Runtime and circuit-resource results
- Failed experiments
- Interpretation
- Whether differences appear meaningful
- Threats to validity

### 5. `qml-mvp-recommendation.md`

Conclude with:

- Recommended primary disease dataset
- Recommended QML model
- Recommended feature map
- Recommended feature count
- Recommended classical baselines
- Recommended simulator
- Proposed real-hardware experiment
- Features to integrate into Quantum Helix Lab
- Features to postpone
- Claims we can safely make
- Claims we must avoid
- Exact next steps for Phase 2

## Testing requirements

Add automated tests for at least:

- Dataset loading
- Leakage-safe preprocessing
- Reproducible splitting
- Kernel self-similarity
- Kernel symmetry
- Kernel-matrix dimensions
- Model prediction shape
- Metric calculations
- Sensitivity and specificity calculations
- Result serialization

Run the relevant tests, formatting and linting before considering the phase complete.

## Execution approach

First inspect the repository and produce a concise implementation plan.

Ask me a question only if a missing decision would materially change the experiment. Otherwise, make a reasonable assumption, document it and continue.

Then execute the work in checkpoints:

### Checkpoint 1

- Repository inspection
- Environment setup
- Literature-search strategy
- Dataset availability verification

### Checkpoint 2

- Binary Iris reproduction
- Quantum-kernel validation
- VQC validation
- Classical sanity baselines

### Checkpoint 3

- WDBC benchmark
- Fair matched-feature comparisons
- Required metrics and plots

### Checkpoint 4

- Heart Disease benchmark
- Noise experiment
- Scaling analysis

### Checkpoint 5

- Literature report
- Result interpretation
- MVP recommendation
- Tests and reproducibility review

After each checkpoint, summarize:

- What was completed
- What was learned
- Commands used
- Files created
- Important results
- Remaining risks

## Completion criteria

Phase 1 is complete only when:

- The literature matrix contains approximately 12–15 credible sources.
- The PennyLane kernel tutorial has been reproduced or carefully adapted.
- Quantum kernel and VQC implementations run successfully.
- At least Iris, WDBC and Heart Disease experiments are completed.
- At least four classical baselines are evaluated.
- Quantum and classical models use matched data and preprocessing.
- Sensitivity and specificity are reported.
- Circuit resources and execution counts are recorded.
- At least one noise experiment is completed.
- Results can be regenerated from saved configurations.
- Automated tests pass.
- The final report identifies a defensible QML configuration for Phase 2.
- The report explicitly states whether quantum advantage was or was not demonstrated.
- No unsupported clinical or quantum-advantage claims remain.

The final goal of this phase is not to force QML to win. The goal is to determine honestly which QML approach is technically feasible, scientifically defensible and impressive enough to integrate into Quantum Helix Lab for the SIH prototype.

# Standalone QML Research Phase 1

## Summary

Create C:\Users\seera\Desktop\Exploration\Rit_Grover_project\qml-research as an independent, reproducible Python research
repository that can later be imported or merged into Quantum Helix Lab.

- Initialize local Git metadata without touching the unusable parent repository, then run git checkout -b qml-research-phase1
  exactly as requested.

- Keep quantum-helix-lab-full unchanged, including its existing unrelated working-tree edits.
- Use Python 3.12 in an isolated .venv; no Hugging Face services, web-platform integration, real-QPU jobs, MedMNIST,
  genomics, or Parkinson’s work in Phase 1.

- Record the branch, implementation decisions, checkpoints, commands, results, and verification in specs/qml-research-
  phase1.md.

## Implementation Changes

### Repository and reproducibility

- Scaffold the prompt’s configs/, data/, notebooks/, src/qml_research/, experiments/, results/, research/, and tests/
  structure.

- Use pyproject.toml plus a pinned requirements lock covering PennyLane, Lightning, NumPy, pandas, SciPy, scikit-learn,
  matplotlib, seaborn, PyYAML, UCI dataset access, pytest, and Ruff.

- Track compact raw JSON/CSV results, configurations, provenance manifests, representative Gram matrices, and generated
  figures. Ignore downloaded datasets, virtual environments, caches, and large transient checkpoints.

- Add resumable, configuration-hashed runs so an interrupted experiment continues without recomputing completed folds.

### Data and leakage-safe evaluation

- Implement typed dataset loaders returning features, disease-positive labels, feature metadata, provenance, and optional
  group IDs:
  - Binary Iris: first two classes, used only for implementation sanity.
  - WDBC: remap malignant to positive class 1; explicitly describe it as diagnostic rather than pre-symptomatic data.
  - Official UCI Cleveland Heart Disease: 303 records, 13 predictors, target num > 0, with source URL, license, retrieval
    timestamp, and checksum recorded. UCI Heart Disease (https://archive.ics.uci.edu/dataset/45/heart%2Bdisease)

- Store reusable stratified fold indices for seeds 42, 123, and 2026. Use five folds and identical indices for matched
  classical and quantum runs.

- Fit imputation, categorical encoding, scaling, PCA, mutual-information selection, and final angle scaling exclusively
  within each training fold.

- Evaluate 2, 4, 6, and 8 selected features. Matched classical runs consume the exact transformed data used by QML; separate
  full-feature classical runs quantify feature-reduction loss.

- Use disease-focused metrics: accuracy, balanced accuracy, precision, sensitivity, specificity, F1, MCC, AUROC, AUPRC,
  confusion matrix, prediction time, and training time. Report fold mean, standard deviation, and paired model deltas;
  provide bootstrap confidence intervals for the final selected configurations.

### Models and experiment matrix

- Classical fixed baselines:
  - Logistic regression: C=1, balanced classes, liblinear, 2,000 maximum iterations.
  - Linear SVM: C=1, balanced classes.
  - RBF SVM: C=1, gamma="scale", balanced classes.
  - Random forest: 500 trees, max_features="sqrt", balanced classes, fixed seed.
  - Do not add XGBoost in Phase 1.

- Quantum-kernel SVM:
  - Primary feature map: AngleEmbedding(rotation="Y"), one qubit per selected feature.
  - Secondary controlled comparison: one-repeat IQPEmbedding.
  - Compute state-overlap kernels with SVC(kernel="precomputed"), reuse symmetric train-kernel entries, and count actual
    circuit executions.

  - Validate diagonal self-similarity, symmetry, dimensions, minimum eigenvalue, and negative eigenvalue count. Never
    silently repair an indefinite noisy kernel.

- VQC:
  - Angle embedding, Pauli-Z binary output, ring-entangled StronglyEntanglingLayers, one- and two-layer variants.
  - Adam with learning rate 0.05, 100 steps, batch size up to 16, binary cross-entropy, parameter-shift differentiation,
    and three initialization seeds.

  - Record loss history, parameters, convergence, predictions, runtime, circuit executions, and failed/non-convergent runs.

- Experiment sequence:
  1. Reproduce the PennyLane Iris tutorial, including its published behavior, then run a separate leakage-corrected version
     used for reporting. The tutorial’s perfect Iris result will be labelled a sanity result, not evidence of usefulness or
     advantage. PennyLane kernel tutorial (https://pennylane.ai/demos/tutorial_kernel_based_training)

  2. Screen WDBC on a stratified 100-sample subset using seed 42 across feature count, PCA/mutual information, Angle/IQP
     kernels, and one-/two-layer VQCs.

  3. Select configurations by balanced accuracy, then sensitivity, then lower circuit/runtime cost. Clearly label this as
     exploratory selection.

  4. Confirm the selected QKSVM and VQC alongside all classical baselines at 50, 100, 200, and full-data sizes for WDBC and
     Heart Disease using all three seeds.

  5. Run full classical data unconditionally. Run full-data QML only when timing extrapolated from the 200-sample run is at
     most eight hours per model family; otherwise retain the matched 200-sample comparison and record the skipped run
     explicitly.

- Controlled noise study:
  - Use one fixed 100-sample WDBC fold and the selected quantum kernel.
  - Compare analytic ideal simulation, 1,024-shot simulation, and 1,024-shot simulation with a documented 1% single-qubit
    depolarizing channel.

  - Retrain the kernel SVM for each kernel condition and report changes in kernel quality, balanced accuracy, sensitivity,
    specificity, and AUROC.

- Record qubits, circuit depth, gates, two-qubit gates, trainable parameters, shots, backend, and executions. Generate
  circuit diagrams from the actual implemented circuits.

## Interfaces and Deliverables

- Provide CLI commands equivalent to:
  - python -m qml_research data verify
  - python -m qml_research run --config <yaml>
  - python -m qml_research plot --results <path>
  - python -m qml_research report --results <path>

- Define YAML configuration fields for dataset, sample size, folds, seed, preprocessing, feature count, model, feature map,
  layers, optimizer, shots, backend, and noise.

- Define stable serialized records containing run/config hashes, dataset provenance, fold identity, preprocessing, model
  settings, metrics, timings, quantum resources, status, warnings, and failure details. These records will be suitable for
  later Quantum Helix Lab ingestion without importing experimental internals.

- Produce all requested plots from serialized results, with dataset, model, seed/aggregation, feature count, and backend
  embedded in labels.

- Create:
  - research/literature-matrix.csv
  - research/phase1-literature-review.md
  - research/research-gaps.md
  - research/experiment-protocol.md
  - research/experiment-results.md
  - research/qml-mvp-recommendation.md

- Use a 15-source corpus: the six requested sources plus Schuld–Killoran feature Hilbert spaces, Havlíček et al., McClean
  barren plateaus, Schuld–Sweke–Meyer data encoding, Huang et al. power of data, Caro et al. generalization, Thanasilp et al.
  kernel concentration, Bowles et al. fair benchmarking, and the 2026 Cleveland heart-disease HQNN study. Separate author
  claims, demonstrated evidence, project interpretation, reproducibility, and whether advantage is established. Relevant
  foundational evidence includes feature Hilbert spaces
  (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504), barren plateaus
  (https://www.nature.com/articles/s41467-018-07090-4), kernel concentration
  (https://www.nature.com/articles/s41467-024-49287-w), and fair QML benchmarking (https://arxiv.org/abs/2403.07059).

- Unit tests will cover dataset loading/provenance, disease-label mappings, reproducible folds, training-only preprocessing,
  kernel self-similarity/symmetry/PSD reporting/dimensions, prediction shapes, metric formulas, sensitivity/specificity,
  circuit telemetry, configuration hashing, and result serialization.

- Add fast smoke tests for Iris QKSVM, Iris VQC, one WDBC fold, plotting, and report regeneration. Mark full research
  experiments as slow rather than running them in routine unit tests.

- Final verification:
  - ruff format --check .
  - ruff check .
  - pytest -m "not slow"
  - configured checkpoint runs and report regeneration from a clean results directory
  - before/after status check proving quantum-helix-lab-full was untouched

- After each of the five prompt checkpoints, update the specification with completed work, commands, files, observations,
  results, and risks.

- The final recommendation must name the dataset, QML model, feature map/count, classical baselines, simulator, frozen future
  hardware experiment, Phase 2 integration boundary, and postponed work. It must explicitly state that quantum advantage was
  not demonstrated unless consistent matched-fold evidence and uncertainty estimates genuinely support that conclusion.

- Default assumptions: local repository only with no remote or push; public non-identifiable data only; no clinical claims;
  no real-hardware execution; optional Parkinson’s work remains postponed until all required datasets, reports, and tests
  pass.
