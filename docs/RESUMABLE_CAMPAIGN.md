# Resumable, full-data laptop campaign

This campaign supersedes the earlier laptop smoke/training queue for sustained
training. Entry point and restart instructions: [RESUME_TRAINING.md](../RESUME_TRAINING.md).
It was configured on 10 September 2026 for this 23.64 GiB Windows laptop.

## Scope and honest interpretation

All available rows participate in evaluation: WDBC 569, Cleveland 303, Kaggle
Framingham 4,238, Pima 768, and BreastMNIST 780. Full-data evaluation means fitting
on the training partition and retaining validation/test partitions, not training
on held-out labels. Tabular datasets use three seeds (42, 123, 2026), five outer
stratified group folds, and one inner validation partition per outer fold. Identical
feature rows stay together. Approximate proportions are 64% train, 16% validation,
20% test. Every tabular row receives an outer-test prediction once per seed.
BreastMNIST retains its official 546/78/156 partitions and repeats initialization
seeds; these are not three independent test cohorts.

Imputation, missing indicators, scaling, one-hot encoding, PCA and projected-feature
scaling are fitted only on training data. Labels are disease-positive, including
reversing BreastMNIST's normal/benign-positive original convention. Hyperparameters
are selected by validation AUROC; ties retain the first candidate. Classification
thresholds maximize validation Youden J. Test predictions are generated only after
the family's fixed search finishes. Neural checkpoints use validation loss for
early stopping and best-epoch selection, then validation AUROC across trials.

## Fixed tuning budget

Each kernel-stage fold has 280 candidates across eight families:

| Family | Candidates | Search |
| --- | ---: | --- |
| Logistic regression | 16 | Regularization, class weighting |
| Full-feature RBF SVM | 48 | C and bandwidth |
| Random forest | 12 | Depth, leaf size, feature fraction; 200 trees |
| Histogram boosting | 12 | Learning rate, leaves, L2; up to 200 iterations |
| PCA-matched RBF SVM | 48 | 2/4/6/8 components, C, bandwidth |
| Angle fidelity SVM | 48 | Components, input scale, C |
| IQP fidelity SVM | 48 | 4/6/8 qubits, scale, repetitions, C |
| Projected IQP RBF SVM | 48 | 4/6/8 qubits, scale, RBF bandwidth, C |

Each neural stage compares a classical MLP, hybrid quantum MLP, entangled
data-reuploading VQC, and separable VQC ablation. Each family has eight candidates:
4/8 reduced features, 1/2 layers, learning rates 0.003/0.01, batch size 32, up to
100 epochs, patience 15, Adam and gradient clipping. The separable ablation is
classified as a classical comparator because it is efficiently simulable.
BreastMNIST additionally has eight hierarchical QCNN candidates (up to 60 epochs)
and four native-image classical CNN candidates (up to 100 epochs).

The full plan comprises 63 dataset/seed/fold evaluations, 17,640 kernel-stage fits
and 2,052 neural fits. Early stopping is intentional convergence control, not a
claim that every fit reaches its maximum epoch. Total runtime is not yet established;
the largest neural searches may require multiple sessions. The supervisor first
finishes kernel pilots, then a WDBC neural pilot, then all kernel and neural repeats.
Per-job time limits are 4, 4, 12 and 24 hours; a timeout stops the queue and retains
checkpoints. Explicit resume continues the remaining work with a fresh time budget.

## Persistence and resource controls

The neural trainer atomically saves `last.pt` after **every mini-batch and epoch**:
model, Adam state, all CPU RNG states, batch permutation/cursor, epoch, best weights,
patience counter, cumulative active time and loss history. `best.pt` stores the
validation-selected weights. Checkpoint identities reject changed data/configuration
or architecture. Keep the frozen environment in `results/laptop/environment-lock.txt`.
Only load the locally produced checkpoints; they use Python object serialization.

Quantum state caches are memory-mapped and checkpointed every 128 input rows.
Completed validation trials and selected classical estimators persist individually.
An interrupted scikit-learn fit restarts that one candidate; it has no mid-fit
optimizer checkpoint. The previous completed candidates remain saved. Atomic file
replacement retries short Windows file-sharing conflicts without deleting the
previous artifact. Matrix allocations have a 512 MiB per-matrix limit and quantum
state simulation has a ten-qubit limit (configured search uses at most eight).

The guard monitors the worker's entire process tree every 0.25 seconds, counting
both Windows Python redirector and real interpreter. It uses below-normal process
priority, four CPU threads, no GPU, 8 GiB maximum sampled RSS/private memory and
6 GiB minimum free RAM, with an extra 1 GiB required at startup. It stops only its
worker tree. These are sampled safeguards, not a hard operating-system allocation
quota. No other person's processes are terminated. Checkpoints overwrite latest
state instead of accumulating every epoch, but all trial-best states are retained.

Initial WDBC kernel tuning completed in 60.45 seconds, peak process-tree RSS
0.372 GiB, minimum free RAM 12.12 GiB. Framingham's first attempt retained all five
classical-family results before a Windows file-sharing error during a quantum cache
progress update; retry support was then added. Consult current logs for later runs.

## Results and evidence rule

[QML_MODELS_WITH_BETTER_METRICS.md](../research/QML_MODELS_WITH_BETTER_METRICS.md)
is regenerated after each completed family. It lists positive differences as
exploratory, retains all classical and quantum results, and shows incomplete counts.
The comparator is the classical family selected by validation AUROC on that split,
including neural/ablation baselines when available. Its identity may change as
previously pending classical families finish; this is disclosed through completion
counts and saved paired records.

The stronger signal rule requires all 15 tabular evaluations, mean balanced-accuracy
gain at least 0.02, sensitivity loss no more than 0.02, and an approximate corrected
repeated-CV interval above zero after Bonferroni correction for 26 planned quantum
family/dataset comparisons. The correction is conservative but does not establish
external validity. BreastMNIST's shared test set cannot meet the 15-fold rule.
Angle kernels have a closed-form classical equivalent. No result is a demonstration
of computational quantum advantage. Grids must not be expanded in response to test
scores; new hypotheses need a new campaign and independent confirmation.

## Manual commands

Prefer the supervisor for the complete queue. For a bounded diagnostic run while
no supervisor is active:

```powershell
.venv/Scripts/python.exe scripts/laptop_guard.py --name framingham-resume --threads 4 --max-gib 8 --reserve-gib 6 --minutes 120 -- .venv/Scripts/python.exe scripts/run_campaign.py --stage kernels --datasets framingham_kaggle --seeds 42 --fold-limit 1
.venv/Scripts/python.exe scripts/run_campaign.py --stage report
.venv/Scripts/python.exe scripts/run_campaign.py --stage audit
```

Exact split indices, fitted preprocessing, validation scores, warnings, selected
specifications and held-out predictions live below
`results/campaign/biomedical_qml_tuning_v1/<dataset>/seed-<seed>-fold-<fold>/`.
They are local ignored artifacts; preserve them for resumption and independent review.
