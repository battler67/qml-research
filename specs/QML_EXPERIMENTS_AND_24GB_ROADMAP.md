# QML experiments, datasets, results, and a 24 GB laptop roadmap

Prepared: 8 September 2026. Evidence checked against the current local `qml-research` checkout.

## 1. Main findings

We have run QML experiments on **binary Iris, Wisconsin Diagnostic Breast Cancer (WDBC), Cleveland Heart Disease, and BreastMNIST**. The model families tried are **quantum-kernel SVM (QKSVM), variational quantum classifier (VQC), hierarchical quantum convolutional neural network (QCNN), Pauli optimized QSVM (OQSVM), and two hybrid quantum MLP (HQMLP) variants**.

The most defensible QML candidate to investigate first is **Angle-encoded QKSVM on WDBC**: its best exploratory result matched a classical logistic regression comparator. **HQMLP has substantial training headroom**, because its heart-disease smoke run used only three epochs and 50 training rows, but it has not yet shown competitive performance. **QCNN is a reasonable image-research experiment**, although its current results are weak and its four-number image representation needs investigation.

For practical prediction, **classical SVM/logistic models on tabular data and pretrained MobileNetV3-Small on images deserve equal or greater priority**. No experiment establishes quantum advantage, clinical usefulness, or a guaranteed improvement from more training.

Here, “24 GB laptop” means **24 GB system RAM**, with CPU and dedicated GPU specifications unknown. Extra RAM can reduce memory pressure and support larger experiments; it does not by itself improve accuracy or make circuit simulation three times faster.

## 2. Datasets actually used

| Dataset | Available data and task | What we actually ran | Input features |
| --- | --- | --- | --- |
| Binary Iris | 100 examples from two of the three Iris classes; nonmedical sanity check | Phase 1 smoke and a separate PennyLane kernel reproduction | Four flower measurements; Phase 1 smoke reduced to two PCA components |
| WDBC | 569 examples, 30 measurements; malignant versus benign breast mass | Phase 1 subsets of 50/100/200; 100-example feature/encoding screen and noise study; separate tiny QCNN control | Nucleus morphology measurements from digitized fine-needle aspirate images; full classical inputs or PCA/mutual-information reduction |
| UCI Cleveland Heart Disease | 303 rows, 13 predictors; disease presence defined as `num > 0` | Phase 1 subsets of 50/100/200; later EHR/IHD smoke using a 212/45/46 train/validation/test split | Age, sex, chest pain, blood pressure, cholesterol, fasting blood sugar, ECG, exercise variables, vessel count, and thallium-test category |
| BreastMNIST | 780 ultrasound images; official partitions of 546 train / 78 validation / 156 test | Smoke subsets of 32/16/32; six model records on the same image split | 28 × 28 grayscale images; QCNN and reduced baselines used four spatial-pooling features; CNNs used image inputs |

Dataset totals are not training counts. For example, Phase 1's 100-example WDBC subset produced an 80-example training fold and a 20-example test fold. The later EHR smoke used all 303 rows to create its split, but limited quantum-model training to 50 training-only rows.

BreastMNIST labels were remapped so **malignant = 1** and **normal/benign = 0**. WDBC and Cleveland are diagnostic classification datasets; they do not establish future disease prediction. BreastMNIST is an educational benchmark with official partitions, not a clinical validation dataset. [Official MedMNIST documentation](https://medmnist.com/)

### Prepared or discussed, without completed training evidence

| Dataset/model direction | Current evidence status |
| --- | --- |
| Framingham longitudinal CHD | Configuration and endpoint/censoring preparation exist; official teaching data was unavailable. No verified cohort or model performance results. |
| MIMIC-IV | Adapter specification exists; no completed MIMIC-IV training results found. |
| Gene-expression/genomic disease QML | No completed benchmark artifacts found in the reviewed `qml-research` result roots. |
| QRNN | Feasibility review only; deferred because static BreastMNIST images lack a justified temporal sequence. |
| QViT | Excluded from the existing experiment scope; not trained. |
| Real quantum hardware | The evidence summarized here is classical simulation. Hardware-ready configuration/export is not QPU training or execution evidence. |

The neighboring Grover/FRQI/DNA-search projects are separate quantum algorithm work, not trained disease-prediction QML models, so they are not counted as QML training experiments here.

## 3. Experiment history and measured results

### A. Phase 1: tabular QKSVM and VQC

There are **121 completed model/fold records**, with no failed records in the normalized Phase 1 table. These cover five experiment groups: Iris smoke (10 records), WDBC checkpoint (30), Cleveland checkpoint (30), WDBC feature/kernel screen (48), and WDBC kernel noise study (3). These are not 121 independent datasets or repeated-seed validations: the saved checkpoint evidence uses **seed 42 and one fixed fold per configuration**.

Models and variants tried:

- **QKSVM:** a classical SVM supplied with a quantum fidelity kernel; Angle and IQP feature maps were evaluated. PCA and mutual-information reductions used 2, 4, 6, or 8 features/qubits in the WDBC screen.
- **VQC:** shallow trainable circuit; the biomedical checkpoints used four PCA features, one variational layer, Adam, learning rate 0.05, and 100 optimization steps with batch size 16. An optimization step is not a full epoch over the dataset.
- **Classical controls:** logistic regression, linear SVM, RBF SVM, and random forest, including both reduced-feature and full-feature comparisons where configured.

| Selected result | Balanced accuracy | AUROC | Interpretation |
| --- | ---: | ---: | --- |
| Iris smoke: Angle QKSVM, 2 PCA components | 0.9800 | 0.9984 | Sanity check only; classical smoke controls reached 1.000 accuracy |
| Iris smoke: VQC | 0.4800 | 0.5392 | Very short functionality run |
| WDBC screen: Angle QKSVM, 2 PCA components, subset 100 | 0.9375 | 0.9896 | Best observed QML balanced accuracy; matched logistic regression also reached 0.9375 |
| WDBC checkpoint: Angle QKSVM, 4 PCA components, subset 200 | 0.9133 | 0.9440 | Encouraging single-fold result; needs repeated validation |
| WDBC checkpoint: VQC, subset 200 | 0.4800 | 0.5760 | Weak in the tested training configuration |
| Cleveland checkpoint: Angle QKSVM, subset 200 | 0.7197 | 0.7323 | Modest performance; not directly comparable to the later EHR split |
| Cleveland checkpoint: VQC, subset 200 | 0.5051 | 0.3687 | Weak; a smaller 50-example checkpoint performed better, showing instability |

The best WDBC configuration was selected after exploring alternatives. Its 20 test predictions gave a descriptive balanced-accuracy bootstrap interval of **0.8125–1.0000**, also obtained by the matched logistic comparator. This does not establish superiority or replace repeated validation.

The separate Iris tutorial reproduction achieved **1.000 accuracy on 25 test examples** with both tutorial-order and corrected fold-local preprocessing. It is stored separately from the 121-record aggregation.

**Noise study:** WDBC, subset 100, four PCA features, Angle kernel; ideal simulation, 1,024 finite shots, and depolarizing strength 0.01 all produced balanced accuracy 0.8958 and AUROC 0.9896 on that fold. The saved kernel matrices changed. Identical scores on this small test set do not demonstrate general noise robustness.

Sources: [normalized records](../results/tables/normalized-results.csv), [summary metrics](../results/tables/summary-metrics.csv), [Phase 1 report](../research/experiment-results.md), [tutorial reproduction](../results/reproductions/pennylane_tutorial_reproduction.json).

### B. QCNN breast-ultrasound experiment

The implemented architecture is a genuine **4 → 2 → 1 hierarchy**, with trainable convolution and pooling blocks, RY input encoding, 36 quantum parameters and 38 total parameters. It uses `lightning.qubit` with analytic adjoint differentiation. An **8 → 4 → 2 → 1** version is implemented/configured, but no completed eight-qubit benchmark was found. The original QCNN paper motivates hierarchical circuits; it is not evidence that our ultrasound implementation will outperform classical models. [Cong, Choi and Lukin](https://arxiv.org/abs/1810.03787)

BreastMNIST smoke: seed 7, 32 training / 16 validation / 32 test images. QCNN received two epochs; the image networks received one epoch. The test set contained only nine malignant images.

| Model actually run | AUROC | AUPRC | Balanced accuracy | Recorded training seconds |
| --- | ---: | ---: | ---: | ---: |
| QCNN, four pooled features | 0.3478 | 0.2323 | 0.5097 | 43.77 |
| Logistic regression, same pooled features | 0.6570 | 0.5774 | 0.5072 | 0.04 |
| Reduced classical MLP | 0.6715 | 0.4331 | 0.7029 | 1.45 |
| Small native CNN | 0.7246 | 0.6339 | 0.5749 | 1.67 |
| ImageNet ResNet-18, frozen-backbone mode | 0.6329 | 0.3698 | 0.5338 | 17.68 |
| ImageNet MobileNetV3-Small, frozen-backbone mode | 0.8164 | 0.6689 | 0.5556 | 1.69 |

Times are recorded fit times on the previous machine, not download/setup/end-to-end times or predictions for the new laptop. The QCNN recorded 144 circuit executions, depth 38, and 24 two-qubit gates.

**Threshold behavior matters:** the QCNN's sensitivity was 0.889 but specificity only 0.130; it produced many false positives. MobileNet's validation-selected threshold gave sensitivity 0.111 despite a much better AUROC. Its separately recorded default-0.5 balanced accuracy was 0.756. This exposes threshold instability with only four malignant validation images; choosing whichever threshold looks best on the test set would be invalid.

A two-trial, validation-only QCNN tuning check completed. The selected trial used learning rate 0.01, batch size 16, initialization scale 0.01, and weight decay 0.0001; validation AUROC was 0.625. It explicitly recorded `test_set_evaluated: false`. This was a tuning functionality check, not demonstrated test improvement.

**WDBC QCNN control:** two folds at seed 7, each limited to 16 train / 8 validation / 16 test examples, PCA features, and one epoch. QCNN AUROCs were 0.4667 and 0.6833; corresponding logistic AUROCs were 0.9333 and 1.0000. QCNN balanced accuracies were 0.4500 and 0.5000. These are tiny controls, separate from the Phase 1 WDBC benchmark.

Ten completed QCNN-experiment model records exist: six BreastMNIST records and four WDBC-control records. Additional models such as reduced linear/RBF SVM, random forest, and randomly initialized ResNet-18 are supported/configured, but are not counted as completed BreastMNIST runs here.

Sources: [local run records](../results/qcnn_breast_cancer/runs/), [tuning record](../results/qcnn_breast_cancer/tuning/864a6690e2dad830.json), [architecture decision](../research/qcnn-architecture-decision.md).

### C. EHR/IHD: Pauli OQSVM and HQMLP

The reviewed result folder contains two completed UCI smoke executions with the same split and selected fields. They are not independent repeated-seed evidence. The documented reference is `uci-smoke-20260829T194302Z-3b55c98a85f0`.

- Cohort: 303 Cleveland rows, 139 disease-positive.
- Split: **212 train / 45 validation / 46 test**; test positives: 21.
- Training-only consensus retained **`thal`, `cp`, `thalach`, `ca`**.
- Both quantum families trained on a balanced subset of **50** training rows; classical models trained on **212**. They share selected input fields and evaluation sets, but not training sample counts. Add sample-matched controls before attributing the gap to architecture.
- Calibration and threshold choice used validation data.

| Model | AUROC | AUPRC | Balanced accuracy | Train seconds |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.909 | 0.875 | 0.845 | 0.007 |
| Linear SVM | 0.910 | 0.878 | 0.865 | 0.049 |
| RBF SVM | 0.897 | 0.896 | 0.852 | 0.004 |
| Random forest | 0.899 | 0.894 | 0.736 | 0.649 |
| Classical MLP | 0.619 | 0.663 | 0.574 | 0.054 |
| Pauli OQSVM | 0.775 | 0.739 | 0.650 | 65.011 |
| Preferred BCE HQMLP | 0.621 | 0.583 | 0.572 | 3.888 |
| Paper-loss HQMLP | 0.619 | 0.558 | 0.572 | 4.592 |

**OQSVM:** four-qubit Pauli Z/ZZ fidelity kernel, two feature-map repetitions, linear entanglement, exact PennyLane statevectors. Validation AUPRC selected SVM `C=0.1` over `1.0`. It recorded 5,825 total kernel evaluations. “Optimized” here refers to the implemented/tuned SVM setup; it is not evidence of a learned quantum feature map or computational speedup, and this is not an exact Qiskit API replication.

**Preferred HQMLP:** classical projection → four-qubit RY/RZ/CNOT circuit → classical output, BCE-with-logits loss, Adam 0.01, weight decay 0.0001, three epochs, 33 total parameters including eight quantum parameters.

**Paper-inspired HQMLP:** Rot/CNOT circuit and sigmoid-MSE loss ablation, three epochs, 37 parameters including 12 quantum parameters. The two variants change both ansatz and loss; their difference does not isolate loss alone.

The best descriptive AUPRC came from RBF SVM. OQSVM's AUPRC bootstrap interval was 0.505–0.896; the HQMLP intervals were also wide. Total reference-run time was 165.77 seconds and reported process RSS peaked around 461 MB. These small circuits were not using anything close to 24 GB RAM.

Sources: [verified smoke report](../experiments/ehr_ihd_qml/docs/smoke_result.md), [local reference metrics](../results/ehr_ihd_qml/uci-smoke-20260829T194302Z-3b55c98a85f0/metrics/model_metrics.csv), [model implementation](../src/qml_research/ehr_ihd/models.py).

## 4. Which models should we prioritize on 24 GB?

The ranking below is a research judgment based on the local evidence, not a prediction of achieved future accuracy.

| Priority | Model/dataset | Why spend time here? | First experiment |
| --- | --- | --- | --- |
| 1: strongest existing QML candidate | Angle QKSVM on WDBC | Competitive exploratory result, straightforward controlled comparison | Repeat 2/4/6/8-feature PCA and MI comparisons with equal classical tuning budgets and train-only selection |
| 2: undertrained hybrid candidate | BCE HQMLP on Cleveland | Only three epochs and 50 rows tested; small parameter count makes a bounded learning-curve study feasible | Four qubits, 30–60 epoch ceiling, patience 8–10, then larger training subsets and repeated seeds |
| 3: kernel follow-up | Pauli OQSVM on Cleveland | Better smoke AUPRC than HQMLP, but still below classical models | Match training sample counts; tune `C`, repetitions, and entanglement before increasing qubits |
| 4: image-research candidate | QCNN on BreastMNIST | Only two epochs and four coarse image features tested | Four-qubit full-partition run; compare spatial pooling with PCA and matched reduced baselines |
| 5: bounded diagnostic follow-up | VQC on WDBC/Cleveland | Current results are weak; training behavior needs investigation | Check loss/gradients/learning curves and smaller learning rates before adding depth |
| Practical tabular priority | RBF/linear SVM, logistic regression, tree ensembles | Strong observed tabular performance and low fit cost | Full-feature versus reduced-feature controls; proper categorical encoding and repeated validation |
| Practical image priority | MobileNetV3-Small, then ResNet-18/small CNN | MobileNet led smoke image AUROC/AUPRC | Full official training partition, train the head, then cautiously unfreeze final blocks |

Do not start by allocating all 24 GB or maximizing qubits. Complex128 state storage alone is `16 × 2^q` bytes: four qubits need 256 bytes, eight need 4 KiB, and ten need 16 KiB. Training graphs, simulation intermediates, image networks, and repeated evaluations dominate actual cost. At 30 qubits a single state alone needs 16 GiB; that is not a practical training recommendation for this laptop.

Use CPU analytic simulation first for these small circuits. QCNN already uses adjoint differentiation, which is supported by PennyLane's CPU simulators. [PennyLane adjoint tutorial](https://pennylane.ai/demos/tutorial_adjoint_diff)

A compatible NVIDIA GPU may help image-network fine-tuning. Quantum GPU simulation requires a separately supported backend and software stack; system RAM is not GPU VRAM. Benchmark an actual workload before assuming GPU acceleration helps tiny circuits. [PennyLane Lightning GPU documentation](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html)

## 5. Features and fine-tuning to investigate

### Tabular feature improvements

| Area | Existing behavior | Proposed experiment |
| --- | --- | --- |
| WDBC reduction | PCA/MI at 2, 4, 6, 8 features already screened once | Repeat selection inside training folds; measure feature stability and retained PCA variance |
| WDBC interpretability | Two PCA components are combinations of all 30 inputs, not two named medical fields | Compare a small selected set of original morphology measurements against PCA and full-feature classical baselines |
| Cleveland categories | Nominal categories are compact numeric codes in the smoke representation | One-hot `cp`, `restecg`, and `thal` for practical classical baselines; evaluate an appropriate categorical representation before quantum compression |
| Cleveland missingness | Training-only median imputation; missing `ca` and `thal` values | Compare imputation with/without missingness indicators; avoid tuning on held-out data |
| Heart feature count | Four selected fields used for training | Compare 4/6/8/10 fields. Existing ten-field selection was `thal`, `thalach`, `cp`, `ca`, `exang`, `oldpeak`, `chol`, `sex`, `age`, `restecg`; this was selection-only, not a completed ten-qubit run |
| Selection method | Consensus ranking with an MI fallback for unavailable mRMR | Record the actual selector; compare PCA, MI, and consensus rather than calling the fallback exact mRMR |
| Feature availability | Some Cleveland fields come from diagnostic investigations | Define the intended information-availability point before a baseline-only ablation; removing fields does not turn this cross-sectional dataset into a future-risk dataset |

Sources: [data dictionary](../experiments/ehr_ihd_qml/docs/data_dictionary.md), [feature/config implementation](../src/qml_research/ehr_ihd/), [feature-screen configuration](../configs/wdbc_feature_screen_checkpoint.yaml).

### Proposed bounded search ranges

These are starting experiments, not validated optimal settings. Use staged searches of approximately 8–12 configurations rather than the full Cartesian product.

| Model | Proposed search | Implementation status |
| --- | --- | --- |
| QKSVM/OQSVM | Feature count 2/4/6/8 for Phase 1 or 4/6/8/10 for EHR; SVM `C` 0.01/0.1/1/10; preserve map-specific comparisons | Existing EHR runner accepts `C_candidates`; Phase 1 model tuning may need extension |
| Pauli OQSVM | Feature-map repetitions 1/2/3; linear versus circular entanglement | Existing EHR configuration/model controls |
| HQMLP | 30–60 epoch ceiling; learning rate 0.001/0.003/0.01; 1–3 layers; weight decay 0/0.0001/0.001; dropout 0/0.1; initialization 0.01/0.05; patience 8–10 | Existing controls; current training is full-batch, so mini-batches require code changes |
| HQMLP loss ablation | Hold ansatz fixed while comparing BCE and MSE, then hold loss fixed while comparing ansatz | Configurable variants can isolate the effects; present smoke comparison does not |
| QCNN | Four qubits first; 30 then 60 epochs; LR 0.003/0.01/0.03; batch 8/16/32; small initialization; balanced class loss | Existing profiles/tuning controls; eight qubits is a later controlled experiment |
| VQC | LR 0.001/0.01/0.05; 1–3 layers; progressively larger step budget; multiple initializations | Start with existing config options; add richer gradient diagnostics if necessary |
| Classical controls | SVM `C`/RBF gamma; tree depth and leaf size; MLP regularization | Must receive comparable validation-search effort |

For HQMLP, a classical network with the quantum layer removed/replaced is an important additional ablation: it tests whether any improvement comes from the quantum circuit or the classical projection and training changes.

### Image features and transfer learning

1. **Use the full official 546/78/156 BreastMNIST partitions.** The smoke run is too small for stable threshold selection.
2. **Compare four versus eight spatial features and train-only PCA.** Four regional averages discard much of the texture and shape information. More features are a hypothesis to test, not a guaranteed improvement.
3. **Proposed new hybrid:** frozen MobileNet/ResNet embeddings → train-only PCA to 4/8 components → QCNN or quantum-kernel head. Also evaluate logistic/SVM/MLP heads on those exact embeddings. The current pretrained models are standalone classical baselines; this hybrid connection is not implemented.
4. **Fine-tune pretrained models gradually:** train the classification head first, then unfreeze only late backbone blocks with a smaller learning rate. Existing QCNN profiles expose frozen and partial-finetune modes. This follows the distinction between fixed feature extraction and fine-tuning in the [official PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
5. **Consider training-only augmentation** with modest, task-appropriate transformations after checking image semantics. Augmentation and richer texture descriptors are proposed extensions, not completed experiments.
6. **Keep model ranking and operating threshold separate.** Track AUROC/AUPRC plus malignant sensitivity, specificity, and calibration. More validation data is more valuable than selecting a favorable threshold after looking at test outcomes.

New image feature extractors/reducers will require checkpoint and inference-contract updates. Preserve the original benchmark as an ablation.

## 6. Existing 24 GB profiles: what is ready and what needs work

The EHR file [workstation_24gb.yaml](../experiments/ehr_ihd_qml/configs/workstation_24gb.yaml) specifies ten features/qubits, three HQMLP layers, 60 epochs, a training limit of 300, five listed seeds, and 5,000 bootstrap resamples. It is **configuration intent, not a completed 24 GB result**.

The current source review found these limitations:

- **Seed orchestration:** the runner uses `experiment.seed`; it does not loop over `training.seeds`. Execute separate runs with distinct `experiment.seed` values or implement a verified multi-seed wrapper.
- **Training count:** the 300-row limit cannot manufacture 300 training examples from a 212-row training partition. Report the actual selected count.
- **Early stopping:** the workstation file omits patience, so the runner falls back to two. Set patience explicitly before interpreting a 60-epoch ceiling as substantial training.
- **SVM search:** it specifies `C=1.0` without `C_candidates`; that alone is not a hyperparameter sweep.
- **Resource limits:** the kernel-evaluation preflight is enforced, but the RAM field is configuration validation rather than a hard process-memory cap. `max_runtime_minutes` is not an implemented watchdog in this runner.
- **Backend:** EHR quantum models instantiate `default.qubit`; selecting a GPU or Lightning path needs an implementation change, not merely a larger laptop.
- **Training fairness:** sample-matched classical controls and full-feature practical baselines need explicit additional runs/implementation.

Source: [EHR runner](../src/qml_research/ehr_ihd/runner.py), [configuration validation](../src/qml_research/ehr_ihd/config.py), [models/training](../src/qml_research/ehr_ihd/models.py).

For QCNN, [full_dataset.yaml](../experiments/qcnn_breast_cancer/configs/full_dataset.yaml) keeps four qubits and requests 60 epochs/five seeds. [high_memory.yaml](../experiments/qcnn_breast_cancer/configs/high_memory.yaml) requests eight qubits, 100 epochs, five seeds, and partial image fine-tuning. The latter is a later-stage benchmark, not the best first run merely because 24 GB RAM is available.

Start with one process and conservative data-loader workers, measure RSS and epoch time, and leave several GB for the operating system. For kernels, pair counts scale approximately quadratically with training size; cache/reuse the same kernel across SVM `C` choices. Do not extrapolate total runtime from RAM capacity.

## 7. Recommended next experiment sequence

1. **Reproduce functionality on the new laptop.** Recreate the documented environment, record CPU/RAM/GPU/package versions, run the small smoke profiles, and measure actual fit time and RSS.
2. **Run a bounded WDBC kernel study.** Start from the two-PCA Angle candidate and paired classical baselines; evaluate repeated folds/seeds with training-only selection. The existing best test result is exploratory evidence, not a fresh tuning target.
3. **Run HQMLP and OQSVM learning curves.** Begin at four qubits with 50/100/available-training rows; add classical controls on each identical subset. Tune on validation data. Expand qubits only if the learning curves justify it.
4. **Run full-partition four-qubit QCNN and practical image baselines.** Start with one seed and a moderate epoch ceiling, then reproduce the selected configuration across three to five seeds. Repeated initializations on the same official test set are not independent test cohorts.
5. **Test richer image features or eight qubits as a separate ablation.** Change one design choice at a time and preserve equal-feature classical comparisons.
6. **Evaluate finite shots/noise after freezing promising ideal models.** Keep this distinct from ideal-simulator accuracy and from real-hardware evidence.

Useful QCNN commands, run from the `qml-research` root; these are **suggested next commands, not executed for this report**:

```powershell
# Estimate a moderate full-data run first.
.\.venv\Scripts\qml-research.exe qcnn estimate --profile full_dataset --set epochs=30 --set seeds=[42] --set num_workers=0

# Run the same bounded configuration and matched reduced controls.
.\.venv\Scripts\qml-research.exe qcnn run --profile full_dataset --set epochs=30 --set seeds=[42] --set num_workers=0 --set models=[qcnn,logistic,rbf_svm,reduced_mlp] --confirm-expensive
```

`--confirm-expensive` acknowledges the existing simulator resource gate; it does not submit a real-hardware job. Check the estimate first. For EHR, address the profile limitations above before treating its workstation YAML as a multi-seed benchmark.

An improvement should mean a reproducible gain on held-out evaluation, with uncertainty and resource cost reported. Record AUROC, AUPRC, balanced accuracy, sensitivity/specificity, calibration, training size, split/seed, parameter count, circuit evaluations, wall time, and measured memory. Use validation or inner folds for selection and evaluate the frozen choice on the outer test data. Bootstrap intervals on the same test examples do not measure all training/split uncertainty.

## 8. Evidence and document record

Plan/scope: inventory completed results, separate configured-only work, inspect resource profiles, and produce this one-file summary. Documentation-only work on the existing `feature/ehr-ihd-qml` branch; existing uncommitted implementation work was preserved. No new training, dataset download, hardware submission, or model-code change was performed for this report.

Verification: cross-checked Phase 1 CSV records, ten QCNN model records and its tuning artifact, two EHR smoke summaries and reference metrics, current YAML profiles, and model/runner code. Primary external documentation was checked for MedMNIST, simulator differentiation/GPU requirements, QCNN architecture provenance, and transfer learning. Recommendations are explicitly proposed experiments, not measured future outcomes.

Further local detail: [Phase 1 protocol](../research/experiment-protocol.md), [QCNN reproduction guide](../experiments/qcnn_breast_cancer/docs/reproduction.md), [EHR research basis](../experiments/ehr_ihd_qml/docs/research_basis.md), [QRNN feasibility](../research/qrnn-feasibility.md). Some linked runtime artifacts are ignored by Git and will only resolve on a machine where those results are present; the aggregate findings needed to read this report are included above.
