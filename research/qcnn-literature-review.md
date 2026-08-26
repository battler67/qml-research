# QCNN medical-image literature review

Reviewed: 2026-08-26

This review records the design evidence for `experiments/qcnn_breast_cancer`. Values marked **not reported** were not recoverable from the supplied paper. Reported results are literature claims, not results reproduced by this repository.

## Supplied papers

| Paper | Question and data | Quantum method | Training/backend | Baselines and reported result | Limitations and use here |
|---|---|---|---|---|---|
| *A quantum convolutional network and ResNet (50)-based classification architecture for the MNIST medical dataset* (BSPC 2024, DOI 10.1016/j.bspc.2023.105560) | Classifies 58,954 64x64 MedNIST images into six **imaging modality/body-region classes**: Abdomen CT, Head CT, Breast MRI, Chest CT, CXR, Hand. Uses an 80/20 train/test split. This is not disease classification. | A vaguely specified random/threshold quantum preprocessing stage combined with a modified frozen ImageNet ResNet-50. No hierarchical trainable QCNN pooling is specified. Qubits, exact gates, depth, trainable quantum parameters, shots, noise, and device are not reported. | Adam, learning rate 1e-6, categorical cross-entropy, 200 epochs; batch 32 for QCNN and 64 for ResNet-50. | Claims MQCNN accuracy 99.6%, ResNet-50 98.9%, QCNN 97.5%; another table gives inconsistent values. | Split/preprocessing and circuit details are insufficient for a faithful reproduction, and the target is modality rather than disease. Used only as evidence that medical-image papers can misuse the QCNN label and that an ImageNet baseline is mandatory. |
| *HQCNN: A Hybrid Quantum-Classical Neural Network for Medical Image Classification* (arXiv:2509.14277v1) | Six MedMNIST datasets. BreastMNIST has 780 ultrasound images with official 546/78/156 splits. | Five-layer classical CNN -> dense layer with four features -> four-qubit VQC -> dense 128 -> classifier. Angle embedding with RY; U3 rotations, cyclic CNOTs, and a proposed attention circuit; X and Z measured on each qubit for eight outputs. This is a CNN plus quantum head, not inverse-MERA QCNN pooling. | Parameter-shift and mini-batch training are described. Figures show 10 epochs. Exact optimizer, learning rate, batch size, loss, simulator, shots, noise, circuit depth, and total parameter count are not reported. | BreastMNIST HQCNN AUC 0.9004, accuracy 0.8718. Paper table lists ResNet-18 0.901/0.863 and ResNet-50 0.857/0.812. | Baseline provenance and whether all baselines were rerun are unclear; no repeated-seed uncertainty or resource accounting. Used to support the four-qubit feasibility target and the need for locally reproduced baselines. |
| *Digital-analog quantum convolutional neural networks for image classification* (Physical Review Research 6, L042060, 2024) | Binary BreastMNIST and PneumoniaMNIST classification, with five-fold cross-validation on training data. | Non-trainable digital-analog quanvolution kernels scan 2x2 (four-qubit) or 3x3 (nine-qubit) image patches and create quantum feature maps for a classical CNN. Pixel angles use H and RY; Rydberg-Ising evolution is Trotterized for four steps. Per-qubit Z expectations are measured. This is quanvolution, not hierarchical QCNN. | Grid: learning rate {1e-4, 3e-4, 5e-4}, activation {ReLU, GELU}, dropout {.50, .55, .60}, weight decay {1e-7, 1e-6}; 30 independent trainings per configuration. Batch, epochs, optimizer, and hardware are not clearly reported. Simulated ideal and optional depolarizing noise. | BreastMNIST best nine-qubit/four-graph AUC 0.926, accuracy 0.897; four-qubit/single-graph 0.922/0.885; equivalent CNN 0.903/0.885. | Simulation and multiple circuit calls dominate cost; scaling patches to high resolution would require many qubits and substantial hardware parallelism. Used as an imaging comparison and noise-design reference, not the selected QCNN. |
| *A Comprehensive Analysis of Accuracy and Robustness in Quantum Neural Networks* (arXiv:2604.26110v1) | QCNN, QRNN, and QViT on binary/multiclass MNIST and CIFAR-10 with varied training sizes; robustness to attacks, shots, and quantum channels. | QCNN resizes images to 8x8 and amplitude-encodes them on six qubits. It uses two U3/IsingXX/YY/ZZ convolution layers, two measurement-conditioned pooling layers, and an ArbitraryUnitary final layer; reported circuit depth 12. QRNN creates a sequence of regional image averages and angle-encodes recurrent steps. | Adam and cross-entropy; figures/results use up to 100 epochs. Exact QCNN learning rate, batch size, total parameter count, simulator, and seeds are not reported. | QCNN reports MNIST 1-vs-7 accuracy 97.3% and CIFAR cat-vs-dog 55.5%; QRNN 96.7% and 57.1%. Noise and adversarial results vary substantially by architecture/channel. | Preprint, binary-only QCNN/QRNN, incomplete hyperparameter reporting, no disease task, and no thorough tuning. Supports the inverse-MERA block choice, shallow circuits, gradient/resource logging, and the decision not to force QRNN onto unordered images. QViT is excluded by scope. |
| *Exponential concentration in quantum kernel methods* (Nature Communications 15, 5200, 2024) | Theoretical and numerical study of quantum-kernel concentration; examples include PCA-reduced binary MNIST and synthetic data. It is not a QCNN or disease study. | Shows concentration can arise from expressive embeddings, global measurements, entanglement in projected kernels, and noise; polynomial shots may then yield data-independent models. | Includes ideal/noisy simulations and finite shots; training hyperparameters are not applicable to the main theorems. | No disease-classification benchmark. | Used as a negative design constraint: keep circuits shallow/local, avoid unstructured repeated encoding, use a local final observable, and record gradient norms/variance, depth, gates, shots, and noise. |

## Additional authoritative sources

### Foundational QCNN definition

Cong, Choi, and Lukin introduced QCNNs as convolution and pooling over a quantum state with only O(log N) variational parameters for N input qubits. Their work targets quantum-state phase recognition and error correction rather than classical medical images, so it provides the architectural definition—not evidence of medical-image advantage.

- Paper: https://arxiv.org/abs/1810.03787
- PennyLane worked implementation: https://pennylane.ai/demos/tutorial_learning_few_data

The PennyLane construction uses shared two-qubit unitaries for convolution, conditioned operations for pooling, and a final small unitary. The experiment here uses the same hierarchy but a fully unitary pooling block so analytic differentiation remains portable across simulators.

### Dataset facts and published classical references

BreastMNIST derives from 780 breast-ultrasound images originally labelled normal, benign, or malignant. MedMNIST combines normal and benign as the negative class and provides fixed 7:1:2 train/validation/test splits at 28x28 grayscale resolution. Official labels encode malignant as `0`; this project explicitly remaps malignant to disease-positive `1`.

- Dataset metadata and checksum: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py
- Dataset paper: https://doi.org/10.1038/s41597-022-01721-8
- Original BUSI source: Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief 28 (2020), https://doi.org/10.1016/j.dib.2019.104863

Official MedMNIST v2 reference results for BreastMNIST include ResNet-18 (28) AUC 0.901/accuracy 0.863 and ResNet-50 (28) 0.857/0.812. These are literature references only. This repository reruns its own pinned models and never merges published and reproduced values in one statistical comparison.

WDBC contains 569 samples, 30 real-valued features computed from digitized fine-needle aspirate images, and 212 malignant/357 benign cases.

- UCI: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

### Benchmarking standard

Bowles, Ahmed, and Schuld found that out-of-the-box classical methods generally outperform simulated quantum classifiers on small binary tasks, and that entanglement is not consistently beneficial. This motivates matched-feature baselines, repeated seeds, resource accounting, and conservative advantage language.

- https://arxiv.org/abs/2403.07059

## Resulting research question

On fixed BreastMNIST splits, does a shallow four-qubit hierarchical QCNN provide competitive discrimination or sample efficiency relative to:

1. classical models using exactly the same four spatial features;
2. a small CNN using the original 28x28 image; and
3. pinned ImageNet transfer-learning models?

The experiment measures predictive quality, uncertainty, resource cost, noise sensitivity, and train-size scaling. It does not presuppose an advantage and does not claim a hardware speedup from statevector simulation.
