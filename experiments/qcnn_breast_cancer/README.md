# QCNN breast-cancer experiment

This directory holds the configuration and experiment-specific documentation for the BreastMNIST/WDBC QCNN benchmark. Runtime code is packaged as `qml_research.qcnn` so it shares the established CLI, metrics, and installation conventions without changing the existing Phase 1 runner.

The default model is a four-qubit, two-stage hierarchical QCNN. BreastMNIST malignant cases are disease-positive. All outputs are research artifacts and are not clinical predictions.

See `docs/reproduction.md` for commands and `configs/` for every adjustable sample size and hyperparameter.
