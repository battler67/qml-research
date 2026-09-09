# Comparison with Maheshwari et al. (2023)

| Component | Original paper | This experiment | Reason for difference |
| --- | --- | --- | --- |
| Dataset | Private Basurto EHR | UCI/Framingham teaching | Reproducibility |
| Prediction | CAD in chest-pain EHR cohort | UCI presence / Framingham incident 10-year CHD | Exact outcome naming |
| Features | 42 to 10 | Dataset-specific to 4/6/8/10 | Scaling and auditability |
| Selection | RFE/RF/mRMR | Train-only RFE/RF/MI consensus | Leakage prevention; mRMR fallback recorded |
| QSVM | Pauli, two repetitions | Pauli Z/ZZ, two repetitions plus configurable variants | Underspecified original gates |
| QSVM API | Qiskit-style statevector | PennyLane exact fidelity kernel in current environment | Qiskit ML is not installed |
| QMLP loss | MSE | BCE preferred plus MSE ablation | Binary classification and faithful ablation |
| Split | 80/20 | 70/15/15 | Separate model selection/calibration |
| Balance | Ambiguous timing | Training-only quantum subset; natural validation/test prevalence | Prevent leakage |
| Metrics | Mainly accuracy | AUROC, AUPRC, threshold metrics, Brier, ECE, bootstrap intervals | Clinical relevance |
| Hardware | Statevector | Statevector; no hardware submission | Reproducible functionality |

This is paper-inspired, not an exact reproduction. The private cohort, full gate sequence, and some
selection/validation details are unavailable. Any comparison is methodological rather than a direct
performance replication.
