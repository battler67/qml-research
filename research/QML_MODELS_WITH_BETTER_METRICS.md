# QML models with better measured metrics

Generated from saved held-out predictions and validation-only model selection. This file includes negative results; it is not a promise of quantum advantage.

## Evidence standard

Each model family selects hyperparameters by validation AUROC, and chooses a threshold on validation data. Test labels are not used for selection. The comparator is the classical family selected by validation AUROC on the same split. All classical test results are also retained below to expose stronger alternatives.

A positive mean difference is exploratory. A stronger signal requires all 3 seeds × 5 outer folds, mean balanced-accuracy gain ≥0.02, no mean sensitivity loss >0.02, and a multiplicity-adjusted corrected repeated-CV interval above zero. The correction uses test/train size and a Bonferroni adjustment over the planned quantum-family/dataset comparisons. It remains an approximate within-dataset analysis, not external clinical validation.

Angle kernels are separable and have an efficient classical product-cosine implementation. Projected quantum kernels have a classical RBF head on simulated quantum features. No simulator result here establishes computational quantum advantage.

Kaggle Framingham is a 4,238-row third-party mirror with unverifiable follow-up/censoring and unspecified source license; it is not the official BioLINCC longitudinal cohort. Pima is a limited demographic benchmark. BreastMNIST uses its official partitions; repeated seeds share the same test images.

## Models with positive point estimates

These rows improved at least one mean metric against the validation-selected classical comparator. They are candidates, not established advantages.

| Dataset | QML family | Paired splits | Δ balanced accuracy | Δ AUROC | Δ AUPRC | Stronger signal rule |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| breastmnist | q_iqp | 1 | -0.0570 | -0.0472 | +0.0125 | No |
| cleveland | q_angle | 15 | -0.0126 | +0.0075 | +0.0128 | No |
| cleveland | q_iqp | 15 | -0.0240 | +0.0000 | +0.0044 | No |
| pima_kaggle | q_angle | 1 | +0.0043 | -0.0104 | -0.0007 | No |
| pima_kaggle | q_iqp | 1 | +0.0120 | -0.0035 | +0.0111 | No |
| pima_kaggle | q_projected | 1 | +0.0313 | -0.0119 | -0.0232 | No |
| wdbc | hqmlp | 1 | -0.0069 | +0.0003 | +0.0005 | No |
| wdbc | q_iqp | 15 | -0.0076 | +0.0002 | -0.0003 | No |
| wdbc | q_projected | 15 | -0.0088 | +0.0010 | -0.0005 | No |

## All completed family results

Means below include every completed split for that family. Unequal counts indicate incomplete coverage and must not be treated as matched comparisons.

| Dataset | Family | Splits | Mean balanced accuracy | Mean AUROC | Mean AUPRC |
| --- | --- | ---: | ---: | ---: | ---: |
| breastmnist | classical_boosting | 1 | 0.7920 | 0.8517 | 0.6913 |
| breastmnist | classical_forest | 1 | 0.6598 | 0.8491 | 0.7042 |
| breastmnist | classical_logistic | 1 | 0.7494 | 0.7765 | 0.4829 |
| breastmnist | classical_rbf_full | 1 | 0.7751 | 0.8546 | 0.6940 |
| breastmnist | classical_rbf_reduced | 1 | 0.7068 | 0.7813 | 0.5560 |
| breastmnist | q_angle | 1 | 0.6648 | 0.6850 | 0.5193 |
| breastmnist | q_iqp | 1 | 0.7180 | 0.8074 | 0.7065 |
| breastmnist | q_projected | 1 | 0.6867 | 0.7930 | 0.5990 |
| cleveland | classical_boosting | 15 | 0.7906 | 0.8929 | 0.8928 |
| cleveland | classical_forest | 15 | 0.8140 | 0.9093 | 0.9068 |
| cleveland | classical_logistic | 15 | 0.8043 | 0.9095 | 0.9061 |
| cleveland | classical_rbf_full | 15 | 0.7924 | 0.9002 | 0.8982 |
| cleveland | classical_rbf_reduced | 15 | 0.8004 | 0.8989 | 0.8907 |
| cleveland | q_angle | 15 | 0.7952 | 0.9038 | 0.8991 |
| cleveland | q_iqp | 15 | 0.7838 | 0.8964 | 0.8907 |
| cleveland | q_projected | 15 | 0.7820 | 0.8802 | 0.8758 |
| framingham_kaggle | classical_boosting | 7 | 0.6366 | 0.6953 | 0.3100 |
| framingham_kaggle | classical_forest | 7 | 0.6399 | 0.7041 | 0.3179 |
| framingham_kaggle | classical_logistic | 7 | 0.6649 | 0.7231 | 0.3413 |
| framingham_kaggle | classical_rbf_full | 7 | 0.6611 | 0.7220 | 0.3349 |
| framingham_kaggle | classical_rbf_reduced | 7 | 0.6533 | 0.7127 | 0.3063 |
| framingham_kaggle | q_angle | 7 | 0.6536 | 0.7121 | 0.3107 |
| framingham_kaggle | q_iqp | 6 | 0.6452 | 0.7102 | 0.3116 |
| framingham_kaggle | q_projected | 6 | 0.6351 | 0.6897 | 0.2817 |
| pima_kaggle | classical_boosting | 1 | 0.7339 | 0.7856 | 0.7019 |
| pima_kaggle | classical_forest | 1 | 0.7467 | 0.8178 | 0.7218 |
| pima_kaggle | classical_logistic | 1 | 0.7161 | 0.8183 | 0.7154 |
| pima_kaggle | classical_rbf_full | 1 | 0.7076 | 0.8213 | 0.7181 |
| pima_kaggle | classical_rbf_reduced | 1 | 0.7361 | 0.8131 | 0.7298 |
| pima_kaggle | q_angle | 1 | 0.7204 | 0.8080 | 0.7147 |
| pima_kaggle | q_iqp | 1 | 0.7281 | 0.8148 | 0.7265 |
| pima_kaggle | q_projected | 1 | 0.7474 | 0.8065 | 0.6922 |
| wdbc | classical_boosting | 15 | 0.9566 | 0.9906 | 0.9886 |
| wdbc | classical_forest | 15 | 0.9463 | 0.9890 | 0.9865 |
| wdbc | classical_logistic | 15 | 0.9597 | 0.9932 | 0.9920 |
| wdbc | classical_mlp | 1 | 0.9762 | 0.9987 | 0.9978 |
| wdbc | classical_rbf_full | 15 | 0.9675 | 0.9914 | 0.9896 |
| wdbc | classical_rbf_reduced | 15 | 0.9443 | 0.9929 | 0.9909 |
| wdbc | hqmlp | 1 | 0.9692 | 0.9993 | 0.9989 |
| wdbc | q_angle | 15 | 0.9531 | 0.9921 | 0.9882 |
| wdbc | q_iqp | 15 | 0.9559 | 0.9926 | 0.9907 |
| wdbc | q_projected | 15 | 0.9548 | 0.9934 | 0.9906 |
| wdbc | reupload_vqc | 1 | 0.9058 | 0.9864 | 0.9790 |
| wdbc | separable_vqc | 1 | 0.9762 | 0.9990 | 0.9984 |

## Current conclusion

0 comparison(s) meet the stronger signal rule. No robust predictive advantage has been demonstrated in this campaign yet.

Raw records, trial warnings/failures, exact split indices, selected specifications, and predictions are under `results/campaign/biomedical_qml_tuning_v1/`. Resume and protocol details: [campaign guide](../docs/RESUMABLE_CAMPAIGN.md).

Saved failed-trial records: 0. Completed family/split records: 314.
