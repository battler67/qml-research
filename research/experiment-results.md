# Phase 1 Experiment Results

This report is generated from `results/raw/*.json`; values are not hardcoded.

## Coverage

- Completed fold/model records: 121
- Failed or skipped records: 0
- Datasets represented: heart, iris, wdbc
- Seeds represented: 42

## Results

| experiment | dataset | sample_size | model | reduction | feature_count | feature_map | quantum_condition | runs | balanced_accuracy_mean | balanced_accuracy_std | sensitivity_mean | specificity_mean | auroc_mean | circuit_executions_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_phase1_checkpoint | heart | 50 | linear_svm | pca | 4 | - | - | 1 | 0.8000 | - | 0.8000 | 0.8000 | 0.8400 | - |
| heart_phase1_checkpoint | heart | 50 | rbf_svm | pca | 4 | - | - | 1 | 0.8000 | - | 0.8000 | 0.8000 | 0.8400 | - |
| heart_phase1_checkpoint | heart | 50 | vqc | pca | 4 | angle | ideal | 1 | 0.8000 | - | 0.6000 | 1.0000 | 0.8400 | 40010.0000 |
| heart_phase1_checkpoint | heart | 200 | rbf_svm | pca | 4 | - | - | 1 | 0.7475 | - | 0.7222 | 0.7727 | 0.7652 | - |
| heart_phase1_checkpoint | heart | 100 | linear_svm | none | full | - | - | 1 | 0.7424 | - | 0.6667 | 0.8182 | 0.7273 | - |
| heart_phase1_checkpoint | heart | 100 | logistic_regression | none | full | - | - | 1 | 0.7424 | - | 0.6667 | 0.8182 | 0.7778 | - |
| heart_phase1_checkpoint | heart | 200 | logistic_regression | pca | 4 | - | - | 1 | 0.7247 | - | 0.7222 | 0.7273 | 0.7854 | - |
| heart_phase1_checkpoint | heart | 200 | logistic_regression | none | full | - | - | 1 | 0.7197 | - | 0.6667 | 0.7727 | 0.7929 | - |
| heart_phase1_checkpoint | heart | 200 | qksvm | pca | 4 | angle | ideal | 1 | 0.7197 | - | 0.6667 | 0.7727 | 0.7323 | 19280.0000 |
| heart_phase1_checkpoint | heart | 200 | linear_svm | none | full | - | - | 1 | 0.7020 | - | 0.7222 | 0.6818 | 0.7828 | - |
| heart_phase1_checkpoint | heart | 200 | random_forest | none | full | - | - | 1 | 0.7020 | - | 0.7222 | 0.6818 | 0.8333 | - |
| heart_phase1_checkpoint | heart | 200 | rbf_svm | none | full | - | - | 1 | 0.7020 | - | 0.7222 | 0.6818 | 0.7854 | - |
| heart_phase1_checkpoint | heart | 50 | qksvm | pca | 4 | angle | ideal | 1 | 0.7000 | - | 0.8000 | 0.6000 | 0.7600 | 1220.0000 |
| heart_phase1_checkpoint | heart | 50 | rbf_svm | none | full | - | - | 1 | 0.7000 | - | 0.8000 | 0.6000 | 0.6000 | - |
| heart_phase1_checkpoint | heart | 100 | random_forest | none | full | - | - | 1 | 0.6970 | - | 0.6667 | 0.7273 | 0.7374 | - |
| heart_phase1_checkpoint | heart | 200 | linear_svm | pca | 4 | - | - | 1 | 0.6970 | - | 0.6667 | 0.7273 | 0.7904 | - |
| heart_phase1_checkpoint | heart | 100 | rbf_svm | none | full | - | - | 1 | 0.6515 | - | 0.6667 | 0.6364 | 0.7172 | - |
| heart_phase1_checkpoint | heart | 200 | random_forest | pca | 4 | - | - | 1 | 0.6237 | - | 0.6111 | 0.6364 | 0.6970 | - |
| heart_phase1_checkpoint | heart | 100 | linear_svm | pca | 4 | - | - | 1 | 0.6061 | - | 0.6667 | 0.5455 | 0.6667 | - |
| heart_phase1_checkpoint | heart | 100 | logistic_regression | pca | 4 | - | - | 1 | 0.6061 | - | 0.6667 | 0.5455 | 0.7071 | - |
| heart_phase1_checkpoint | heart | 100 | rbf_svm | pca | 4 | - | - | 1 | 0.6061 | - | 0.6667 | 0.5455 | 0.5859 | - |
| heart_phase1_checkpoint | heart | 50 | logistic_regression | none | full | - | - | 1 | 0.6000 | - | 0.8000 | 0.4000 | 0.5200 | - |
| heart_phase1_checkpoint | heart | 50 | logistic_regression | pca | 4 | - | - | 1 | 0.6000 | - | 0.6000 | 0.6000 | 0.7600 | - |
| heart_phase1_checkpoint | heart | 50 | random_forest | pca | 4 | - | - | 1 | 0.6000 | - | 0.6000 | 0.6000 | 0.7600 | - |
| heart_phase1_checkpoint | heart | 100 | qksvm | pca | 4 | angle | ideal | 1 | 0.5505 | - | 0.5556 | 0.5455 | 0.5556 | 4840.0000 |
| heart_phase1_checkpoint | heart | 100 | random_forest | pca | 4 | - | - | 1 | 0.5505 | - | 0.5556 | 0.5455 | 0.5657 | - |
| heart_phase1_checkpoint | heart | 200 | vqc | pca | 4 | angle | ideal | 1 | 0.5051 | - | 0.0556 | 0.9545 | 0.3687 | 40040.0000 |
| heart_phase1_checkpoint | heart | 50 | linear_svm | none | full | - | - | 1 | 0.5000 | - | 0.6000 | 0.4000 | 0.4800 | - |
| heart_phase1_checkpoint | heart | 50 | random_forest | none | full | - | - | 1 | 0.5000 | - | 0.6000 | 0.4000 | 0.5200 | - |
| heart_phase1_checkpoint | heart | 100 | vqc | pca | 4 | angle | ideal | 1 | 0.4848 | - | 0.3333 | 0.6364 | 0.5051 | 40020.0000 |
| smoke | iris | full | linear_svm | none | full | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | linear_svm | pca | 2 | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | logistic_regression | none | full | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | logistic_regression | pca | 2 | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | random_forest | none | full | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | random_forest | pca | 2 | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | rbf_svm | none | full | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | rbf_svm | pca | 2 | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| smoke | iris | full | qksvm | pca | 2 | angle | ideal | 1 | 0.9800 | - | 0.9600 | 1.0000 | 0.9984 | 3775.0000 |
| smoke | iris | full | vqc | pca | 2 | angle | ideal | 1 | 0.4800 | - | 0.5200 | 0.4400 | 0.5392 | 570.0000 |
| wdbc_phase1_checkpoint | wdbc | 50 | linear_svm | none | full | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | linear_svm | pca | 4 | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | logistic_regression | none | full | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | logistic_regression | pca | 4 | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | random_forest | none | full | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | rbf_svm | pca | 4 | - | - | 1 | 1.0000 | - | 1.0000 | 1.0000 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | logistic_regression | none | full | - | - | 1 | 0.9667 | - | 0.9333 | 1.0000 | 0.9653 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | rbf_svm | none | full | - | - | 1 | 0.9467 | - | 0.9333 | 0.9600 | 0.9947 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | pca | 2 | - | - | 1 | 0.9375 | - | 0.8750 | 1.0000 | 0.9896 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 2 | angle | ideal | 1 | 0.9375 | - | 0.8750 | 1.0000 | 0.9896 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | pca | 8 | - | - | 1 | 0.9375 | - | 0.8750 | 1.0000 | 0.9792 | - |
| wdbc_phase1_checkpoint | wdbc | 100 | linear_svm | none | full | - | - | 1 | 0.9375 | - | 0.8750 | 1.0000 | 0.9688 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | linear_svm | pca | 4 | - | - | 1 | 0.9333 | - | 0.8667 | 1.0000 | 0.9973 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | random_forest | none | full | - | - | 1 | 0.9333 | - | 0.8667 | 1.0000 | 0.9947 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | rbf_svm | pca | 4 | - | - | 1 | 0.9333 | - | 0.8667 | 1.0000 | 1.0000 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | pca | 4 | - | - | 1 | 0.9167 | - | 1.0000 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | mutual_info | 8 | - | - | 1 | 0.9167 | - | 1.0000 | 0.8333 | 0.9583 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | rbf_svm | none | full | - | - | 1 | 0.9167 | - | 1.0000 | 0.8333 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 100 | linear_svm | pca | 4 | - | - | 1 | 0.9167 | - | 1.0000 | 0.8333 | 0.9583 | - |
| wdbc_phase1_checkpoint | wdbc | 100 | random_forest | none | full | - | - | 1 | 0.9167 | - | 1.0000 | 0.8333 | 0.9583 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | qksvm | pca | 4 | angle | ideal | 1 | 0.9133 | - | 0.8667 | 0.9600 | 0.9440 | 19280.0000 |
| wdbc_phase1_checkpoint | wdbc | 200 | linear_svm | none | full | - | - | 1 | 0.9000 | - | 0.8000 | 1.0000 | 1.0000 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | logistic_regression | pca | 4 | - | - | 1 | 0.9000 | - | 0.8000 | 1.0000 | 0.9760 | - |
| wdbc_phase1_checkpoint | wdbc | 200 | random_forest | pca | 4 | - | - | 1 | 0.9000 | - | 0.8000 | 1.0000 | 0.9920 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | pca | 2 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9896 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | pca | 4 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9688 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | pca | 8 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 4 | angle | ideal | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9896 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | pca | 2 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9792 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | pca | 8 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9688 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | pca | 2 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9792 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | pca | 4 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9792 | - |
| wdbc_kernel_noise | wdbc | 100 | qksvm | pca | 4 | angle | depolarizing | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9896 | 4840.0000 |
| wdbc_kernel_noise | wdbc | 100 | qksvm | pca | 4 | angle | finite_shot | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9896 | 4840.0000 |
| wdbc_kernel_noise | wdbc | 100 | qksvm | pca | 4 | angle | ideal | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9896 | 4840.0000 |
| wdbc_phase1_checkpoint | wdbc | 100 | logistic_regression | pca | 4 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9688 | - |
| wdbc_phase1_checkpoint | wdbc | 100 | qksvm | pca | 4 | angle | ideal | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9896 | 4840.0000 |
| wdbc_phase1_checkpoint | wdbc | 100 | rbf_svm | none | full | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9792 | - |
| wdbc_phase1_checkpoint | wdbc | 100 | rbf_svm | pca | 4 | - | - | 1 | 0.8958 | - | 0.8750 | 0.9167 | 0.9792 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | pca | 6 | - | - | 1 | 0.8750 | - | 0.7500 | 1.0000 | 0.9688 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | pca | 8 | - | - | 1 | 0.8750 | - | 0.7500 | 1.0000 | 0.9792 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | pca | 6 | - | - | 1 | 0.8750 | - | 0.7500 | 1.0000 | 0.9688 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 6 | angle | ideal | 1 | 0.8750 | - | 0.7500 | 1.0000 | 0.9375 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | mutual_info | 4 | - | - | 1 | 0.8542 | - | 0.8750 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | mutual_info | 6 | - | - | 1 | 0.8542 | - | 0.8750 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | mutual_info | 8 | - | - | 1 | 0.8542 | - | 0.8750 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | pca | 4 | - | - | 1 | 0.8542 | - | 0.8750 | 0.8333 | 0.9688 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | pca | 6 | - | - | 1 | 0.8542 | - | 0.8750 | 0.8333 | 0.9688 | - |
| wdbc_phase1_checkpoint | wdbc | 100 | logistic_regression | none | full | - | - | 1 | 0.8542 | - | 0.8750 | 0.8333 | 0.9583 | - |
| wdbc_phase1_checkpoint | wdbc | 100 | random_forest | pca | 4 | - | - | 1 | 0.8542 | - | 0.8750 | 0.8333 | 0.9688 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 6 | iqp | ideal | 1 | 0.8333 | - | 0.7500 | 0.9167 | 0.9583 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 8 | iqp | ideal | 1 | 0.8333 | - | 0.7500 | 0.9167 | 0.9479 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 2 | iqp | ideal | 1 | 0.8333 | - | 0.7500 | 0.9167 | 0.9167 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 8 | angle | ideal | 1 | 0.8333 | - | 0.7500 | 0.9167 | 0.9167 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | pca | 6 | - | - | 1 | 0.8125 | - | 0.6250 | 1.0000 | 1.0000 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | mutual_info | 4 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | mutual_info | 6 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | mutual_info | 8 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | mutual_info | 4 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9479 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | mutual_info | 6 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9479 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | mutual_info | 8 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 4 | angle | ideal | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 4 | iqp | ideal | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.8125 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 6 | angle | ideal | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 8 | angle | ideal | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | mutual_info | 4 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | mutual_info | 6 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | random_forest | pca | 4 | - | - | 1 | 0.7917 | - | 0.7500 | 0.8333 | 0.9583 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | random_forest | mutual_info | 2 | - | - | 1 | 0.7708 | - | 0.8750 | 0.6667 | 0.9375 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | linear_svm | mutual_info | 2 | - | - | 1 | 0.7500 | - | 0.7500 | 0.7500 | 0.9375 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | logistic_regression | mutual_info | 2 | - | - | 1 | 0.7500 | - | 0.7500 | 0.7500 | 0.9271 | - |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 2 | angle | ideal | 1 | 0.7500 | - | 0.7500 | 0.7500 | 0.8854 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | rbf_svm | mutual_info | 2 | - | - | 1 | 0.7500 | - | 0.7500 | 0.7500 | 0.8854 | - |
| wdbc_phase1_checkpoint | wdbc | 50 | qksvm | pca | 4 | angle | ideal | 1 | 0.7083 | - | 0.7500 | 0.6667 | 0.7917 | 1220.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 6 | iqp | ideal | 1 | 0.6875 | - | 0.3750 | 1.0000 | 0.8229 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 8 | iqp | ideal | 1 | 0.6250 | - | 0.2500 | 1.0000 | 0.7500 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | pca | 4 | iqp | ideal | 1 | 0.6042 | - | 0.3750 | 0.8333 | 0.7604 | 4840.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | mutual_info | 2 | iqp | ideal | 1 | 0.5208 | - | 0.3750 | 0.6667 | 0.5312 | 4840.0000 |
| wdbc_phase1_checkpoint | wdbc | 200 | vqc | pca | 4 | angle | ideal | 1 | 0.4800 | - | 0.2000 | 0.7600 | 0.5760 | 40040.0000 |
| wdbc_phase1_checkpoint | wdbc | 50 | vqc | pca | 4 | angle | ideal | 1 | 0.4167 | - | 0.0000 | 0.8333 | 0.2500 | 40010.0000 |
| wdbc_phase1_checkpoint | wdbc | 100 | vqc | pca | 4 | angle | ideal | 1 | 0.4167 | - | 0.0000 | 0.8333 | 0.4688 | 40020.0000 |

## Quantum resource telemetry

Circuit structures are expanded at the PennyLane device level. Execution totals are
actual QNode executions counted by the implemented run, not asymptotic estimates.

| experiment | dataset | sample_size | model | feature_count | feature_map | quantum_condition | num_qubits_mean | circuit_depth_mean | total_gate_count_mean | two_qubit_gate_count_mean | trainable_parameters_mean | circuit_executions_mean | training_time_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_phase1_checkpoint | heart | 50 | qksvm | 4 | angle | ideal | 4.0000 | - | - | - | 0.0000 | 1220.0000 | 1.1709 |
| heart_phase1_checkpoint | heart | 50 | vqc | 4 | angle | ideal | 4.0000 | - | - | - | 13.0000 | 40010.0000 | 91.2455 |
| heart_phase1_checkpoint | heart | 100 | qksvm | 4 | angle | ideal | 4.0000 | - | - | - | 0.0000 | 4840.0000 | 5.0708 |
| heart_phase1_checkpoint | heart | 100 | vqc | 4 | angle | ideal | 4.0000 | - | - | - | 13.0000 | 40020.0000 | 94.7364 |
| heart_phase1_checkpoint | heart | 200 | qksvm | 4 | angle | ideal | 4.0000 | - | - | - | 0.0000 | 19280.0000 | 1647.6758 |
| heart_phase1_checkpoint | heart | 200 | vqc | 4 | angle | ideal | 4.0000 | - | - | - | 13.0000 | 40040.0000 | 72.8219 |
| smoke | iris | full | qksvm | 2 | angle | ideal | 2.0000 | 2.0000 | 4.0000 | 0.0000 | 0.0000 | 3775.0000 | 0.9756 |
| smoke | iris | full | vqc | 2 | angle | ideal | 2.0000 | 4.0000 | 6.0000 | 2.0000 | 7.0000 | 570.0000 | 0.5366 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 2 | angle | ideal | 2.0000 | 2.0000 | 4.0000 | 0.0000 | 0.0000 | 4840.0000 | 7.3254 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 2 | iqp | ideal | 2.0000 | 6.0000 | 10.0000 | 2.0000 | 0.0000 | 4840.0000 | 10.8857 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 4 | angle | ideal | 4.0000 | 2.0000 | 8.0000 | 0.0000 | 0.0000 | 4840.0000 | 10.4791 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 4 | iqp | ideal | 4.0000 | 14.0000 | 28.0000 | 12.0000 | 0.0000 | 4840.0000 | 19.2244 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 6 | angle | ideal | 6.0000 | 2.0000 | 12.0000 | 0.0000 | 0.0000 | 4840.0000 | 12.2802 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 6 | iqp | ideal | 6.0000 | 22.0000 | 54.0000 | 30.0000 | 0.0000 | 4840.0000 | 38.3046 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 8 | angle | ideal | 8.0000 | 2.0000 | 16.0000 | 0.0000 | 0.0000 | 4840.0000 | 15.8426 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 8 | iqp | ideal | 8.0000 | 30.0000 | 88.0000 | 56.0000 | 0.0000 | 4840.0000 | 55.0345 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 2 | angle | ideal | 2.0000 | 2.0000 | 4.0000 | 0.0000 | 0.0000 | 4840.0000 | 7.6141 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 2 | iqp | ideal | 2.0000 | 6.0000 | 10.0000 | 2.0000 | 0.0000 | 4840.0000 | 11.9772 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 4 | angle | ideal | 4.0000 | 2.0000 | 8.0000 | 0.0000 | 0.0000 | 4840.0000 | 10.0985 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 4 | iqp | ideal | 4.0000 | 14.0000 | 28.0000 | 12.0000 | 0.0000 | 4840.0000 | 18.3448 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 6 | angle | ideal | 6.0000 | 2.0000 | 12.0000 | 0.0000 | 0.0000 | 4840.0000 | 12.9508 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 6 | iqp | ideal | 6.0000 | 22.0000 | 54.0000 | 30.0000 | 0.0000 | 4840.0000 | 28.3261 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 8 | angle | ideal | 8.0000 | 2.0000 | 16.0000 | 0.0000 | 0.0000 | 4840.0000 | 11.4512 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | qksvm | 8 | iqp | ideal | 8.0000 | 30.0000 | 88.0000 | 56.0000 | 0.0000 | 4840.0000 | 55.5415 |
| wdbc_kernel_noise | wdbc | 100 | qksvm | 4 | angle | depolarizing | 4.0000 | 4.0000 | 16.0000 | 0.0000 | 0.0000 | 4840.0000 | 16.0483 |
| wdbc_kernel_noise | wdbc | 100 | qksvm | 4 | angle | finite_shot | 4.0000 | 2.0000 | 8.0000 | 0.0000 | 0.0000 | 4840.0000 | 7.3458 |
| wdbc_kernel_noise | wdbc | 100 | qksvm | 4 | angle | ideal | 4.0000 | 2.0000 | 8.0000 | 0.0000 | 0.0000 | 4840.0000 | 3.2010 |
| wdbc_phase1_checkpoint | wdbc | 50 | qksvm | 4 | angle | ideal | 4.0000 | - | - | - | 0.0000 | 1220.0000 | 1.1965 |
| wdbc_phase1_checkpoint | wdbc | 50 | vqc | 4 | angle | ideal | 4.0000 | - | - | - | 13.0000 | 40010.0000 | 82.9779 |
| wdbc_phase1_checkpoint | wdbc | 100 | qksvm | 4 | angle | ideal | 4.0000 | - | - | - | 0.0000 | 4840.0000 | 4.5183 |
| wdbc_phase1_checkpoint | wdbc | 100 | vqc | 4 | angle | ideal | 4.0000 | - | - | - | 13.0000 | 40020.0000 | 83.5347 |
| wdbc_phase1_checkpoint | wdbc | 200 | qksvm | 4 | angle | ideal | 4.0000 | - | - | - | 0.0000 | 19280.0000 | 19.1649 |
| wdbc_phase1_checkpoint | wdbc | 200 | vqc | 4 | angle | ideal | 4.0000 | - | - | - | 13.0000 | 40040.0000 | 85.1666 |

## Controlled noise study

Kernel-value deltas are calculated directly from the saved Gram matrices against the
ideal matrix for the identical fixed fold. Unchanged predictions on one small fold do
not establish noise robustness.

| condition | backend | shots | noise_strength | train_kernel_mean_absolute_delta_from_ideal | train_kernel_max_absolute_delta_from_ideal | balanced_accuracy | sensitivity | specificity | auroc | diagonal_max_error | minimum_eigenvalue | negative_eigenvalues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ideal | default.qubit | - | 0.0000 | 0.0000 | 0.0000 | 0.8958 | 0.8750 | 0.9167 | 0.9896 | 0.0000 | 0.0000 | 0 |
| finite_shot | default.qubit | 1024.0000 | 0.0000 | 0.0058 | 0.0502 | 0.8958 | 0.8750 | 0.9167 | 0.9896 | 0.0000 | -0.1190 | 20 |
| depolarizing | default.mixed | 1024.0000 | 0.0100 | 0.0099 | 0.0757 | 0.8958 | 0.8750 | 0.9167 | 0.9896 | 0.0752 | -0.1230 | 20 |

## Paired model deltas

Positive deltas favor QML. Every row uses the same dataset subset, fold, reduction,
feature count, preprocessing and seed as its classical comparator.

| experiment | dataset | sample_size | seed | fold | feature_count | qml_model | qml_feature_map | classical_model | delta_balanced_accuracy | delta_sensitivity | delta_specificity | delta_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | qksvm | angle | logistic_regression | -0.0051 | -0.0556 | 0.0455 | -0.0530 |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | qksvm | angle | random_forest | 0.0960 | 0.0556 | 0.1364 | 0.0354 |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | qksvm | angle | linear_svm | 0.0227 | 0.0000 | 0.0455 | -0.0581 |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | qksvm | angle | rbf_svm | -0.0278 | -0.0556 | 0.0000 | -0.0328 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | rbf_svm | -0.1250 | -0.2500 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | random_forest | -0.0625 | -0.1250 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | linear_svm | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | qksvm | angle | random_forest | 0.1000 | 0.2000 | 0.0000 | 0.0000 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | qksvm | angle | linear_svm | -0.1000 | 0.0000 | -0.2000 | -0.0800 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | qksvm | angle | logistic_regression | 0.1000 | 0.2000 | 0.0000 | 0.0000 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | qksvm | angle | rbf_svm | -0.1000 | 0.0000 | -0.2000 | -0.0800 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | linear_svm | -0.2292 | -0.3750 | -0.0833 | -0.4062 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | random_forest | -0.2500 | -0.5000 | 0.0000 | -0.4062 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | rbf_svm | -0.2292 | -0.3750 | -0.0833 | -0.3542 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | logistic_regression | -0.2292 | -0.3750 | -0.0833 | -0.3958 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | rbf_svm | -0.0833 | -0.2500 | 0.0833 | -0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | random_forest | -0.0208 | -0.1250 | 0.0833 | -0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | linear_svm | 0.0417 | 0.0000 | 0.0833 | -0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | logistic_regression | 0.0417 | 0.0000 | 0.0833 | -0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | linear_svm | -0.0625 | -0.1250 | 0.0000 | -0.0729 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | random_forest | -0.0625 | -0.1250 | 0.0000 | -0.0625 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | rbf_svm | -0.0625 | -0.1250 | 0.0000 | -0.0625 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | iqp | logistic_regression | -0.1042 | -0.1250 | -0.0833 | -0.0729 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | qksvm | angle | rbf_svm | -0.0556 | -0.1111 | 0.0000 | -0.0303 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | qksvm | angle | linear_svm | -0.0556 | -0.1111 | 0.0000 | -0.1111 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | qksvm | angle | logistic_regression | -0.0556 | -0.1111 | 0.0000 | -0.1515 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | qksvm | angle | random_forest | 0.0000 | 0.0000 | 0.0000 | -0.0101 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | 0.0208 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | rbf_svm | 0.0000 | 0.0000 | 0.0000 | 0.0104 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | linear_svm | -0.0208 | -0.1250 | 0.0833 | 0.0312 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | random_forest | 0.0417 | 0.0000 | 0.0833 | 0.0208 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | qksvm | angle | logistic_regression | 0.0133 | 0.0667 | -0.0400 | -0.0320 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | qksvm | angle | linear_svm | -0.0200 | 0.0000 | -0.0400 | -0.0533 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | qksvm | angle | rbf_svm | -0.0200 | 0.0000 | -0.0400 | -0.0560 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | qksvm | angle | random_forest | 0.0133 | 0.0667 | -0.0400 | -0.0480 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | 0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | linear_svm | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | rbf_svm | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | random_forest | -0.0625 | -0.1250 | 0.0000 | 0.0000 |
| smoke | iris | full | 42 | 0 | 2 | qksvm | angle | rbf_svm | -0.0200 | -0.0400 | 0.0000 | -0.0016 |
| smoke | iris | full | 42 | 0 | 2 | qksvm | angle | logistic_regression | -0.0200 | -0.0400 | 0.0000 | -0.0016 |
| smoke | iris | full | 42 | 0 | 2 | qksvm | angle | random_forest | -0.0200 | -0.0400 | 0.0000 | -0.0016 |
| smoke | iris | full | 42 | 0 | 2 | qksvm | angle | linear_svm | -0.0200 | -0.0400 | 0.0000 | -0.0016 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | random_forest | -0.1667 | -0.5000 | 0.1667 | -0.1458 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | rbf_svm | -0.1250 | -0.2500 | 0.0000 | -0.1771 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | logistic_regression | -0.1875 | -0.3750 | 0.0000 | -0.1458 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | linear_svm | -0.1875 | -0.3750 | 0.0000 | -0.1458 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | qksvm | angle | linear_svm | -0.2917 | -0.2500 | -0.3333 | -0.2083 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | qksvm | angle | rbf_svm | -0.2917 | -0.2500 | -0.3333 | -0.2083 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | qksvm | angle | random_forest | -0.0833 | 0.0000 | -0.1667 | -0.1667 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | qksvm | angle | logistic_regression | -0.2917 | -0.2500 | -0.3333 | -0.2083 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | vqc | angle | linear_svm | -0.5833 | -1.0000 | -0.1667 | -0.7500 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | vqc | angle | rbf_svm | -0.5833 | -1.0000 | -0.1667 | -0.7500 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | vqc | angle | random_forest | -0.3750 | -0.7500 | 0.0000 | -0.7083 |
| wdbc_phase1_checkpoint | wdbc | 50 | 42 | 0 | 4 | vqc | angle | logistic_regression | -0.5833 | -1.0000 | -0.1667 | -0.7500 |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | vqc | angle | logistic_regression | -0.2197 | -0.6667 | 0.2273 | -0.4167 |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | vqc | angle | random_forest | -0.1187 | -0.5556 | 0.3182 | -0.3283 |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | vqc | angle | linear_svm | -0.1919 | -0.6111 | 0.2273 | -0.4217 |
| heart_phase1_checkpoint | heart | 200 | 42 | 0 | 4 | vqc | angle | rbf_svm | -0.2424 | -0.6667 | 0.1818 | -0.3965 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | linear_svm | 0.0000 | 0.0000 | 0.0000 | -0.0521 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | random_forest | -0.0208 | -0.1250 | 0.0833 | -0.0521 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | rbf_svm | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | -0.0417 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | random_forest | -0.2500 | -0.5000 | 0.0000 | -0.2083 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | rbf_svm | -0.2917 | -0.5000 | -0.0833 | -0.2188 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | logistic_regression | -0.2917 | -0.5000 | -0.0833 | -0.2083 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | linear_svm | -0.3125 | -0.6250 | 0.0000 | -0.1979 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | logistic_regression | -0.2708 | -0.6250 | 0.0833 | -0.2083 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | random_forest | -0.2708 | -0.6250 | 0.0833 | -0.2188 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | linear_svm | -0.2500 | -0.5000 | 0.0000 | -0.2292 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | iqp | rbf_svm | -0.3125 | -0.6250 | 0.0000 | -0.2292 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | linear_svm | 0.0417 | 0.0000 | 0.0833 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | random_forest | 0.0417 | 0.0000 | 0.0833 | 0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | rbf_svm | 0.0417 | 0.0000 | 0.0833 | 0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 2 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | linear_svm | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | rbf_svm | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | 0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | random_forest | -0.0625 | -0.1250 | 0.0000 | 0.0000 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | vqc | angle | rbf_svm | -0.1212 | -0.3333 | 0.0909 | -0.0808 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | vqc | angle | linear_svm | -0.1212 | -0.3333 | 0.0909 | -0.1616 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | vqc | angle | logistic_regression | -0.1212 | -0.3333 | 0.0909 | -0.2020 |
| heart_phase1_checkpoint | heart | 100 | 42 | 0 | 4 | vqc | angle | random_forest | -0.0657 | -0.2222 | 0.0909 | -0.0606 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | logistic_regression | 0.0417 | 0.0000 | 0.0833 | 0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | linear_svm | 0.0417 | 0.0000 | 0.0833 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | rbf_svm | 0.0417 | 0.0000 | 0.0833 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | iqp | random_forest | -0.0208 | -0.1250 | 0.0833 | 0.0000 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | vqc | angle | random_forest | 0.2000 | 0.0000 | 0.4000 | 0.0800 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | vqc | angle | linear_svm | 0.0000 | -0.2000 | 0.2000 | 0.0000 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | vqc | angle | logistic_regression | 0.2000 | 0.0000 | 0.4000 | 0.0800 |
| heart_phase1_checkpoint | heart | 50 | 42 | 0 | 4 | vqc | angle | rbf_svm | 0.0000 | -0.2000 | 0.2000 | 0.0000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | logistic_regression | -0.0625 | -0.1250 | 0.0000 | -0.0417 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | random_forest | -0.0625 | -0.1250 | 0.0000 | -0.0521 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | linear_svm | -0.0417 | 0.0000 | -0.0833 | -0.0625 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 8 | qksvm | angle | rbf_svm | -0.1042 | -0.1250 | -0.0833 | -0.0625 |
| smoke | iris | full | 42 | 0 | 2 | vqc | angle | rbf_svm | -0.5200 | -0.4800 | -0.5600 | -0.4608 |
| smoke | iris | full | 42 | 0 | 2 | vqc | angle | logistic_regression | -0.5200 | -0.4800 | -0.5600 | -0.4608 |
| smoke | iris | full | 42 | 0 | 2 | vqc | angle | random_forest | -0.5200 | -0.4800 | -0.5600 | -0.4608 |
| smoke | iris | full | 42 | 0 | 2 | vqc | angle | linear_svm | -0.5200 | -0.4800 | -0.5600 | -0.4608 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | vqc | angle | logistic_regression | -0.4200 | -0.6000 | -0.2400 | -0.4000 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | vqc | angle | linear_svm | -0.4533 | -0.6667 | -0.2400 | -0.4213 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | vqc | angle | rbf_svm | -0.4533 | -0.6667 | -0.2400 | -0.4240 |
| wdbc_phase1_checkpoint | wdbc | 200 | 42 | 0 | 4 | vqc | angle | random_forest | -0.4200 | -0.6000 | -0.2400 | -0.4160 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | vqc | angle | logistic_regression | -0.4792 | -0.8750 | -0.0833 | -0.5000 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | vqc | angle | rbf_svm | -0.4792 | -0.8750 | -0.0833 | -0.5104 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | vqc | angle | linear_svm | -0.5000 | -1.0000 | 0.0000 | -0.4896 |
| wdbc_phase1_checkpoint | wdbc | 100 | 42 | 0 | 4 | vqc | angle | random_forest | -0.4375 | -0.8750 | 0.0000 | -0.5000 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | random_forest | 0.0417 | 0.0000 | 0.0833 | 0.0208 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | rbf_svm | 0.0000 | 0.0000 | 0.0000 | 0.0104 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | 0.0208 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | angle | linear_svm | -0.0208 | -0.1250 | 0.0833 | 0.0312 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | random_forest | 0.0208 | -0.1250 | 0.1667 | -0.0312 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | rbf_svm | 0.0625 | 0.1250 | 0.0000 | -0.0625 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | logistic_regression | 0.0000 | 0.0000 | 0.0000 | -0.0312 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 6 | qksvm | angle | linear_svm | 0.0000 | 0.0000 | 0.0000 | -0.0312 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | linear_svm | 0.0000 | 0.0000 | 0.0000 | -0.1458 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | rbf_svm | 0.0000 | 0.0000 | 0.0000 | -0.1458 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | logistic_regression | 0.0000 | 0.0000 | 0.0000 | -0.1354 |
| wdbc_feature_kernel_checkpoint | wdbc | 100 | 42 | 0 | 4 | qksvm | iqp | random_forest | -0.0625 | -0.1250 | 0.0000 | -0.1458 |

## Selected-configuration bootstrap intervals

These deterministic 2,000-resample intervals bootstrap saved predictions within
class for the strongest observed WDBC QML and classical configurations. They are
descriptive uncertainty checks; they do not replace repeated cross-validation or
external clinical validation.

| family | experiment | model | sample_size | prediction_count | balanced_accuracy_pooled | balanced_accuracy_bootstrap_low | balanced_accuracy_bootstrap_high | sensitivity_pooled | sensitivity_bootstrap_low | sensitivity_bootstrap_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qml | wdbc_feature_kernel_checkpoint | qksvm | 100 | 20 | 0.9375 | 0.8125 | 1.0000 | 0.8750 | 0.6250 | 1.0000 |
| classical_matched | wdbc_feature_kernel_checkpoint | logistic_regression | 100 | 20 | 0.9375 | 0.8125 | 1.0000 | 0.8750 | 0.6250 | 1.0000 |

The uncertainty columns in `results/tables/summary-metrics.csv` use two-sided 95%
t-intervals across the available fold records. Repeated-CV folds are correlated, so
these intervals are descriptive and must not be presented as independent clinical
validation.

## Failed and incomplete experiments

Every failure is retained in `results/tables/failed-runs.csv`. Missing planned runs
must be described as incomplete evidence, not silently treated as negative results.

## Interpretation and threats to validity

- These are small public tabular benchmarks and simulator experiments.
- Feature reduction can remove clinically useful classical information.
- Kernel simulation cost is not QPU runtime, and model parameters are not total cost.
- Model selection on WDBC is exploratory; Heart Disease provides a limited transfer
  check, not external clinical validation.
- The checked-in checkpoint evidence uses one seed and one fixed fold to bound
  simulator runtime. The three-seed, five-fold configs are pre-registered but their
  unexecuted combinations must not be described as completed evidence.
- No result here alone establishes quantum advantage or clinical utility.
