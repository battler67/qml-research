# gated_residual_ckd_confirm_v1

Locked test patients: 80; seeds: [42, 123, 2026].

Values are mean per-seed metrics. ROC-AUC/PR-AUC intervals use a stratified paired patient bootstrap. Sensitivity/specificity use conservative binomial intervals: mean endpoints of seed-wise Clopper-Pearson intervals with Bonferroni coverage across seeds. No seed independence is assumed. Intervals are conditional on these fitted models, not independent cohort or seed evidence.

PR-AUC here means average precision. Thresholds use validation Youden J.

| Family | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | Tuning s/seed | Peak RSS GiB |
| --- | --- | --- | --- | --- | ---: | ---: |
| logistic_original | 0.9980 [0.9913, 1.0000] | 0.9988 [0.9952, 1.0000] | 0.9667 [0.8492, 0.9981] | 1.0000 [0.8525, 1.0000] | 0.21 | 0.439 |
| logistic_reduced | 0.9913 [0.9760, 1.0000] | 0.9952 [0.9871, 1.0000] | 0.9600 [0.8393, 0.9963] | 0.9111 [0.7116, 0.9892] | 0.16 | 0.439 |
| rbf_svm_original | 0.9956 [0.9858, 1.0000] | 0.9974 [0.9920, 1.0000] | 0.9533 [0.8299, 0.9937] | 0.9667 [0.7958, 0.9984] | 0.16 | 0.439 |
| rbf_svm_reduced | 0.9920 [0.9733, 1.0000] | 0.9959 [0.9867, 1.0000] | 0.9667 [0.8492, 0.9981] | 0.9444 [0.7590, 0.9969] | 0.13 | 0.439 |
| random_forest_original | 0.9984 [0.9924, 1.0000] | 0.9991 [0.9957, 1.0000] | 0.9800 [0.8726, 0.9990] | 0.9667 [0.7928, 0.9997] | 8.45 | 0.439 |
| random_forest_reduced | 0.9949 [0.9833, 1.0000] | 0.9972 [0.9912, 1.0000] | 0.9667 [0.8492, 0.9981] | 0.9667 [0.7928, 0.9997] | 7.51 | 0.439 |
| gradient_boosting_original | 0.9978 [0.9920, 1.0000] | 0.9987 [0.9957, 1.0000] | 0.9867 [0.8834, 0.9999] | 0.9556 [0.7862, 0.9910] | 4.23 | 0.440 |
| gradient_boosting_reduced | 0.9924 [0.9769, 1.0000] | 0.9963 [0.9890, 1.0000] | 0.9600 [0.8393, 0.9963] | 0.9667 [0.7958, 0.9984] | 3.17 | 0.440 |
| mlp_original | 0.9980 [0.9909, 1.0000] | 0.9988 [0.9949, 1.0000] | 0.9667 [0.8492, 0.9981] | 0.9889 [0.8326, 0.9999] | 73.89 | 0.440 |
| mlp_reduced | 0.9913 [0.9755, 1.0000] | 0.9952 [0.9868, 1.0000] | 0.9667 [0.8501, 0.9971] | 0.9111 [0.7116, 0.9892] | 75.91 | 0.440 |
| residual_quantum | 0.9984 [0.9931, 1.0000] | 0.9991 [0.9961, 1.0000] | 0.9667 [0.8492, 0.9981] | 1.0000 [0.8525, 1.0000] | 174.57 | 0.438 |
| residual_ungated | 0.9982 [0.9924, 1.0000] | 0.9990 [0.9957, 1.0000] | 0.9667 [0.8492, 0.9981] | 1.0000 [0.8525, 1.0000] | 128.80 | 0.439 |
| residual_separable | 0.9982 [0.9924, 1.0000] | 0.9990 [0.9957, 1.0000] | 0.9667 [0.8492, 0.9981] | 1.0000 [0.8525, 1.0000] | 76.49 | 0.439 |
| residual_frozen | 0.9984 [0.9931, 1.0000] | 0.9991 [0.9961, 1.0000] | 0.9667 [0.8492, 0.9981] | 1.0000 [0.8525, 1.0000] | 61.42 | 0.439 |
| residual_classical | 0.9982 [0.9924, 1.0000] | 0.9990 [0.9957, 1.0000] | 0.9667 [0.8492, 0.9981] | 1.0000 [0.8525, 1.0000] | 33.96 | 0.439 |
| residual_no_quantum | 0.9982 [0.9924, 1.0000] | 0.9990 [0.9957, 1.0000] | 0.9667 [0.8492, 0.9981] | 1.0000 [0.8525, 1.0000] | 30.96 | 0.439 |
| validation_selected_classical | 0.9980 [0.9920, 1.0000] | 0.9988 [0.9955, 1.0000] | 0.9733 [0.8618, 0.9982] | 0.9889 [0.8326, 0.9999] | 0.00 | 0.000 |

## Prespecified paired comparisons

Positive differences favor the proposed quantum residual. Six ROC-AUC comparisons use Bonferroni-adjusted intervals; other metric intervals are descriptive 95%.

| Comparator | ROC-AUC difference [adjusted CI] | Sensitivity difference [95% CI] |
| --- | --- | --- |
| validation_selected_classical | +0.0004 [-0.0022, +0.0044] | -0.0067 [-0.0200, +0.0000] |
| residual_ungated | +0.0002 [+0.0000, +0.0022] | +0.0000 [+0.0000, +0.0000] |
| residual_separable | +0.0002 [+0.0000, +0.0022] | +0.0000 [+0.0000, +0.0000] |
| residual_frozen | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| residual_classical | +0.0002 [+0.0000, +0.0022] | +0.0000 [+0.0000, +0.0000] |
| residual_no_quantum | +0.0002 [+0.0000, +0.0022] | +0.0000 [+0.0000, +0.0000] |

## Conclusion

The prespecified predictive signal rule was not met; no reliable improvement is established.
This rule requires ROC-AUC gain >=0.01, adjusted interval above zero, and sensitivity-difference lower bound >=-0.02. It is not a computational advantage claim.

Validation-selected classical families by seed: ['random_forest_original', 'logistic_original', 'logistic_original'].
Costs include all tuning trials; shared preprocessing/one fixed logistic-anchor fit is recorded separately per seed. Peak RSS includes the interpreter and libraries. The selected-reference row has no standalone cost (zero is a placeholder, not free training).
