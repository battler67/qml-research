# Quantum scaling and advantage limits

For `n` QSVM training participants, the symmetric training Gram matrix requires approximately
`n(n+1)/2` unique pair evaluations. Validation and test each add `n_reference * n_evaluation`
evaluations. Kernel costs are estimated before execution, capped by configuration, and stored with
matrix dimensions and actual counts. A 24 GB workstation does not make a full large quantum kernel
cheap: time grows quadratically in training samples even when the statevector itself fits in memory.

Five questions must remain separate:

1. Predictive utility: paired test AUROC, AUPRC, calibration, and uncertainty.
2. Sample efficiency: stability across controlled training sizes and seeds.
3. Runtime/memory utility: wall time, peak process memory, and circuit evaluations.
4. Hardware utility: behaviour under shots/noise and real-device constraints.
5. Asymptotic advantage: scaling evidence against the strongest known classical methods.

The bounded grid is 50, 100, 200, 300, and 500 training participants crossed with 4, 6, 8, and 10
features, but each cell runs only after estimation. QSVM is paired with linear and RBF SVM; QMLP is
paired with a classical MLP on identical data. Reduced and full-feature comparisons remain labelled
separately. Statevector simulation cannot demonstrate computational quantum advantage. A better
metric on one small split is not quantum advantage, and overlapping intervals do not justify a
superiority claim. Feature-selection gains must also be available to matched classical controls.
