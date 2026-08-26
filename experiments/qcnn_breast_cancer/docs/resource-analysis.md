# QCNN resource and hardware analysis

For `q` statevector qubits, state storage grows as `O(2^q)`. A complex128 state alone is 256 bytes at four qubits and 4,096 bytes at eight qubits; simulator intermediates, differentiation tapes, Python/Torch objects, images, and classical models dominate actual RAM here. The practical qubit limit is driven more by repeated differentiated circuit evaluations than by the bare statevector.

The shared hierarchy has 38 total trainable parameters at four qubits and 56 at eight. Analytic adjoint differentiation avoids the approximately two shifted executions per quantum parameter required by parameter-shift gradients. Mini-batches bound retained Torch graphs, but each sample remains a circuit evaluation.

Runtime scales approximately with `((train + validation) x epochs + validation + test) x seeds x folds`. The estimate counts validation during training plus the final validation/test predictions and uses a conservative CPU calibration from the verified smoke run. Depth and two-qubit gates grow with convolution/pooling stages; `qml.specs` records device-level values. The CLI blocks more than 25,000 estimated quantum forwards, more than two projected hours, any eight-qubit run, or a noisy sweep until `--confirm-expensive` is present.

Analytic results contain neither shot noise nor device noise. The optional `default.mixed` profile is a controlled sensitivity test, not a calibrated hardware model. Real execution additionally requires provider selection, native-basis/connectivity transpilation, queue/cost review, recalculated two-qubit depth, credentials outside Git, and separate authorization.

Shallow local blocks, local measurement, small initialization, early stopping, and recorded gradient variance reduce—but do not eliminate—barren-plateau and trainability risks. A favorable simulator score or smaller parameter count is not computational quantum advantage.
