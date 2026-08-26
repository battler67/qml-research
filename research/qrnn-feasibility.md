# QRNN feasibility for BreastMNIST

Decision: document only; do not implement in the initial experiment.

BreastMNIST contains static 2D ultrasound images. A QRNN requires an ordered sequence whose recurrence has domain meaning. Raster order or an arbitrary list of image patches would impose an order that is not part of the diagnosis task and can change under flips, translations, or crop choices.

The supplied QNN comparison constructs image sequences from regional averages, but its evidence is MNIST/CIFAR rather than breast ultrasound and it does not establish that the ordering is clinically meaningful. Adding that model now would also dilute the controlled QCNN/classical benchmark and multiply simulator cost.

A later QRNN experiment is justified only if one of these inputs becomes available:

- a temporal ultrasound sweep with preserved frame order;
- longitudinal imaging from the same subject with leakage-safe patient grouping; or
- a preregistered anatomical scan path validated as meaningful independently of labels.

Until then, QCNN spatial pooling is the defensible quantum inductive bias. QViT is excluded by the task requirements.
