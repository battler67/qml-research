"""Deterministic QCNN and image-model training loops."""

from __future__ import annotations

import copy
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qml_research.qcnn.config import QCNNConfig
from qml_research.qcnn.models import prepare_image_tensor
from qml_research.qcnn.quantum import HierarchicalQCNN


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    stopping_reason: str
    training_time_seconds: float
    peak_memory_bytes: int


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _positive_weight(y: np.ndarray) -> torch.Tensor:
    positives = max(int(np.sum(y == 1)), 1)
    negatives = max(int(np.sum(y == 0)), 1)
    return torch.tensor(negatives / positives, dtype=torch.float64)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def predict_qcnn(model: HierarchicalQCNN, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float64))
        return torch.sigmoid(logits).detach().cpu().numpy().astype(float)


def train_qcnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: QCNNConfig,
    seed: int,
) -> TrainingResult:
    seed_everything(seed)
    model = HierarchicalQCNN(
        config.qubits,
        backend=config.backend,
        shots=config.shots,
        gradient_method=config.gradient_method,
        init_scale=config.init_scale,
        seed=seed,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(y_train))
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float64),
        torch.as_tensor(y_train, dtype=torch.float64),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0
    )
    validation_X = torch.as_tensor(X_validation, dtype=torch.float64)
    validation_y = torch.as_tensor(y_validation, dtype=torch.float64)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    stopping_reason = "epoch_limit"
    tracemalloc.start()
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        gradient_values: list[float] = []
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            gradients = [
                parameter.grad.detach().reshape(-1)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if gradients:
                gradient_values.extend(torch.cat(gradients).cpu().numpy().tolist())
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_X)
            validation_loss = float(criterion(validation_logits, validation_y))
            validation_scores = torch.sigmoid(validation_logits).cpu().numpy()
        gradient_array = np.asarray(gradient_values, dtype=float)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "validation_loss": validation_loss,
                "validation_auroc": _safe_auc(y_validation, validation_scores),
                "gradient_norm": float(np.linalg.norm(gradient_array)),
                "gradient_variance": float(np.var(gradient_array)),
                "circuit_executions": model.circuit_executions,
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            stopping_reason = "early_stopping"
            break
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    model.load_state_dict(best_state)
    return TrainingResult(model, history, best_epoch, stopping_reason, elapsed, int(peak))


def predict_image_model(
    model: nn.Module,
    images: np.ndarray,
    representation: str,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor = prepare_image_tensor(images, representation)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = normalize_binary_logits(model(batch.to(device)))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(float)


def normalize_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize binary heads while preserving a one-sample batch dimension."""
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim != 1:
        raise ValueError(f"binary model must return [batch] or [batch, 1], got {logits.shape}")
    return logits


def train_image_model(
    model: nn.Module,
    representation: str,
    train_images: np.ndarray,
    y_train: np.ndarray,
    validation_images: np.ndarray,
    y_validation: np.ndarray,
    config: QCNNConfig,
    seed: int,
) -> TrainingResult:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_X = prepare_image_tensor(train_images, representation)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=config.image_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=config.num_workers,
    )
    learning_rate = (
        config.fine_tune_learning_rate
        if config.pretrained_mode == "partial_finetune" and representation == "imagenet_224"
        else config.image_learning_rate
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    positive_weight = _positive_weight(y_train).to(dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    validation_X = prepare_image_tensor(validation_images, representation).to(device)
    validation_y = torch.as_tensor(y_validation, dtype=torch.float32, device=device)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    stopping_reason = "epoch_limit"
    tracemalloc.start()
    started = time.perf_counter()
    for epoch in range(1, config.image_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = normalize_binary_logits(model(batch_X.to(device)))
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits = normalize_binary_logits(model(validation_X))
            validation_loss = float(criterion(validation_logits, validation_y).cpu())
            validation_scores = torch.sigmoid(validation_logits).cpu().numpy()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "validation_auroc": _safe_auc(y_validation, validation_scores),
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            stopping_reason = "early_stopping"
            break
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    model.load_state_dict(best_state)
    return TrainingResult(model, history, best_epoch, stopping_reason, elapsed, int(peak))
