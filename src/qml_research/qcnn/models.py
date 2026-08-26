"""Controlled classical and practical image baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch import nn


class PretrainedWeightsUnavailable(RuntimeError):
    """Raised when a profile forbids the required first-time weights download."""


def make_reduced_classical_model(name: str, seed: int) -> Any:
    if name == "logistic":
        return LogisticRegression(
            class_weight="balanced", max_iter=2_000, random_state=seed, solver="liblinear"
        )
    if name == "linear_svm":
        return SVC(kernel="linear", class_weight="balanced", probability=True, random_state=seed)
    if name == "rbf_svm":
        return SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
    if name == "reduced_mlp":
        return MLPClassifier(
            hidden_layer_sizes=(8, 4),
            activation="relu",
            alpha=1e-4,
            max_iter=1_000,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"not a reduced-feature classical model: {name}")


def classical_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    raw = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-raw))


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images)).squeeze(-1)


def build_image_model(
    name: str, *, pretrained_download: bool, pretrained_mode: str
) -> tuple[nn.Module, str]:
    if name == "small_cnn":
        return SmallCNN(), "native_28x28"
    try:
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            ResNet18_Weights,
            mobilenet_v3_small,
            resnet18,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('image benchmarks require `pip install -e ".[qcnn]"`') from exc

    pretrained = name.endswith("_imagenet")
    if pretrained and not pretrained_download:
        raise PretrainedWeightsUnavailable(
            f"{name} needs pinned ImageNet weights; enable pretrained_download"
        )
    if name in {"resnet18_random", "resnet18_imagenet"}:
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, 1)
        if pretrained:
            _configure_transfer(model, model.fc, pretrained_mode, partial_module=model.layer4)
        return model, "imagenet_224"
    if name == "mobilenet_v3_small_imagenet":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, 1)
        _configure_transfer(
            model, model.classifier, pretrained_mode, partial_module=model.features[-3:]
        )
        return model, "imagenet_224"
    raise ValueError(f"not an image model: {name}")


def _configure_transfer(
    model: nn.Module,
    classifier: nn.Module,
    mode: str,
    *,
    partial_module: nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    if mode == "partial_finetune":
        for parameter in partial_module.parameters():
            parameter.requires_grad = True


def prepare_image_tensor(images: np.ndarray, representation: str) -> torch.Tensor:
    tensor = torch.as_tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    if representation == "native_28x28":
        return tensor
    if representation != "imagenet_224":
        raise ValueError(f"unknown image representation: {representation}")
    tensor = functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = tensor.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
