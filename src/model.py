"""Model construction (transfer learning), freezing helpers and checkpoint loading."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ("efficientnet_b0", "resnet18", "mobilenet_v3_small")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}")

    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    else:  # mobilenet_v3_small
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


def _head(model: nn.Module, name: str) -> nn.Module:
    return model.fc if name == "resnet18" else model.classifier


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """trainable=False -> only the new classifier head learns (stage 1).
    trainable=True  -> whole network learns (stage 2, fine-tuning)."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in _head(model, name).parameters():
        p.requires_grad = True


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block, used by Grad-CAM."""
    if name == "resnet18":
        return model.layer4[-1]
    return model.features[-1]  # efficientnet_b0 / mobilenet_v3_small


def load_checkpoint(path: str, device: torch.device | str = "cpu"):
    """Rebuild the model from a checkpoint saved by train.py. Returns (model, meta)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
