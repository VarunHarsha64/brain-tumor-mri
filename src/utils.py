"""Shared constants and small helpers."""
from __future__ import annotations

import random

import numpy as np

# Default order (alphabetical folder names, which is what the Kaggle dataset uses).
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CHECKPOINT = "models/best_model.pt"
DEFAULT_ONNX = "models/model.onnx"
OUTPUT_DIR = "outputs"


def display_name(cls: str) -> str:
    return DISPLAY_NAMES.get(cls, cls)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str = "auto"):
    """Return a torch.device. 'auto' picks CUDA, then Apple MPS, then CPU."""
    import torch

    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
