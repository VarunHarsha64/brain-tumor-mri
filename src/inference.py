"""Inference helpers used by the dashboard and the command-line tools.

Predictor      -> PyTorch checkpoint, supports Grad-CAM heat-maps.
OnnxPredictor  -> ONNX Runtime, fastest on CPU, no heat-maps.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import IMAGENET_MEAN, IMAGENET_STD


def preprocess_np(img: Image.Image, img_size: int = 224):
    """PIL image -> (float32 array [1,3,H,W] normalised, uint8 RGB array [H,W,3] for display).

    Same pipeline as training: grayscale -> 3 channels -> resize -> ImageNet normalisation.
    """
    img = img.convert("L").convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
    return x, rgb


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _result(class_names, probs, rgb, cam=None, overlay=None):
    idx = int(np.argmax(probs))
    return {
        "label": class_names[idx],
        "confidence": float(probs[idx]),
        "probs": {c: float(p) for c, p in zip(class_names, probs)},
        "image": rgb,
        "cam": cam,
        "overlay": overlay,
    }


class Predictor:
    def __init__(self, checkpoint: str, device: str = "auto"):
        import torch

        from .gradcam import GradCAM
        from .model import get_target_layer, load_checkpoint
        from .utils import get_device

        self.torch = torch
        self.device = get_device(device)
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.class_names = list(self.meta["class_names"])
        self.img_size = int(self.meta.get("img_size", 224))
        self.model_name = self.meta["model_name"]
        self._cam = GradCAM(self.model, get_target_layer(self.model, self.model_name))

    def predict(self, img: Image.Image, with_cam: bool = False, alpha: float = 0.45):
        from .gradcam import overlay_cam

        x_np, rgb = preprocess_np(img, self.img_size)
        x = self.torch.from_numpy(x_np).to(self.device)
        if with_cam:
            cam, logits = self._cam(x)
            probs = _softmax(logits.cpu().numpy()[0])
            return _result(self.class_names, probs, rgb, cam, overlay_cam(rgb, cam, alpha))
        with self.torch.inference_mode():
            logits = self.model(x)
        probs = _softmax(logits.cpu().numpy()[0])
        return _result(self.class_names, probs, rgb)


class OnnxPredictor:
    def __init__(self, onnx_path: str, class_names: list[str], img_size: int = 224):
        import onnxruntime as ort

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.img_size = img_size

    def predict(self, img: Image.Image, with_cam: bool = False, alpha: float = 0.45):
        x, rgb = preprocess_np(img, self.img_size)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        return _result(self.class_names, _softmax(logits), rgb)
