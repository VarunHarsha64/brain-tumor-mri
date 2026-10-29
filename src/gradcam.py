"""Grad-CAM: highlights the image regions that drove the model's prediction."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._on_grad)

    def _on_grad(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: [1,3,H,W]. Returns (cam [H,W] in 0..1 as numpy, logits [1,C] detached)."""
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # guarantees a graph even if params are frozen
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), logits.detach()


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """rgb: uint8 [H,W,3]; cam: float [h,w] in 0..1. Returns uint8 RGB heat-map overlay."""
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam.astype(np.float32), (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)
