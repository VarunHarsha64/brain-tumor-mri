"""Create a tiny SYNTHETIC "MRI-like" dataset so you can test the whole pipeline in a minute
(training -> evaluation -> dashboard) before downloading the real data.

The images are fake (a grey ellipse 'brain' with bright blobs); accuracy on them means nothing
medically. Use the real Kaggle dataset for actual results.

Usage:
    python scripts/make_dummy_data.py --out data/dummy --per-class 60
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def make_image(cls: str, rng: np.random.Generator, size: int = 256) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.normal(0, 4), size / 2 + rng.normal(0, 4)
    brain = (((xx - cx) / (size * 0.38)) ** 2 + ((yy - cy) / (size * 0.44)) ** 2) < 1
    img = np.zeros((size, size), np.float32)
    img[brain] = 90 + 25 * np.sin(xx[brain] / 9.0) * np.cos(yy[brain] / 11.0)

    def blob(bx, by, r, val):
        mask = ((xx - bx) ** 2 + (yy - by) ** 2) < r ** 2
        img[mask & brain] = val

    if cls == "glioma":        # large irregular bright mass in the upper hemisphere
        blob(cx + rng.uniform(-50, 50), cy - rng.uniform(20, 60), rng.uniform(22, 36), 215)
        blob(cx + rng.uniform(-50, 50), cy - rng.uniform(20, 60), rng.uniform(12, 20), 235)
    elif cls == "meningioma":  # round bright mass near the skull edge
        ang = rng.uniform(0, 2 * np.pi)
        blob(cx + 0.78 * size * 0.38 * np.cos(ang), cy + 0.78 * size * 0.44 * np.sin(ang),
             rng.uniform(16, 26), 225)
    elif cls == "pituitary":   # small bright mass at the centre-bottom
        blob(cx + rng.normal(0, 3), cy + size * 0.22 + rng.normal(0, 3), rng.uniform(9, 14), 230)
    img += rng.normal(0, 6, img.shape)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return pil.filter(ImageFilter.GaussianBlur(1.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, n in (("Training", args.per_class), ("Testing", max(10, args.per_class // 4))):
        for cls in CLASSES:
            d = Path(args.out) / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_image(cls, rng).save(d / f"{cls}_{i:04d}.jpg", quality=95)
    print(f"Synthetic dataset written to {args.out}")
    print(f"Try: python -m src.train --data-dir {args.out} --no-pretrained --epochs-head 1 --epochs-ft 3 --img-size 128")


if __name__ == "__main__":
    main()
