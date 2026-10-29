"""Evaluate the trained model on the held-out Testing folder.

Writes to outputs/:
  metrics.json, confusion_matrix.png, test_predictions.csv, gradcam_samples/*.png
The Streamlit dashboard reads these files.

Example:
    python -m src.evaluate --data-dir data/brain-tumor-mri-dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from .data import build_test_loader
from .inference import Predictor
from .model import load_checkpoint
from .utils import DEFAULT_CHECKPOINT, OUTPUT_DIR, display_name, get_device


@torch.inference_mode()
def predict_all(model, loader, device):
    probs_all, y_all = [], []
    for x, y in tqdm(loader, desc="testing"):
        logits = model(x.to(device))
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(probs_all), np.concatenate(y_all)


def save_confusion_matrix(cm, class_names, path):
    labels = [display_name(c) for c in class_names]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_gradcam_samples(predictor, items, preds, out_dir, per_class=3, n_wrong=6):
    """Save side-by-side (original | Grad-CAM) images for correct and wrong predictions."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    class_names = predictor.class_names
    chosen, count = [], {c: 0 for c in class_names}
    wrong = 0
    for i, ((path, true), pred) in enumerate(zip(items, preds)):
        pred_name = class_names[pred]
        if pred_name == true and count[true] < per_class:
            count[true] += 1
            chosen.append(i)
        elif pred_name != true and wrong < n_wrong:
            wrong += 1
            chosen.append(i)

    meta = []
    for k, i in enumerate(chosen):
        path, true = items[i]
        res = predictor.predict(Image.open(path), with_cam=True)
        pair = np.concatenate([res["image"], res["overlay"]], axis=1)
        fname = f"{k:02d}_true-{true}_pred-{res['label']}.png"
        Image.fromarray(pair).save(out_dir / fname)
        meta.append({"file": fname, "true": true, "pred": res["label"],
                     "confidence": res["confidence"], "correct": res["label"] == true})
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/brain-tumor-mri-dataset")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--gradcam-per-class", type=int, default=3)
    args = ap.parse_args()

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, device)
    class_names = list(meta["class_names"])
    img_size = int(meta.get("img_size", 224))

    loader, items = build_test_loader(args.data_dir, class_names, img_size, args.batch_size,
                                      args.num_workers, args.max_per_class)
    print(f"Test images: {len(items)} | classes: {class_names}")
    probs, y_true = predict_all(model, loader, device)
    y_pred = probs.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    save_confusion_matrix(cm, class_names, out / "confusion_matrix.png")

    pd.DataFrame({
        "path": [p for p, _ in items],
        "true": [class_names[i] for i in y_true],
        "pred": [class_names[i] for i in y_pred],
        "confidence": probs.max(1),
    }).to_csv(out / "test_predictions.csv", index=False)

    print("Generating Grad-CAM samples ...")
    predictor = Predictor(args.checkpoint, args.device)
    samples = save_gradcam_samples(predictor, items, y_pred, out / "gradcam_samples",
                                   per_class=args.gradcam_per_class)

    metrics = {
        "model_name": meta["model_name"],
        "n_test": len(items),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "class_names": class_names,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "gradcam_samples": samples,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nTest accuracy: {acc:.4f} | macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    print("Per-class recall matters most (missing a tumour is the costly error).")
    print("Saved results to outputs/. Next: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
