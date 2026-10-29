"""Streamlit dashboard for the brain-tumour MRI classifier.

Run from the project root:
    streamlit run app/dashboard.py

Tabs: Results | Analyze image
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import DEFAULT_CHECKPOINT, DEFAULT_ONNX, OUTPUT_DIR, display_name  # noqa: E402

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

OUT = ROOT / OUTPUT_DIR
IMG_W = 360  # display width of images in pixels


# --------------------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model ...")
def load_predictor(kind: str, ckpt: str, onnx_path: str):
    if kind == "ONNX Runtime (fast, no heat-map)":
        from src.inference import OnnxPredictor

        meta_file = Path(onnx_path).with_suffix(".json")
        meta = json.loads(meta_file.read_text())
        return OnnxPredictor(onnx_path, meta["class_names"], meta["img_size"])
    from src.inference import Predictor

    return Predictor(ckpt)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def show_prediction(res: dict, threshold: float, show_cam: bool, caption: str | None = None):
    """Render original image, optional Grad-CAM overlay, the predicted class and probabilities."""
    cols = st.columns([2, 2, 2] if show_cam and res.get("overlay") is not None else [2, 3])
    cols[0].image(res["image"], caption=caption or "Input (grayscale, resized)", width=IMG_W)
    if show_cam and res.get("overlay") is not None:
        cols[1].image(res["overlay"], caption="Grad-CAM: regions driving the prediction",
                      width=IMG_W)
    out_col = cols[-1]
    name = display_name(res["label"])
    if res["label"] == "notumor":
        out_col.success(f"**{name}** — {res['confidence']:.1%} confidence")
    else:
        out_col.error(f"**{name}** — {res['confidence']:.1%} confidence")
    if res["confidence"] < threshold:
        out_col.warning(f"Low confidence (< {threshold:.0%}). The image may be out-of-distribution "
                        "(not a brain MRI, or from a different scanner/protocol).")
    for c, p in sorted(res["probs"].items(), key=lambda kv: -kv[1]):
        out_col.progress(min(max(p, 0.0), 1.0), text=f"{display_name(c)}: {p:.1%}")


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.title("🧠 Brain Tumor MRI")
st.sidebar.caption("4-class classifier: glioma · meningioma · pituitary · no tumor")

backend = st.sidebar.radio("Inference backend", ["PyTorch (Grad-CAM available)",
                                                 "ONNX Runtime (fast, no heat-map)"])
backend_key = "ONNX Runtime (fast, no heat-map)" if backend.startswith("ONNX") else "PyTorch"
ckpt_path = st.sidebar.text_input("Checkpoint", str(ROOT / DEFAULT_CHECKPOINT))
onnx_path = st.sidebar.text_input("ONNX file", str(ROOT / DEFAULT_ONNX))
threshold = st.sidebar.slider("Low-confidence warning below", 0.3, 0.95, 0.70, 0.05)
alpha = st.sidebar.slider("Heat-map opacity", 0.1, 0.9, 0.45, 0.05)
st.sidebar.warning("Educational demo only. **Not a medical device** and not for diagnosis.")

predictor = None
model_error = None
try:
    if backend_key == "PyTorch" and not Path(ckpt_path).exists():
        raise FileNotFoundError(f"No trained model at {ckpt_path}")
    if backend_key != "PyTorch" and not Path(onnx_path).exists():
        raise FileNotFoundError(f"No ONNX model at {onnx_path} (run: python -m src.export_onnx)")
    predictor = load_predictor(backend_key, ckpt_path, onnx_path)
except Exception as e:
    model_error = str(e)

can_cam = predictor is not None and backend_key == "PyTorch"


def predict(img: Image.Image, with_cam: bool):
    if with_cam and can_cam:
        return predictor.predict(img, with_cam=True, alpha=alpha)
    return predictor.predict(img)


def need_model() -> bool:
    if predictor is None:
        st.error(f"Model not available: {model_error}")
        st.code("python -m src.train --data-dir data/brain-tumor-mri-dataset\n"
                "python -m src.evaluate --data-dir data/brain-tumor-mri-dataset", language="bash")
        return True
    return False


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
st.title("Brain Tumor MRI Analysis")
tab_results, tab_upload = st.tabs(["📊 Results", "🖼️ Analyze image"])

# ---------------------------------- Results ------------------------------------------
with tab_results:
    metrics = read_json(OUT / "metrics.json")
    hist = read_json(OUT / "history.json")
    if not metrics:
        st.info("No evaluation results yet. Train and evaluate first:")
        st.code("python -m src.train --data-dir data/brain-tumor-mri-dataset\n"
                "python -m src.evaluate --data-dir data/brain-tumor-mri-dataset", language="bash")
    else:
        names = metrics["class_names"]
        rep = metrics["report"]
        recalls = {c: rep[c]["recall"] for c in names}
        weakest = min(recalls, key=recalls.get)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
        m2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        m3.metric("Test images", f"{metrics['n_test']:,}")
        m4.metric("Lowest class recall", f"{recalls[weakest]:.2%}", help=f"Class: {display_name(weakest)}")

        left, right = st.columns(2)
        with left:
            st.subheader("Per-class metrics")
            df = pd.DataFrame(
                [{"Class": display_name(c), "Precision": round(rep[c]["precision"], 4),
                  "Recall": round(rep[c]["recall"], 4), "F1": round(rep[c]["f1-score"], 4),
                  "Images": int(rep[c]["support"])} for c in names])
            st.dataframe(df, hide_index=True)
            st.caption("Recall matters most here: a missed tumour is the costly error.")

            if hist:
                st.subheader("Training curves")
                h = pd.DataFrame(hist["history"]).set_index("epoch")
                c1, c2 = st.columns(2)
                c1.line_chart(h[["train_loss", "val_loss"]])
                c2.line_chart(h[["train_acc", "val_acc"]])
                info = hist.get("data_info", {})
                if info:
                    st.caption(
                        f"Model: {metrics['model_name']} · best epoch {hist['best_epoch']} · "
                        f"train time {hist['train_time_sec'] / 60:.1f} min · "
                        f"duplicates removed: {info.get('duplicates_within_train', 0)} within train, "
                        f"{info.get('train_images_also_in_test', 0)} train/test leaks")
        with right:
            st.subheader("Confusion matrix")
            cm_file = OUT / "confusion_matrix.png"
            if cm_file.exists():
                st.image(str(cm_file), width=480)

        samples = metrics.get("gradcam_samples", [])
        if samples:
            st.subheader("Grad-CAM examples (left: MRI, right: heat-map)")
            only_wrong = st.checkbox("Show only misclassified examples", value=False)
            shown = [s for s in samples if (not s["correct"]) or not only_wrong]
            gcols = st.columns(3)
            for k, s in enumerate(shown):
                f = OUT / "gradcam_samples" / s["file"]
                if f.exists():
                    tag = "✅" if s["correct"] else "❌"
                    gcols[k % 3].image(
                        str(f), width=440,
                        caption=f"{tag} true: {display_name(s['true'])} · "
                                f"pred: {display_name(s['pred'])} ({s['confidence']:.0%})")

# ---------------------------------- Upload -------------------------------------------
with tab_upload:
    st.subheader("Upload an MRI slice")
    up = st.file_uploader("JPG / PNG brain MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if up is not None and not need_model():
        img = Image.open(up)
        with st.spinner("Analysing ..."):
            t0 = time.perf_counter()
            res = predict(img, with_cam=True)
            dt = time.perf_counter() - t0
        show_prediction(res, threshold, show_cam=can_cam)
        st.caption(f"Inference time: {dt * 1000:.0f} ms ({backend_key})")
    elif up is None:
        st.info("Upload an image to classify it. Try an image from the dataset's Testing folder.")