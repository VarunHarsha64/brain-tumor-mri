"""Export the trained PyTorch model to ONNX for fast CPU inference.

Example:
    python -m src.export_onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint
from .utils import DEFAULT_CHECKPOINT, DEFAULT_ONNX


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out", default=DEFAULT_ONNX)
    args = ap.parse_args()

    model, meta = load_checkpoint(args.checkpoint, "cpu")
    size = int(meta.get("img_size", 224))
    dummy = torch.randn(1, 3, size, size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:  # newer torch versions: use the stable (non-dynamo) exporter
        torch.onnx.export(model, dummy, args.out, dynamo=False, **kwargs)
    except TypeError:  # older torch has no 'dynamo' argument
        torch.onnx.export(model, dummy, args.out, **kwargs)
    print(f"Exported -> {args.out}")

    # Save class names next to the model so the ONNX file can be used standalone.
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps({"class_names": meta["class_names"], "img_size": size,
                                     "model_name": meta["model_name"]}, indent=2))

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        with torch.no_grad():
            torch_out = model(dummy).numpy()
        diff = float(np.abs(onnx_out - torch_out).max())
        print(f"Verification: max |ONNX - PyTorch| = {diff:.2e}  ({'OK' if diff < 1e-3 else 'CHECK!'})")
    except ImportError:
        print("onnxruntime not installed; skipped verification.")


if __name__ == "__main__":
    main()
