"""Classify one or more MRI images from the command line.

Examples:
    python -m src.predict path/to/mri.jpg
    python -m src.predict img1.jpg img2.jpg --gradcam          # also saves heat-map overlays
    python -m src.predict path/to/mri.jpg --backend onnx       # fast CPU inference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .utils import DEFAULT_CHECKPOINT, DEFAULT_ONNX, display_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--gradcam", action="store_true", help="save Grad-CAM overlays (torch backend only)")
    ap.add_argument("--out-dir", default="outputs/predictions")
    args = ap.parse_args()

    if args.backend == "onnx":
        from .inference import OnnxPredictor

        meta = json.loads(Path(args.onnx).with_suffix(".json").read_text())
        predictor = OnnxPredictor(args.onnx, meta["class_names"], meta["img_size"])
    else:
        from .inference import Predictor

        predictor = Predictor(args.checkpoint)

    out_dir = Path(args.out_dir)
    if args.gradcam and args.backend == "torch":
        out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        res = predictor.predict(Image.open(path), with_cam=args.gradcam and args.backend == "torch")
        print(f"\n{path}\n  -> {display_name(res['label'])}  ({res['confidence']:.1%})")
        for c, p in sorted(res["probs"].items(), key=lambda kv: -kv[1]):
            print(f"     {display_name(c):<16} {p:6.1%}")
        if res["overlay"] is not None:
            dst = out_dir / f"{Path(path).stem}_gradcam.png"
            Image.fromarray(res["overlay"]).save(dst)
            print(f"  heat-map saved: {dst}")
    print("\nEducational demo only - not a medical device.")


if __name__ == "__main__":
    main()
