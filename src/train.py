"""Train a brain-tumour MRI classifier with two-stage transfer learning.

Stage 1: freeze the pretrained backbone, train only the new classifier head.
Stage 2: unfreeze everything and fine-tune with a small learning rate.

Example:
    python -m src.train --data-dir data/brain-tumor-mri-dataset --model efficientnet_b0
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import build_train_val_loaders
from .model import SUPPORTED_MODELS, build_model, set_backbone_trainable
from .utils import DEFAULT_CHECKPOINT, OUTPUT_DIR, get_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, desc=""):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    grad_ctx = torch.enable_grad() if train else torch.inference_mode()
    with grad_ctx:
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            amp = torch.autocast("cuda") if use_amp else contextlib.nullcontext()
            with amp:
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def save_history_plot(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/brain-tumor-mri-dataset")
    ap.add_argument("--model", default="efficientnet_b0", choices=SUPPORTED_MODELS)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=3, help="stage 1 epochs (backbone frozen)")
    ap.add_argument("--epochs-ft", type=int, default=12, help="stage 2 epochs (full fine-tuning)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--max-per-class", type=int, default=None, help="use fewer images (quick tests)")
    ap.add_argument("--no-dedup", action="store_true", help="skip duplicate removal")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline / smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = device.type == "cuda"
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | model: {args.model} | AMP: {use_amp}")

    train_loader, val_loader, class_names, info = build_train_val_loaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        dedup=not args.no_dedup,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print(f"Classes: {class_names}")
    print(f"Data info: {info}")

    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history, best_val_loss, best_epoch, bad_epochs, epoch = [], float("inf"), 0, 0, 0
    t0 = time.time()

    stages = [
        ("head", args.epochs_head, args.lr_head, False),
        ("finetune", args.epochs_ft, args.lr_ft, True),
    ]
    for stage_name, n_epochs, lr, backbone_trainable in stages:
        if n_epochs <= 0:
            continue
        set_backbone_trainable(model, args.model, backbone_trainable)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        bad_epochs = 0  # early-stopping counter restarts for each stage
        print(f"\n=== Stage: {stage_name} | epochs={n_epochs} | lr={lr} | "
              f"trainable params={sum(p.numel() for p in params):,} ===")

        for _ in range(n_epochs):
            epoch += 1
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler,
                                        use_amp, desc=f"epoch {epoch} train")
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp,
                                        desc=f"epoch {epoch} val")
            scheduler.step()
            history.append({"epoch": epoch, "stage": stage_name, "train_loss": tr_loss,
                            "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
            print(f"[{epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}")

            if va_loss < best_val_loss:
                best_val_loss, best_epoch, bad_epochs = va_loss, epoch, 0
                torch.save({
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "epoch": epoch,
                }, args.out)
                print(f"    saved best model -> {args.out}")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"    early stopping (no improvement for {args.patience} epochs)")
                    break

    elapsed = time.time() - t0
    summary = {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
               "train_time_sec": elapsed, "args": vars(args), "data_info": info,
               "class_names": class_names}
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
    if history:
        save_history_plot(history, out_dir / "training_curves.png")
    print(f"\nDone in {elapsed / 60:.1f} min. Best epoch: {best_epoch} (val loss {best_val_loss:.4f}).")
    print("Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
