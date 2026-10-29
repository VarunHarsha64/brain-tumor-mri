"""PyTorch datasets, transforms and data loaders."""
from __future__ import annotations

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .files import deduplicate, find_split_dirs, limit_per_class, list_images
from .utils import IMAGENET_MEAN, IMAGENET_STD


def get_transforms(img_size: int = 224, train: bool = False):
    """Grayscale MRI -> 3 channels, resize, (light) augmentation, ImageNet normalisation."""
    base = [T.Grayscale(num_output_channels=3), T.Resize((img_size, img_size))]
    if train:
        base += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    base += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(base)


class MRIDataset(Dataset):
    def __init__(self, items, class_to_idx, transform=None):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]


def build_train_val_loaders(
    data_dir,
    img_size=224,
    batch_size=32,
    val_split=0.15,
    num_workers=2,
    dedup=True,
    max_per_class=None,
    seed=42,
):
    """Returns train_loader, val_loader, class_names, info_dict."""
    train_dir, test_dir = find_split_dirs(data_dir)
    train_items = list_images(train_dir)
    test_items = list_images(test_dir)
    if not train_items:
        raise RuntimeError(f"No images found in {train_dir}")

    info = {"train_before": len(train_items)}
    if dedup:
        train_items, info = deduplicate(train_items, test_items)
    train_items = limit_per_class(train_items, max_per_class, seed)
    info["train_used"] = len(train_items)

    class_names = sorted({lbl for _, lbl in train_items})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    labels = [lbl for _, lbl in train_items]
    tr_items, va_items = train_test_split(
        train_items, test_size=val_split, stratify=labels, random_state=seed
    )
    info["n_train"], info["n_val"] = len(tr_items), len(va_items)

    tr_ds = MRIDataset(tr_items, class_to_idx, get_transforms(img_size, train=True))
    va_ds = MRIDataset(va_items, class_to_idx, get_transforms(img_size, train=False))

    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kw["persistent_workers"] = True
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False, **kw)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, class_names, info


def build_test_loader(data_dir, class_names, img_size=224, batch_size=32, num_workers=2, max_per_class=None):
    _, test_dir = find_split_dirs(data_dir)
    items = limit_per_class(list_images(test_dir), max_per_class)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    items = [it for it in items if it[1] in class_to_idx]
    ds = MRIDataset(items, class_to_idx, get_transforms(img_size, train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, items
