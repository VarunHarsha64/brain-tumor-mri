"""File discovery and de-duplication (no torch needed here)."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image

from .utils import IMG_EXTS


def _find_dir(parent: Path, name: str) -> Path | None:
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def find_split_dirs(data_dir: str | Path) -> tuple[Path, Path]:
    """Locate the Training/ and Testing/ folders, even if nested one or two levels down."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Data folder '{root}' not found. Download the dataset first "
            "(see README, step 3) or run scripts/make_dummy_data.py for a quick test."
        )
    level1 = [p for p in root.iterdir() if p.is_dir()]
    level2 = [q for p in level1 for q in p.iterdir() if q.is_dir()]
    for c in [root] + level1 + level2:
        tr, te = _find_dir(c, "Training"), _find_dir(c, "Testing")
        if tr and te:
            return tr, te
    raise FileNotFoundError(
        f"Could not find 'Training' and 'Testing' folders inside '{root}'. "
        "Expected: <data_dir>/Training/<class>/*.jpg and <data_dir>/Testing/<class>/*.jpg"
    )


def list_images(split_dir: str | Path) -> list[tuple[str, str]]:
    """Return [(path, class_name), ...] for <split_dir>/<class_name>/*.img"""
    items = []
    for cls_dir in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        for f in sorted(cls_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                items.append((str(f), cls_dir.name))
    return items


def image_hash(path: str, size: int = 64) -> str:
    """Hash of a 64x64 grayscale thumbnail: catches exact copies and re-saved/resized copies."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return hashlib.md5(im.tobytes()).hexdigest()


def deduplicate(train_items, test_items):
    """Drop duplicates inside the training set and any training image that also appears in the test set.

    The test set is never modified. Returns (clean_train_items, stats_dict).
    """
    test_hashes = {image_hash(p) for p, _ in test_items}
    seen: set[str] = set()
    clean, dup_within, dup_leak = [], 0, 0
    for path, label in train_items:
        h = image_hash(path)
        if h in test_hashes:
            dup_leak += 1
            continue
        if h in seen:
            dup_within += 1
            continue
        seen.add(h)
        clean.append((path, label))
    stats = {
        "train_before": len(train_items),
        "train_after": len(clean),
        "duplicates_within_train": dup_within,
        "train_images_also_in_test": dup_leak,
    }
    return clean, stats


def limit_per_class(items, max_per_class: int | None, seed: int = 42):
    """Optional subsampling for quick experiments on a slow laptop."""
    if not max_per_class:
        return items
    rng = random.Random(seed)
    by_cls: dict[str, list] = {}
    for it in items:
        by_cls.setdefault(it[1], []).append(it)
    out = []
    for cls in sorted(by_cls):
        lst = by_cls[cls][:]
        rng.shuffle(lst)
        out.extend(lst[:max_per_class])
    return out
