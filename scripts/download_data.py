"""Download the Brain Tumor MRI dataset from Kaggle (needs a free Kaggle account + API token).

One-time setup:
  1. kaggle.com -> your profile picture -> Settings -> "Create New Token" (downloads kaggle.json)
  2. Put kaggle.json in:
       Windows : C:\\Users\\<you>\\.kaggle\\kaggle.json
       macOS/Linux : ~/.kaggle/kaggle.json   (then: chmod 600 ~/.kaggle/kaggle.json)

Usage:
    python scripts/download_data.py

Prefer a manual download? Get the zip from
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
and extract it so you have data/brain-tumor-mri-dataset/Training and .../Testing
"""
import argparse
import sys
from pathlib import Path

SLUG = "masoudnickparvar/brain-tumor-mri-dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/brain-tumor-mri-dataset")
    args = ap.parse_args()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The 'kaggle' package is missing. Run: pip install kaggle")
    except OSError as e:  # the kaggle package raises this on import when credentials are missing
        sys.exit(f"Kaggle credentials not found: {e}\nSee the instructions at the top of this file.")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(f"Kaggle authentication failed: {e}\nSee the instructions at the top of this file.")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SLUG} -> {dest} (about 150 MB) ...")
    api.dataset_download_files(SLUG, path=str(dest), unzip=True, quiet=False)
    print("Done. Folder contents:", sorted(p.name for p in dest.iterdir()))


if __name__ == "__main__":
    main()
