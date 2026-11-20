"""Official GTSRB loading, ROI crops, track-separated splits, deterministic evaluation."""

import csv
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from occlusion import overlay, KINDS

BASE = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ARCHIVES = (
    "GTSRB_Final_Training_Images.zip",
    "GTSRB_Final_Test_Images.zip",
    "GTSRB_Final_Test_GT.zip",
)


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for filename in ARCHIVES:
        target = root / filename
        if not target.exists():
            print("Downloading", filename, flush=True)
            temp = target.with_suffix(".part")
            try:
                with urllib.request.urlopen(
                    BASE + filename, timeout=60
                ) as src, temp.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        with zipfile.ZipFile(target) as archive:
            for name in archive.namelist():
                resolved = (root / name).resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(root)


def records(root, split):
    root = Path(root)
    if split == "train":
        files = sorted(root.glob("GTSRB/Final_Training/Images/*/GT-*.csv"))
    else:
        files = sorted(root.glob("**/GT-final_test.csv"))
    if not files:
        raise FileNotFoundError("GTSRB missing. Run: python train.py --download")
    rows = []
    for path in files:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter=";"):
                image = (
                    path.parent / row["Filename"]
                    if split == "train"
                    else root / "GTSRB/Final_Test/Images" / row["Filename"]
                )
                rows.append(
                    dict(
                        path=str(image),
                        label=int(row["ClassId"]),
                        track=f"{row['ClassId']}:{row['Filename'].split('_')[0]}",
                        roi=[
                            int(row[k])
                            for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
                        ],
                    )
                )
    return rows


def split_tracks(rows, seed=42):
    """Keep neighboring frames of the same physical sign in the same subset."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        tracks = sorted({r["track"] for r in group})
        if len(tracks) < 2:
            raise ValueError("Need at least two tracks per class")
        rng.shuffle(tracks)
        val_tracks = set(tracks[: max(1, round(0.2 * len(tracks)))])
        train.extend(r for r in group if r["track"] not in val_tracks)
        val.extend(r for r in group if r["track"] in val_tracks)
    return train, val


def limit_per_class(rows, limit, seed=42):
    if limit <= 0:
        return rows
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        selected.extend(group[i] for i in rng.permutation(len(group))[:limit])
    return selected


def crop_record(row):
    with Image.open(row["path"]) as im:
        x1, y1, x2, y2 = row["roi"]
        return (
            im.convert("RGB")
            .crop((x1, y1, x2 + 1, y2 + 1))
            .resize((64, 64), Image.Resampling.BILINEAR)
        )


def tensor(image):
    a = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((a - 0.5) / 0.5)


class SignDataset(Dataset):
    def __init__(self, rows, training=False, seed=42):
        self.rows = rows
        self.training = training
        self.seed = seed
        self.epoch = 0
        # Small crops cached in RAM, not decoded repeatedly each epoch.
        self.images = [crop_record(r) for r in rows]

    def __len__(self):
        return len(self.rows) if self.training else 2 * len(self.rows)

    def __getitem__(self, index):
        base = index if self.training else index // 2
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.epoch if self.training else 0, index]
            )
        )
        im = self.images[base].copy()
        if self.training:
            # No flips: traffic-sign meanings can change under reflection.
            im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.8, 1.2)))
            occluded = int(rng.random() < 0.5)
            kind = str(rng.choice(KINDS))
        else:
            occluded = index % 2
            kind = KINDS[base % 3]
        coverage = 0.0
        if occluded:
            im, coverage = overlay(im, rng, kind)
        return (
            tensor(im),
            self.rows[base]["label"],
            float(occluded),
            kind if occluded else "clean",
            coverage,
        )


def write_manifest(path, train, val, test):
    Path(path).write_text(
        json.dumps({"train": train, "validation": val, "test": test}, indent=2)
    )
