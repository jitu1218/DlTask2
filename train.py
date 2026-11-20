"""Train and evaluate a custom two-head CNN on GTSRB with synthetic occlusions."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)
from model import SignOcclusionCNN
from data import (
    download,
    records,
    split_tracks,
    limit_per_class,
    SignDataset,
    write_manifest,
)


def collect(model, loader, device):
    model.eval()
    targets, classes, scores, predictions, kinds, coverages = [], [], [], [], [], []
    total = 0.0
    with torch.no_grad():
        for images, labels, occluded, kind, coverage in loader:
            images, labels, occluded = (
                images.to(device),
                labels.to(device),
                occluded.float().to(device),
            )
            sign_logits, occ_logits = model(images)
            loss = nn.functional.cross_entropy(
                sign_logits, labels
            ) + nn.functional.binary_cross_entropy_with_logits(occ_logits, occluded)
            total += loss.item() * len(images)
            targets.extend(occluded.cpu().tolist())
            classes.extend(labels.cpu().tolist())
            scores.extend(occ_logits.sigmoid().cpu().tolist())
            predictions.extend(sign_logits.argmax(1).cpu().tolist())
            kinds.extend(kind)
            coverages.extend(coverage.tolist())
    return dict(
        loss=total / len(targets),
        target=np.array(targets, dtype=int),
        true_sign=np.array(classes),
        score=np.array(scores),
        sign=np.array(predictions),
        kind=np.array(kinds),
        coverage=np.array(coverages),
    )


def choose_threshold(result):
    # Validation only. Prefer the highest F1; ties use the lower threshold.
    grid = np.linspace(0.05, 0.95, 91)
    values = [
        f1_score(result["target"], result["score"] >= t, zero_division=0) for t in grid
    ]
    return float(grid[int(np.argmax(values))])


def metrics(result, threshold):
    y = result["target"]
    pred = (result["score"] >= threshold).astype(int)
    result_metrics = dict(
        threshold=threshold,
        sample_count=len(y),
        occlusion_precision=float(precision_score(y, pred, zero_division=0)),
        occlusion_recall=float(recall_score(y, pred, zero_division=0)),
        occlusion_f1=float(f1_score(y, pred, zero_division=0)),
        occlusion_average_precision=float(average_precision_score(y, result["score"])),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        sign_accuracy=float(accuracy_score(result["true_sign"], result["sign"])),
    )
    result_metrics["sign_accuracy_clean"] = float(
        accuracy_score(result["true_sign"][y == 0], result["sign"][y == 0])
    )
    result_metrics["sign_accuracy_occluded"] = float(
        accuracy_score(result["true_sign"][y == 1], result["sign"][y == 1])
    )
    result_metrics["recall_by_occluder"] = {
        k: float(pred[result["kind"] == k].mean())
        for k in ("leaves", "mud", "graffiti")
        if np.any(result["kind"] == k)
    }
    result_metrics["recall_by_coverage"] = {}
    for name, lo, hi in [
        ("under_10_percent", 0, 0.1),
        ("10_to_25_percent", 0.1, 0.25),
        ("over_25_percent", 0.25, 1.01),
    ]:
        mask = (y == 1) & (result["coverage"] >= lo) & (result["coverage"] < hi)
        result_metrics["recall_by_coverage"][name] = {
            "count": int(mask.sum()),
            "recall": float(pred[mask].mean()) if mask.any() else None,
        }
    return result_metrics


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(args.data)
    train, val = split_tracks(records(args.data, "train"), args.seed)
    train = limit_per_class(train, args.train_per_class, args.seed)
    val = limit_per_class(val, args.val_per_class, args.seed + 1)
    test = limit_per_class(
        records(args.data, "test"), args.test_per_class, args.seed + 2
    )
    write_manifest(out / "split_manifest.json", train, val, test)
    config = vars(args).copy()
    config.update(
        split_sizes={"train": len(train), "validation": len(val), "test": len(test)},
        dataset="GTSRB official final training/test",
        label_definition="added synthetic overlay, not natural occlusion",
        torch_version=torch.__version__,
        input_size=64,
        classes=43,
        protocol="Track-disjoint training/validation; official test. Evaluation pairs clean + one deterministic overlay.",
        subset_run=any([args.train_per_class, args.val_per_class, args.test_per_class]),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    train_ds = SignDataset(train, training=True, seed=args.seed)
    val_ds = SignDataset(val, seed=100000 + args.seed)
    test_ds = SignDataset(test, seed=200000 + args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    model = SignOcclusionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = float("inf")
    waiting = 0
    history = []
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_ds.epoch = epoch
        model.train()
        total = 0.0
        for images, labels, occluded, _, _ in train_loader:
            optimizer.zero_grad()
            sign_logits, occ_logits = model(images.to(device))
            loss = nn.functional.cross_entropy(
                sign_logits, labels.to(device)
            ) + nn.functional.binary_cross_entropy_with_logits(
                occ_logits, occluded.float().to(device)
            )
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        result = collect(model, val_loader, device)
        history.append(
            dict(
                epoch=epoch,
                train_objective=total / len(train_ds),
                val_loss=result["loss"],
                val_sign_accuracy=float((result["true_sign"] == result["sign"]).mean()),
                val_occlusion_f1_at_05=float(
                    f1_score(result["target"], result["score"] >= 0.5, zero_division=0)
                ),
            )
        )
        print(
            f'Epoch {epoch:02d}: train={history[-1]["train_objective"]:.4f} val={result["loss"]:.4f}',
            flush=True,
        )
        if result["loss"] < best - 1e-4:
            best = result["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            waiting = 0
        else:
            waiting += 1
        if waiting >= args.patience:
            break
    model.load_state_dict(best_state)
    val_result = collect(model, val_loader, device)
    threshold = choose_threshold(val_result)
    # Freeze checkpoint and threshold before any test evaluation.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "best_epoch": best_epoch,
            "num_classes": 43,
            "input_size": 64,
        },
        out / "model.pt",
    )
    (out / "selection.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "threshold": threshold,
                "selection": "validation joint loss for checkpoint; validation occlusion F1 for threshold",
            },
            indent=2,
        )
    )
    test_result = collect(
        model, DataLoader(test_ds, batch_size=args.batch_size, num_workers=0), device
    )
    summary = {
        "validation": metrics(val_result, threshold),
        "test": metrics(test_result, threshold),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_csv(out / "history.csv", history)
    predrows = [
        dict(
            image=test[i // 2]["path"],
            true_sign=int(test_result["true_sign"][i]),
            predicted_sign=int(test_result["sign"][i]),
            synthetic_occlusion=int(test_result["target"][i]),
            occlusion_score=float(test_result["score"][i]),
            predicted_occlusion=int(test_result["score"][i] >= threshold),
            kind=str(test_result["kind"][i]),
            crop_coverage=float(test_result["coverage"][i]),
        )
        for i in range(len(test_ds))
    ]
    write_csv(out / "test_predictions.csv", predrows)
    from report import generate

    generate(out, history, test_result, summary, config, test_ds)
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="results")
    p.add_argument("--download", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    for name in ["train", "val", "test"]:
        p.add_argument(
            f"--{name}-per-class",
            type=int,
            default=0,
            help="0 = all images; positive = subset demo",
        )
    a = p.parse_args()
    if (
        min(a.epochs, a.patience, a.batch_size, a.threads) < 1
        or min(a.train_per_class, a.val_per_class, a.test_per_class) < 0
    ):
        p.error("Counts must be positive; subset limits may be zero for all data.")
    run(a)
