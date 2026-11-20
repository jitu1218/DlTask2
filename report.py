"""Plots and a readable report generated from recorded predictions."""

import html
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def generate(out, history, result, summary, config, test_ds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["train_objective"] for r in history],
        label="Training (augmentation/dropout on)",
    )
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["val_loss"] for r in history],
        label="Validation (fixed views)",
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Sign CE + occlusion BCE",
        title="Joint learning objective",
    )
    axes[0].legend(fontsize=8)
    p, r, _ = precision_recall_curve(result["target"], result["score"])
    axes[1].plot(r, p, color="#087f8c")
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Synthetic occlusion: test PR curve",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    fig.savefig(out / "learning_and_pr.png", dpi=150)
    plt.close(fig)
    matrix = np.array(summary["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No overlay", "Overlay"],
        yticklabels=["No overlay", "Overlay"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Occlusion confusion matrix",
    )
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), constrained_layout=True)
    example_indices = [
        2 * int(base) + view
        for base in np.linspace(0, len(test_ds.rows) - 1, 6, dtype=int)
        for view in (0, 1)
    ]
    for i, ax in zip(example_indices, axes.flat):
        image, label, occluded, kind, coverage = test_ds[i]
        ax.imshow(np.clip((image.numpy().transpose(1, 2, 0) + 1) / 2, 0, 1))
        ax.set_title(
            f"Class {label} • {kind}\nCrop covered: {coverage:.0%}", fontsize=9
        )
        ax.axis("off")
    fig.suptitle("GTSRB crops with procedural synthetic occlusion")
    fig.savefig(out / "occlusion_examples.png", dpi=150)
    plt.close(fig)
    m = summary["test"]
    status = "SUBSET DEMONSTRATION" if config["subset_run"] else "FULL DATASET RUN"
    lines = [
        "# Road Sign Occlusion — Experiment Report",
        "",
        f"Run status: **{status}**.",
        f'Original-image split sizes: {config["split_sizes"]}. Each validation/test image has one clean and one occluded view. Paired views are correlated, not independent source images.',
        "## Method",
        "Custom three-block convolutional backbone (16/32/64 channels), pooled shared 128-dimensional representation, 43-class sign head and binary occlusion head. Training uses equal-weight cross-entropy and binary cross-entropy losses, AdamW, dropout 0.25, and early stopping on joint validation loss.",
        "Training/validation are separated by class + physical sign track before augmenting. Test images come from the official final test split. All models see 64×64 crops using annotated sign ROIs. No horizontal flips are used.",
        "Leaves, mud and graffiti are procedural textured approximations with randomized shapes, colors and positions. They are not photorealistic assets. Coverage measures the affected fraction of the crop, not a segmented sign surface.",
        "## Held-out results",
        f'- Occlusion precision: {m["occlusion_precision"]:.4f}',
        f'- Occlusion recall: {m["occlusion_recall"]:.4f}',
        f'- Occlusion F1: {m["occlusion_f1"]:.4f}',
        f'- Occlusion average precision: {m["occlusion_average_precision"]:.4f}',
        f'- Sign accuracy on clean crops: {m["sign_accuracy_clean"]:.4f}',
        f'- Sign accuracy on occluded crops: {m["sign_accuracy_occluded"]:.4f}',
        f'- Validation-selected threshold: {m["threshold"]:.2f}',
        "The threshold maximizes validation F1 over a predefined grid. The test set is not used to select it. A recall-oriented application would need a separately specified operating requirement and validation threshold rule.",
        "## Interpretation and limitations",
        f'The subset/full run recorded clean sign accuracy {m["sign_accuracy_clean"]:.1%} versus {m["sign_accuracy_occluded"]:.1%} with overlays. The difference shows recognition degradation under occlusion. Strong overlay-detection scores do not imply strong sign recognition.',
        "Precision measures how often an occlusion flag is correct; recall measures how many synthetic occlusions are found. False negatives are missed overlays. Sign accuracy on occluded images tests whether the classifier still recognizes a partially covered sign.",
        "These metrics concern added synthetic overlays. Original GTSRB images are not annotated here for natural occlusion, so “clean” means no added overlay, not verified unobstructed. A model may learn generator artifacts instead of general physical occlusion. Unseen occluder types, natural occlusions, weather, and full road scenes require independent evaluation.",
        "This is sign-crop classification, not bounding-box detection in a full scene. The model does not localize signs. Its scores are uncalibrated and must not be interpreted as reliable uncertainty or used to control a vehicle. A road-scene pipeline would need a separate sign detector and real-world validation.",
        "Subset runs demonstrate execution only. Low sign accuracy should be reported honestly; use the documented full-data training command to investigate improvements. Results from one seed do not establish robustness.",
        "## Visuals",
        "![Learning and precision-recall curves](learning_and_pr.png)",
        "![Confusion matrix](confusion_matrix.png)",
        "![Synthetic examples](occlusion_examples.png)",
        "## Sources",
        "https://benchmark.ini.rub.de/",
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html",
    ]
    (out / "REPORT.md").write_text("\n\n".join(lines), encoding="utf-8")
    elements = []
    for line in lines:
        if line.startswith("!["):
            alt, src = line[2:].split("](")
            elements.append(f'<img alt="{html.escape(alt)}" src="{src[:-1]}">')
        elif line.startswith("# "):
            elements.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line.startswith("## "):
            elements.append("<h2>" + html.escape(line[3:]) + "</h2>")
        else:
            elements.append("<p>" + html.escape(line.replace("**", "")) + "</p>")
    (out / "REPORT.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Road Sign Occlusion Report</title><style>body{font:16px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:20px;color:#19343a}h1,h2{color:#087f8c}img{max-width:100%}</style>'
        + "".join(elements)
        + "</html>",
        encoding="utf-8",
    )
