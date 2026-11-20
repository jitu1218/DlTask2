"""Predict a sign class and synthetic-occlusion score for an already-cropped sign."""

import argparse
import json
from PIL import Image
import torch
from data import tensor
from model import SignOcclusionCNN


def predict(checkpoint, image):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SignOcclusionCNN(saved["num_classes"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with Image.open(image) as im:
        batch = tensor(
            im.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        ).unsqueeze(0)
    with torch.no_grad():
        logits, occ = model(batch)
    score = float(occ.sigmoid().item())
    return dict(
        sign_class_id=int(logits.argmax(1).item()),
        sign_softmax_score=float(logits.softmax(1).max().item()),
        occlusion_score=score,
        occlusion_threshold=saved["threshold"],
        occlusion_detected=score >= saved["threshold"],
        note="Scores are uncalibrated. Trained on synthetic occlusions in sign crops, not whole road scenes.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/model.pt")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image), indent=2))
