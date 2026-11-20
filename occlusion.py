"""Procedural leaf, mud and graffiti overlays. Labels mean added synthetic occlusion."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

KINDS = ("leaves", "mud", "graffiti")


def overlay(image, rng, kind=None, severity=None):
    """Overlay an irregular texture; return RGB image and measured crop coverage.

    Coverage is mask pixels / crop pixels, NOT fraction of a segmented sign.
    Shapes are generated on a patch within the sign ROI crop. No text/label leak.
    """
    image = image.convert("RGB")
    kind = kind or str(rng.choice(KINDS))
    if kind not in KINDS:
        raise ValueError("Unknown occluder type")
    severity = float(rng.uniform(0.25, 0.6) if severity is None else severity)
    if not 0.1 <= severity <= 0.8:
        raise ValueError("severity must be between .1 and .8")
    w, h = image.size
    side = max(8, int(min(w, h) * severity))
    x = int(rng.integers(0, w - side + 1))
    y = int(rng.integers(0, h - side + 1))
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "leaves":
        color = np.array([int(rng.integers(25, 95)), int(rng.integers(65, 145)), 25])
        for _ in range(9):
            cx = x + int(rng.integers(0, side))
            cy = y + int(rng.integers(0, side))
            rx, ry = max(2, side // 5), max(3, side // 3)
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    elif kind == "mud":
        color = np.array([90, 65, 40]) + rng.integers(-20, 21, 3)
        for _ in range(12):
            cx = x + int(rng.integers(0, side))
            cy = y + int(rng.integers(0, side))
            r = max(2, int(rng.integers(max(2, side // 5), max(3, side // 2))))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    else:
        color = rng.integers(0, 256, 3)
        points = [
            (x + int(rng.integers(side)), y + int(rng.integers(side)))
            for _ in range(12)
        ]
        draw.line(points, fill=255, width=max(3, side // 4), joint="curve")
    mask = mask.filter(ImageFilter.GaussianBlur(0.4))
    texture = np.clip(color + rng.normal(0, 15, (h, w, 3)), 0, 255).astype("uint8")
    coverage = float((np.asarray(mask) > 127).mean())
    return Image.composite(Image.fromarray(texture), image, mask), coverage
