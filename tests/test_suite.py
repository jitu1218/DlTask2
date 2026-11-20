import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import SignOcclusionCNN
from occlusion import overlay, KINDS
from data import split_tracks, SignDataset
from train import choose_threshold, metrics


class SuiteTests(unittest.TestCase):
    def test_two_heads_and_gradient_flow(self):
        torch.set_num_threads(1)
        model = SignOcclusionCNN()
        sign, occ = model(torch.rand(3, 3, 64, 64))
        self.assertEqual(tuple(sign.shape), (3, 43))
        self.assertEqual(tuple(occ.shape), (3,))
        loss = torch.nn.functional.cross_entropy(
            sign, torch.tensor([0, 1, 2])
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            occ, torch.tensor([0.0, 1.0, 0.0])
        )
        loss.backward()
        for layer in (model.sign_head, model.occlusion_head, model.features[0]):
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_overlay_types_and_reproducibility(self):
        original = Image.new("RGB", (64, 64), (180, 180, 180))
        for kind in KINDS:
            a, c = overlay(original, np.random.default_rng(42), kind)
            b, d = overlay(original, np.random.default_rng(42), kind)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(c, d)
            self.assertGreater(c, 0)
            self.assertLess(c, 1)
            self.assertFalse(np.array_equal(a, original))
        np.testing.assert_array_equal(np.asarray(original), np.full((64, 64, 3), 180))

    def test_group_split_no_track_leakage(self):
        rows = [
            {"label": label, "track": f"{label}:{track}"}
            for label in range(3)
            for track in range(10)
            for frame in range(3)
        ]
        train, val = split_tracks(rows)
        self.assertFalse({r["track"] for r in train} & {r["track"] for r in val})
        self.assertEqual(len(train) + len(val), len(rows))
        self.assertEqual({r["label"] for r in val}, {0, 1, 2})

    def test_eval_pairs_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sign.png"
            Image.new("RGB", (80, 80), "red").save(path)
            ds = SignDataset(
                [dict(path=str(path), label=2, track="2:0", roi=[0, 0, 79, 79])]
            )
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0][2], 0)
            self.assertEqual(ds[1][2], 1)
            self.assertEqual(ds[0][1], ds[1][1])
            self.assertTrue(torch.equal(ds[1][0], ds[1][0]))
            self.assertGreater(ds[1][4], 0)

    def test_threshold_and_metrics(self):
        result = dict(
            target=np.array([0, 0, 1, 1]),
            score=np.array([0.1, 0.2, 0.8, 0.9]),
            true_sign=np.array([0, 1, 2, 3]),
            sign=np.array([0, 1, 2, 3]),
            kind=np.array(["clean", "clean", "mud", "leaves"]),
            coverage=np.array([0, 0, 0.2, 0.3]),
        )
        threshold = choose_threshold(result)
        summary = metrics(result, threshold)
        self.assertEqual(summary["occlusion_precision"], 1.0)
        self.assertEqual(summary["occlusion_recall"], 1.0)
        self.assertEqual(summary["confusion_matrix"], [[2, 0], [0, 2]])


if __name__ == "__main__":
    unittest.main()
