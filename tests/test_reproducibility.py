"""Focused checks for portable model exports and aggregate feature statistics."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from export_inference_checkpoint import export_checkpoint
from summarize_feature_pool import moments
from train_fasterrcnn_resnet101 import build_model
from evaluate_released_model import validate_categories


class ReproducibilityTests(unittest.TestCase):
    def test_export_excludes_private_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            out = Path(directory) / "inference.pth"
            state = {"weight": torch.arange(12, dtype=torch.float32).view(3, 4)}
            torch.save({"model_ema": {"shadow": state, "decay": 0.9998, "updates": 4},
                        "epoch": 1, "args": {"data": "/path/to/Test-Data"},
                        "optimizer": {"private": "metadata"}}, source)
            result = export_checkpoint(source, out)
            restored = torch.load(out, weights_only=True)
            self.assertTrue(result["tensor_equality"])
            self.assertEqual(set(restored), {"epoch", "model_ema"})
            self.assertTrue(torch.equal(restored["model_ema"]["shadow"]["weight"], state["weight"]))
            self.assertNotIn(b"/path/to/Test-Data", out.read_bytes())
            with self.assertRaises(FileExistsError):
                export_checkpoint(source, out)

    def test_missing_ema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            torch.save({"model": {"weight": torch.ones(2)}}, source)
            with self.assertRaises(ValueError):
                export_checkpoint(source, Path(directory) / "out.pth")

    def test_nonfinite_export_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            torch.save({"epoch": 1, "model_ema": {"shadow": {"weight": torch.tensor([float("nan")])},
                        "decay": 0.9998, "updates": 1}}, source)
            with self.assertRaises(ValueError):
                export_checkpoint(source, Path(directory) / "out.pth")

    def test_batched_moments_match_direct_computation(self):
        values = np.random.default_rng(7).normal(size=(37, 5)) + 1000
        count, mean, variance = moments(values, chunk_size=6)
        self.assertEqual(count, len(values))
        np.testing.assert_allclose(mean, values.mean(0), rtol=1e-12)
        np.testing.assert_allclose(variance, values.var(0), rtol=1e-12)
        with self.assertRaises(ValueError):
            moments(np.empty((0, 5)))
        with self.assertRaises(ValueError):
            moments(np.array([[np.nan]]))

    def test_category_mapping_is_checked(self):
        names = ["bicycle", "bus", "car", "motorcycle", "person", "rider", "truck"]
        categories = [{"id": index + 1, "name": name} for index, name in enumerate(names)]
        validate_categories(categories)
        categories[0]["name"] = "car"
        with self.assertRaises(ValueError):
            validate_categories(categories)

    def test_detector_constructs_without_download(self):
        with patch("torchvision.models._api.load_state_dict_from_url", side_effect=AssertionError("Unexpected download")):
            model = build_model(8, 3, 800, 1333, pretrained_backbone=False)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 8)
        self.assertEqual(model.transform.min_size, (800,))
        self.assertEqual(model.transform.max_size, 1333)

    def test_reference_metric_counts_match_model_cards(self):
        import csv
        registry = json.loads((ROOT / "reproducibility/models.json").read_text())
        with (ROOT / "reproducibility/statistics/reference_metrics.csv").open() as stream:
            for row in csv.DictReader(stream):
                self.assertIn(row["model"], registry["models"])
                self.assertEqual(int(row["images"]), registry["benchmarks"][row["benchmark"]]["images"])


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
