"""Train/decode/data/checkpoint smoke tests, all on synthetic headless images."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from config import Config
from modules.custom_detector import (
    FORMAT,
    GridNet,
    box_iou,
    decode_predictions,
    detector_loss,
    encode_targets,
    image_tensor,
)
from modules.object_detection import ObjectDetectionModule
from modules.training_data import active_classes, load_manifest, split_records
from predict import predict
from train import parse_args, train


torch.set_num_threads(1)


class GridNetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def test_letterbox_forward_loss_backprop_and_decoding(self):
        tensor, transform = image_tensor(self.frame, 160)
        self.assertEqual(tuple(tensor.shape), (3, 160, 160))
        box = (30, 20, 60, 40)
        objects, coords, classes, skipped = encode_targets(
            [("gun", box)], ["gun"], transform
        )
        self.assertEqual(int(objects.sum()), 1)
        self.assertEqual(skipped, 0)
        _, _, _, crowded = encode_targets([("gun", box)] * 3, ["gun"], transform)
        self.assertEqual(crowded, 1)
        model = GridNet(1)
        logits = model(tensor.unsqueeze(0))
        self.assertEqual(tuple(logits.shape), (1, 2, 6, 10, 10))
        loss, _ = detector_loss(
            logits, objects.unsqueeze(0), coords.unsqueeze(0), classes.unsqueeze(0)
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.head[-1].weight.grad)

        cx, cy, w, h = transform.encode_box(box)
        cell_x, cell_y = int(cx * 10), int(cy * 10)
        predicted = torch.full((1, 2, 6, 10, 10), -20.0)
        offsets = torch.tensor([cx * 10 - cell_x, cy * 10 - cell_y, w, h])
        for slot in range(2):
            predicted[0, slot, 0, cell_y, cell_x] = 12.0
            predicted[0, slot, 1:5, cell_y, cell_x] = torch.logit(
                offsets.clamp(1e-5, 1 - 1e-5)
            )
        detections = decode_predictions(predicted, transform, ["gun"])
        self.assertEqual(len(detections), 1)  # duplicate suppressed
        self.assertEqual(detections[0][0], "gun")
        self.assertGreater(box_iou(detections[0][2], (30, 20, 90, 60)), 0.95)

    def test_manifest_validates_labels_paths_and_negative_images(self):
        images = self.root / "images"
        images.mkdir()
        for name in ("one.png", "two.png", "empty.png"):
            cv2.imwrite(str(images / name), self.frame)
        entries = [
            {"file": "one.png", "boxes": [{"label": "gun", "bbox": [10, 10, 30, 30]}]},
            {"file": "two.png", "boxes": [{"label": "gun", "bbox": [20, 20, 30, 30]}]},
            {"file": "empty.png", "boxes": []},
        ]
        manifest = self.root / "annotations.json"
        manifest.write_text(
            json.dumps({"version": 1, "images": entries}), encoding="utf-8"
        )
        records = load_manifest(manifest, images)
        self.assertEqual(active_classes(records), ["gun"])
        training, validation = split_records(records, ["gun"])
        self.assertTrue(training and validation)
        self.assertTrue(any(record.boxes for record in training))
        self.assertTrue(any(record.boxes for record in validation))
        entries[0]["file"] = "../outside.png"
        manifest.write_text(
            json.dumps({"version": 1, "images": entries}), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            load_manifest(manifest, images)
        entries[0]["file"] = "one.png"
        entries[0]["boxes"][0]["label"] = "imaginary hazard"
        manifest.write_text(
            json.dumps({"version": 1, "images": entries}), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            load_manifest(manifest, images)

    def test_one_epoch_from_scratch_checkpoint_and_gun_inference(self):
        images = self.root / "images"
        images.mkdir()
        entries = []
        for index in range(3):
            frame = self.frame.copy()
            cv2.rectangle(
                frame, (20 + index * 5, 15), (60 + index * 5, 60), (255, 255, 255), -1
            )
            name = f"scene_{index}.png"
            cv2.imwrite(str(images / name), frame)
            boxes = (
                [{"label": "gun", "bbox": [20 + index * 5, 15, 40, 45]}]
                if index < 2
                else []
            )
            entries.append({"file": name, "boxes": boxes})
        manifest = self.root / "annotations.json"
        manifest.write_text(
            json.dumps({"version": 1, "images": entries}), encoding="utf-8"
        )
        checkpoint = self.root / "trained.pt"
        args = parse_args(
            [
                "--annotations",
                str(manifest),
                "--images-dir",
                str(images),
                "--output",
                str(checkpoint),
                "--image-size",
                "128",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--threads",
                "1",
                "--from-scratch",
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(saved["format"], FORMAT)
        self.assertEqual(saved["classes"], ["gun"])
        config = type("TestDetectorConfig", (Config,), {"DETECTOR_MODEL": checkpoint})
        detector = ObjectDetectionModule(config)
        self.assertTrue(detector.is_available)
        self.assertIn("knife", detector.unsupported_hazards)
        self.assertNotIn("gun", detector.unsupported_hazards)
        preview = self.root / "preview.png"
        with contextlib.redirect_stdout(io.StringIO()):
            _, output = predict(images / "scene_2.png", preview, checkpoint)
        self.assertEqual(output, preview)
        self.assertIsNotNone(cv2.imread(str(preview)))

        # Replace the network's predictions with a deterministic gun detection
        # to test the inference adapter without claiming the toy model learned.
        class PredictGun(torch.nn.Module):
            def forward(self, images):
                raw = torch.full((1, 2, 6, 8, 8), -20.0, device=images.device)
                raw[0, 0, 0, 4, 4] = 12.0
                raw[0, 0, 1:5, 4, 4] = torch.tensor(
                    [0.0, 0.0, -1.0, -1.0], device=images.device
                )
                return raw

        detector.model = PredictGun()
        results = detector.detect(self.frame)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].label, "Gun")
        self.assertEqual(results[0].threat_modifier, "CRITICAL")


if __name__ == "__main__":
    unittest.main()
