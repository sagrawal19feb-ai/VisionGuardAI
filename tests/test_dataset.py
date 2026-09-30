"""Manifest validation for the supported Faster R-CNN training workflow."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visionguard.training.dataset import active_classes, load_manifest, split_records


class DatasetTests(unittest.TestCase):
    def test_boxes_negatives_split_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            frame = np.zeros((100, 120, 3), dtype=np.uint8)
            names = ("one.png", "two.png", "background.png")
            for name in names:
                self.assertTrue(cv2.imwrite(str(images / name), frame))
            entries = [
                {
                    "file": "one.png",
                    "boxes": [{"label": "gun", "bbox": [10, 15, 30, 30]}],
                },
                {
                    "file": "two.png",
                    "boxes": [{"label": "gun", "bbox": [25, 20, 30, 30]}],
                },
                {"file": "background.png", "boxes": []},
            ]
            manifest = root / "annotations.json"

            def save():
                manifest.write_text(
                    json.dumps({"version": 1, "images": entries}), encoding="utf-8"
                )

            save()
            records = load_manifest(manifest)
            self.assertEqual(active_classes(records), ["gun"])
            training, validation = split_records(records, ["gun"])
            self.assertTrue(any(record.boxes for record in training))
            self.assertTrue(any(record.boxes for record in validation))
            self.assertTrue(any(not record.boxes for record in records))
            entries[0]["file"] = "../outside.png"
            save()
            with self.assertRaises(ValueError):
                load_manifest(manifest)
            entries[0]["file"] = "one.png"
            entries[0]["boxes"][0]["label"] = "unknown hazard"
            save()
            with self.assertRaises(ValueError):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
