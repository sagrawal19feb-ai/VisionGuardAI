"""A lower official-test threshold must reach the gun detector too."""

import contextlib
import io
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from visionguard.detection.detector import DetectionResult
from visionguard.training.evaluate import evaluate


class EvaluateTests(unittest.TestCase):
    def test_uniform_cutoff_applies_to_gun_and_coco_classes(self):
        class FakeDetector:
            is_available = True
            unavailable_reason = ""
            class_names = ["gun", "knife"]
            last_error = False
            gun_threshold = 0.4

            def __init__(self, config):
                self.config = config
                self.detect_calls = 0
                FakeDetector.instance = self

            def detect(self, _frame):
                self.detect_calls += 1
                if self.gun_threshold > 0.3:
                    return []
                return [DetectionResult("Gun", 0.3, (10, 10, 40, 40))]

        record = SimpleNamespace(
            path=Path("test.jpg"), boxes=(("gun", (10, 10, 30, 30)),)
        )
        output = io.StringIO()
        with (
            mock.patch(
                "visionguard.training.evaluate.load_manifest", return_value=[record]
            ),
            mock.patch(
                "visionguard.training.evaluate.read_image",
                return_value=np.zeros((80, 80, 3), dtype=np.uint8),
            ),
            mock.patch(
                "visionguard.training.evaluate.FasterRCNNDetector", FakeDetector
            ),
            contextlib.redirect_stdout(output),
        ):
            evaluate(Path("unused.json"), thresholds=(0.2, 0.5))
        self.assertEqual(FakeDetector.instance.gun_threshold, 0.2)
        self.assertEqual(FakeDetector.instance.config.DETECTION_CLASS_THRESHOLDS, {})
        self.assertEqual(FakeDetector.instance.detect_calls, 1)
        self.assertEqual(len(re.findall(r"gun\s+TP=\s*1", output.getvalue())), 1)
        self.assertEqual(len(re.findall(r"gun\s+TP=\s*0", output.getvalue())), 1)


if __name__ == "__main__":
    unittest.main()
