"""Headless tests for the non-YOLO model and self-contained checkpoint loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from config import Config
from modules.object_detection import ObjectDetectionModule
from modules.quality_detector import (
    FORMAT,
    LABEL_TO_ID,
    build_model,
    train_gun_head_only,
    validate_checkpoint,
)


class QualityDetectorTests(unittest.TestCase):
    def test_gun_head_gradient_does_not_change_pretrained_class_rows(self):
        torch.set_num_threads(1)
        model = build_model(pretrained=False)
        train_gun_head_only(model)
        model.rpn._pre_nms_top_n["training"] = 50
        model.rpn._post_nms_top_n["training"] = 30
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 32
        model.rpn.fg_bg_sampler.batch_size_per_image = 32
        model.train()
        image = torch.rand(3, 160, 160)
        target = {
            "boxes": torch.tensor([[20.0, 30.0, 95.0, 105.0]]),
            "labels": torch.tensor([LABEL_TO_ID["gun"]]),
        }
        loss = sum(model([image], [target]).values())
        loss.backward()
        predictor = model.roi_heads.box_predictor
        self.assertEqual(
            int(torch.count_nonzero(predictor.cls_score.weight.grad[:91])), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(predictor.bbox_pred.weight.grad[:364])), 0
        )
        self.assertTrue(torch.isfinite(loss))

    def test_weights_only_checkpoint_loads_without_downloading_weights(self):
        torch.set_num_threads(1)
        model = build_model(pretrained=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frcnn.pt"
            record = {
                "format": FORMAT,
                "label_to_id": LABEL_TO_ID,
                "enabled_classes": ["knife", "scissors", "baseball bat"],
                "gun_threshold": 0.4,
                "state_dict": {
                    name: tensor.detach().cpu().half()
                    if tensor.is_floating_point()
                    else tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            }
            self.assertNotIn("gun", validate_checkpoint(record))
            torch.save(record, path)
            config = type("QualityTestConfig", (Config,), {"DETECTOR_MODEL": path})
            detector = ObjectDetectionModule(config)
            self.assertTrue(detector.is_available, detector.unavailable_reason)
            self.assertEqual(detector.backend, "frcnn")
            self.assertIn("gun", detector.unsupported_hazards)
            results = detector.detect(np.zeros((160, 160, 3), dtype=np.uint8))
            self.assertIsInstance(results, list)
            self.assertFalse(detector.last_error)


if __name__ == "__main__":
    unittest.main()
