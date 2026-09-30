"""Live inference for the bundled non-YOLO Faster R-CNN checkpoint only."""

import logging
from dataclasses import dataclass

import cv2
import torch

from visionguard.detection.model import ID_TO_LABEL, build_model, validate_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    is_hazardous: bool = True
    threat_modifier: str = "NONE"
    color: tuple[int, int, int] = (0, 255, 0)


class FasterRCNNDetector:
    """No architecture switching or downloads: load this project's trained weights."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = None
        self.class_names = []
        self.backend = "frcnn"
        self.gun_threshold = 0.4
        self.is_available = False
        self.unsupported_hazards = sorted(config.HAZARDOUS_OBJECTS)
        self.last_error = False
        self.unavailable_reason = "Detector unavailable"
        self.reload_model()

    def reload_model(self):
        """Validate a weights-only checkpoint and enable only declared classes."""
        self.model = None
        self.is_available = False
        self.class_names = []
        self.unsupported_hazards = sorted(self.config.HAZARDOUS_OBJECTS)
        self.last_error = False
        path = self.config.DETECTOR_MODEL
        if not path.is_file():
            self.unavailable_reason = "No detector checkpoint · run python train.py"
            logger.warning("No detector at %s", path)
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            classes = validate_checkpoint(checkpoint)
            model = build_model(pretrained=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            self.model = model.to(device).eval()
            self.device = device
            self.gun_threshold = checkpoint["gun_threshold"]
            self.class_names = classes
            self.unsupported_hazards = sorted(
                set(self.config.HAZARDOUS_OBJECTS) - set(classes)
            )
            self.is_available = True
            self.unavailable_reason = ""
            logger.info("Faster R-CNN loaded: %s (%s)", path, device)
            if self.unsupported_hazards:
                logger.warning("Not validated: %s", ", ".join(self.unsupported_hazards))
            return True
        except Exception as exc:
            self.unavailable_reason = f"Detector load failed: {str(exc)[:100]}"
            logger.exception("Cannot load detector checkpoint")
            return False

    def detect(self, frame):
        if not self.is_available or self.model is None:
            return []
        try:
            tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy())
            tensor = tensor.permute(2, 0, 1).float() / 255.0
            with torch.inference_mode():
                predictions = self.model([tensor.to(self.device)])[0]
            height, width = frame.shape[:2]
            results = []
            for box, score, cls in zip(
                predictions["boxes"],
                predictions["scores"],
                predictions["labels"],
                strict=True,
            ):
                label = ID_TO_LABEL.get(int(cls))
                if label not in self.class_names:
                    continue
                confidence = float(score)
                threshold = (
                    self.gun_threshold
                    if label == "gun"
                    else self.config.DETECTION_CONFIDENCE_THRESHOLD
                )
                if confidence < threshold:
                    continue
                x1, y1, x2, y2 = [round(float(v)) for v in box]
                coords = (
                    max(0, min(width, x1)),
                    max(0, min(height, y1)),
                    max(0, min(width, x2)),
                    max(0, min(height, y2)),
                )
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    continue
                info = self.config.HAZARDOUS_OBJECTS[label]
                results.append(
                    DetectionResult(
                        label=label.title(),
                        confidence=confidence,
                        bbox=coords,
                        threat_modifier=info["threat_modifier"],
                        color=info["color"],
                    )
                )
            self.last_error = False
            return results
        except Exception:
            self.last_error = True
            logger.exception("Faster R-CNN inference failed")
            return []
