"""Inference adapter for the project's own GridNet neural network."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    is_hazardous: bool = False
    threat_modifier: str = "NONE"
    color: Tuple[int, int, int] = (0, 255, 0)


class ObjectDetectionModule:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = None
        self.class_names = []
        self.image_size = None
        self.is_available = False
        self.unsupported_hazards = []
        self.last_error = False
        self.unavailable_reason = "Detector not trained"
        self.reload_model()

    def reload_model(self):
        """Load a trusted, weights-only GridNet checkpoint (or stay disabled)."""
        self.model = None
        self.is_available = False
        self.unsupported_hazards = []
        self.last_error = False
        path = self.config.DETECTOR_MODEL
        if not path.is_file():
            self.unavailable_reason = "No trained detector · run python train.py"
            logger.warning("No trained detector at %s", path)
            return False
        try:
            import torch
            from modules.custom_detector import GridNet, validate_checkpoint

            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            classes, size = validate_checkpoint(
                checkpoint, self.config.HAZARDOUS_OBJECTS
            )
            model = GridNet(len(classes))
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            self.model = model.to(device).eval()
            self.device = device
            self.class_names = classes
            self.image_size = size
            self.unsupported_hazards = sorted(
                set(self.config.HAZARDOUS_OBJECTS) - set(classes)
            )
            self.is_available = True
            self.unavailable_reason = ""
            if self.unsupported_hazards:
                logger.warning(
                    "Untrained hazard labels: %s", ", ".join(self.unsupported_hazards)
                )
            logger.info("GridNet loaded: %s (%s)", path, device)
            return True
        except Exception as exc:
            self.unavailable_reason = f"Detector load failed: {str(exc)[:100]}"
            logger.exception("Cannot load GridNet checkpoint")
            return False

    def detect(self, frame) -> List[DetectionResult]:
        if not self.is_available or self.model is None:
            return []
        try:
            import torch
            from modules.custom_detector import decode_predictions, image_tensor

            tensor, transform = image_tensor(frame, self.image_size)
            with torch.inference_mode():
                logits = self.model(tensor.unsqueeze(0).to(self.device))
            results = decode_predictions(
                logits,
                transform,
                self.class_names,
                threshold=self.config.DETECTION_CONFIDENCE_THRESHOLD,
                nms_iou=self.config.DETECTION_NMS_IOU,
            )
            detections = []
            for label, confidence, box in results:
                info = self.config.HAZARDOUS_OBJECTS[label]
                detections.append(
                    DetectionResult(
                        label=label.title(),
                        confidence=confidence,
                        bbox=box,
                        is_hazardous=True,
                        threat_modifier=info["threat_modifier"],
                        color=info["color"],
                    )
                )
            self.last_error = False
            return detections
        except Exception:
            self.last_error = True
            logger.exception("GridNet inference failed")
            return []

    def highest_threat(self, detections) -> Optional[str]:
        priority = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return max(
            (d.threat_modifier for d in detections if d.is_hazardous),
            key=lambda threat: priority.get(threat, 0),
            default=None,
        )
