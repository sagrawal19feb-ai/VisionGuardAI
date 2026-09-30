"""Inference adapter for non-YOLO TorchVision or custom GridNet checkpoints."""

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
        self.backend = None
        self.model_warning = None
        self.gun_threshold = 0.4
        self.is_available = False
        self.unsupported_hazards = []
        self.last_error = False
        self.unavailable_reason = "Detector not trained"
        self.reload_model()

    def reload_model(self):
        """Load a weights-only checkpoint; never download models on startup."""
        self.model = None
        self.is_available = False
        self.backend = None
        self.model_warning = None
        self.class_names = []
        self.unsupported_hazards = []
        self.last_error = False
        path = self.config.DETECTOR_MODEL
        if not path.is_file():
            self.unavailable_reason = (
                "No verified detector · run python train_quality.py"
            )
            logger.warning("No detector at %s", path)
            return False
        try:
            import torch
            from modules.custom_detector import FORMAT as GRID_FORMAT
            from modules.custom_detector import GridNet, validate_checkpoint
            from modules.quality_detector import FORMAT as FRCNN_FORMAT
            from modules.quality_detector import build_model
            from modules.quality_detector import validate_checkpoint as validate_frcnn

            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            if checkpoint.get("format") == GRID_FORMAT:
                classes, size = validate_checkpoint(
                    checkpoint, self.config.HAZARDOUS_OBJECTS
                )
                model = GridNet(len(classes))
                self.backend, self.image_size = "gridnet", size
            elif checkpoint.get("format") == FRCNN_FORMAT:
                classes = validate_frcnn(checkpoint)
                model = build_model(pretrained=False)
                self.backend, self.image_size = "frcnn", None
                self.gun_threshold = checkpoint["gun_threshold"]
                self.model_warning = (
                    "Experimental public-data detector · not webcam-validated"
                )
            else:
                raise ValueError("Unsupported detector checkpoint format")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            self.model = model.to(device).eval()
            self.device = device
            self.class_names = classes
            self.unsupported_hazards = sorted(
                set(self.config.HAZARDOUS_OBJECTS) - set(classes)
            )
            self.is_available = True
            self.unavailable_reason = ""
            if self.unsupported_hazards:
                logger.warning("Not validated: %s", ", ".join(self.unsupported_hazards))
            logger.info("%s detector loaded: %s (%s)", self.backend, path, device)
            return True
        except Exception as exc:
            self.unavailable_reason = f"Detector load failed: {str(exc)[:100]}"
            logger.exception("Cannot load detector checkpoint")
            return False

    def detect(self, frame) -> List[DetectionResult]:
        if not self.is_available or self.model is None:
            return []
        try:
            if self.backend == "frcnn":
                results = self._detect_frcnn(frame)
            else:
                results = self._detect_gridnet(frame)
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
            logger.exception("Object inference failed")
            return []

    def _detect_frcnn(self, frame):
        import cv2
        import torch
        from modules.quality_detector import ID_TO_LABEL

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
            if coords[2] > coords[0] and coords[3] > coords[1]:
                results.append((label, confidence, coords))
        return results

    def _detect_gridnet(self, frame):
        import torch
        from modules.custom_detector import decode_predictions, image_tensor

        tensor, transform = image_tensor(frame, self.image_size)
        with torch.inference_mode():
            logits = self.model(tensor.unsqueeze(0).to(self.device))
        return decode_predictions(
            logits,
            transform,
            self.class_names,
            threshold=self.config.DETECTION_CONFIDENCE_THRESHOLD,
            nms_iou=self.config.DETECTION_NMS_IOU,
        )

    def highest_threat(self, detections) -> Optional[str]:
        priority = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return max(
            (d.threat_modifier for d in detections if d.is_hazardous),
            key=lambda threat: priority.get(threat, 0),
            default=None,
        )
