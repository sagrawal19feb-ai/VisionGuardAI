"""YOLO object detection and policy labels for configured hazardous objects."""

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
        self.is_available = False
        self.unsupported_hazards = []
        self.last_error = False
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            path = self.config.YOLO_MODEL
            if not path.exists():
                logger.error("YOLO model not found: %s", path)
                return
            self.model = YOLO(str(path))
            names = self.model.names
            supported = {
                str(n).lower()
                for n in (names.values() if isinstance(names, dict) else names)
            }
            self.unsupported_hazards = sorted(
                set(self.config.HAZARDOUS_OBJECTS) - supported
            )
            if self.unsupported_hazards:
                logger.warning(
                    "Model has no labels for: %s", ", ".join(self.unsupported_hazards)
                )
            self.is_available = True
            logger.info("YOLO model loaded: %s", path)
        except ImportError:
            logger.error("Install ultralytics to enable object detection")
        except Exception:
            logger.exception("YOLO model could not be loaded")

    def detect(self, frame) -> List[DetectionResult]:
        if not self.is_available or self.model is None:
            return []
        try:
            results = self.model(
                frame, conf=self.config.YOLO_CONFIDENCE_THRESHOLD, verbose=False
            )
            detections = []
            for result in results:
                for box in result.boxes:
                    label = str(result.names[int(box.cls[0])]).lower()
                    info = self.config.HAZARDOUS_OBJECTS.get(label)
                    detections.append(
                        DetectionResult(
                            label=label.title(),
                            confidence=float(box.conf[0]),
                            bbox=tuple(int(v) for v in box.xyxy[0]),
                            is_hazardous=info is not None,
                            threat_modifier=info["threat_modifier"] if info else "NONE",
                            color=info["color"] if info else (0, 255, 0),
                        )
                    )
            self.last_error = False
            return detections
        except Exception:
            self.last_error = True
            logger.exception("YOLO inference failed")
            return []

    def highest_threat(self, detections) -> Optional[str]:
        priority = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return max(
            (d.threat_modifier for d in detections if d.is_hazardous),
            key=lambda t: priority.get(t, 0),
            default=None,
        )
