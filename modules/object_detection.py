"""
Object Detection Module
=======================
YOLOv8 wrapper for real-time object detection.
Loads model from data/models/yolov8n.pt
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

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

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            model_path = self.config.YOLO_MODEL

            if not model_path.exists():
                logger.error(
                    f"YOLO model not found:\n{model_path}\n\n"
                    f"Download yolov8n.pt and place it in:\n"
                    f"{self.config.MODELS_DIR}"
                )
                return

            self.model = YOLO(str(model_path))

            self.is_available = True

            logger.info(
                f"YOLO model loaded successfully: {model_path}"
            )

        except ImportError:
            logger.error(
                "Ultralytics not installed.\n"
                "Run: pip install ultralytics"
            )

        except Exception as e:
            logger.error(
                f"YOLO load failed: {e}"
            )

    def detect(self, frame) -> List[DetectionResult]:
        if not self.is_available or self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf=self.config.YOLO_CONFIDENCE_THRESHOLD,
                verbose=False
            )

            detections = []

            for result in results:

                for box in result.boxes:

                    cls_id = int(box.cls[0])

                    label = result.names[cls_id].lower()

                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    is_hazardous = (
                        label in self.config.HAZARDOUS_OBJECTS
                    )

                    threat = "NONE"
                    color = (0, 255, 0)

                    if is_hazardous:
                        info = self.config.HAZARDOUS_OBJECTS[label]

                        threat = info["threat_modifier"]
                        color = info["color"]

                    detections.append(
                        DetectionResult(
                            label=label.title(),
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            is_hazardous=is_hazardous,
                            threat_modifier=threat,
                            color=color
                        )
                    )

            return detections

        except Exception as e:
            logger.error(
                f"Detection error: {e}"
            )
            return []

    def highest_threat(
        self,
        detections
    ) -> Optional[str]:

        priority = {
            "NONE": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4
        }

        highest = None
        highest_score = -1

        for det in detections:

            score = priority.get(
                det.threat_modifier,
                0
            )

            if score > highest_score:

                highest_score = score
                highest = det.threat_modifier

        return highest