"""Incident alerts, screenshots and annotated video overlays."""

import logging
import time
import uuid
from datetime import datetime, timezone

import cv2

logger = logging.getLogger(__name__)


class AlertSystem:
    def __init__(self, config, database):
        self.config = config
        self.database = database
        self.config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.last_alert_time = float("-inf")
        self.last_alert_level = None
        self.session_id = uuid.uuid4().hex

    def process(self, assessment, frame):
        """Return one new incident or None. Only HIGH/CRITICAL produce alerts."""
        level = assessment["threat_level"]
        if level not in ("HIGH", "CRITICAL"):
            return None
        now = time.monotonic()
        escalation = level == "CRITICAL" and self.last_alert_level != "CRITICAL"
        if (
            not escalation
            and now - self.last_alert_time < self.config.ALERT_COOLDOWN_SECONDS
        ):
            return None
        self.last_alert_time, self.last_alert_level = now, level
        description = assessment["description"]
        screenshot = None
        if self.config.ALERT_SCREENSHOT_ON_HIGH_RISK or (
            self.config.ALERT_SCREENSHOT_ON_UNKNOWN and assessment["unknown_count"] > 0
        ):
            screenshot = self.save_incident_screenshot(frame, level)
        objects = [
            {"label": label, "confidence": confidence}
            for label, confidence in assessment["all_objects"]
        ]
        try:
            self.database.log_alert(level, description, level)
            self.database.log_detection(
                event_type="incident",
                person_name=(assessment["known_persons"] or [None])[0],
                person_is_known=(
                    0
                    if assessment["unknown_count"]
                    else 1
                    if assessment["known_persons"]
                    else None
                ),
                objects=objects,
                threat_level=level,
                screenshot_path=screenshot,
                session_id=self.session_id,
            )
        except Exception:
            logger.exception("Failed to record incident")
        logger.warning("[%s] %s", level, description)
        return {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "threat_level": level,
            "description": description,
            "screenshot_path": screenshot,
        }

    def save_incident_screenshot(self, frame, label="EVENT"):
        filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        path = self.config.SCREENSHOTS_DIR / filename
        try:
            ok, data = cv2.imencode(".jpg", frame)
            if not ok:
                raise OSError("OpenCV could not encode screenshot")
            data.tofile(str(path))
            return str(path)
        except (OSError, cv2.error):
            logger.exception("Screenshot save failed")
            return None

    def draw_overlays(self, frame, faces, objects, assessment, fps=0):
        output = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            color = (0, 210, 0) if face.is_known else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            name = face.name if face.is_known else "Unknown"
            cv2.putText(
                output,
                name,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), obj.color, 2)
            label = f"{obj.label} {obj.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                obj.color,
                2,
            )
        level = assessment["threat_level"]
        color = {
            "LOW": (0, 130, 0),
            "MEDIUM": (0, 170, 230),
            "HIGH": (0, 120, 240),
            "CRITICAL": (0, 0, 220),
        }[level]
        width, height = output.shape[1], output.shape[0]
        cv2.rectangle(output, (0, 0), (width, min(42, height)), color, -1)
        cv2.putText(
            output,
            f"THREAT: {level}",
            (10, min(29, height - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        if fps > 0 and width >= 360:
            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (width - 150, min(29, height - 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        if height >= 100:
            description = assessment["description"][: max(0, (width - 20) // 12)]
            cv2.rectangle(output, (0, height - 35), (width, height), (10, 15, 25), -1)
            cv2.putText(
                output,
                description,
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
        return output
