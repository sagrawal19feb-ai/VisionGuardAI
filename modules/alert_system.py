"""
Alert System
============

Handles:
- Threat alerts
- Screenshot capture
- Overlay drawing
- Alert cooldown
"""

import cv2
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertSystem:

    def __init__(self, config, database):

        self.config = config
        self.database = database

        self.last_alert_time = 0

        # Absolute project path
        self.base_dir = Path(__file__).resolve().parent.parent

        self.screenshot_dir = self.base_dir / "data" / "screenshots"

        try:
            self.screenshot_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            logger.info(
                f"Screenshots folder: {self.screenshot_dir}"
            )

        except Exception as exc:
            logger.error(
                f"Failed creating screenshot directory: {exc}"
            )

    # --------------------------------------------------
    # Alert Processing
    # --------------------------------------------------

    def process(
        self,
        assessment,
        frame
    ):

        try:

            threat_level = assessment.get(
                "threat_level",
                "LOW"
            )

            if threat_level not in (
                "HIGH",
                "CRITICAL"
            ):
                return

            now = time.time()

            cooldown = getattr(
                self.config,
                "ALERT_COOLDOWN_SECONDS",
                10
            )

            if now - self.last_alert_time < cooldown:
                return

            self.last_alert_time = now

            description = assessment.get(
                "description",
                ""
            )

            logger.warning(
                f"[{threat_level}] {description}"
            )

            try:
                self.database.log_alert(
                    alert_type=threat_level,
                    message=description,
                    threat_level=threat_level
                )
            except Exception as exc:
                logger.error(
                    f"Database alert logging failed: {exc}"
                )

            self.save_incident_screenshot(
                frame,
                threat_level
            )

        except Exception as exc:
            logger.error(
                f"Alert processing error: {exc}"
            )

    # --------------------------------------------------
    # Screenshot
    # --------------------------------------------------

    def save_incident_screenshot(
        self,
        frame,
        label="EVENT"
    ):

        try:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                f"{label}_{timestamp}.jpg"
            )

            path = (
                self.screenshot_dir /
                filename
            )

            success = cv2.imwrite(
                str(path),
                frame
            )

            if success:
                logger.info(
                    f"Screenshot saved: {path}"
                )
                return str(path)

            return None

        except Exception as exc:
            logger.error(
                f"Screenshot save failed: {exc}"
            )
            return None

    # --------------------------------------------------
    # Overlay Drawing
    # --------------------------------------------------

    def draw_overlays(
        self,
        frame,
        faces,
        objects,
        assessment,
        fps=0
    ):

        output = frame.copy()

        # ---------------------------------
        # Faces
        # ---------------------------------

        for face in faces:

            try:

                x1, y1, x2, y2 = face.bbox

                color = (
                    (0, 255, 0)
                    if face.is_known
                    else
                    (0, 0, 255)
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    face.name
                    if face.is_known
                    else
                    "Unknown"
                )

                cv2.putText(
                    output,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            except Exception:
                pass

        # ---------------------------------
        # Objects
        # ---------------------------------

        for obj in objects:

            try:

                x1, y1, x2, y2 = obj.bbox

                color = getattr(
                    obj,
                    "color",
                    (255, 255, 0)
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                text = (
                    f"{obj.label} "
                    f"{obj.confidence:.2f}"
                )

                cv2.putText(
                    output,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            except Exception:
                pass

        # ---------------------------------
        # Threat Banner
        # ---------------------------------

        level = assessment.get(
            "threat_level",
            "LOW"
        )

        description = assessment.get(
            "description",
            ""
        )

        banner_color = {
            "LOW": (0, 180, 0),
            "MEDIUM": (0, 200, 255),
            "HIGH": (0, 140, 255),
            "CRITICAL": (0, 0, 255)
        }.get(level, (0, 180, 0))

        cv2.rectangle(
            output,
            (0, 0),
            (650, 45),
            banner_color,
            -1
        )

        cv2.putText(
            output,
            f"THREAT: {level}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if fps > 0:

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (500, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.putText(
            output,
            description[:100],
            (10, output.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return output