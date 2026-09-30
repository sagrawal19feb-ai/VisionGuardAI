"""Application lifecycle and commands shared by desktop and loopback web UI."""

import logging
import threading
import time
from pathlib import Path

from visionguard.core.alerts import AlertSystem
from visionguard.core.camera import Camera
from visionguard.core.database import Database
from visionguard.core.faces import FaceRecognitionModule
from visionguard.core.monitor import MonitoringWorker
from visionguard.detection.detector import FasterRCNNDetector
from visionguard.core.registration import register_person
from visionguard.core.threat import ThreatAssessment

logger = logging.getLogger(__name__)


class MonitorService:
    def __init__(self, config):
        config.create_dirs()
        self.config = config
        self.database = Database(config.DATABASE_PATH)
        self.camera = Camera(
            config.CAMERA_INDEX,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.CAMERA_FPS,
        )
        self.faces = FaceRecognitionModule(config, self.database)
        self.objects = FasterRCNNDetector(config)
        self.worker = MonitoringWorker(
            self.camera,
            self.faces,
            self.objects,
            ThreatAssessment(config),
            AlertSystem(config, self.database),
            config,
            self.database,
        )
        self._lifecycle = threading.Lock()
        self._closed = False

    def start(self):
        with self._lifecycle:
            if self._closed:
                raise RuntimeError("Monitor has been closed")
            if self.camera.is_available:
                return self.worker.start()
            if not self.camera.start():
                return False
            return self.worker.start()

    def retry_camera(self):
        with self._lifecycle:
            if self._closed:
                raise RuntimeError("Monitor has been closed")
            self.worker.stop()
            self.worker.state.clear()
            self.camera.stop()
            if not self.camera.start():
                return False
            return self.worker.start()

    def reload_faces(self):
        self.worker.reload_faces()

    def reload_detector(self):
        if self.worker.is_running:
            self.worker.reload_detector()
        else:
            self.objects.reload_model()

    def register_face(self, name, image_path):
        path = register_person(name, image_path, self.config, self.database)
        self.reload_faces()
        return path

    def deactivate_face(self, person_id):
        person = self.database.get_person_by_id(person_id)
        changed = self.database.deactivate_person(person_id)
        if changed:
            # Remove only our own registration file; never delete an arbitrary
            # pathname stored in a modified or old SQLite database.
            if person and person["image_path"]:
                path = Path(person["image_path"])
                if not path.is_absolute():
                    path = self.config.BASE_DIR / path
                if path.resolve().is_relative_to(self.config.FACES_DIR.resolve()):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.exception("Could not remove deactivated face photo")
            self.reload_faces()
        return changed

    def status(self):
        snapshot = self.worker.state.read()
        camera_ok = self.camera.is_available
        monitoring = self.worker.is_running and camera_ok
        fresh = monitoring and time.monotonic() - snapshot.updated_at < 4
        result = snapshot.result if fresh else None
        assessment = result.assessment if result else None
        return {
            "camera_available": camera_ok,
            "monitoring": monitoring,
            "frame_sequence": snapshot.sequence if fresh else 0,
            "fps": round(result.fps, 1) if result else 0,
            "threat_level": assessment["threat_level"] if assessment else None,
            "description": assessment["description"]
            if assessment
            else "No live assessment",
            "faces": len(result.faces) if result else 0,
            "unknown_faces": assessment["unknown_count"] if assessment else 0,
            "hazards": [
                {"label": label, "confidence": round(score, 3)}
                for label, score in assessment["hazardous_objects"]
            ]
            if assessment
            else [],
            "warnings": list(result.warnings) if result else self._warnings(),
            "model": {
                "name": "Faster R-CNN MobileNetV3-320"
                if self.objects.is_available
                else "Unavailable",
                "available": self.objects.is_available,
                "classes": self.objects.class_names,
                "missing": self.objects.unsupported_hazards,
            },
            "statistics": self.database.get_statistics(),
        }

    def _warnings(self):
        warnings = []
        if not self.camera.is_available:
            warnings.append("Camera unavailable · check permissions or retry")
        elif self.worker.is_running:
            warnings.append("Waiting for a recent analyzed frame")
        else:
            warnings.append("Inference worker is not running · retry camera")
        if not self.objects.is_available:
            warnings.append(self.objects.unavailable_reason)
        elif self.objects.model_warning:
            warnings.append(self.objects.model_warning)
        if self.objects.unsupported_hazards:
            warnings.append(
                "Model cannot detect: " + ", ".join(self.objects.unsupported_hazards)
            )
        if not self.faces.recognition_available:
            warnings.append("Face recognition unavailable")
        return warnings

    def close(self):
        with self._lifecycle:
            if self._closed:
                return
            self._closed = True
            self.worker.stop()
            self.camera.stop()
            self.database.close()
