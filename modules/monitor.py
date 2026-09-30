"""Run slow vision inference off the Tkinter thread; publish only the latest frame."""

import logging
import queue
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame: object
    faces: list
    objects: list
    assessment: dict
    fps: float
    warnings: tuple


class MonitoringWorker:
    def __init__(
        self,
        camera,
        face_module,
        object_module,
        threat_module,
        alert_system,
        config,
        database,
    ):
        self.camera = camera
        self.face_module = face_module
        self.object_module = object_module
        self.threat_module = threat_module
        self.alert_system = alert_system
        self.config = config
        self.database = database
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()  # Incidents must not be dropped with stale frames.
        self._stop_event = threading.Event()
        self._reload_event = threading.Event()
        self._thread = None
        self._counter = 0
        self._faces = []
        self._objects = []

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            if self._stop_event.is_set():
                logger.error("Previous inference worker has not stopped yet")
                return False
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="vision-inference", daemon=True
        )
        self._thread.start()
        return True

    def reload_faces(self):
        self._reload_event.set()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("Inference thread did not stop within 2 seconds")

    def _run(self):
        last_frame_number = -1
        try:
            while not self._stop_event.is_set():
                if self._reload_event.is_set():
                    self._reload_event.clear()
                    try:
                        self.face_module.reload_database()
                        self._faces = []
                        self._counter = 0
                    except Exception:
                        logger.exception("Could not reload registered faces")
                frame_number = self.camera.frame_count
                if frame_number == last_frame_number:
                    self._stop_event.wait(0.02)
                    continue
                frame = self.camera.get_frame()
                if frame is None:
                    self._stop_event.wait(0.02)
                    continue
                last_frame_number = frame_number
                try:
                    result = self.process_frame(frame)
                    if self.frames.full():
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                    self.frames.put_nowait(result)
                except Exception:
                    logger.exception("Frame processing failed")
        finally:
            self.database.close()  # Close the connection belonging to this thread.

    def process_frame(self, frame):
        warnings = []
        if self._counter % max(1, self.config.FACE_DETECTION_INTERVAL) == 0:
            try:
                self._faces = self.face_module.detect_and_recognize(frame)
            except Exception:
                logger.exception("Face detection failed")
                self._faces = []
                warnings.append("Face detection failed")
        if not self.face_module.recognition_available:
            warnings.append("Face recognition unavailable")
        if self._counter % max(1, self.config.YOLO_DETECTION_INTERVAL) == 0:
            try:
                self._objects = self.object_module.detect(frame)
            except Exception:
                logger.exception("Object detection failed")
                self._objects = []
                warnings.append("Object detection failed")
        if not self.object_module.is_available:
            warnings.append("Object detection unavailable")
        elif self.object_module.last_error:
            warnings.append("Object inference failed")
        if self.object_module.unsupported_hazards:
            warnings.append(
                "Model cannot detect: "
                + ", ".join(self.object_module.unsupported_hazards)
            )
        self._counter += 1
        assessment = self.threat_module.assess(self._faces, self._objects)
        fps = self.camera.actual_fps
        annotated = self.alert_system.draw_overlays(
            frame, self._faces, self._objects, assessment, fps=fps
        )
        incident = self.alert_system.process(assessment, annotated)
        if incident is not None:
            self.events.put(incident)
        return FrameResult(
            annotated, self._faces, self._objects, assessment, fps, tuple(warnings)
        )
