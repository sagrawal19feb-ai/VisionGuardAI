"""Background camera capture with a thread-safe latest-frame buffer."""

import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        self.camera_index = camera_index
        self.width, self.height, self.fps = width, height, fps
        self.cap = None
        self.frame = None
        self.frame_count = 0
        self.actual_fps = 0.0
        self.running = False
        self.lock = threading.Lock()
        self._thread = None

    def start(self):
        if self.running:
            return True
        if self._thread is not None and self._thread.is_alive():
            logger.error("Previous camera capture has not stopped yet")
            return False
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error("Failed to open camera %s", self.camera_index)
            self.cap.release()
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="camera-capture", daemon=True
        )
        self._thread.start()
        return True

    def stop(self):
        self.running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        if self._thread and not self._thread.is_alive():
            self._thread = None
        elif self._thread:
            logger.warning("Camera capture did not stop within 2 seconds")

    @property
    def is_available(self):
        return self.running and self.cap is not None and self.cap.isOpened()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def _capture_loop(self):
        last_time = time.monotonic()
        frames = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
            except cv2.error:
                logger.exception("Camera read failed")
                break
            if not ret:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = frame
                self.frame_count += 1
            frames += 1
            now = time.monotonic()
            if now - last_time >= 1:
                self.actual_fps = frames / (now - last_time)
                frames = 0
                last_time = now
        self.running = False
