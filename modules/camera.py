"""
Camera Module
=============
Thread-safe camera capture.
"""

import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)


class Camera:
    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720,
        fps=30
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.frame = None

        self.running = False
        self.lock = threading.Lock()

        self.frame_count = 0
        self.actual_fps = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.running = True

        threading.Thread(
            target=self._capture_loop,
            daemon=True
        ).start()

        return True

    def stop(self):
        self.running = False

        if self.cap:
            self.cap.release()

    @property
    def is_available(self):
        return (
            self.cap is not None
            and self.cap.isOpened()
        )

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def _capture_loop(self):
        last_time = time.time()
        frames = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

            frames += 1
            self.frame_count += 1

            now = time.time()

            if now - last_time >= 1:
                self.actual_fps = frames / (now - last_time)
                frames = 0
                last_time = now