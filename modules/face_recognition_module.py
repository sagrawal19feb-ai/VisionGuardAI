"""Haar face detection and LBPH matching against active database registrations."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    name: str
    confidence: float  # relative LBPH match score, NOT a calibrated probability
    bbox: Tuple[int, int, int, int]
    is_known: bool


class FaceRecognitionModule:
    def __init__(self, config, database):
        self.config = config
        self.database = database
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("OpenCV face cascade is unavailable")
        self.recognizer = None
        self.label_to_name = {}
        self.trained = False
        self.train_model()

    @property
    def recognition_available(self):
        return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")

    def train_model(self):
        # A new model ensures removed/deactivated persons no longer match.
        self.label_to_name = {}
        self.trained = False
        if not self.recognition_available:
            self.recognizer = None
            logger.warning(
                "LBPH unavailable; install opencv-contrib-python. "
                "Faces will be detected as unknown."
            )
            return
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        images, labels = [], []
        for person in self.database.get_all_persons():
            if not person["image_path"]:
                continue
            path = Path(person["image_path"])
            if not path.is_absolute():
                path = self.config.BASE_DIR / path
            try:
                image = cv2.imdecode(
                    np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
            except (OSError, cv2.error):
                image = None
            if image is None:
                logger.warning("Cannot read registered face image: %s", path)
                continue
            found = self.face_cascade.detectMultiScale(
                image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            if len(found) != 1:
                logger.warning("Expected one face in registered image: %s", path)
                continue
            x, y, w, h = found[0]
            roi = cv2.resize(image[y : y + h, x : x + w], (200, 200))
            label = len(images)
            images.append(roi)
            labels.append(label)
            self.label_to_name[label] = person["name"]
        if images:
            self.recognizer.train(images, np.asarray(labels, dtype=np.int32))
            self.trained = True
        logger.info("Face model trained on %d active person(s)", len(images))

    def reload_database(self):
        self.train_model()

    def detect_and_recognize(self, frame) -> List[FaceResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        results = []
        threshold = self.config.FACE_RECOGNITION_DISTANCE_THRESHOLD
        for x, y, w, h in faces:
            name, known, score = "Unknown", False, 0.0
            if self.trained:
                roi = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
                try:
                    label, distance = self.recognizer.predict(roi)
                    if distance < threshold and label in self.label_to_name:
                        name = self.label_to_name[label]
                        known = True
                        score = max(0.0, min(1.0, 1.0 - distance / threshold))
                except cv2.error:
                    logger.exception(
                        "Face recognition failed; treating face as unknown"
                    )
            results.append(FaceResult(name, score, (x, y, x + w, y + h), known))
        return results

    def count_registered(self):
        return len(self.label_to_name)

    def has_registered_faces(self):
        return self.trained
