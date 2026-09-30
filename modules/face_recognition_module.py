import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class FaceResult:
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    is_known: bool


class FaceRecognitionModule:

    def __init__(self, config, database):

        self.config = config
        self.database = database

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.label_to_name = {}
        self.trained = False

        self.train_model()

    def train_model(self):

        faces_dir = Path("data/faces")

        if not faces_dir.exists():
            return

        images = []
        labels = []

        current_label = 0

        for image_file in faces_dir.iterdir():

            if image_file.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                continue

            name = image_file.stem

            image = cv2.imread(
                str(image_file),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            detected = self.face_cascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5
            )

            if len(detected) == 0:
                continue

            x, y, w, h = detected[0]

            face = image[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (200, 200)
            )

            images.append(face)
            labels.append(current_label)

            self.label_to_name[current_label] = name

            current_label += 1

        if len(images) > 0:

            self.recognizer.train(
                images,
                np.array(labels)
            )

            self.trained = True

            print(
                f"Trained on {len(images)} person(s)"
            )

    def reload_database(self):
        self.train_model()

    def detect_and_recognize(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        results = []

        for (x, y, w, h) in faces:

            name = "Unknown"
            is_known = False
            confidence_score = 0.0

            if self.trained:

                roi = gray[y:y+h, x:x+w]

                roi = cv2.resize(
                    roi,
                    (200, 200)
                )

                label, confidence = (
                    self.recognizer.predict(roi)
                )

                if confidence < 80:

                    name = self.label_to_name.get(
                        label,
                        "Unknown"
                    )

                    is_known = True

                    confidence_score = max(
                        0,
                        1 - (confidence / 100)
                    )

            results.append(
                FaceResult(
                    name=name,
                    confidence=confidence_score,
                    bbox=(x, y, x+w, y+h),
                    is_known=is_known
                )
            )

        return results

    def register_face(self, name, frames):
        return True

    def count_registered(self):
        return len(self.label_to_name)

    def has_registered_faces(self):
        return len(self.label_to_name) > 0