"""Validate and register a single face image without depending on Tkinter."""

import logging
import uuid
from pathlib import Path

import cv2
import numpy as np

from config import Config
from visionguard.core.database import Database

logger = logging.getLogger(__name__)


class RegistrationError(ValueError):
    pass


def register_person(name, image_path, config=Config, database=None):
    name = name.strip()
    if not name or len(name) > 80 or any(ord(c) < 32 for c in name):
        raise RegistrationError("Enter a name of 1–80 printable characters")
    source = Path(image_path)
    if not source.is_file():
        raise RegistrationError("Select an existing image")
    if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise RegistrationError("Select a JPG, PNG or BMP image")
    if source.stat().st_size > 20 * 1024 * 1024:
        raise RegistrationError("The image must be smaller than 20 MB")

    try:
        image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise RegistrationError("Could not read the selected image") from exc
    if image is None:
        raise RegistrationError("The selected file is not a valid image")
    if image.shape[0] > 6000 or image.shape[1] > 6000:
        raise RegistrationError("The image must be at most 6000 pixels on each side")
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RegistrationError(
            "Face detection is unavailable in this OpenCV installation"
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    if len(faces) != 1:
        raise RegistrationError("Use a clear photo showing exactly one detectable face")

    config.create_dirs()
    owns_database = database is None
    db = database if database is not None else Database(config.DATABASE_PATH)
    target = config.FACES_DIR / f"{uuid.uuid4().hex}.jpg"
    try:
        previous = db.get_person(name)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not success:
            raise RegistrationError("Could not encode the selected image")
        encoded.tofile(str(target))  # Supports Unicode paths on Windows; strips EXIF.
        try:
            db.register_person(name, engine_used="opencv-lbph", image_path=str(target))
        except Exception:
            target.unlink(missing_ok=True)
            raise
        # Never delete arbitrary paths stored in a pre-existing database.
        if previous and previous["image_path"]:
            old = Path(previous["image_path"])
            if not old.is_absolute():
                old = config.BASE_DIR / old
            if (
                old.resolve().is_relative_to(config.FACES_DIR.resolve())
                and old != target
            ):
                try:
                    old.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove old face image: %s", old)
        return target
    finally:
        if owns_database:
            db.close()
