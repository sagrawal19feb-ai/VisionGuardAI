"""Simple annotated-image manifest shared by the labeler and the trainer."""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import Config


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    boxes: tuple  # (label, (x, y, width, height)) in original pixels


def read_image(path):
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f"Could not read image: {path}") from exc
    if image is None:
        raise ValueError(f"Invalid image: {path}")
    return image


def load_manifest(annotations, images_dir=None, allowed_classes=None):
    """Reject invalid images, boxes, labels, duplicates and unsafe paths."""
    annotations = Path(annotations)
    images_dir = Path(images_dir or annotations.parent / "images").resolve()
    allowed = set(allowed_classes or Config.HAZARDOUS_OBJECTS)
    try:
        manifest = json.loads(annotations.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read annotations: {annotations}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("images"), list)
    ):
        raise ValueError("Annotations must have version 1 and an images list")
    records, seen = [], set()
    for entry in manifest["images"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("Every image needs a file name")
        path = (images_dir / entry["file"]).resolve()
        if not path.is_relative_to(images_dir) or path in seen:
            raise ValueError(f"Duplicate or outside-images path: {entry['file']}")
        seen.add(path)
        image = read_image(path)
        height, width = image.shape[:2]
        if not isinstance(entry.get("boxes"), list):
            raise ValueError(
                f"Image needs a boxes list (use [] for no hazards): {path}"
            )
        boxes = []
        for box in entry["boxes"]:
            if not isinstance(box, dict) or box.get("label") not in allowed:
                raise ValueError(f"Unknown hazard label in {path}: {box}")
            values = box.get("bbox")
            if (
                not isinstance(values, list)
                or len(values) != 4
                or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in values
                )
            ):
                raise ValueError(f"Invalid [x, y, width, height] box in {path}")
            x, y, w, h = values
            if x < 0 or y < 0 or w < 2 or h < 2 or x + w > width or y + h > height:
                raise ValueError(f"Box is outside the image: {path}: {values}")
            boxes.append((box["label"], tuple(float(v) for v in values)))
        records.append(ImageRecord(path, tuple(boxes)))
    if len(records) < 2:
        raise ValueError(
            "Label at least two different images (and include background images)"
        )
    if not any(record.boxes for record in records):
        raise ValueError("No hazardous-object boxes found in annotations")
    return records


def active_classes(records):
    labels = {label for record in records for label, _ in record.boxes}
    return [label for label in Config.HAZARDOUS_OBJECTS if label in labels]


def split_records(records, classes, val_fraction=0.2, seed=42):
    """Keep every trained class in both train and validation sets."""
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    for label in classes:
        count = sum(
            any(name == label for name, _ in record.boxes) for record in records
        )
        if count < 2:
            raise ValueError(f"Add at least two different images containing {label}")
    rng = random.Random(seed)
    val_count = min(
        len(records) - 1, max(1, round(len(records) * val_fraction), len(classes))
    )
    shuffled = list(records)
    for _ in range(1000):
        rng.shuffle(shuffled)
        val, train = shuffled[:val_count], shuffled[val_count:]
        if all(
            any(any(name == label for name, _ in record.boxes) for record in group)
            for label in classes
            for group in (train, val)
        ):
            return list(train), list(val)
    raise ValueError(
        "Cannot split every class into train and validation; add more varied images"
    )
