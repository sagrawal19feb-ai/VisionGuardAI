"""Build an attributed, labeled Open Images V7 subset without YOLO/FiftyOne.

The public CSVs are streamed, never stored in full. Images and annotations
remain local under dataset/ (Git-ignored). Individual image licenses are NOT
verified by this utility; see DATA_PROVENANCE.md before redistribution.
"""

import argparse
import csv
import json
import logging
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

BOX_CSV = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"
# Official Open Images MID -> VisionGuard policy class (gun combines 2 labels).
CLASS_IDS = {
    "/m/0gxl3": "gun",  # Handgun
    "/m/06c54": "gun",  # Rifle
    "/m/04ctx": "knife",  # Knife
    "/m/058qzx": "knife",  # Kitchen knife
    "/m/01lsmm": "scissors",  # Scissors
    "/m/03g8mr": "baseball bat",  # Baseball bat
}
BACKGROUND_NAMES = {"Laptop", "Chair", "Bottle", "Computer keyboard"}
CLASS_CSV = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
)


def background_mids():
    r = requests.get(CLASS_CSV, timeout=30)
    r.raise_for_status()
    return {
        mid for mid, name in csv.reader(r.text.splitlines()) if name in BACKGROUND_NAMES
    }


def read_boxes(split, background_labels, seed):
    positives = defaultdict(list)
    negative_candidates = []
    seen_negatives = set()
    rng = random.Random(seed)
    background_seen = 0
    url = BOX_CSV[split]
    logger.info("Streaming %s bounding box annotations from %s", split, url)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for row in csv.DictReader(response.iter_lines(decode_unicode=True)):
            mid = row["LabelName"]
            image_id = row["ImageID"]
            if mid in CLASS_IDS:
                if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                    continue
                x1, x2 = float(row["XMin"]), float(row["XMax"])
                y1, y2 = float(row["YMin"]), float(row["YMax"])
                if x2 <= x1 or y2 <= y1:
                    continue
                positives[image_id].append((CLASS_IDS[mid], x1, y1, x2, y2))
            elif mid in background_labels and image_id not in seen_negatives:
                seen_negatives.add(image_id)
                background_seen += 1
                if len(negative_candidates) < 5000:
                    negative_candidates.append(image_id)
                else:
                    slot = rng.randrange(background_seen)
                    if slot < 5000:
                        negative_candidates[slot] = image_id
    logger.info(
        "%s: %d target images, %d background candidates",
        split,
        len(positives),
        len(negative_candidates),
    )
    return positives, negative_candidates


def select_images(positives, backgrounds, per_class, negative_count, seed):
    rng = random.Random(seed)
    selected = set()
    counts = Counter()
    for label in ("gun", "knife", "scissors", "baseball bat"):
        candidates = [
            image_id
            for image_id, boxes in positives.items()
            if any(box[0] == label for box in boxes)
        ]
        rng.shuffle(candidates)
        for image_id in candidates:
            if counts[label] >= per_class:
                break
            if image_id not in selected:
                selected.add(image_id)
                counts.update({name for name, *_ in positives[image_id]})
        if counts[label] < per_class:
            logger.warning(
                "Only %d unique %s photos (asked for %d)",
                counts[label],
                label,
                per_class,
            )
    background_only = [
        image_id for image_id in backgrounds if image_id not in positives
    ]
    rng.shuffle(background_only)
    if len(background_only) < negative_count:
        logger.warning("Only %d usable background photos", len(background_only))
    for image_id in background_only[:negative_count]:
        selected.add(image_id)
    print(
        "Selected images by class (photos, not boxes):",
        dict(counts),
        "+ backgrounds:",
        min(negative_count, len(background_only)),
    )
    return [(image_id, positives.get(image_id, [])) for image_id in sorted(selected)]


def fetch_one(split, image_id, boxes, images_dir, max_dimension):
    url = IMAGE_URL.format(split=split, id=image_id)
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 50))
            response.raise_for_status()
            if len(response.content) > 10_000_000:
                raise ValueError("Image exceeds 10 MB")
            image = cv2.imdecode(
                np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError("Unsupported image format")
            height, width = image.shape[:2]
            ratio = min(1.0, max_dimension / max(height, width))
            if ratio < 1:
                image = cv2.resize(
                    image,
                    (round(width * ratio), round(height * ratio)),
                    interpolation=cv2.INTER_AREA,
                )
            height, width = image.shape[:2]
            annotations = []
            for label, left, top, right, bottom in boxes:
                x1, x2 = round(left * width), round(right * width)
                y1, y2 = round(top * height), round(bottom * height)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    annotations.append(
                        {"label": label, "bbox": [x1, y1, x2 - x1, y2 - y1]}
                    )
            if boxes and not annotations:
                raise ValueError("No object survives resize")
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise ValueError("Cannot encode image")
            filename = f"{split}_{image_id}.jpg"
            encoded.tofile(str(images_dir / filename))
            return {
                "file": filename,
                "boxes": annotations,
                "source_id": f"{split}/{image_id}",
                "source_url": url,
            }
        except (requests.RequestException, ValueError, cv2.error) as exc:
            if attempt == 2:
                logger.warning("Skipped %s: %s", url, exc)
                return None
            time.sleep(0.4 * (attempt + 1))


def prepare(split, desired_per_class, backgrounds, destination, size, workers, seed):
    mids = background_mids()
    positives, negative_candidates = read_boxes(split, mids, seed)
    selected = select_images(
        positives, negative_candidates, desired_per_class, backgrounds, seed
    )
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_one, split, image_id, boxes, images_dir, size)
            for image_id, boxes in selected
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                records.append(result)
            if index % 100 == 0:
                print(f"Downloaded {index}/{len(futures)} {split} photos")
    records.sort(key=lambda item: item["file"])
    manifest = destination / "annotations.json"
    manifest.write_text(
        json.dumps({"version": 1, "images": records}, indent=2) + "\n", encoding="utf-8"
    )
    counts = Counter(box["label"] for row in records for box in row["boxes"])
    report = {
        "dataset": "Open Images V7",
        "split": split,
        "boxes_source": BOX_CSV[split],
        "image_url_template": IMAGE_URL,
        "photographs": len(records),
        "boxes_per_class": counts,
        "manifest": str(manifest),
        "license_notice": (
            "See DATA_PROVENANCE.md; source image licenses are NOT "
            "individually verified."
        ),
    }
    (destination / "provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared:", manifest, "photos:", len(records), "boxes:", dict(counts))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--per-class", type=int, default=160)
    parser.add_argument("--backgrounds", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-dimension", type=int, default=640)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        min(args.per_class, args.max_dimension, args.workers) < 1
        or args.backgrounds < 0
    ):
        parser.error("Counts/dimension/workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    destination = args.output or Path("dataset") / f"openimages_{args.split}"
    prepare(
        args.split,
        args.per_class,
        args.backgrounds,
        destination,
        args.max_dimension,
        args.workers,
        args.seed,
    )


if __name__ == "__main__":
    main()
