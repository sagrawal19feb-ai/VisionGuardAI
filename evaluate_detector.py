"""Report class-wise held-out precision/recall and background false alerts."""

import argparse
from collections import Counter
from pathlib import Path

from config import Config
from modules.custom_detector import box_iou
from modules.object_detection import ObjectDetectionModule
from modules.training_data import load_manifest, read_image


def evaluate(
    manifest, model_path=Config.DETECTOR_MODEL, images_dir=None, thresholds=(0.45,)
):
    records = load_manifest(manifest, images_dir)
    settings = type("EvalSettings", (Config,), {"DETECTOR_MODEL": Path(model_path)})
    detector = ObjectDetectionModule(settings)
    if not detector.is_available:
        raise ValueError(detector.unavailable_reason)
    print(
        f"Held-out photos: {len(records)} · model classes: "
        f"{', '.join(detector.class_names)}"
    )
    predictions = []
    for record in records:
        detector.config.DETECTION_CONFIDENCE_THRESHOLD = min(thresholds)
        found = detector.detect(read_image(record.path))
        if detector.last_error:
            raise ValueError("Model inference failed; check logs")
        predictions.append(found)
    for threshold in thresholds:
        stats = {label: Counter() for label in detector.class_names}
        negatives = negative_alarms = 0
        for record, found in zip(records, predictions, strict=True):
            expected = [
                (label, (x, y, x + w, y + h)) for label, (x, y, w, h) in record.boxes
            ]
            filtered = [det for det in found if det.confidence >= threshold]
            if not expected:
                negatives += 1
                negative_alarms += bool(filtered)
            for label in detector.class_names:
                target = [box for name, box in expected if name == label]
                candidates = sorted(
                    (det for det in filtered if det.label.lower() == label),
                    key=lambda det: -det.confidence,
                )
                used = set()
                for det in candidates:
                    match = next(
                        (
                            idx
                            for idx, box in enumerate(target)
                            if idx not in used and box_iou(det.bbox, box) >= 0.5
                        ),
                        None,
                    )
                    if match is None:
                        stats[label]["FP"] += 1
                    else:
                        stats[label]["TP"] += 1
                        used.add(match)
                stats[label]["FN"] += len(target) - len(used)
        print(f"\nConfidence >= {threshold:.2f} | matching IoU >= 0.50")
        for label, counts in stats.items():
            tp, fp, fn = (counts[k] for k in ("TP", "FP", "FN"))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            print(
                f"{label:13} TP={tp:3} FP={fp:4} FN={fn:3}"
                f" precision={precision:6.1%} recall={recall:6.1%}"
            )
        print(f"Background images with >=1 alarm: {negative_alarms}/{negatives}")
    print("Public-image results do NOT measure performance on your webcam.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("dataset/openimages_test/annotations.json"),
    )
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--model", type=Path, default=Config.DETECTOR_MODEL)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.45, 0.7])
    args = parser.parse_args()
    if not args.thresholds or any(not 0 < n < 1 for n in args.thresholds):
        parser.error("Every threshold must be between 0 and 1")
    try:
        evaluate(args.annotations, args.model, args.images_dir, tuple(args.thresholds))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
