"""Check your trained GridNet on a NEW still image (no webcam needed)."""

import argparse
from pathlib import Path

import cv2

from config import Config
from modules.object_detection import ObjectDetectionModule
from modules.training_data import read_image


def predict(image_path, output_path=None, model_path=None, threshold=None):
    image_path = Path(image_path)
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    settings = type(
        "PredictionSettings",
        (Config,),
        {
            "DETECTOR_MODEL": Path(model_path or Config.DETECTOR_MODEL),
            "DETECTION_CONFIDENCE_THRESHOLD": (
                threshold
                if threshold is not None
                else Config.DETECTION_CONFIDENCE_THRESHOLD
            ),
        },
    )
    detector = ObjectDetectionModule(settings)
    if not detector.is_available:
        raise ValueError(detector.unavailable_reason)
    frame = read_image(image_path)
    results = detector.detect(frame)
    if detector.last_error:
        raise ValueError("Inference failed; see the error logged above")
    for item in results:
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), item.color, 2)
        cv2.putText(
            frame,
            f"{item.label} {item.confidence:.2f}",
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            item.color,
            2,
        )
        print(f"{item.label}: {item.confidence:.2%} at {item.bbox}")
    if not results:
        print("No hazard above threshold. This does NOT mean the image is safe.")
    output = Path(
        output_path
        or Config.DATA_DIR / "predictions" / f"{image_path.stem}_detections.jpg"
    )
    if output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Output must end in .jpg, .jpeg or .png")
    output.parent.mkdir(parents=True, exist_ok=True)
    extension = ".png" if output.suffix.lower() == ".png" else ".jpg"
    ok, encoded = cv2.imencode(extension, frame)
    if not ok:
        raise ValueError("Could not encode the annotated image")
    encoded.tofile(str(output))
    print(f"Annotated image: {output}")
    return results, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    try:
        predict(args.image, args.output, args.model, args.threshold)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Prediction error: {exc}") from exc


if __name__ == "__main__":
    main()
