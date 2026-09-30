"""Legacy GridNet experiment only; NOT used by the live application."""

import argparse
from pathlib import Path

import cv2
import torch

from config import Config
from visionguard.legacy.gridnet import (
    GridNet,
    decode_predictions,
    image_tensor,
    validate_checkpoint,
)
from visionguard.training.dataset import read_image


def predict(image_path, output_path=None, model_path=None, threshold=0.45):
    checkpoint_path = Path(model_path or Config.MODELS_DIR / "visionguard_gridnet.pt")
    if not checkpoint_path.is_file():
        raise ValueError("No legacy GridNet checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes, size = validate_checkpoint(checkpoint, Config.HAZARDOUS_OBJECTS)
    model = GridNet(len(classes)).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    frame = read_image(Path(image_path))
    tensor, transform = image_tensor(frame, size)
    with torch.inference_mode():
        boxes = decode_predictions(
            model(tensor.unsqueeze(0)), transform, classes, threshold=threshold
        )
    for label, confidence, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 100, 255),
            2,
        )
    output = Path(output_path or Config.DATA_DIR / "predictions" / "gridnet.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(
        ".png" if output.suffix.lower() == ".png" else ".jpg", frame
    )
    if not ok:
        raise ValueError("Cannot encode preview")
    data.tofile(str(output))
    return boxes, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predict(args.image, args.output, args.model)


if __name__ == "__main__":
    main()
