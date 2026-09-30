"""Train VisionGuard GridNet from YOUR labeled hazard images (no pretrained YOLO)."""

import argparse
import logging
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from modules.custom_detector import (
    FORMAT,
    STRIDE,
    GridNet,
    box_iou,
    decode_predictions,
    detector_loss,
    encode_targets,
    image_tensor,
    validate_checkpoint,
)
from modules.training_data import (
    active_classes,
    load_manifest,
    read_image,
    split_records,
)

logger = logging.getLogger(__name__)


class HazardDataset(Dataset):
    def __init__(self, records, class_names, image_size, augment=False, seed=42):
        self.records = records
        self.class_names = class_names
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_image(record.path)
        boxes = list(record.boxes)
        if self.augment:
            if self.rng.random() < 0.5:
                image = cv2.flip(image, 1)
                boxes = [
                    (label, (image.shape[1] - x - w, y, w, h))
                    for label, (x, y, w, h) in boxes
                ]
            brightness = self.rng.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(
                np.uint8
            )
        tensor, transform = image_tensor(image, self.image_size)
        objects, locations, classes, skipped = encode_targets(
            boxes, self.class_names, transform
        )
        return tensor, objects, locations, classes, record, transform, skipped


def collate_samples(samples):
    images, objects, boxes, classes, records, transforms, skipped = zip(
        *samples, strict=True
    )
    return (
        torch.stack(images),
        torch.stack(objects),
        torch.stack(boxes),
        torch.stack(classes),
        records,
        transforms,
        sum(skipped),
    )


def evaluate(model, loader, device, class_names):
    model.eval()
    loss_sum = 0.0
    true_pos = false_pos = false_neg = dropped = 0
    with torch.inference_mode():
        for images, objects, boxes, labels, records, transforms, skipped in loader:
            logits = model(images.to(device))
            loss, _ = detector_loss(
                logits, objects.to(device), boxes.to(device), labels.to(device)
            )
            loss_sum += float(loss) * len(records)
            dropped += skipped
            for index, (record, transform) in enumerate(
                zip(records, transforms, strict=True)
            ):
                predictions = decode_predictions(
                    logits[index : index + 1],
                    transform,
                    class_names,
                    threshold=0.3,
                )
                expected = [
                    (label, (x, y, x + w, y + h))
                    for label, (x, y, w, h) in record.boxes
                ]
                used = set()
                for label, _, box in predictions:
                    match = next(
                        (
                            i
                            for i, (true_label, true_box) in enumerate(expected)
                            if i not in used
                            and label == true_label
                            and box_iou(box, true_box) >= 0.5
                        ),
                        None,
                    )
                    if match is None:
                        false_pos += 1
                    else:
                        true_pos += 1
                        used.add(match)
                false_neg += len(expected) - len(used)
    precision = true_pos / max(1, true_pos + false_pos)
    recall = true_pos / max(1, true_pos + false_neg)
    return loss_sum / len(loader.dataset), precision, recall, dropped


def train(args):
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.image_size < 128
        or args.image_size > 640
        or args.image_size % STRIDE
    ):
        raise ValueError(
            "Epochs/batch must be positive; image size 128–640, multiple of 16"
        )
    if not 0 < args.learning_rate <= 1:
        raise ValueError("Learning rate must be > 0 and <= 1")
    records = load_manifest(args.annotations, args.images_dir)
    classes = active_classes(records)
    training, validation = split_records(records, classes, args.val_fraction, args.seed)
    counts = Counter(label for record in records for label, _ in record.boxes)
    print(
        f"Images: {len(records)} ({len(training)} train / {len(validation)} validation)"
    )
    print("Training classes: " + ", ".join(f"{c}: {counts[c]} boxes" for c in classes))
    excluded = sorted(set(Config.HAZARDOUS_OBJECTS) - set(classes))
    if excluded:
        print("NOT trained (dashboard will warn): " + ", ".join(excluded))
    print("Include empty/background images and varied real-world examples.")
    if min(counts.values()) < 100:
        print(
            "WARNING: fewer than 100 labeled boxes/class; expect poor generalization."
        )
    if args.dry_run:
        print("Data validation passed. No model was trained.")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    use_pretrained = not args.from_scratch and not args.resume
    model = GridNet(len(classes), pretrained=use_pretrained).to(device)
    print(
        "Feature initialization:",
        "ImageNet MobileNetV3"
        if use_pretrained
        else "checkpoint"
        if args.resume
        else "random",
    )
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        old_classes, old_size = validate_checkpoint(previous, Config.HAZARDOUS_OBJECTS)
        if old_classes != classes or old_size != args.image_size:
            raise ValueError("Resume needs the same class list/order and image size")
        model.load_state_dict(previous["state_dict"], strict=True)
        print(f"Continuing from {args.resume}; optimizer starts fresh.")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": args.learning_rate * (1.0 if args.from_scratch else 0.1),
            },
            {"params": model.head.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=1e-4,
    )
    train_loader = DataLoader(
        HazardDataset(training, classes, args.image_size, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_samples,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        HazardDataset(validation, classes, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    patience = 0
    print(
        f"Device: {device} · model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        dropped = 0
        for images, objects, boxes, labels, records, _, skipped in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss, _ = detector_loss(
                logits, objects.to(device), boxes.to(device), labels.to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(records)
            dropped += skipped
        val_loss, precision, recall, val_dropped = evaluate(
            model, val_loader, device, classes
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}"
            f" · train loss {total_loss / len(training):.3f}"
            f" · val loss {val_loss:.3f} · precision@0.5 {precision:.2%}"
            f" · recall@0.5 {recall:.2%}"
        )
        if dropped or val_dropped:
            logger.warning(
                "Skipped %d train / %d validation boxes: >2 object centers in one cell",
                dropped,
                val_dropped,
            )
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            checkpoint = {
                "format": FORMAT,
                "classes": classes,
                "image_size": args.image_size,
                "epoch": epoch,
                "val_loss": float(val_loss),
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
            }
            temporary = output.with_name(output.name + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, output)
        else:
            patience += 1
        if patience >= args.patience:
            print(f"Stopped early after {args.patience} epochs without improvement.")
            break
    print(f"Saved best weights to: {output}")
    print("Press Reload detector in the open monitor, or run python main.py.")
    print("Always verify detection/false alarms on NEW scenes not used for training.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=Path("dataset/annotations.json")
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Config.MODELS_DIR / "visionguard_gridnet.pt"
    )
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--resume", type=Path, help="Resume weights with same classes/size"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Skip ImageNet feature initialization (usually less accurate)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate labels without training"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        train(parse_args())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Training error: {exc}") from exc
