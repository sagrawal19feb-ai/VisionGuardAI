"""Fine-tune a non-YOLO Faster R-CNN gun head; retain COCO's 3 other hazards."""

import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from visionguard.training.metrics import box_iou
from visionguard.detection.model import (
    COCO_WEIGHTS,
    FORMAT,
    LABEL_TO_ID,
    build_model,
    train_gun_head_only,
    validate_checkpoint,
)
from visionguard.training.dataset import load_manifest, read_image, split_records

logger = logging.getLogger(__name__)


class DetectionDataset(Dataset):
    def __init__(self, records, augment=False, seed=42):
        self.records = records
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_image(record.path)
        boxes = list(record.boxes)
        if self.augment and self.rng.random() < 0.5:
            image = cv2.flip(image, 1)
            boxes = [
                (label, (image.shape[1] - x - w, y, w, h))
                for label, (x, y, w, h) in boxes
            ]
        tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy())
        tensor = tensor.permute(2, 0, 1).float() / 255.0
        coords = [[x, y, x + w, y + h] for _, (x, y, w, h) in boxes]
        target = {
            "boxes": torch.tensor(coords, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(
                [LABEL_TO_ID[label] for label, _ in boxes], dtype=torch.int64
            ),
        }
        return tensor, target


def collate(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def evaluate_gun(model, records, threshold=0.4):
    """Gun-specific validation. Do not use the held-out official test set here."""
    model.eval()
    tp = fp = fn = background_alarms = background_count = 0
    with torch.inference_mode():
        for record in records:
            image, _ = DetectionDataset([record])[0]
            output = model([image])[0]
            expected = [
                (x, y, x + w, y + h)
                for label, (x, y, w, h) in record.boxes
                if label == "gun"
            ]
            selected = [
                box.tolist()
                for box, score, label in zip(
                    output["boxes"], output["scores"], output["labels"], strict=True
                )
                if int(label) == LABEL_TO_ID["gun"] and float(score) >= threshold
            ]
            matched = set()
            for box in selected:
                found = next(
                    (
                        i
                        for i, truth in enumerate(expected)
                        if i not in matched and box_iou(box, truth) >= 0.5
                    ),
                    None,
                )
                if found is None:
                    fp += 1
                else:
                    tp += 1
                    matched.add(found)
            fn += len(expected) - len(matched)
            if not record.boxes:
                background_count += 1
                background_alarms += bool(selected)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "background_alarms": background_alarms,
        "background_count": background_count,
    }


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size and patience must be positive")
    records = load_manifest(args.annotations)
    classes = list(LABEL_TO_ID)
    training, validation = split_records(records, classes, args.val_fraction, args.seed)
    # Existing three COCO labels are already strong. Prioritize gun photos and
    # retain representative other objects/empty scenes as hard negatives.
    rng = random.Random(args.seed)
    selected = {r for r in training if any(label == "gun" for label, _ in r.boxes)}
    for label in ("knife", "scissors", "baseball bat"):
        candidates = [
            r
            for r in training
            if r not in selected and any(name == label for name, _ in r.boxes)
        ]
        rng.shuffle(candidates)
        selected.update(candidates[:80])
    backgrounds = [r for r in training if not r.boxes]
    rng.shuffle(backgrounds)
    selected.update(backgrounds[:100])
    training = sorted(selected, key=lambda r: r.path)
    # Evaluate every gun-containing validation photo and a seeded sample of
    # non-gun photos to see false alerts, without making each epoch too slow.
    validation_guns = [
        r for r in validation if any(label == "gun" for label, _ in r.boxes)
    ]
    validation_others = [r for r in validation if r not in validation_guns]
    random.Random(args.seed).shuffle(validation_others)
    validation_sample = (
        validation_guns + validation_others[: min(80, len(validation_others))]
    )
    print(
        f"Train photos: {len(training)} · internal validation: {len(validation)}"
        f" · gun-check photos: {len(validation_sample)}"
    )
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model = build_model(pretrained=not args.resume)
    # The default 2,000 training proposals make CPU fine-tuning ~25s/step.
    # ROI heads always add each ground-truth box during training; keep the
    # inference proposal settings unchanged for COCO-compatible evaluation.
    model.rpn._pre_nms_top_n["training"] = 100
    model.rpn._post_nms_top_n["training"] = 50
    model.roi_heads.fg_bg_sampler.batch_size_per_image = 64
    model.rpn.fg_bg_sampler.batch_size_per_image = 64
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print("Continuing from", args.resume)
    parameters = train_gun_head_only(model)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=0)
    loader = DataLoader(
        DetectionDataset(training, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    no_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_number, (images, targets) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            if batch_number % 60 == 0:
                print(
                    f"Epoch {epoch}: {batch_number}/{len(loader)} batches", flush=True
                )
        report = evaluate_gun(model, validation_sample, threshold=args.gun_threshold)
        print(
            f"Epoch {epoch:02}/{args.epochs}"
            f" · train loss {total_loss / len(training):.3f}"
            f" · gun TP={report['TP']} FP={report['FP']} FN={report['FN']}"
            f" · P={report['precision']:.1%} R={report['recall']:.1%}"
            f" · negative alarms {report['background_alarms']}"
            f"/{report['background_count']}",
            flush=True,
        )
        if report["f1"] > best:
            best = report["f1"]
            no_improvement = 0
            # Always preserve the COCO-trained three. Enable guns only if
            # internal validation passes minimum quality gates; public data
            # cannot certify webcam accuracy.
            gun_enabled = (
                report["precision"] >= 0.4
                and report["recall"] >= 0.2
                and report["TP"] >= 8
            )
            enabled = ["knife", "scissors", "baseball bat"] + (
                ["gun"] if gun_enabled else []
            )
            checkpoint = {
                "format": FORMAT,
                "label_to_id": LABEL_TO_ID,
                "enabled_classes": enabled,
                "gun_threshold": args.gun_threshold,
                "gun_internal_validation": report,
                "epoch": epoch,
                "pretrained_weights_url": COCO_WEIGHTS.url,
                "state_dict": {
                    k: (
                        v.detach().cpu().half()
                        if v.is_floating_point()
                        else v.detach().cpu()
                    )
                    for k, v in model.state_dict().items()
                },
            }
            temporary = output.with_name(output.name + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, output)
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            print(f"Early stop after {args.patience} unimproved epochs")
            break
    print("Saved:", output)
    saved = torch.load(output, map_location="cpu", weights_only=True)
    print("Active hazard classes:", ", ".join(saved["enabled_classes"]))
    print("External official test set is untouched by this training script.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("dataset/openimages_train/annotations.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/visionguard_retrained.pt")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gun-threshold", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
