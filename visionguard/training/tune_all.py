"""Positive-ROI head fine-tuning of four hazards from the bundled checkpoint.

Because public box annotations can be incomplete, train on labeled positive
regions instead of treating every other proposal as verified background.
Select checkpoints on Open Images TRAIN-side validation only; compare against
the separate public test split after selection.
"""

import argparse
import hashlib
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from visionguard.detection.model import (
    COCO_WEIGHTS,
    FORMAT,
    LABEL_TO_ID,
    build_model,
    train_head_rows,
    validate_checkpoint,
)
from visionguard.training.dataset import load_manifest, split_records
from visionguard.training.finetune import DetectionDataset, collate
from visionguard.training.metrics import box_iou


def positive_roi_loss(model, images, targets):
    """Increase true hazard ROI scores without treating unlabeled photos as BG.

    Open Images may omit hazards outside the annotated labels. Standard R-CNN
    training on incomplete labels aggressively trains every other proposal as
    background and erased useful COCO detections in a first two-epoch probe.
    Ground-truth ROIs give a conservative, positive-only classifier update.
    """
    with torch.no_grad():
        transformed, scaled = model.transform(images, targets)
        features = model.backbone(transformed.tensors)
        embeddings = model.roi_heads.box_roi_pool(
            features, [item["boxes"] for item in scaled], transformed.image_sizes
        )
        embeddings = model.roi_heads.box_head(embeddings)
    logits = model.roi_heads.box_predictor.cls_score(embeddings.detach())
    labels = torch.cat([item["labels"] for item in scaled])
    return F.cross_entropy(logits, labels)


def measure(model, records, threshold=0.4):
    """Greedy per-class IoU>=0.5; report every class and empty-scene alarms."""
    model.eval()
    totals = {label: {"TP": 0, "FP": 0, "FN": 0} for label in LABEL_TO_ID}
    background_alarms = backgrounds = 0
    with torch.inference_mode():
        for record in records:
            image, _ = DetectionDataset([record])[0]
            predicted = model([image])[0]
            selected = [
                (int(cls), float(score), box.tolist())
                for box, score, cls in zip(
                    predicted["boxes"],
                    predicted["scores"],
                    predicted["labels"],
                    strict=True,
                )
                if int(cls) in LABEL_TO_ID.values() and float(score) >= threshold
            ]
            if not record.boxes:
                backgrounds += 1
                background_alarms += bool(selected)
            for label, index in LABEL_TO_ID.items():
                truth = [
                    (x, y, x + w, y + h)
                    for name, (x, y, w, h) in record.boxes
                    if name == label
                ]
                found = sorted(
                    ((score, box) for cls, score, box in selected if cls == index),
                    reverse=True,
                )
                matched = set()
                for _, box in found:
                    match = next(
                        (
                            i
                            for i, target in enumerate(truth)
                            if i not in matched and box_iou(box, target) >= 0.5
                        ),
                        None,
                    )
                    if match is None:
                        totals[label]["FP"] += 1
                    else:
                        totals[label]["TP"] += 1
                        matched.add(match)
                totals[label]["FN"] += len(truth) - len(matched)
    for counts in totals.values():
        tp, fp, fn = (counts[key] for key in ("TP", "FP", "FN"))
        counts["precision"] = tp / max(1, tp + fp)
        counts["recall"] = tp / max(1, tp + fn)
        counts["f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "classes": totals,
        "background_alarms": background_alarms,
        "backgrounds": backgrounds,
    }


def selection_score(report, baseline):
    """Require real knife recall gain and no material other-class regression."""
    classes = report["classes"]
    reference = baseline["classes"]
    # Five scissors on the official test are insufficient for model selection;
    # use TRAIN-side validation, but require no class to lose many matches.
    if classes["knife"]["TP"] <= reference["knife"]["TP"]:
        return None
    for label in LABEL_TO_ID:
        if classes[label]["recall"] < reference[label]["recall"] - 0.06:
            return None
        if classes[label]["f1"] < reference[label]["f1"] - 0.06:
            return None
    if report["background_alarms"] > baseline["background_alarms"] + 2:
        return None
    old_macro = sum(c["f1"] for c in reference.values()) / len(reference)
    new_macro = sum(c["f1"] for c in classes.values()) / len(classes)
    return new_macro if new_macro > old_macro + 0.005 else None


def format_report(name, report):
    print(
        f"{name}: background alarms "
        f"{report['background_alarms']}/{report['backgrounds']}",
        flush=True,
    )
    for label, counts in report["classes"].items():
        print(
            f"  {label:13} TP={counts['TP']:3} FP={counts['FP']:3} FN={counts['FN']:3}"
            f" P={counts['precision']:.1%} R={counts['recall']:.1%}"
            f" F1={counts['f1']:.1%}",
            flush=True,
        )


def run(args):
    if args.epochs < 1 or args.per_class < 1 or args.threads < 1:
        raise ValueError("Epochs, per-class images and threads must be positive")
    if "knife" not in args.classes:
        raise ValueError("Include knife: the selection gate requires a knife gain")
    if not 0 < args.learning_rate <= 0.001:
        raise ValueError("Use a conservative learning rate between 0 and 0.001")
    if Path(args.output).resolve() == Path(args.checkpoint).resolve():
        raise ValueError("Output must differ from the current model (safe rollback)")
    records = load_manifest(args.annotations)
    training, validation = split_records(records, list(LABEL_TO_ID), seed=args.seed)
    rng = random.Random(args.seed)
    chosen = set()
    for label in args.classes:
        candidates = [r for r in training if any(name == label for name, _ in r.boxes)]
        rng.shuffle(candidates)
        chosen.update(candidates[: args.per_class])
    # Positive-only updates must never pass empty/background examples to CE.
    training = sorted((r for r in chosen if r.boxes), key=lambda record: record.path)
    print(
        f"Training on {len(training)} photos; internal validation: {len(validation)}",
        flush=True,
    )
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    original = Path(args.checkpoint)
    checkpoint = torch.load(original, map_location="cpu", weights_only=True)
    enabled = validate_checkpoint(checkpoint)
    if set(enabled) != set(LABEL_TO_ID):
        raise ValueError(
            "All four hazard classes must be enabled in the parent checkpoint"
        )
    parent_sha256 = hashlib.sha256(original.read_bytes()).hexdigest()
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    baseline = measure(model, validation, threshold=args.threshold)
    format_report("Starting checkpoint (train-side validation)", baseline)

    parameters = train_head_rows(model, args.classes)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=0)
    loader = DataLoader(
        DetectionDataset(training, augment=True, seed=args.seed),
        batch_size=1,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.eval()  # Only the prediction rows receive gradients.
        loss_total = 0.0
        for batch_no, (images, targets) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = positive_roi_loss(model, images, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()
            loss_total += float(loss.detach())
            if batch_no % 100 == 0:
                print(f"Epoch {epoch} · {batch_no}/{len(loader)}", flush=True)
        report = measure(model, validation, threshold=args.threshold)
        format_report(
            f"Epoch {epoch} · mean loss {loss_total / len(loader):.3f}", report
        )
        score = selection_score(report, baseline)
        if score is None or (best is not None and score <= best):
            print(
                "  Not selected: insufficient gain or another class regressed",
                flush=True,
            )
            continue
        best = score
        state = {
            "format": FORMAT,
            "label_to_id": LABEL_TO_ID,
            "enabled_classes": enabled,
            "gun_threshold": checkpoint["gun_threshold"],
            "pretrained_weights_url": checkpoint.get(
                "pretrained_weights_url", COCO_WEIGHTS.url
            ),
            "parent_sha256": parent_sha256,
            "training_mode": "positive_rois_head_rows",
            "fine_tuned_classes": list(args.classes),
            "fine_tune_epoch": epoch,
            "internal_validation_before": baseline,
            "internal_validation_after": report,
            "state_dict": {
                name: tensor.detach().cpu().half()
                if tensor.is_floating_point()
                else tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        }
        temporary = output.with_name(output.name + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, output)
        print(f"  Saved selected candidate: {output}", flush=True)
    if best is None:
        print(
            "No candidate passed train-side selection; original model unchanged.",
            flush=True,
        )
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("dataset/openimages_train/annotations.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Config.MODELS_DIR / "visionguard_frcnn.pt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/visionguard_frcnn_candidate.pt"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=list(LABEL_TO_ID),
        default=list(LABEL_TO_ID),
        help="Head rows to update (default: all four hazards)",
    )
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("Confidence threshold must be between zero and one")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
