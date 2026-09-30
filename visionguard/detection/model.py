"""Non-YOLO quality detector: COCO-trained Faster R-CNN with a learned gun class.

The pretrained architecture/weights belong to TorchVision, not this project.
VisionGuard extends its ROI classifier to detect guns and preserves the
pretrained COCO classes for knife, scissors and baseball bat.
"""

import math

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

FORMAT = "visionguard-frcnn-v1"
LABEL_TO_ID = {"baseball bat": 39, "knife": 49, "scissors": 87, "gun": 91}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
COCO_WEIGHTS = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT


def build_model(pretrained=False):
    """Build a 92-class model; only pretrained=True downloads initial weights."""
    if not pretrained:
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=92
        )
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=COCO_WEIGHTS)
    old = model.roi_heads.box_predictor
    new = FastRCNNPredictor(old.cls_score.in_features, 92)
    with torch.no_grad():
        new.cls_score.weight[:91].copy_(old.cls_score.weight)
        new.cls_score.bias[:91].copy_(old.cls_score.bias)
        new.bbox_pred.weight[:364].copy_(old.bbox_pred.weight)
        new.bbox_pred.bias[:364].copy_(old.bbox_pred.bias)
        # Start gun near a similar existing localization prior, but with a
        # conservative score until enough labeled gun photos are learned.
        new.cls_score.weight[91].copy_(old.cls_score.weight[LABEL_TO_ID["knife"]])
        new.cls_score.bias[91].copy_(old.cls_score.bias[LABEL_TO_ID["knife"]] - 3)
        gun = slice(91 * 4, 92 * 4)
        knife = slice(LABEL_TO_ID["knife"] * 4, (LABEL_TO_ID["knife"] + 1) * 4)
        new.bbox_pred.weight[gun].copy_(old.bbox_pred.weight[knife])
        new.bbox_pred.bias[gun].copy_(old.bbox_pred.bias[knife])
    model.roi_heads.box_predictor = new
    return model


def train_head_rows(model, labels):
    """Update only named hazard predictor rows, preserving all other COCO rows.

    Use zero weight decay: optimizer weight decay could modify frozen rows even
    when their gradients are masked. The backbone and proposal network remain
    frozen; the model's weights-only checkpoint format stays unchanged.
    """
    selected = {LABEL_TO_ID[label] for label in labels}
    if not selected or len(selected) != len(labels):
        raise ValueError("Select distinct trained hazard labels")
    model.requires_grad_(False)
    predictor = model.roi_heads.box_predictor
    predictor.requires_grad_(True)
    for param in (predictor.cls_score.weight, predictor.cls_score.bias):
        mask = torch.zeros_like(param)
        for cls in selected:
            mask[cls] = 1
        param.register_hook(lambda grad, keep=mask: grad * keep)
    for param in (predictor.bbox_pred.weight, predictor.bbox_pred.bias):
        mask = torch.zeros_like(param)
        for cls in selected:
            mask[cls * 4 : (cls + 1) * 4] = 1
        param.register_hook(lambda grad, keep=mask: grad * keep)
    return list(predictor.parameters())


def train_gun_head_only(model):
    """Keep trusted COCO rows fixed when learning a gun class from scratch."""
    return train_head_rows(model, ["gun"])


def validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError("Not a VisionGuard Faster R-CNN checkpoint")
    if checkpoint.get("label_to_id") != LABEL_TO_ID:
        raise ValueError("Checkpoint class mapping mismatch")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Checkpoint has no model weights")
    enabled = checkpoint.get("enabled_classes")
    if (
        not isinstance(enabled, list)
        or not enabled
        or not all(isinstance(c, str) and c in LABEL_TO_ID for c in enabled)
        or len(enabled) != len(set(enabled))
    ):
        raise ValueError("Checkpoint coverage metadata is invalid")
    threshold = checkpoint.get("gun_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("Checkpoint gun threshold must be between zero and one")
    return enabled
