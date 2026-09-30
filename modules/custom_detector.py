"""VisionGuard GridNet: a small, original anchor-free convolutional detector.

Each 16-pixel grid cell predicts up to two independently supervised objects.
This module depends on PyTorch only; it does not load any pretrained weights.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "visionguard-grid-v1"
STRIDE = 16
SLOTS = 2


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outgoing),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GridNet(nn.Module):
    """Trainable CNN written for this project; output is not a YOLO model."""

    def __init__(self, num_classes):
        super().__init__()
        if num_classes < 1:
            raise ValueError("Train at least one class")
        self.num_classes = num_classes
        self.features = nn.Sequential(
            ConvBlock(3, 24, 2),
            ConvBlock(24, 48, 2),
            ConvBlock(48, 80, 2),
            ConvBlock(80, 128, 2),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )
        self.head = nn.Conv2d(128, SLOTS * (5 + num_classes), 1)
        # New models should not raise confident detections before training.
        with torch.no_grad():
            self.head.bias.view(SLOTS, 5 + num_classes)[:, 0].fill_(-4.5)

    def forward(self, images):
        features = self.features(images)
        output = self.head(features)
        batch, _, height, width = output.shape
        return output.reshape(batch, SLOTS, 5 + self.num_classes, height, width)


@dataclass(frozen=True)
class ImageTransform:
    size: int
    width: int
    height: int
    scaled_width: int
    scaled_height: int
    left: int
    top: int

    @property
    def scale_x(self):
        return self.scaled_width / self.width

    @property
    def scale_y(self):
        return self.scaled_height / self.height

    def encode_box(self, xywh):
        x, y, w, h = xywh
        return (
            (x + w / 2) * self.scale_x / self.size + self.left / self.size,
            (y + h / 2) * self.scale_y / self.size + self.top / self.size,
            w * self.scale_x / self.size,
            h * self.scale_y / self.size,
        )

    def decode_box(self, cx, cy, width, height):
        x1 = (cx * self.size - self.left - width * self.size / 2) / self.scale_x
        y1 = (cy * self.size - self.top - height * self.size / 2) / self.scale_y
        x2 = (cx * self.size - self.left + width * self.size / 2) / self.scale_x
        y2 = (cy * self.size - self.top + height * self.size / 2) / self.scale_y
        return (
            max(0, min(self.width, round(x1))),
            max(0, min(self.height, round(y1))),
            max(0, min(self.width, round(x2))),
            max(0, min(self.height, round(y2))),
        )


def letterbox(frame, size):
    """Pad a BGR image to a square without distorting aspect ratio."""
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Image must have nonzero width and height")
    factor = min(size / width, size / height)
    scaled_width = min(size, max(1, round(width * factor)))
    scaled_height = min(size, max(1, round(height * factor)))
    left = (size - scaled_width) // 2
    top = (size - scaled_height) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(frame, (scaled_width, scaled_height))
    padded[top : top + scaled_height, left : left + scaled_width] = resized
    return padded, ImageTransform(
        size, width, height, scaled_width, scaled_height, left, top
    )


def image_tensor(frame, size):
    padded, transform = letterbox(frame, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return tensor, transform


def encode_targets(boxes, class_names, transform):
    """Return objectness/box/class targets and count of crowded-cell drops.

    boxes: iterable of (class_name, [x, y, width, height]) in source pixels.
    """
    grid = transform.size // STRIDE
    objectness = torch.zeros(SLOTS, grid, grid, dtype=torch.float32)
    locations = torch.zeros(SLOTS, grid, grid, 4, dtype=torch.float32)
    classes = torch.full((SLOTS, grid, grid), -1, dtype=torch.long)
    skipped = 0
    for label, box in boxes:
        cx, cy, width, height = transform.encode_box(box)
        cell_x = min(grid - 1, max(0, int(cx * grid)))
        cell_y = min(grid - 1, max(0, int(cy * grid)))
        free = next(
            (slot for slot in range(SLOTS) if objectness[slot, cell_y, cell_x] == 0),
            None,
        )
        if free is None:
            skipped += 1
            continue
        objectness[free, cell_y, cell_x] = 1
        locations[free, cell_y, cell_x] = torch.tensor(
            [cx * grid - cell_x, cy * grid - cell_y, width, height]
        )
        classes[free, cell_y, cell_x] = class_names.index(label)
    return objectness, locations, classes, skipped


def detector_loss(logits, objectness, locations, classes):
    """Balanced objectness + coordinate regression + positive-only class loss."""
    prediction = logits.permute(0, 1, 3, 4, 2)
    positive = objectness.bool()
    negative = ~positive
    obj_error = F.binary_cross_entropy_with_logits(
        prediction[..., 0], objectness, reduction="none"
    )
    obj_loss = 0.25 * obj_error[negative].mean()
    if positive.any():
        obj_loss = obj_loss + obj_error[positive].mean()
        box_loss = F.smooth_l1_loss(
            prediction[..., 1:5][positive].sigmoid(), locations[positive]
        )
        class_loss = F.cross_entropy(prediction[..., 5:][positive], classes[positive])
    else:
        box_loss = logits.sum() * 0
        class_loss = logits.sum() * 0
    total = obj_loss + 5.0 * box_loss + class_loss
    return total, (obj_loss.detach(), box_loss.detach(), class_loss.detach())


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def decode_predictions(
    logits, transform, class_names, threshold=0.45, nms_iou=0.45, max_detections=40
):
    """Decode one image to (label, confidence, xyxy) with class-aware NMS."""
    if logits.ndim == 5:
        if logits.shape[0] != 1:
            raise ValueError("Decode one image at a time")
        logits = logits[0]
    grid = logits.shape[-1]
    if logits.shape[-2] != grid:
        raise ValueError("Expected a square output grid")
    raw = logits.detach().float().cpu().permute(0, 2, 3, 1)
    objectness = raw[..., 0].sigmoid()
    probabilities = raw[..., 5:].softmax(dim=-1)
    scores, indices = probabilities.max(dim=-1)
    scores = scores * objectness
    candidates = (scores >= threshold).nonzero(as_tuple=False)
    candidates = sorted(
        candidates.tolist(),
        key=lambda p: float(scores[p[0], p[1], p[2]]),
        reverse=True,
    )
    detections = []
    for slot, cell_y, cell_x in candidates[:300]:
        offset_x, offset_y, width, height = (
            raw[slot, cell_y, cell_x, 1:5].sigmoid().tolist()
        )
        box = transform.decode_box(
            (cell_x + offset_x) / grid,
            (cell_y + offset_y) / grid,
            width,
            height,
        )
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        label = class_names[int(indices[slot, cell_y, cell_x])]
        if any(
            existing[0] == label and box_iou(box, existing[2]) > nms_iou
            for existing in detections
        ):
            continue
        detections.append((label, float(scores[slot, cell_y, cell_x]), box))
        if len(detections) == max_detections:
            break
    return detections


def validate_checkpoint(data, allowed_classes):
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Not a VisionGuard GridNet checkpoint")
    classes = data.get("classes")
    size = data.get("image_size")
    if (
        not isinstance(classes, list)
        or not classes
        or not all(isinstance(c, str) and c in allowed_classes for c in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Checkpoint class labels do not match configured hazards")
    if not isinstance(size, int) or not (128 <= size <= 640) or size % STRIDE:
        raise ValueError("Invalid checkpoint image size")
    if not isinstance(data.get("state_dict"), dict):
        raise ValueError("Checkpoint has no model weights")
    return classes, size
