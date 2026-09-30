# Public training data, pretrained weights and validation limits

**This is a trained prototype, not a security certification.** No webcam footage was available. A photo benchmark cannot tell us how it performs on your laptop webcam or how often it will miss a small or hidden object. A human must verify any alert, and an absence of alerts is not evidence that a scene is safe.

## Sources and licensing

1. [1](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) **Open Images V7**, Google LLC: the annotations are CC BY 4.0. Its images are **listed** as CC BY 2.0, but Google explicitly makes no warranty that each individual image has that license; verify the rights for each photo before redistributing it. Source box CSVs and official image IDs are linked from the [2](https://storage.googleapis.com/openimages/web/download_v7.html) official download page. Each local `dataset/openimages_*/annotations.json` entry carries its original `source_id` and S3 `source_url`. Images and annotations are ignored by Git; **photos are not included in this repository**.
2. [3](https://docs.pytorch.org/vision/stable/models.html) **TorchVision Faster R-CNN MobileNetV3-320 COCO weights**: TorchVision cautions that pretrained weights may have their own dataset-derived terms; determine whether you have permission for your intended use before redistributing or commercially deploying the adapted checkpoint. Original weight URL: `https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth`. This is **not YOLO and not an original detector architecture**; the repo adds a gun classifier/regressor and preserves COCO's knife, scissors and baseball-bat detections.

`prepare_openimages.py` streams official box CSVs, filters group/depiction boxes, downloads only selected images, resizes them, and writes box annotations in original-resized pixel coordinates. Handgun and rifle are grouped as `gun`; `knife` and `kitchen knife` are grouped as `knife`. Sampling is deterministic for the given seed, but public image hosting can change. Background photos were selected by unrelated Open Images labels and absence of *target* box annotations; they were **not individually verified to contain no hazards**. This can introduce label errors.

## What was actually trained

- Open Images **train** subset: 1,292 photos, including 120 selected background photos; 518 knife, 338 scissors, 338 baseball-bat and 443 gun boxes. `train.py` internally split these and trained the added gun head on 580 selected photos (all available gun positives on its training side plus other labeled objects/backgrounds). The original COCO prediction rows were held fixed. Gun-head checkpoint selected from the training-side *internal* validation after **epoch 5** (fixed confidence 0.40, box IoU 0.50).
- Open Images **test** split: 154 distinct photos, including 30 selected backgrounds. It was not used by the gun-head trainer for gradient updates or checkpoint selection. Box counts on this evaluation subset: knife 86, scissors **5**, baseball bat 45, gun 52. Five scissors are far too few to estimate robust scissors performance.
- The model under `data/models/visionguard_frcnn.pt` saves full weights in half precision, along with label mapping, original weight URL, selected epoch, internal gun metrics, and explicitly enabled classes. The app loads it with PyTorch's `weights_only=True` and does not download pretrained weights at runtime. SHA-256: `16f20a2812472212800b7fc7f5e80fcb8f8e7d10ff707fd289367da67405d85f`.

## Separate Open Images test results

For comparison, the table uses a fixed confidence **0.40** across all classes and box-match IoU **0.50**. After train-side threshold checks, the live app uses **0.40** for knife and gun, **0.45** for scissors and baseball bat; its exact numbers will differ slightly. The pretrained weights did not change during these later trials. A prediction that does not match a labeled object of the same class counts as a false positive; an unmatched labeled object counts as a miss. Open Images' annotations may not be exhaustive, so treat these figures as approximate.

| Class | Matched / actual boxes | False positives | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Knife | 31 / 86 | 6 | 83.8% | 36.0% |
| Scissors | 4 / 5 | 5 | 44.4% | 80.0% *very small sample* |
| Baseball bat | 22 / 45 | 7 | 75.9% | 48.9% |
| Gun | 27 / 52 | 16 | 62.8% | 51.9% |

Background photos with any alert: **0 / 30** at that threshold; 30 is too few to establish a real-world false-alarm rate. Raising the threshold to 0.50 improved gun precision to 75.0% but reduced gun recall to 46.2%. These figures are neither a webcam test nor an assurance of real-time threat detection. In particular, the model missed **55 of 86 knives** and **25 of 52 guns** in the selected public test photos at 0.40. It must **not** be the sole basis for a safety decision.

An earlier fully custom GridNet experiment performed too poorly on public images and was removed from the live codebase. The app uses the explicitly attributed non-YOLO TorchVision detector. Future improvement requires independent labeled webcam frames, more verified negatives and another untouched test set after changes.
