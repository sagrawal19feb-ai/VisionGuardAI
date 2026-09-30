# VisionGuardAI — training result

**Outcome:** A four-class **non-YOLO** checkpoint has been trained and installed at `data/models/visionguard_frcnn.pt`. It is an **experimental baseline, not a reliably safe detector**. The desktop webcam was not available for a real-world acceptance test.

## What changed

- Built a labeled public subset using `prepare_openimages.py`: 1,292 Open Images **train** photos (including 120 background photos); 154 separate Open Images **test** photos (including 30 backgrounds). No dataset photos are committed. [Sources, license cautions and class mappings](DATA_PROVENANCE.md).
- Trained the original `GridNet` from scratch on a smaller balanced subset first. At a 0.45 score cutoff it produced large numbers of false positives and had very low precision; it was **rejected**. A later MobileNet-initialized GridNet run missed all selected public-test hazards at that cutoff; **also rejected**. Those models are not the default checkpoint.
- Adapted **TorchVision Faster R-CNN MobileNetV3-320 pretrained on COCO**, which already knows knife, scissors and baseball bat. `train_quality.py` learned a new gun class from public images while freezing the pretrained class rows. Best gun-head checkpoint selected at internal-validation epoch **5**, not on the official test split. The user authorized accuracy-first use of a non-YOLO pretrained detector rather than an entirely original architecture.
- `modules/object_detection.py` now loads the resulting checkpoint in the live monitor. The UI flags it as **experimental / not webcam-validated**, and still reports unavailable classes if a different checkpoint omits them. A bad/missing checkpoint never falls back silently.

## Separate public test result

Selected Open Images official **test** photos, match IoU ≥ 0.50, all class scores ≥ **0.40**:

| Object | True matches | False alarms | Misses | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Knife | 31 | 6 | 55 | 83.8% | **36.0%** |
| Scissors | 4 | 5 | 1 | 44.4% | 80.0% (only **5** labeled boxes) |
| Baseball bat | 22 | 7 | 23 | 75.9% | **48.9%** |
| Gun | 27 | 16 | 25 | 62.8% | **51.9%** |

0 / 30 selected background photos raised an alert at that threshold. **Thirty is too few to predict actual false-alarm rates.** The live config uses 0.45 for the first three and checkpoint-stored 0.40 for gun. On the public test at uniform 0.50, gun precision increased to 75.0% but recall fell to 46.2%. The labels may be incomplete, so these are indicative, not deployment-grade metrics. There is no honest basis to call this “really good” or suitable as the sole security control. In particular, many knives and guns were missed.

## To improve it responsibly

1. Record and label **consented photos from the actual laptop webcam**, with different distances, lighting, backgrounds, occupied/unoccupied scenes and lookalike objects. Keep entire recording sessions out of training for an independent final test.
2. Check all label boxes, including empty scenes, and audit apparent false positives. More reliable labels and harder negatives matter at least as much as more epochs.
3. Rebuild the model using `python train_quality.py` (see [README](README.md)), then test a **new, untouched** dataset, measuring per-class precision/recall and misses before adjusting thresholds. Keep a human monitoring the system.

**Verification:** 16 automated tests pass; a trained checkpoint loaded and detected a labeled gun image without downloading any weights at runtime. The GUI and webcam could not be run in this environment. Local public photos were removed after evaluation to keep the deliverable compact; the manifests and source IDs remain, and the README gives the download commands.
