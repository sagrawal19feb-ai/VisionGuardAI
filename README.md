# VisionGuardAI

A local Python desktop security-monitoring **prototype** with a webcam, OpenCV face recognition, a **non-YOLO** trained object detector, a Tkinter dashboard, and incident logs/screenshots. Created for the 13th Gurugram Police Cyber Security Summer Internship Program (GPCSSI 2026) by Shivansh Agrawal and Kushagra Singh.

> **Human-in-the-loop only.** No image detector is a guarantee of safety. This model **missed 55/86 knives and 25/52 guns** on selected public test images. It has **not** been tested on your webcam. Treat a clear scene or a LOW label as *unverified*, not safe. See [data sources and full measurements](DATA_PROVENANCE.md).

## Quick start (Windows laptop)

Install Python 3.10+ and create a fresh virtual environment. CPU-only setup:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows Command Prompt: .venv\Scripts\activate.bat
# macOS/Linux: source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python main.py
```

For a supported GPU, use [PyTorch's installer](https://pytorch.org/get-started/locally/) instead of the CPU-install line. Linux also needs a working Tkinter GUI (e.g. `python3-tk`). Use the installed `opencv-contrib-python`, not a conflicting `opencv-python`, for LBPH face recognition. A display and webcam are required to test the desktop dashboard.

The bundled **trained** `data/models/visionguard_frcnn.pt` detects four classes: knife, scissors, baseball bat and gun. The model is experimental; its score is not a safety probability. If the weight file is missing or invalid the UI **does not silently substitute a model**; object detection remains off and the status warns you. To try it on a still image without a webcam:

```sh
python predict.py path/to/a/photo.jpg
```

That saves an annotated copy under `data/predictions/`. Absence of a predicted box does *not* mean the object is absent. `DETECTION_CONFIDENCE_THRESHOLD` in `config.py` controls the original COCO classes; the gun threshold is stored in the checkpoint. Higher thresholds reduce false alarms but miss more real objects.

### Registering faces

Click **Register face** in the app, or run `python face_register.py` separately. Supply one clear face per photo. Click **Reload faces** after using the standalone tool. Face recognition uses OpenCV Haar detection and LBPH from **active** SQLite registrations, not the object detector. Its displayed match scores are **not probabilities**. If your webcam does not open, check OS permissions and `CAMERA_INDEX` in `config.py`, then click Retry camera.

## How the detector was built

There is **no YOLO / Ultralytics dependency**. Accuracy was more important than claiming a completely original architecture: the default model uses **TorchVision Faster R-CNN MobileNetV3-320 pretrained on COCO** for the existing knife, scissors and baseball-bat classes. `modules/quality_detector.py` adds and fine-tunes a separate gun class using public bounding-box images; `modules/object_detection.py` serves the resulting checkpoint through the same dashboard pipeline. [The exact data and pretrained-weight sources and rights cautions](DATA_PROVENANCE.md) are documented. All four classes are enabled in the current experimental checkpoint because the gun class passed a limited internal validation gate; **none is certified for deployment**.

The repository also keeps `modules/custom_detector.py` (GridNet), a hand-written trainable grid architecture. That from-scratch experiment produced too many false alarms and is **not used by default**. You can still train it with `python train.py`; do not mistake it for the bundled default model.

## Rebuild/fine-tune it yourself

The downloaded public **photos are not committed**. To build reproducible local data from Open Images (streams large official box CSVs, then downloads selected photos):

```sh
python prepare_openimages.py --split train --per-class 300 --backgrounds 120
python prepare_openimages.py --split test --per-class 40 --backgrounds 30
python train_quality.py --epochs 8
python evaluate_detector.py
```

These create local, Git-ignored `dataset/openimages_train/`, `dataset/openimages_test/` and a fine-tuned `data/models/visionguard_frcnn.pt`. Training on a CPU takes time. The train script uses an internal validation split and keeps COCO's original class rows fixed while fitting gun; the **official test split** is used only by the evaluation script. Do not repeatedly tune against the test split and present its scores as unbiased. `python train_quality.py --help` lists the batch, epoch, seed and resume options. After updating weights, click **Reload detector** or restart the monitor. Verify the resulting detection scores on **new footage from your actual webcam** before considering alerts.

To prepare your own (consented) images for the experimental original GridNet, place them under `dataset/images/`, run `python annotate.py` to drag boxes around each object (including photos with no hazards), then `python train.py --dry-run` and `python train.py --epochs 30`. That workflow is separate from the pretrained Faster R-CNN and needs substantial data and independent validation to compete.

## Threats, privacy and checks

The live UI and overlay share one policy: unknown face alone -> HIGH; scissors -> MEDIUM; knife/baseball bat -> HIGH; unknown face plus a HIGH hazard -> CRITICAL; gun -> CRITICAL. A recognized person is not inherently safe. Detection does not establish that a person holds an object. HIGH/CRITICAL alerts are rate-limited, written to `data/security.db`, optionally saved as annotated screenshots in `data/screenshots/`, and listed in Recent Alerts. It does not send email, push, or audio notifications. Missing detector or untrained labels produce a visible degraded-coverage warning.

`dataset/`, face photos, screenshots, predictions and SQLite records are Git-ignored; none is encrypted. Obtain consent, restrict filesystem access, and set a retention policy. Individual Open Images photo rights are **not guaranteed** by its listing; do not redistribute images without checking them. Pretrained weights can have separate use terms; see [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

Headless tests (including synthetic training and model-load checks):

```sh
python -m unittest discover -s tests -v
```

These tests do not replace a real webcam/display test, an independent held-out evaluation, or a professional security review.
