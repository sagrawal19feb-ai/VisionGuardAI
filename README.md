# VisionGuardAI

A local Python security-monitoring **prototype** with webcam capture, face recognition, a train-it-yourself object-detection neural network, a Tkinter dashboard, and incident logs/screenshots. No YOLO, Ultralytics or pretrained object-detection model is used.

Created for the **13th Gurugram Police Cyber Security Summer Internship Program (GPCSSI 2026)** by Shivansh Agrawal and Kushagra Singh.

> **This is not a certified safety or identity-verification system.** A missed detection is not proof of safety. Keep a human in the loop. Before training, the dashboard explicitly warns that object detection is unavailable; training on a few photos does **not** make it reliable.

## 1. Install once

Use Python 3.10+ on a desktop with a webcam and a GUI display (Windows 10/11 is the primary target). For CPU training/inference:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows Command Prompt: .venv\Scripts\activate.bat
# macOS/Linux: source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If you have a supported GPU, follow [PyTorch's installer](https://pytorch.org/get-started/locally/) instead of the CPU-install line. On Debian/Ubuntu also install `python3-tk`. Install `opencv-contrib-python` (from the requirements), **not** the conflicting `opencv-python`: `cv2.face` needs the contrib build. A fresh environment avoids OpenCV conflicts.

## 2. Collect your pictures

Create `dataset/images/` and copy in photos or extracted camera frames (JPG, PNG or BMP). **You supply and label the training data; none is bundled.** Supported hazard names are `knife`, `scissors`, `baseball bat`, and `gun`. Start with whichever of those you want to train; untrained classes are visibly flagged on the dashboard.

Aim for **hundreds of varied, correctly labeled images per class**, plus plenty of ordinary **background images with no hazards**. Include different lighting, sizes, occlusion, viewpoints, people, and objects that look similar. Use footage from the actual camera environment where possible. Ask permission before collecting identifiable images. Avoid near-identical video frames appearing on both training and validation; keep a separate set of new scenes for a final real-world check. A webcam feed or two labeled images is not sufficient to claim a working security detector.

## 3. Label images (no coding needed)

```sh
python annotate.py
```

Choose the object class in the dropdown, **drag a rectangle tightly around each visible hazard**, and press **Next**. Multiple boxes per picture are supported. If a picture has no hazards, leave it blank and press Next (it becomes a valuable background example). Previous, Undo, and Save buttons are provided. The tool saves as you go in `dataset/annotations.json`; images and labels are Git-ignored by default. Use your own `--images-dir` and `--annotations` paths if desired.

The JSON format is intentionally simple and editable:

```json
{
  "version": 1,
  "images": [
    {"file": "photo_001.jpg", "boxes": [
      {"label": "gun", "bbox": [20, 35, 65, 40]}
    ]},
    {"file": "empty_scene.jpg", "boxes": []}
  ]
}
```

A box is `[left, top, width, height]` in **original-image pixels**; image paths are relative to `dataset/images/`. The labeler creates this file for you. Each trained class needs at least two different labeled images so one can go into training and one into validation. More data is strongly recommended.

## 4. Train your OWN network

```sh
python train.py --dry-run
python train.py --epochs 30 --batch-size 8
```

The dry run checks filenames, boxes, class names and train/validation split **without training**. Training prints loss and validation precision/recall at IoU 0.5 (using a 0.3 score cutoff for these metrics). It saves the best-validation-loss model at `data/models/visionguard_gridnet.pt`. A CPU can take a while; run `python train.py --help` for batch size, image size, CPU/GPU device, resume, and custom dataset/model paths. Start with `--batch-size 2` if you run out of memory. Resume only works when the active classes and image size are unchanged; adding a new class requires retraining from scratch. The model file is ignored by Git unless you explicitly choose to publish it.

**What's inside:** `modules/custom_detector.py` defines **GridNet**, a small convolutional network written for this project. It predicts objectness, class and bounding box on a 16-pixel grid with two predictions per cell, uses balanced objectness/box/class losses, and applies class-aware non-maximum suppression. Training is from random initialization using your photos; PyTorch supplies tensor operations and autograd, **not** a pretrained model. The simple architecture can miss small/occluded objects or more than two object centers in one grid cell (skipped boxes are reported during training). Good labels, sufficient data and independent testing matter more than changing a threshold.

## 5. Test on a NEW image, then open the monitor

```sh
python predict.py path/to/new_photo.jpg
python main.py
```

`predict.py` prints labels and scores and writes an annotated image to `data/predictions/`. It does not need a webcam. Test photos that were **not** used for labeling or training. An empty result is not proof of safety. You can specify `--threshold 0.3` for a different confidence cutoff.

Press **Reload detector** after training if the dashboard is already open, or restart the monitor. `DETECTION_CONFIDENCE_THRESHOLD` in `config.py` controls how confident a detection must be. If the model does not exist or fails to load, object detection stays off and the status reads **DEGRADED**. If you trained only `gun`, the other three class names remain untrained and are listed in the warning.

The monitor uses **one threat policy** for the overlay and dashboard: unknown face alone -> HIGH; scissors with a known/no face -> MEDIUM; knife/baseball bat with a known/no face -> HIGH; unknown face plus a HIGH hazard -> CRITICAL; trained gun detection -> CRITICAL. A recognized person is not inherently safe. A hazard detection does not prove someone is holding the object. HIGH and CRITICAL incidents are rate-limited, saved to SQLite and optionally annotated screenshots, and displayed under Recent Alerts. No email, push or alarm audio is sent.

Face registration is a separate option: press **Register face** in the app or run `python face_register.py`. Supply one clear face per photo; use **Reload faces** for updates made from the standalone utility. Face recognition uses OpenCV Haar detection and LBPH trained from active SQLite registrations, **not** GridNet. LBPH match scores are *not probabilities*. For a fresh checkout either registration path creates the database. Set `CAMERA_INDEX` in `config.py` or press Retry camera if necessary.

## Files, tests and privacy

- `annotate.py` — click-and-drag image labeling; `train.py` — validation/training/metrics; `predict.py` — single-image check.
- `modules/custom_detector.py` — custom PyTorch GridNet, transforms, loss, prediction decoder.
- `modules/training_data.py`, `modules/object_detection.py` — dataset checks and live inference.
- `modules/camera.py`, `modules/monitor.py`, `modules/threat_assessment.py`, `modules/alert_system.py` — capture, processing, risk policy, logging.
- `ui/main_window.py` — desktop dashboard; `modules/database.py` — SQLite.
- `data/models/visionguard_gridnet.pt` — **created by you** after training; not shipped.
- `data/faces/`, `data/screenshots/`, `data/predictions/`, `data/security.db`, and `dataset/` are local runtime files ignored by Git.

Headless automated tests (including one tiny synthetic one-epoch training run) are available:

```sh
python -m unittest discover -s tests -v
```

A synthetic training test verifies the code path, **not** that it can identify real hazards. Do a real webcam/display acceptance test and measure misses and false alarms on new scenes before relying on any alert. Treat face images, footage, annotations and incident screenshots as sensitive: obtain consent, restrict access, and choose a retention/deletion policy. `.gitignore` does not encrypt data or remove files you committed previously.
