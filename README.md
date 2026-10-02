GPCSSI Shivansh Agrawal (26CA057)  Kushagra Singh (26CA065)

# VisionGuardAI

A local webcam-monitoring prototype with face registration, incident logging, and a trained **R-CNN** detector for knife, scissors, baseball bat, and gun. Use the desktop app, a localhost browser page, or both at once from **one process**.

> Prototype stage — detections may be inaccurate. [Measured results and limitations](TRAINING_REPORT.md).

## Set up

Use **Python 3.10+** and a webcam. Clone/download this repository and run the commands **from its root**, where `main.py` and `requirements.txt` live. The trained checkpoint `data/models/visionguard_frcnn.pt` is included; you do **not** need to train before first use.

**Windows (PowerShell):**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

**macOS / Linux:**

```sh
python3 -m venv .venv
source .venv/bin/activate
# Install matching torch + torchvision from https://pytorch.org/get-started/locally/
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python main.py
```

On Linux the desktop app also needs Tkinter (`sudo apt install python3-tk` on Ubuntu/Debian). For GPU installation, follow [PyTorch's installer](https://pytorch.org/get-started/locally/) instead of the CPU wheel command. Face recognition uses `opencv-contrib-python` from `requirements.txt`; avoid installing a second, conflicting OpenCV package.

### Choose an interface

| Command | What opens |
| --- | --- |
| `python main.py` | Desktop app **and** the web page at **http://127.0.0.1:8765/**; the desktop has an **Open web dashboard** button. |
| `python web.py` | Browser page only (no Tkinter display needed); open **http://127.0.0.1:8765/** on the same computer. |
| `python main.py --no-web` | Desktop only. |

Replace `python` with `.\.venv\Scripts\python.exe` in PowerShell if the virtual environment is not activated. To use another local web port, add `--port 8766`. **Do not start `main.py` and `web.py` simultaneously**: each would try to open the webcam. The browser server binds to **127.0.0.1 only**, never the LAN; it is not intended for remote access.

## First run

1. Grant your operating system's camera permission if prompted. The dashboard remains open if no camera is present; choose **Retry camera** after connecting one. If needed, change `CAMERA_INDEX` in `config.py` (usually `0`).
2. Click **Fullscreen** above the live camera feed in either interface; press **Esc** or **Exit fullscreen** to return. Both views use the same camera and detector. Check that the page shows **4/4** detector classes. If the model is missing or invalid, detection is disabled and a warning appears rather than silently using an untrained model.
3. To recognize a familiar person, use **Register face** and choose a clear image showing exactly one face. Changes appear in both interfaces. **Manage faces** (desktop) / **Deactivate** (browser) removes a stored registration photo; historical incidents remain.
4. HIGH and CRITICAL events appear under **Recent incidents** and in local `data/security.db`; annotated screenshots may be saved to `data/screenshots/`. The app does not send emails or push notifications.

To check a still image without a webcam: `python predict.py path/to/photo.jpg`. Its annotated copy is saved under `data/predictions/`.

## Train or evaluate

Normal monitoring uses the **bundled** checkpoint. Public training photos are *not* committed. To download a new Open Images subset, fine-tune the gun class, and evaluate on a separate public split:

```sh
python prepare_openimages.py --split train --per-class 300 --backgrounds 120
python prepare_openimages.py --split test --per-class 40 --backgrounds 30
python train.py --epochs 8 --output dataset/retrained.pt
python evaluate_detector.py --model dataset/retrained.pt --thresholds 0.4 0.5 0.7
```

The official train annotation CSV is large and streamed; CPU training can take tens of minutes. The trainer selects a checkpoint using an **internal validation split**. To *experiment* with additional head-only training for all four classes from the bundled checkpoint, run `python tune_hazards.py --epochs 2`. It writes an optional candidate under `dataset/` **only if** train-side validation improves knife matches without material regressions; it never overwrites the installed model. In our public-photo attempts, extra training increased false alarms or reduced other classes' scores, so the bundled weights were **kept unchanged**. [Details](TRAINING_REPORT.md). Do not repeatedly adjust the model against the official test split and call it an independent result. `python train.py --help` and `python tune_hazards.py --help` list options. Reload the model from either dashboard if you install new validated weights.

For your **own consented photos**, place images in `dataset/images/`, draw boxes with `python annotate.py` (including empty/background scenes), then use `python train.py --annotations dataset/annotations.json`. Include multiple photos of **all four** labels, with at least two photos per class to allow a train/validation split; more varied webcam examples and a new untouched test set are needed for meaningful improvement.

## Project map

```text
main.py / web.py             desktop + web / browser-only entry points
config.py                   camera index, model path, alert policy
visionguard/detection/      Faster R-CNN model definition and live inference
visionguard/core/           one camera, monitoring worker, state, faces, alerts, SQLite
visionguard/interfaces/     Tkinter desktop and localhost HTML dashboard
visionguard/training/       public-data preparation, datasets, fine-tuning, evaluation
train.py / tune_hazards.py / predict.py   training, further tuning, still images
```

Both interfaces use **one** camera and inference worker, so they show the same monitoring state. The live app loads only the packaged Faster R-CNN model, with no YOLO/Ultralytics dependency or automatic model download.

## Privacy, limits, and troubleshooting

- The browser is **local-only** and uses a session cookie, CSRF protection, and Host checks. Do not port-forward it. Other software on the same computer can access loopback; secure your OS account.
- Face photos, screenshots, predictions, and the SQLite database are Git-ignored but **not encrypted**. Obtain consent and define a retention policy. Deactivating a face removes its registration photo but does not erase past incident logs.
- A LOW reading or an empty frame **does not prove safety**. Scores are not calibrated safety probabilities. Public-photo tests do not measure how this model works on your webcam. See [TRAINING_REPORT.md](TRAINING_REPORT.md) for the measured misses and [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for licensing cautions before redistributing data or adapted weights.
- **No camera:** check OS permission, close other apps using it, then retry. **Port occupied:** use `--port 8766` or stop the other service. **Faces always unknown:** install `opencv-contrib-python` without a conflicting OpenCV installation, then register a clear single-face photo. **Detector unavailable:** check `data/models/visionguard_frcnn.pt` and the torch/torchvision installation.

Run the headless checks with `python -m unittest discover -s tests -v`. They do not replace a real webcam/display test.
