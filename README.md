# VisionGuardAI

A **local-first, human-in-the-loop monitoring prototype** for a laptop webcam. It combines face registration, one trained **non-YOLO Faster R-CNN** detector, a shared threat policy, incident records, and two redesigned ways to view the **same running monitor**: a desktop app and a private localhost dashboard.

> **Prototype stage — detections may be inaccurate.** See the [test results and limitations](TRAINING_REPORT.md) and [data provenance](DATA_PROVENANCE.md) for details.

## Quick start

Requires **Python 3.10+**, a working webcam, and (for desktop mode) a Tkinter GUI. Windows PowerShell example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python main.py
```

For macOS/Linux activate with `source .venv/bin/activate` instead. On Ubuntu-like systems, install `python3-tk` if Tkinter is absent. For a compatible GPU, select the appropriate [PyTorch installer](https://pytorch.org/get-started/locally/) rather than the CPU line. `opencv-contrib-python` (in `requirements.txt`) is needed for LBPH face matching; conflicting OpenCV packages can prevent it from loading.

**Desktop mode:** `python main.py` opens the native dashboard **and** starts the web dashboard in the same process. Click **Open web dashboard**, or visit **http://127.0.0.1:8765/** on *this computer*. They share a single camera, inference worker, incident log and set of registered people. Closing the desktop window also stops its web server.

**Browser-only mode:** `python web.py` starts the same engine without Tkinter or a desktop display. Open **http://127.0.0.1:8765/** in a browser on the computer running the command. Press Ctrl+C to stop. Use `python web.py --port 8766` to choose another free local port. **Do not run `main.py` and `web.py` at the same time**—only one process can own a webcam reliably. `python main.py --no-web` disables the web server. If the port is occupied, desktop mode continues without it.

**Localhost is intentional:** the HTTP server binds **127.0.0.1 only**, not `0.0.0.0` or your LAN IP. It validates the HTTP Host, uses a per-launch cookie and CSRF token for changes, and does not enable cross-origin access. This is **not a remote-access service**: do not port-forward or reverse-proxy it. Other software running *on this computer* can still access loopback; lock your OS account. The dashboard uses no CDN or cloud service. Faces, incident screenshots and SQLite records are stored unencrypted in `data/` and ignored by Git—obtain consent and establish retention and access rules.

### Using the monitor

- **Retry camera** if the camera is disconnected. Check OS webcam permission and `CAMERA_INDEX` in `config.py` if retry fails. A disconnected or stale camera shows **no live assessment**, not a misleading last-known LOW status.
- **Register face** from either dashboard with a name and one clear photo showing exactly one detectable face. The same registered gallery is used by both. **Manage faces** (desktop) or **Deactivate** (web) stops recognizing a person and removes that person's stored registration photo; past incident records remain. A newly registered face is reloaded into the running worker. Face match scores are not calibrated probabilities.
- **Reload model** after replacing `data/models/visionguard_frcnn.pt`. Missing or invalid weights disable hazard detection with an explicit coverage warning; the app never silently substitutes an untrained model. The bundled checkpoint enables knife, scissors, baseball bat and gun, but **none is certified**. The dashboard remains usable without a camera or valid model.
- **Incidents**: HIGH and CRITICAL assessments are rate-limited, logged in `data/security.db`, and can create annotated JPG screenshots under `data/screenshots/`. Alerts have no email or push notification. Detection does not show who *holds* an object. An unknown face alone rates HIGH; scissors MEDIUM; knife or baseball bat HIGH; gun CRITICAL; unknown face plus HIGH hazard CRITICAL. A known face does not negate a hazard.

Still-image check (no webcam required):

```sh
python predict.py path/to/photo.jpg
```

This saves a marked image under `data/predictions/`. No drawn box **does not** mean the image is safe. `DETECTION_CONFIDENCE_THRESHOLD` in `config.py` sets the COCO classes' threshold (default 0.45); gun's 0.40 default is stored in the checkpoint. Raising either threshold can reduce false alerts while missing more real objects.

## How the code is organized

The live application is built around **one model family**, rather than selecting from experimental architectures at runtime:

```text
main.py / web.py                desktop + web  /  browser-only entry points
config.py                      local paths, camera settings, alert policy
visionguard/
  detection/model.py            TorchVision Faster R-CNN architecture + checkpoint validation
  detection/detector.py         trained-checkpoint loading and BGR-frame inference
  core/service.py               single camera/worker/database lifecycle + UI commands
  core/monitor.py               face + object inference, overlays, threat + incidents
  core/state.py                 bounded, thread-safe latest frame and incident publication
  core/{camera,faces,registration,threat,alerts,database}.py
  interfaces/desktop.py         native Tkinter dashboard
  interfaces/web/{server.py,index.html}  loopback HTTP API and self-contained dashboard
  training/{finetune,openimages,evaluate,dataset,metrics}.py
  legacy/gridnet.py             archived from-scratch experiment; never loaded live
train.py / evaluate_detector.py / prepare_openimages.py  short CLI entry points
```

The worker publishes each annotated frame **once**, then the desktop polls the latest snapshot while localhost streams JPEGs from that same snapshot. Neither UI reads the webcam directly, runs a second model, or steals frames from the other. Both read the same SQLite incident records; the browser cannot request arbitrary file paths or screenshots. HTTP endpoints are localhost-only (`/api/status`, `/api/stream.mjpeg`, `/api/alerts`, `/api/people` and protected POST actions). There is no YOLO or Ultralytics dependency.

The TorchVision **Faster R-CNN MobileNetV3-320 COCO** weights supply knife, scissors and baseball-bat predictions. `train.py` fine-tunes an added gun output while protecting the pretrained class rows, then writes a weights-only checkpoint; the live app does **not** download pretrained weights. The original GridNet work remains isolated as `train_gridnet.py`, `predict_gridnet.py` and `visionguard/legacy/gridnet.py` for experiments only. Those weights were not adequate for live use and cannot be loaded by the default monitor.

## Rebuild and measure the detector

Public photos are **not committed** and were removed from this packaged workspace. The source IDs/manifests remain only in this workspace's Git-ignored `dataset/`; new clones do not contain them. To re-download images, fine-tune, and independently measure the separate Open Images *test* split:

```sh
python prepare_openimages.py --split train --per-class 300 --backgrounds 120
python prepare_openimages.py --split test --per-class 40 --backgrounds 30
python train.py --epochs 8
python evaluate_detector.py --thresholds 0.4 0.5 0.7
```

The official train box CSV is large and **streamed**, not saved to disk. Training takes time on CPU and selects a checkpoint using *internal train-side validation*. The Open Images **test** subset must not be used to select an epoch or repeatedly tune thresholds and then reported as an untouched evaluation. To annotate **your own consented** images, put them under `dataset/images/`, run `python annotate.py`, and label *all four* classes plus empty scenes. Then `python train.py --annotations dataset/annotations.json`; each class must appear in at least two images so it can be split into training and validation. For genuine improvement, use varied labeled laptop-webcam footage and keep entire sessions out of training for a fresh independent test. `python train.py --help` describes options; `train_quality.py` is an alias for the earlier CLI.

Data licensing and limitations: Open Images annotations are attributed in [DATA_PROVENANCE.md](DATA_PROVENANCE.md); an image being **listed** as CC BY does **not** verify its individual license. TorchVision's pretrained weights may have dataset-derived terms. Verify your rights before redistributing photographs or the adapted checkpoint.

## Verification and troubleshooting

```sh
python -m unittest discover -s tests -v
python -m compileall -q .
```

Headless tests cover checkpoint loading, the shared state, incident policy, database/face registration, localhost/CSRF controls and the HTTP dashboard. They **do not** verify the physical webcam, face-matching reliability, or real-world detection safety. If the dashboard shows *No live frame*, check camera privacy permissions and retry; if the model says *Unavailable*, verify the checkpoint path and installed TorchVision version. If port 8765 is taken, pass `--port 8766` (or close the other service). Neither UI requires the Open Images photos for normal inference.
