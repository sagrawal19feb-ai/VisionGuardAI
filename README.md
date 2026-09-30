# VisionGuardAI

A local Python desktop security-monitoring prototype: live webcam capture, face detection/recognition, YOLO object detection, configurable threat policy, a Tkinter dashboard, and incident screenshots and SQLite logs.

Created for the **13th Gurugram Police Cyber Security Summer Internship Program (GPCSSI 2026)** by Shivansh Agrawal and Kushagra Singh.

> **Important:** This is a prototype, not a certified safety or identity-verification system. Detection may miss people or objects, and a recognized person is not inherently safe. Keep a human in the loop. The bundled COCO YOLOv8n model **does not recognize guns**; a custom model with a `gun` label is needed for that configured rule. The dashboard displays a degraded-coverage warning for unsupported hazard labels.

## Requirements

- Python 3.10+; Windows 10/11 is the primary desktop target. Other platforms need a working Tkinter installation (for example, `python3-tk` on Debian/Ubuntu), webcam and GUI display.
- Webcam and local disk space for logs/screenshots.
- Install dependencies in an isolated environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use **`opencv-contrib-python`**, not `opencv-python`: LBPH face recognition needs `cv2.face`. If both are installed, remove the conflicting OpenCV package and reinstall the requirements in a fresh environment. CPU-only YOLO inference can be slow; the UI stays responsive because inference runs in a worker thread. Use trusted model checkpoints only: PyTorch `.pt` loading may execute untrusted pickle code.

## Run

```sh
python main.py
```

The app initializes its SQLite database on startup. If your camera is not found, check OS permissions and `CAMERA_INDEX` in `config.py`, then click **Retry camera**. To register a person, click **Register face**, choose a photo containing exactly one detectable face, and enter a name. Registrations made in the separate utility can be applied without restarting by clicking **Reload faces**:

```sh
python face_register.py
```

A freshly cloned repo needs no pre-existing database: either registration path initializes it. Face photos are re-encoded without EXIF and placed under `data/faces/` using generated filenames; active SQLite registrations determine who can be recognized. Deactivating an entry through the `Database` API and reloading faces removes it from the face model. The displayed face-match score is derived from LBPH distance; **it is not a probability**. For production use, collect multiple consented images per identity and validate match thresholds on your actual camera and environment.

## What the monitor records

On each new frame, a background capture thread supplies an image to the inference worker. Face and object detectors run at the intervals configured in `config.py`. `ThreatAssessment` applies a single policy to their latest results, which drives **both** the video overlay and the dashboard:

| People | Highest detected hazard | Result |
| --- | --- | --- |
| No face / known face | None | LOW |
| Any unknown face | None | HIGH |
| Known face | Scissors | MEDIUM |
| Known face / no face | Knife or baseball bat | HIGH |
| Any unknown face | Knife or baseball bat | CRITICAL |
| Any face status | Gun, if a compatible custom model is supplied | CRITICAL |

Rules are defined in `Config.THREAT_RULES` and have a fallback for hazards with no detected face. **LOW means no configured threat was found, not that a scene is safe.** Hazard classification describes an object in the scene, not whether a person is holding it. The dashboard flags disabled/unavailable detection or unsupported model classes as **DEGRADED**.

HIGH and CRITICAL incidents write to `data/security.db` (`alert_history` and `detection_logs`), may save an annotated screenshot in `data/screenshots/`, and appear in **Recent Alerts**. The global alert cooldown is 5 seconds by default; escalation to CRITICAL bypasses a HIGH cooldown. MEDIUM and LOW scenes are not saved to the incident log, and the app does not send email/push notifications or play audio. Database/files are local and are ignored by Git. Set `ALERT_SCREENSHOT_ON_HIGH_RISK` and `ALERT_SCREENSHOT_ON_UNKNOWN` in `config.py` to control screenshot capture.

## Project layout

- `main.py` — initialization and shutdown.
- `config.py` — paths, camera parameters, detection intervals and risk policy.
- `face_register.py`, `modules/registration.py` — standalone and integrated registration.
- `modules/camera.py`, `modules/monitor.py` — capture and off-UI-thread inference.
- `modules/face_recognition_module.py`, `modules/object_detection.py` — detectors.
- `modules/threat_assessment.py`, `modules/alert_system.py`, `modules/database.py` — policy, incidents and storage.
- `ui/main_window.py` — Tkinter dashboard.
- `data/models/yolov8n.pt` — bundled COCO weights; `data/faces/`, `data/screenshots/` and `data/security.db` are generated at runtime.

## Tests and privacy

Run headless regression tests with:

```sh
python -m unittest discover -s tests -v
```

Tests do not require a webcam, display or Ultralytics installation, but do require OpenCV and NumPy. A full desktop acceptance test still requires a webcam, installed requirements and a real Tkinter display. Verify visual accuracy and false-positive rates before relying on any alert.

Face photos, screenshots and SQLite entries are sensitive personal data. Obtain consent, restrict filesystem access, set a retention/deletion policy and do not commit runtime data. `.gitignore` prevents accidental **new** Git additions but does not encrypt data or remove files already committed elsewhere.
