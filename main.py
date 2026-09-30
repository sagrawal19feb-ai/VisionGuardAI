"""
AI Security Monitor
Main Entry Point
"""

import logging
import tkinter as tk
import traceback
import time

from config import Config

from modules.database import Database
from modules.camera import Camera
from modules.face_recognition_module import FaceRecognitionModule
from modules.object_detection import ObjectDetectionModule
from modules.threat_assessment import ThreatAssessment
from modules.alert_system import AlertSystem

from ui.main_window import MainWindow


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def main():
    setup_logging()

    logger = logging.getLogger("main")

    print("\n" + "=" * 70)
    print("AI SECURITY MONITOR STARTING...")
    print("=" * 70)
    time.sleep(2)

    config = Config()

    database = Database(config.DATABASE_PATH)

    camera = Camera(
        camera_index=config.CAMERA_INDEX,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS
    )

    camera_ok = camera.start()

    if not camera_ok:
        logger.warning("Camera failed to start")

    face_module = FaceRecognitionModule(
        config=config,
        database=database
    )

    object_module = ObjectDetectionModule(
        config=config
    )

    threat_module = ThreatAssessment(
        config=config
    )

    alert_system = AlertSystem(
        config=config,
        database=database
    )

    root = tk.Tk()

    app = MainWindow(
        root=root,
        config=config
    )

    app.set_modules(
        camera=camera,
        face_module=face_module,
        object_module=object_module,
        threat_module=threat_module,
        alert_system=alert_system,
        database=database
    )

    if camera_ok:
        app.start()
    else:
        app.show_no_camera()

    logger.info("System ready")

    root.mainloop()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
        print("Press ENTER to exit...")
        input()

    except Exception:
        print("\n")
        print("=" * 70)
        print("FATAL ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("\n")
        print("=" * 70)
        print("Program crashed.")
        print("Copy the error above and send it here.")
        print("=" * 70)

        input("\nPress ENTER to close...")