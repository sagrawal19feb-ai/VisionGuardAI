"""VisionGuardAI desktop monitor entry point. Run with: python main.py"""

import logging
import tkinter as tk

from config import Config
from modules.alert_system import AlertSystem
from modules.camera import Camera
from modules.database import Database
from modules.face_recognition_module import FaceRecognitionModule
from modules.monitor import MonitoringWorker
from modules.object_detection import ObjectDetectionModule
from modules.threat_assessment import ThreatAssessment
from ui.main_window import MainWindow


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    Config.create_dirs()
    config = Config()
    root = tk.Tk()
    database = None
    camera = None
    app = None
    try:
        database = Database(config.DATABASE_PATH)
        camera = Camera(
            config.CAMERA_INDEX,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.CAMERA_FPS,
        )
        faces = FaceRecognitionModule(config, database)
        objects = ObjectDetectionModule(config)
        threat = ThreatAssessment(config)
        alerts = AlertSystem(config, database)
        worker = MonitoringWorker(
            camera, faces, objects, threat, alerts, config, database
        )
        app = MainWindow(root, config)
        app.set_modules(camera, worker, database)
        if camera.start():
            app.start()
        else:
            app.show_no_camera()
        root.mainloop()
    finally:
        if app is not None:
            app.close()
        else:
            if camera is not None:
                camera.stop()
            if database is not None:
                database.close()
            root.destroy()


if __name__ == "__main__":
    main()
