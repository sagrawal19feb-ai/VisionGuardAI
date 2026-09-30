"""Headless regression tests. Run: python -m unittest discover -s tests -v"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from config import Config
from modules.alert_system import AlertSystem
from modules.database import Database
from modules.face_recognition_module import FaceRecognitionModule, FaceResult
from modules.monitor import MonitoringWorker
from modules.object_detection import DetectionResult, ObjectDetectionModule
from modules.registration import RegistrationError, register_person
from modules.threat_assessment import ThreatAssessment


class TempConfig(Config):
    """Override paths so tests never touch the repo's runtime data."""

    @classmethod
    def create_dirs(cls):
        for path in (cls.DATA_DIR, cls.FACES_DIR, cls.SCREENSHOTS_DIR, cls.MODELS_DIR):
            path.mkdir(parents=True, exist_ok=True)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = type(
            "TestConfig",
            (TempConfig,),
            {
                "BASE_DIR": root,
                "DATA_DIR": root / "data",
                "FACES_DIR": root / "data" / "faces",
                "SCREENSHOTS_DIR": root / "data" / "screenshots",
                "MODELS_DIR": root / "data" / "models",
                "DATABASE_PATH": root / "data" / "security.db",
            },
        )
        self.db = Database(self.config.DATABASE_PATH)
        self.addCleanup(self.db.close)
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assessor = ThreatAssessment(self.config)
        self.known = FaceResult("Alice", 0.5, (10, 10, 90, 90), True)
        self.unknown = FaceResult("Unknown", 0, (10, 10, 90, 90), False)
        self.knife = DetectionResult("Knife", 0.9, (10, 10, 80, 80), True, "HIGH")
        self.gun = DetectionResult("Gun", 0.9, (10, 10, 80, 80), True, "CRITICAL")
        self.scissors = DetectionResult(
            "Scissors", 0.9, (10, 10, 80, 80), True, "MEDIUM"
        )

    def test_database_counts_upsert_and_deactivation(self):
        self.assertEqual(self.db.get_statistics()["registered_persons"], 0)
        self.db.register_person("Alice", image_path="first.jpg")
        person = self.db.get_person("Alice")
        self.db.register_person("Alice", image_path="second.jpg")
        self.assertEqual(self.db.get_person("Alice")["id"], person["id"])
        self.assertEqual(self.db.get_person("Alice")["image_path"], "second.jpg")
        self.assertEqual(self.db.get_statistics()["registered_persons"], 1)
        self.assertTrue(self.db.deactivate_person(person["id"]))
        self.assertEqual(self.db.get_all_persons(), [])
        self.assertEqual(self.db.get_statistics()["registered_persons"], 0)
        self.db.register_person("Alice", image_path="third.jpg")
        self.assertEqual(self.db.get_statistics()["registered_persons"], 1)
        self.assertEqual(self.db.get_person("Alice")["id"], person["id"])

    def test_assessment_policy_for_faces_and_hazards(self):
        examples = [
            ([], [], "LOW"),
            ([self.known], [], "LOW"),
            ([self.unknown], [], "HIGH"),
            ([], [self.scissors], "MEDIUM"),
            ([], [self.knife], "HIGH"),
            ([self.known], [self.gun], "CRITICAL"),
            ([self.unknown], [self.knife], "CRITICAL"),
        ]
        for faces, objects, expected in examples:
            with self.subTest(faces=faces, objects=objects):
                self.assertEqual(
                    self.assessor.assess(faces, objects)["threat_level"], expected
                )

    def test_alert_logs_and_screenshots_with_cooldown_and_escalation(self):
        alerts = AlertSystem(self.config, self.db)
        high = self.assessor.assess([self.unknown], [])
        critical = self.assessor.assess([self.unknown], [self.knife])
        with mock.patch(
            "modules.alert_system.time.monotonic", side_effect=[100, 101, 102]
        ):
            first = alerts.process(high, self.frame)
            self.assertIsNone(alerts.process(high, self.frame))
            second = alerts.process(critical, self.frame)
        self.assertEqual(first["threat_level"], "HIGH")
        self.assertEqual(second["threat_level"], "CRITICAL")
        self.assertTrue(Path(first["screenshot_path"]).is_file())
        self.assertTrue(Path(second["screenshot_path"]).is_file())
        self.assertEqual(self.db.get_statistics()["total_alerts"], 2)
        self.assertEqual(self.db.get_statistics()["total_detections"], 2)
        self.assertEqual(len(self.db.get_recent_alerts()), 2)
        rows = (
            self.db._conn()
            .execute("SELECT screenshot_path FROM detection_logs")
            .fetchall()
        )
        self.assertTrue(all(Path(row[0]).is_file() for row in rows))

    def test_worker_uses_shared_threat_policy_and_logs_once(self):
        camera = SimpleNamespace(actual_fps=15)
        faces = SimpleNamespace(
            detect_and_recognize=mock.Mock(return_value=[self.unknown]),
            recognition_available=True,
        )
        objects = SimpleNamespace(
            detect=mock.Mock(return_value=[self.knife]),
            is_available=True,
            last_error=False,
            unsupported_hazards=[],
        )
        alerts = AlertSystem(self.config, self.db)
        worker = MonitoringWorker(
            camera, faces, objects, self.assessor, alerts, self.config, self.db
        )
        levels = [
            worker.process_frame(self.frame).assessment["threat_level"]
            for _ in range(3)
        ]
        self.assertEqual(levels, ["CRITICAL"] * 3)
        self.assertEqual(faces.detect_and_recognize.call_count, 1)
        self.assertEqual(objects.detect.call_count, 2)
        self.assertEqual(self.db.get_statistics()["total_alerts"], 1)
        self.assertEqual(worker.events.qsize(), 1)
        self.assertEqual(worker.events.get_nowait()["threat_level"], "CRITICAL")

    def test_background_worker_publishes_frame_and_incident(self):
        camera = SimpleNamespace(
            frame_count=1,
            actual_fps=12.0,
            get_frame=lambda: self.frame.copy(),
        )
        faces = SimpleNamespace(
            detect_and_recognize=lambda frame: [self.unknown],
            recognition_available=True,
        )
        objects = SimpleNamespace(
            detect=lambda frame: [self.knife],
            is_available=True,
            last_error=False,
            unsupported_hazards=[],
        )
        worker = MonitoringWorker(
            camera,
            faces,
            objects,
            self.assessor,
            AlertSystem(self.config, self.db),
            self.config,
            self.db,
        )
        self.assertTrue(worker.start())
        try:
            incident = worker.events.get(timeout=3)
            displayed = worker.frames.get(timeout=3)
        finally:
            worker.stop()
        self.assertEqual(incident["threat_level"], "CRITICAL")
        self.assertEqual(displayed.assessment["threat_level"], "CRITICAL")
        self.assertEqual(self.db.get_statistics()["total_alerts"], 1)

    def test_registration_validates_and_keeps_photos_in_faces_dir(self):
        image_path = Path(self.temp.name) / "source.png"
        cv2.imwrite(str(image_path), self.frame)
        fake_cascade = SimpleNamespace(
            empty=lambda: False,
            detectMultiScale=lambda *args, **kwargs: [(10, 10, 80, 80)],
        )
        with mock.patch(
            "modules.registration.cv2.CascadeClassifier", return_value=fake_cascade
        ):
            first = register_person("../Alice", image_path, self.config, self.db)
            self.assertEqual(first.parent, self.config.FACES_DIR)
            self.assertTrue(first.exists())
            identity = self.db.get_person("../Alice")["id"]
            second = register_person("../Alice", image_path, self.config, self.db)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(self.db.get_person("../Alice")["id"], identity)
            self.assertEqual(self.db.get_statistics()["registered_persons"], 1)
        with mock.patch(
            "modules.registration.cv2.CascadeClassifier",
            return_value=SimpleNamespace(
                empty=lambda: False, detectMultiScale=lambda *a, **k: []
            ),
        ):
            with self.assertRaises(RegistrationError):
                register_person("Bob", image_path, self.config, self.db)
            self.assertIsNone(self.db.get_person("Bob"))

    def test_failed_registration_removes_new_photo(self):
        image_path = Path(self.temp.name) / "source.png"
        cv2.imwrite(str(image_path), self.frame)
        fake_cascade = SimpleNamespace(
            empty=lambda: False, detectMultiScale=lambda *a, **k: [(0, 0, 80, 80)]
        )
        with (
            mock.patch(
                "modules.registration.cv2.CascadeClassifier", return_value=fake_cascade
            ),
            mock.patch.object(
                self.db, "register_person", side_effect=RuntimeError("db failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "db failed"):
                register_person("Alice", image_path, self.config, self.db)
        self.assertEqual(list(self.config.FACES_DIR.iterdir()), [])

    def test_fresh_standalone_registration_creates_database(self):
        image_path = Path(self.temp.name) / "source.png"
        cv2.imwrite(str(image_path), self.frame)
        root = Path(self.temp.name) / "fresh"
        fresh = type(
            "FreshConfig",
            (TempConfig,),
            {
                "BASE_DIR": root,
                "DATA_DIR": root / "data",
                "FACES_DIR": root / "data" / "faces",
                "SCREENSHOTS_DIR": root / "data" / "screenshots",
                "MODELS_DIR": root / "data" / "models",
                "DATABASE_PATH": root / "data" / "security.db",
            },
        )
        fake_cascade = SimpleNamespace(
            empty=lambda: False, detectMultiScale=lambda *a, **k: [(0, 0, 80, 80)]
        )
        with mock.patch(
            "modules.registration.cv2.CascadeClassifier", return_value=fake_cascade
        ):
            path = register_person("Alice", image_path, fresh)
        db = Database(fresh.DATABASE_PATH)
        self.addCleanup(db.close)
        self.assertEqual(db.get_statistics()["registered_persons"], 1)
        self.assertTrue(path.exists())

    def test_object_detector_reports_unsupported_hazards(self):
        model_path = Path(self.temp.name) / "fake.pt"
        model_path.touch()
        config = type("ObjectTestConfig", (self.config,), {"YOLO_MODEL": model_path})
        box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[1, 2, 50, 60]])
        result = SimpleNamespace(names={0: "knife", 1: "person"}, boxes=[box])

        class FakeModel:
            names = result.names

            def __call__(self, *args, **kwargs):
                return [result]

        with mock.patch.dict(
            "sys.modules",
            {"ultralytics": SimpleNamespace(YOLO=lambda path: FakeModel())},
        ):
            detector = ObjectDetectionModule(config)
        detections = detector.detect(self.frame)
        self.assertEqual(len(detections), 1)
        self.assertTrue(detections[0].is_hazardous)
        self.assertEqual(detections[0].threat_modifier, "HIGH")
        self.assertIn("gun", detector.unsupported_hazards)

    def test_recognition_uses_only_active_people_and_reloads_cleanly(self):
        alice_image = Path(self.temp.name) / "alice.jpg"
        bob_image = Path(self.temp.name) / "bob.jpg"
        cv2.imwrite(str(alice_image), self.frame)
        cv2.imwrite(str(bob_image), self.frame)
        self.db.register_person("Alice", image_path=str(alice_image))
        self.db.register_person("Bob", image_path=str(bob_image))
        bob = self.db.get_person("Bob")
        self.db.deactivate_person(bob["id"])
        fake_cascade = SimpleNamespace(
            empty=lambda: False, detectMultiScale=lambda *a, **k: [(0, 0, 80, 80)]
        )
        fake_recognizer = SimpleNamespace(
            train=mock.Mock(), predict=lambda image: (0, 40)
        )
        face_cv = SimpleNamespace(LBPHFaceRecognizer_create=lambda: fake_recognizer)
        with (
            mock.patch(
                "modules.face_recognition_module.cv2.CascadeClassifier",
                return_value=fake_cascade,
            ),
            mock.patch.object(cv2, "face", face_cv, create=True),
        ):
            recognition = FaceRecognitionModule(self.config, self.db)
            self.assertEqual(recognition.count_registered(), 1)
            self.assertEqual(
                recognition.detect_and_recognize(self.frame)[0].name, "Alice"
            )
            self.db.deactivate_person(self.db.get_person("Alice")["id"])
            recognition.reload_database()
            self.assertFalse(recognition.trained)
            self.assertEqual(
                recognition.detect_and_recognize(self.frame)[0].name, "Unknown"
            )

    @unittest.skipUnless(
        hasattr(getattr(cv2, "face", None), "LBPHFaceRecognizer_create"),
        "requires opencv-contrib-python",
    )
    def test_real_lbph_can_train_match_and_forget_a_face(self):
        photo = np.random.default_rng(42).integers(0, 256, (100, 100), dtype=np.uint8)
        photo_path = Path(self.temp.name) / "face.png"
        cv2.imwrite(str(photo_path), photo)
        self.db.register_person("Alice", image_path=str(photo_path))
        fake_cascade = SimpleNamespace(
            empty=lambda: False, detectMultiScale=lambda *a, **k: [(0, 0, 80, 80)]
        )
        with mock.patch(
            "modules.face_recognition_module.cv2.CascadeClassifier",
            return_value=fake_cascade,
        ):
            recognizer = FaceRecognitionModule(self.config, self.db)
            video = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
            self.assertEqual(recognizer.detect_and_recognize(video)[0].name, "Alice")
            self.db.deactivate_person(self.db.get_person("Alice")["id"])
            recognizer.reload_database()
            self.assertEqual(recognizer.detect_and_recognize(video)[0].name, "Unknown")


if __name__ == "__main__":
    unittest.main()
