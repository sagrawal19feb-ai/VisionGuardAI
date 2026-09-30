"""Localhost privacy, dual UI state, and HTTP lifecycle without a webcam."""

import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from config import Config
from visionguard.core.service import MonitorService
from visionguard.core.state import MonitorState
from visionguard.interfaces.web.server import LocalWebServer, create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        config = type(
            "WebTestConfig",
            (Config,),
            {
                "BASE_DIR": root,
                "DATA_DIR": root / "data",
                "FACES_DIR": root / "data" / "faces",
                "SCREENSHOTS_DIR": root / "data" / "screenshots",
                "MODELS_DIR": root / "data" / "models",
                "DATABASE_PATH": root / "data" / "security.db",
                "DETECTOR_MODEL": root / "data" / "models" / "absent.pt",
            },
        )
        self.service = MonitorService(config)
        self.addCleanup(self.service.close)
        self.app = create_app(self.service, port=8765)
        self.app.testing = True
        self.client = self.app.test_client()
        self.origin = "http://127.0.0.1:8765"

    def get(self, path, **kwargs):
        return self.client.get(path, base_url=self.origin, **kwargs)

    def post(self, path, token=None, **kwargs):
        headers = {"Origin": self.origin}
        if token is not None:
            headers["X-CSRF-Token"] = token
        return self.client.post(path, base_url=self.origin, headers=headers, **kwargs)

    def authorize(self):
        page = self.get("/")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn("<h1>VisionGuardAI</h1>", html)
        self.assertEqual(
            html.count("Prototype stage — detections may be inaccurate."), 1
        )
        self.assertNotIn("command center", html.lower())
        self.index_page = page
        return re.search(
            r'name="csrf-token" content="([^"]+)"', page.get_data(as_text=True)
        ).group(1)

    def test_local_host_session_csrf_and_csp(self):
        self.assertEqual(self.get("/api/status").status_code, 401)
        self.assertEqual(
            self.client.get("/", base_url="http://other.local:8765").status_code,
            403,
        )
        token = self.authorize()
        status = self.get("/api/status")
        self.assertEqual(status.status_code, 200)
        self.assertFalse(status.json["camera_available"])
        self.assertFalse(status.json["model"]["available"])
        self.assertIsNone(status.json["threat_level"])
        self.assertEqual(self.get("/api/frame.jpg").status_code, 503)
        self.assertIn("no-store", status.headers["Cache-Control"])
        self.assertIn(
            "frame-ancestors 'none'", status.headers["Content-Security-Policy"]
        )
        self.assertIn("SameSite=Strict", self.index_page.headers["Set-Cookie"])
        self.assertEqual(self.post("/api/actions/reload-faces").status_code, 403)
        self.assertEqual(
            self.client.post(
                "/api/actions/reload-faces",
                base_url=self.origin,
                headers={"X-CSRF-Token": token, "Origin": "http://evil.example"},
            ).status_code,
            403,
        )
        self.assertEqual(self.post("/api/actions/reload-faces", token).status_code, 200)
        self.assertTrue(self.service.worker._reload_event.is_set())
        self.assertEqual(self.post("/api/actions/unknown", token).status_code, 404)

    def test_registration_and_deactivation_uses_shared_database(self):
        token = self.authorize()
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", frame)
        self.assertTrue(success)
        fake_cascade = SimpleNamespace(
            empty=lambda: False,
            detectMultiScale=lambda *args, **kwargs: [(20, 20, 70, 70)],
        )
        with mock.patch(
            "visionguard.core.registration.cv2.CascadeClassifier",
            return_value=fake_cascade,
        ):
            response = self.post(
                "/api/people",
                token,
                data={
                    "name": "<Safe & Alice>",
                    "photo": (io.BytesIO(encoded.tobytes()), "face.png"),
                },
            )
        self.assertEqual(response.status_code, 201, response.json)
        people = self.get("/api/people").json
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["name"], "<Safe & Alice>")
        self.assertNotIn("image_path", people[0])
        self.assertEqual(len(list(self.service.config.FACES_DIR.iterdir())), 1)
        self.assertEqual(
            self.post(f"/api/people/{people[0]['id']}/deactivate", token).status_code,
            200,
        )
        self.assertEqual(self.get("/api/people").json, [])
        self.assertEqual(list(self.service.config.FACES_DIR.iterdir()), [])
        self.assertEqual(
            self.post(f"/api/people/{people[0]['id']}/deactivate", token).status_code,
            404,
        )
        self.assertEqual(
            self.service.database.get_statistics()["registered_persons"], 0
        )
        self.assertEqual(self.get("/api/alerts").json, [])
        self.assertEqual(self.post("/api/people", token, data={}).status_code, 400)

    def test_both_readers_keep_latest_frame_and_event_without_stealing(self):
        state = MonitorState()
        result = SimpleNamespace(frame=np.zeros((32, 32, 3), dtype=np.uint8))
        state.publish(result, {"threat_level": "HIGH"})
        one = state.read()
        two = state.read()
        self.assertEqual(one.sequence, two.sequence)
        self.assertEqual(one.jpeg, two.jpeg)
        self.assertEqual(state.events_after(0)[0][1]["threat_level"], "HIGH")
        self.assertEqual(state.events_after(0)[0][1]["threat_level"], "HIGH")
        for _ in range(205):
            state.publish(result, {"threat_level": "HIGH"})
        self.assertEqual(len(state.events_after(0)), 200)
        self.assertEqual(state.read().sequence, 206)
        state.clear()
        self.assertIsNone(state.read().result)
        self.assertIsNone(state.read().jpeg)
        self.assertEqual(state.read().sequence, 207)

    def test_stream_and_status_require_a_fresh_shared_frame(self):
        self.authorize()
        result = SimpleNamespace(
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            faces=[],
            fps=12.3,
            warnings=("Experimental detector",),
            assessment={
                "threat_level": "LOW",
                "description": "No threats detected",
                "unknown_count": 0,
                "hazardous_objects": [],
            },
        )
        with (
            mock.patch.object(
                type(self.service.camera),
                "is_available",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                type(self.service.worker),
                "is_running",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
        ):
            self.service.worker.state.publish(result)
            self.assertEqual(self.get("/api/status").json["threat_level"], "LOW")
            image = self.get("/api/frame.jpg")
            self.assertEqual(image.status_code, 200)
            self.assertTrue(image.data.startswith(bytes([0xFF, 0xD8])))
            stream = self.get("/api/stream.mjpeg", buffered=False)
            try:
                self.assertIn(b"Content-Type: image/jpeg", next(stream.response))
            finally:
                stream.close()
            self.service.worker.state.clear()
            self.assertIsNone(self.get("/api/status").json["threat_level"])
            self.assertEqual(self.get("/api/frame.jpg").status_code, 503)

    def test_server_binds_to_loopback_not_lan(self):
        server = LocalWebServer(self.service, 8765)
        with mock.patch("visionguard.interfaces.web.server.make_server") as factory:
            server.start()
            try:
                self.assertEqual(factory.call_args.args[0], "127.0.0.1")
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
