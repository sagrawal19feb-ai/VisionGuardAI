"""Private loopback dashboard; camera/inference are owned by MonitorService.

The HTTP surface is deliberately *not* a general network service. Strict
cookies, a per-process CSRF secret, host validation and no CORS protect the
webcam and registered faces from cross-site requests in a local browser.
"""

import hmac
import logging
import secrets
import tempfile
import time
import threading
from pathlib import Path

from flask import Flask, abort, g, jsonify, render_template, request, session, Response
from werkzeug.serving import make_server

from visionguard.core.registration import RegistrationError

logger = logging.getLogger(__name__)
ALLOWED_PHOTOS = {".jpg", ".jpeg", ".png", ".bmp"}


def create_app(service, port=8765):
    app = Flask(__name__, template_folder=".", static_folder=None)
    app.secret_key = secrets.token_bytes(32)  # Changes every launch; no disk secret.
    app.config.update(
        MAX_CONTENT_LENGTH=21 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.before_request
    def private_request():
        g.nonce = secrets.token_urlsafe(16)
        if request.host not in hosts:
            abort(403)
        if request.endpoint == "index":
            return
        if not session.get("authorized"):
            abort(401)
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in hosts}:
                abort(403)
            sent = request.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(sent, session.get("csrf", "missing")):
                abort(403)

    @app.after_request
    def private_headers(response):
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; base-uri 'none'; form-action 'self'; "
            "connect-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; "
            f"script-src 'nonce-{g.nonce}'; frame-ancestors 'none'"
        )
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Upload exceeds 21 MB"), 413

    @app.get("/")
    def index():
        session["authorized"] = True
        session["csrf"] = secrets.token_urlsafe(32)
        return render_template("index.html", csrf=session["csrf"], nonce=g.nonce)

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.get("/api/alerts")
    def alerts():
        return jsonify(service.database.get_recent_alerts(30))

    @app.get("/api/people")
    def people():
        return jsonify(
            [
                {key: p[key] for key in ("id", "name", "registered_at")}
                for p in service.database.get_all_persons()
            ]
        )

    @app.get("/api/frame.jpg")
    def frame():
        snapshot = service.worker.state.read()
        image = snapshot.jpeg
        if (
            image is None
            or not service.camera.is_available
            or not service.worker.is_running
            or time.monotonic() - snapshot.updated_at >= 4
        ):
            return ("No live frame", 503)
        return Response(image, mimetype="image/jpeg")

    @app.get("/api/stream.mjpeg")
    def stream():
        def frames():
            revision = -1
            while True:
                current = service.worker.state.wait_after(revision, timeout=5)
                if current.sequence == revision:
                    continue
                revision = current.sequence
                if (
                    not service.camera.is_available
                    or not service.worker.is_running
                    or time.monotonic() - current.updated_at >= 4
                ):
                    continue
                if current.jpeg is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + current.jpeg
                        + b"\r\n"
                    )

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.post("/api/actions/<action>")
    def action(action):
        actions = {
            "retry-camera": service.retry_camera,
            "reload-faces": service.reload_faces,
            "reload-detector": service.reload_detector,
        }
        if action not in actions:
            abort(404)
        try:
            outcome = actions[action]()
        except RuntimeError:
            logger.exception("Monitor command failed: %s", action)
            return jsonify(error="Monitor is busy or has stopped"), 503
        if action == "retry-camera" and not outcome:
            return jsonify(error="Camera unavailable; check permissions/index"), 503
        return jsonify(ok=True)

    @app.post("/api/people")
    def register():
        photo = request.files.get("photo")
        name = request.form.get("name", "")
        suffix = Path(photo.filename or "").suffix.lower() if photo else ""
        if not photo or suffix not in ALLOWED_PHOTOS:
            return jsonify(error="Choose a JPG, PNG or BMP face photo"), 400
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=service.config.FACES_DIR, suffix=suffix, delete=False
            ) as file:
                temporary = Path(file.name)
                photo.save(file)
            service.register_face(name, temporary)
        except RegistrationError as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            logger.exception("Cannot register face")
            return jsonify(error="Could not save face; check server logs"), 500
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return jsonify(ok=True), 201

    @app.post("/api/people/<int:person_id>/deactivate")
    def deactivate(person_id):
        if not service.deactivate_face(person_id):
            return jsonify(error="Active person not found"), 404
        return jsonify(ok=True)

    @app.teardown_request
    def close_thread_connection(_error):
        service.database.close()  # SQLite connection belongs to request thread.

    return app


class LocalWebServer:
    def __init__(self, service, port=8765):
        if not 1 <= port <= 65535:
            raise ValueError("Invalid local web port")
        self.port = port
        self.app = create_app(service, port)
        self._server = None
        self._thread = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/"

    def start(self):
        # Never bind to 0.0.0.0: registered faces and camera must stay local.
        self._server = make_server("127.0.0.1", self.port, self.app, threaded=True)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="loopback-dashboard", daemon=True
        )
        self._thread.start()
        logger.info("Private web dashboard: %s", self.url)

    def serve_forever(self):
        self._server = make_server("127.0.0.1", self.port, self.app, threaded=True)
        logger.info("Private web dashboard: %s", self.url)
        self._server.serve_forever()

    def close(self):
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=3)
