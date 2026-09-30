"""Application settings and the policy used by the live threat assessor."""

from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    FACES_DIR = DATA_DIR / "faces"
    SCREENSHOTS_DIR = DATA_DIR / "screenshots"
    MODELS_DIR = DATA_DIR / "models"
    DATABASE_PATH = DATA_DIR / "security.db"

    WEB_PORT = 8765  # Always bound to 127.0.0.1, never the LAN.
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30

    # LBPH returns a distance (lower is a better match), not a probability.
    FACE_RECOGNITION_DISTANCE_THRESHOLD = 80.0
    FACE_DETECTION_INTERVAL = 3
    DETECTOR_MODEL = MODELS_DIR / "visionguard_frcnn.pt"
    DETECTION_CONFIDENCE_THRESHOLD = 0.45
    DETECTION_INTERVAL = 2

    ALERT_COOLDOWN_SECONDS = 5
    ALERT_SCREENSHOT_ON_HIGH_RISK = True
    ALERT_SCREENSHOT_ON_UNKNOWN = True
    UI_UPDATE_INTERVAL_MS = 50
    EVENT_LOG_MAX_LINES = 100

    THREAT_LEVELS = {
        "LOW": {"hex": "#3fb950", "bgr": (80, 200, 63), "priority": 1},
        "MEDIUM": {"hex": "#d29922", "bgr": (34, 153, 210), "priority": 2},
        "HIGH": {"hex": "#f85149", "bgr": (73, 81, 248), "priority": 3},
        "CRITICAL": {"hex": "#bc8cff", "bgr": (255, 140, 188), "priority": 4},
    }

    # Only labels validated in a loaded checkpoint can be detected.
    # Missing classes are explicitly flagged as degraded in the dashboard.
    HAZARDOUS_OBJECTS = {
        "knife": {"threat_modifier": "HIGH", "color": (0, 0, 255)},
        "scissors": {"threat_modifier": "MEDIUM", "color": (0, 165, 255)},
        "baseball bat": {"threat_modifier": "HIGH", "color": (0, 0, 255)},
        "gun": {"threat_modifier": "CRITICAL", "color": (255, 0, 255)},
    }

    THREAT_RULES = {
        ("none", None): "LOW",
        ("known", None): "LOW",
        ("unknown", None): "HIGH",
        ("known", "LOW"): "LOW",
        ("known", "MEDIUM"): "MEDIUM",
        ("known", "HIGH"): "HIGH",
        ("known", "CRITICAL"): "CRITICAL",
        ("unknown", "LOW"): "HIGH",
        ("unknown", "MEDIUM"): "HIGH",
        ("unknown", "HIGH"): "CRITICAL",
        ("unknown", "CRITICAL"): "CRITICAL",
    }

    @classmethod
    def create_dirs(cls):
        for directory in (
            cls.DATA_DIR,
            cls.FACES_DIR,
            cls.SCREENSHOTS_DIR,
            cls.MODELS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
