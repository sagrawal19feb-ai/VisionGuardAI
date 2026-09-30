from pathlib import Path


class Config:
    # ==========================================================
    # PROJECT PATHS
    # ==========================================================
    BASE_DIR = Path(__file__).resolve().parent

    DATA_DIR = BASE_DIR / "data"
    FACES_DIR = DATA_DIR / "faces"
    SCREENSHOTS_DIR = DATA_DIR / "screenshots"
    MODELS_DIR = DATA_DIR / "models"

    DATABASE_PATH = DATA_DIR / "security.db"

    # ==========================================================
    # CAMERA SETTINGS
    # ==========================================================
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30

    # ==========================================================
    # FACE RECOGNITION
    # ==========================================================
    FACE_THRESHOLD = 0.55
    FACE_DETECTION_INTERVAL = 3

    # ==========================================================
    # YOLO OBJECT DETECTION
    # ==========================================================
    YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD = 0.45
    YOLO_DETECTION_INTERVAL = 2

    # ==========================================================
    # ALERT SETTINGS
    # ==========================================================
    ALERT_COOLDOWN_SECONDS = 5
    ALERT_SOUND_ENABLED = True
    ALERT_SCREENSHOT_ON_HIGH_RISK = True
    ALERT_SCREENSHOT_ON_UNKNOWN = True

    # ==========================================================
    # UI SETTINGS
    # ==========================================================
    UI_MIN_WIDTH = 1280
    UI_MIN_HEIGHT = 800

    EVENT_LOG_MAX_LINES = 1000

    # ==========================================================
    # UI COLORS
    # ==========================================================
    C_BG = "#0d1117"
    C_PANEL = "#161b22"
    C_HEADER_BG = "#010409"

    C_TEXT = "#f0f6fc"
    C_TEXT_DIM = "#8b949e"

    C_BORDER = "#30363d"

    C_ACCENT = "#58a6ff"

    C_SUCCESS = "#3fb950"
    C_WARNING = "#d29922"
    C_DANGER = "#f85149"
    C_CRITICAL = "#bc8cff"

    # ==========================================================
    # THREAT LEVEL DEFINITIONS
    # ==========================================================
    THREAT_LEVELS = {
        "LOW": {
            "hex": "#3fb950",
            "bgr": (80, 200, 63),
            "priority": 1
        },
        "MEDIUM": {
            "hex": "#d29922",
            "bgr": (34, 153, 210),
            "priority": 2
        },
        "HIGH": {
            "hex": "#f85149",
            "bgr": (73, 81, 248),
            "priority": 3
        },
        "CRITICAL": {
            "hex": "#bc8cff",
            "bgr": (255, 140, 188),
            "priority": 4
        }
    }

    # ==========================================================
    # HAZARDOUS OBJECTS
    # ==========================================================
    HAZARDOUS_OBJECTS = {
        "knife": {
            "threat_modifier": "HIGH",
            "color": (0, 0, 255)
        },
        "scissors": {
            "threat_modifier": "MEDIUM",
            "color": (0, 165, 255)
        },
        "baseball bat": {
            "threat_modifier": "HIGH",
            "color": (0, 0, 255)
        },
        "gun": {
            "threat_modifier": "CRITICAL",
            "color": (255, 0, 255)
        }
    }

    # ==========================================================
    # THREAT RULES
    # ==========================================================
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
        ("unknown", "CRITICAL"): "CRITICAL"
    }

    # ==========================================================
    # CREATE REQUIRED FOLDERS
    # ==========================================================
    @classmethod
    def create_dirs(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.FACES_DIR.mkdir(parents=True, exist_ok=True)
        cls.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)