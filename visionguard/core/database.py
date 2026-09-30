"""SQLite storage for registered persons, incident detections and alerts."""

import json
import sqlite3
import threading
from pathlib import Path


class Database:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self):
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    face_encoding BLOB,
                    engine_used TEXT,
                    image_path TEXT,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    person_name TEXT,
                    person_is_known INTEGER,
                    objects_json TEXT,
                    threat_level TEXT,
                    confidence REAL,
                    screenshot_path TEXT,
                    session_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT,
                    message TEXT,
                    threat_level TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        finally:
            conn.close()

    def register_person(self, name, face_encoding=None, engine_used="", image_path=""):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO persons (name, face_encoding, engine_used, image_path)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    face_encoding=excluded.face_encoding,
                    engine_used=excluded.engine_used,
                    image_path=excluded.image_path,
                    is_active=1
            """,
                (name, face_encoding, engine_used, str(image_path)),
            )
        return True

    def get_person(self, name):
        row = (
            self._conn()
            .execute("SELECT * FROM persons WHERE name=?", (name,))
            .fetchone()
        )
        return dict(row) if row is not None else None

    def get_person_by_id(self, person_id):
        row = (
            self._conn()
            .execute("SELECT * FROM persons WHERE id=?", (person_id,))
            .fetchone()
        )
        return dict(row) if row is not None else None

    def get_all_persons(self):
        rows = (
            self._conn()
            .execute("SELECT * FROM persons WHERE is_active=1 ORDER BY name")
            .fetchall()
        )
        return [dict(row) for row in rows]

    def deactivate_person(self, person_id):
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE persons SET is_active=0 WHERE id=? AND is_active=1",
                (person_id,),
            )
        return cursor.rowcount > 0

    def log_detection(
        self,
        event_type,
        person_name=None,
        person_is_known=None,
        objects=None,
        threat_level=None,
        confidence=None,
        screenshot_path=None,
        session_id=None,
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO detection_logs
                    (event_type, person_name, person_is_known, objects_json,
                     threat_level, confidence, screenshot_path, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event_type,
                    person_name,
                    person_is_known,
                    json.dumps(objects) if objects is not None else None,
                    threat_level,
                    confidence,
                    screenshot_path,
                    session_id,
                ),
            )

    def log_alert(self, alert_type, message, threat_level):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO alert_history (alert_type, message, threat_level)
                VALUES (?, ?, ?)
            """,
                (alert_type, message, threat_level),
            )

    def get_recent_alerts(self, limit=20):
        rows = (
            self._conn()
            .execute(
                """
            SELECT timestamp, threat_level, message FROM alert_history
            ORDER BY id DESC LIMIT ?
        """,
                (max(0, int(limit)),),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    def get_statistics(self):
        conn = self._conn()
        return {
            "registered_persons": conn.execute(
                "SELECT COUNT(*) FROM persons WHERE is_active=1"
            ).fetchone()[0],
            "total_detections": conn.execute(
                "SELECT COUNT(*) FROM detection_logs"
            ).fetchone()[0],
            "total_alerts": conn.execute(
                "SELECT COUNT(*) FROM alert_history"
            ).fetchone()[0],
        }

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            del self._local.conn
