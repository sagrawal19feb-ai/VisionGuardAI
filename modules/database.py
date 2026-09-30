"""
Database Module
===============
Thread-safe SQLite wrapper.
"""

import sqlite3
import threading
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()

        self._init_schema()

    def _conn(self):
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = sqlite3.connect(str(self.db_path))

        conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            face_encoding BLOB,
            engine_used TEXT,
            image_path TEXT,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS detection_logs(
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

        CREATE TABLE IF NOT EXISTS alert_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT,
            message TEXT,
            threat_level TEXT,
            acknowledged INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        conn.close()

    def register_person(
        self,
        name,
        face_encoding,
        engine_used="",
        image_path=""
    ):
        conn = self._conn()

        conn.execute(
            """
            INSERT OR REPLACE INTO persons
            (name, face_encoding, engine_used, image_path)
            VALUES (?, ?, ?, ?)
            """,
            (name, face_encoding, engine_used, image_path)
        )

        conn.commit()
        return True

    def get_all_persons(self):
        rows = self._conn().execute(
            "SELECT * FROM persons WHERE is_active=1"
        ).fetchall()

        return [dict(r) for r in rows]

    def deactivate_person(self, person_id):
        self._conn().execute(
            "UPDATE persons SET is_active=0 WHERE id=?",
            (person_id,)
        )

        self._conn().commit()
        return True

    def log_detection(
        self,
        event_type,
        person_name=None,
        person_is_known=None,
        objects=None,
        threat_level=None,
        confidence=None,
        screenshot_path=None,
        session_id=None
    ):
        self._conn().execute(
            """
            INSERT INTO detection_logs
            (
                event_type,
                person_name,
                person_is_known,
                objects_json,
                threat_level,
                confidence,
                screenshot_path,
                session_id
            )
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                event_type,
                person_name,
                person_is_known,
                json.dumps(objects) if objects else None,
                threat_level,
                confidence,
                screenshot_path,
                session_id
            )
        )

        self._conn().commit()

    def log_alert(self, alert_type, message, threat_level):
        self._conn().execute(
            """
            INSERT INTO alert_history
            (alert_type,message,threat_level)
            VALUES (?,?,?)
            """,
            (alert_type, message, threat_level)
        )

        self._conn().commit()

    def get_statistics(self):
        conn = self._conn()

        return {
            "registered_persons":
                conn.execute(
                    "SELECT COUNT(*) FROM persons WHERE is_active=1"
                ).fetchone()[0]-1,

            "total_detections":
                conn.execute(
                    "SELECT COUNT(*) FROM detection_logs"
                ).fetchone()[0],

            "total_alerts":
                conn.execute(
                    "SELECT COUNT(*) FROM alert_history"
                ).fetchone()[0]
        }