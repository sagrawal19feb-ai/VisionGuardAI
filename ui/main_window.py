"""Tkinter dashboard; inference runs outside the Tk event loop."""

import logging
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from modules.registration import RegistrationError, register_person

logger = logging.getLogger(__name__)


class MainWindow:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.worker = None
        self.camera = None
        self.database = None
        self._closed = False
        self._poll_id = None
        self._last_stats = 0.0
        self._last_frame = 0.0
        self._monitoring = False

        root.title("VisionGuardAI · Security Monitor")
        root.geometry("1400x850")
        root.minsize(1024, 650)
        root.configure(bg="#111827")
        root.protocol("WM_DELETE_WINDOW", self.close)

        header = tk.Frame(root, bg="#1f2937", height=66)
        header.pack(fill="x")
        tk.Label(
            header,
            text="VISIONGUARDAI",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 19, "bold"),
        ).pack(side="left", padx=22, pady=14)
        tk.Button(
            header,
            text="Register face",
            command=self.open_registration,
            bg="#2563eb",
            fg="white",
            padx=12,
        ).pack(side="right", padx=(0, 15))
        tk.Button(
            header,
            text="Reload faces",
            command=self.reload_faces,
            bg="#374151",
            fg="white",
            padx=12,
        ).pack(side="right", padx=8)
        self.retry_button = tk.Button(
            header,
            text="Retry camera",
            command=self.retry_camera,
            bg="#374151",
            fg="white",
            padx=12,
            state="disabled",
        )
        self.retry_button.pack(side="right", padx=8)

        main = tk.Frame(root, bg="#111827")
        main.pack(fill="both", expand=True)
        left = tk.Frame(main, bg="#111827")
        left.pack(side="left", fill="both", expand=True)
        self.video = tk.Label(
            left,
            bg="#030712",
            fg="#cbd5e1",
            text="Waiting for camera…",
            font=("Segoe UI", 20),
        )
        self.video.pack(fill="both", expand=True, padx=12, pady=12)

        side = tk.Frame(main, bg="#1f2937", width=325)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        tk.Label(
            side,
            text="SYSTEM STATUS",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 12))
        self.threat_label = self._label(side, "Threat: —", 17, "#cbd5e1")
        self.persons_label = self._label(side, "Registered persons: 0")
        self.alerts_label = self._label(side, "Alerts: 0")
        self.detections_label = self._label(side, "Logged incidents: 0")
        self.faces_label = self._label(side, "Faces: 0")
        self.unknown_label = self._label(side, "Unknown faces: 0", fg="#fbbf24")
        self.object_label = self._label(side, "Objects: 0")
        self.hazard_label = self._label(side, "Hazardous objects: 0", fg="#fb7185")
        tk.Label(
            side,
            text="RECENT ALERTS",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=18, pady=(24, 5))
        self.log_box = tk.Listbox(
            side, bg="#111827", fg="#e2e8f0", borderwidth=0, selectbackground="#334155"
        )
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.status = tk.Label(
            root,
            text="Starting…",
            bg="#374151",
            fg="white",
            anchor="w",
            padx=12,
            font=("Segoe UI", 10),
        )
        self.status.pack(fill="x")

    @staticmethod
    def _label(parent, text, size=11, fg="white"):
        widget = tk.Label(
            parent,
            text=text,
            bg="#1f2937",
            fg=fg,
            font=("Segoe UI", size),
            anchor="w",
            justify="left",
        )
        widget.pack(fill="x", padx=18, pady=5)
        return widget

    def set_modules(self, camera, worker, database):
        self.camera, self.worker, self.database = camera, worker, database
        try:
            for row in reversed(
                database.get_recent_alerts(self.config.EVENT_LOG_MAX_LINES)
            ):
                self._add_event(
                    row["timestamp"][11:19] + " UTC",
                    row["threat_level"],
                    row["message"],
                )
            self._refresh_stats()
        except Exception:
            logger.exception("Could not load dashboard statistics")
        self._schedule_poll()

    def start(self):
        if not self.worker.start():
            self._monitoring = False
            self.retry_button.config(state="normal")
            self.status.config(text="Inference worker is busy · retry in a moment")
            return
        self._monitoring = True
        self._last_frame = time.monotonic()
        self.retry_button.config(state="disabled")
        self.status.config(text="Monitoring active · waiting for frames", bg="#374151")

    def show_no_camera(self):
        self._monitoring = False
        self.retry_button.config(state="normal")
        self.status.config(
            text="Camera unavailable · check permissions and camera index, then retry",
            bg="#92400e",
        )
        self.video.configure(image="", text="NO CAMERA DETECTED")
        self.video.image = None
        self.threat_label.config(text="Threat: —", fg="#cbd5e1")

    def retry_camera(self):
        self.retry_button.config(state="disabled")
        self.status.config(text="Opening camera…")
        if self.worker:
            self.worker.stop()
        self.camera.stop()
        try:
            if self.camera.start():
                self.start()
            else:
                self.show_no_camera()
        except Exception:
            logger.exception("Camera restart failed")
            self.show_no_camera()

    def reload_faces(self):
        if self.worker and self._monitoring:
            self.worker.reload_faces()
            self.status.config(text="Reloading registered faces…")
        else:
            messagebox.showinfo(
                "Face gallery",
                "Faces will load when the camera starts.",
                parent=self.root,
            )

    def open_registration(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Register a face")
        dialog.geometry("480x245")
        dialog.configure(bg="#1f2937")
        dialog.transient(self.root)
        name, image = tk.StringVar(), tk.StringVar()
        tk.Label(dialog, text="Person name", bg="#1f2937", fg="white").pack(
            anchor="w", padx=20, pady=(14, 2)
        )
        tk.Entry(dialog, textvariable=name).pack(fill="x", padx=20)
        tk.Label(
            dialog, text="Photo with exactly one visible face", bg="#1f2937", fg="white"
        ).pack(anchor="w", padx=20, pady=(12, 2))
        row = tk.Frame(dialog, bg="#1f2937")
        row.pack(fill="x", padx=20)
        tk.Entry(row, textvariable=image).pack(side="left", fill="x", expand=True)

        def browse():
            path = filedialog.askopenfilename(
                parent=dialog, filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
            )
            if path:
                image.set(path)

        tk.Button(row, text="Browse", command=browse).pack(side="left", padx=(8, 0))

        def save():
            try:
                register_person(
                    name.get(), image.get(), config=self.config, database=self.database
                )
            except RegistrationError as exc:
                messagebox.showerror("Invalid registration", str(exc), parent=dialog)
                return
            except Exception:
                logger.exception("Face registration failed")
                messagebox.showerror(
                    "Registration failed",
                    "Could not save this registration. Check the console.",
                    parent=dialog,
                )
                return
            self._refresh_stats()
            if self.worker and self._monitoring:
                self.worker.reload_faces()
            dialog.destroy()
            messagebox.showinfo(
                "Registered", "The face was registered successfully.", parent=self.root
            )

        tk.Button(
            dialog, text="Register", command=save, bg="#2563eb", fg="white", padx=14
        ).pack(pady=18)

    def _refresh_stats(self):
        if not self.database:
            return
        try:
            stats = self.database.get_statistics()
            self.persons_label.config(
                text=f"Registered persons: {stats['registered_persons']}"
            )
            self.alerts_label.config(text=f"Alerts: {stats['total_alerts']}")
            self.detections_label.config(
                text=f"Logged incidents: {stats['total_detections']}"
            )
        except Exception:
            logger.exception("Could not read dashboard statistics")
        self._last_stats = time.monotonic()

    def _add_event(self, timestamp, level, description):
        self.log_box.insert(tk.END, f"{timestamp}  [{level}] {description}")
        while self.log_box.size() > self.config.EVENT_LOG_MAX_LINES:
            self.log_box.delete(0)
        self.log_box.yview_moveto(1)

    def _schedule_poll(self):
        if not self._closed:
            self._poll_id = self.root.after(
                self.config.UI_UPDATE_INTERVAL_MS, self._poll
            )

    def _poll(self):
        if self._closed:
            return
        try:
            if self.worker:
                while True:
                    try:
                        event = self.worker.events.get_nowait()
                    except queue.Empty:
                        break
                    self._add_event(
                        event["timestamp"], event["threat_level"], event["description"]
                    )
                    self._refresh_stats()
                latest = None
                while True:
                    try:
                        latest = self.worker.frames.get_nowait()
                    except queue.Empty:
                        break
                if latest is not None and self._monitoring:
                    self._apply_result(latest)
                    self._last_frame = time.monotonic()
            if self._monitoring and self.camera and not self.camera.is_available:
                self.show_no_camera()
            elif self._monitoring and time.monotonic() - self._last_frame > 5:
                self.retry_button.config(state="normal")
                self.status.config(
                    text="No recent camera frames · check the camera or retry",
                    bg="#92400e",
                )
            if time.monotonic() - self._last_stats >= 1:
                self._refresh_stats()
        except Exception:
            logger.exception("Dashboard update failed")
        self._schedule_poll()

    def _apply_result(self, result):
        self.retry_button.config(state="disabled")
        assessment = result.assessment
        level = assessment["threat_level"]
        self.threat_label.config(
            text=f"Threat: {level}", fg=self.config.THREAT_LEVELS[level]["hex"]
        )
        self.faces_label.config(text=f"Faces: {len(result.faces)}")
        self.unknown_label.config(text=f"Unknown faces: {assessment['unknown_count']}")
        self.object_label.config(text=f"Objects: {len(result.objects)}")
        self.hazard_label.config(
            text=f"Hazardous objects: {len(assessment['hazardous_objects'])}"
        )
        if result.warnings:
            self.status.config(
                text="DEGRADED · " + "; ".join(result.warnings), bg="#92400e"
            )
        else:
            self.status.config(
                text=f"Monitoring active · {assessment['description']}", bg="#374151"
            )
        rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(640, self.video.winfo_width() - 8)
        height = max(400, self.video.winfo_height() - 8)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.video.configure(image=photo, text="")
        self.video.image = photo

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
        if self.worker:
            self.worker.stop()
        if self.camera:
            self.camera.stop()
        if self.database:
            self.database.close()
        self.root.destroy()
