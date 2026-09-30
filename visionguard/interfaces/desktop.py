"""Desktop command center; reads the same bounded state as the web dashboard."""

import logging
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from visionguard.core.registration import RegistrationError

logger = logging.getLogger(__name__)
BG = "#090e15"
PANEL = "#111a26"
PANEL2 = "#172431"
BORDER = "#263341"
TEXT = "#eef6f6"
MUTED = "#95a9b6"
ACCENT = "#c5f46e"
CYAN = "#65ddde"
AMBER = "#f4bb72"
RED = "#fa7d83"


class MainWindow:
    def __init__(self, root, service, web_url=None, stop_web=None):
        self.root, self.service = root, service
        self.config = service.config
        self.web_url, self.stop_web = web_url, stop_web
        self._closed = False
        self._poll_id = None
        self._sequence = 0
        self._event_number = 0
        self._last_stats = 0.0
        self._last_frame = 0.0

        root.title("VisionGuard · Local Command Center")
        root.geometry("1450x880")
        root.minsize(1060, 690)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._load_history()
        self._poll()

    @staticmethod
    def label(parent, text, size=11, color=TEXT, bold=False, bg=PANEL):
        return tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=color,
            anchor="w",
            justify="left",
            font=("Segoe UI", size, "bold" if bold else "normal"),
        )

    def button(self, parent, text, command, highlight=False):
        return tk.Button(
            parent,
            text=text,
            command=command,
            cursor="hand2",
            relief="flat",
            bg=ACCENT if highlight else PANEL2,
            fg="#142016" if highlight else TEXT,
            activebackground="#d7ffa1" if highlight else "#324353",
            activeforeground="#142016" if highlight else TEXT,
            font=("Segoe UI", 10, "bold"),
            padx=13,
            pady=9,
        )

    def _build(self):
        header = tk.Frame(self.root, bg=PANEL, height=70)
        header.pack(fill="x")
        brand = tk.Frame(header, bg=PANEL)
        brand.pack(side="left", padx=25, pady=14)
        self.label(brand, "◈", 24, ACCENT, True).pack(side="left", padx=(0, 12))
        brand_text = tk.Frame(brand, bg=PANEL)
        brand_text.pack(side="left")
        self.label(brand_text, "VISIONGUARD", 15, TEXT, True).pack(anchor="w")
        self.label(brand_text, "LOCAL COMMAND CENTER", 8, MUTED).pack(anchor="w")
        self.label(header, "●  PRIVATE SESSION", 10, CYAN, True).pack(
            side="right", padx=28
        )

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=25, pady=(21, 16))
        title = tk.Frame(content, bg=BG)
        title.pack(fill="x", pady=(0, 19))
        self.label(title, "Operational overview", 26, TEXT, True, BG).pack(side="left")
        controls = tk.Frame(title, bg=BG)
        controls.pack(side="right")
        for text, command in (
            ("Register face", self.open_registration),
            ("Manage faces", self.manage_faces),
            ("Reload model", self.reload_detector),
            ("Reload faces", self.reload_faces),
        ):
            self.button(controls, text, command).pack(side="left", padx=4)
        if self.web_url:
            self.button(controls, "Open web dashboard ↗", self.open_browser).pack(
                side="left", padx=4
            )
        self.retry_button = self.button(
            controls, "Retry camera", self.retry_camera, highlight=True
        )
        self.retry_button.pack(side="left", padx=(4, 0))

        warning = tk.Frame(
            content, bg="#372b23", highlightbackground="#76563d", highlightthickness=1
        )
        warning.pack(fill="x", pady=(0, 15))
        self.label(
            warning,
            "⚠   EXPERIMENTAL — The detector misses many hazards. "
            "A LOW reading never means a scene is safe. Not tested on your webcam.",
            10,
            "#ffdaa9",
            True,
            "#372b23",
        ).pack(anchor="w", padx=15, pady=11)

        metrics = tk.Frame(content, bg=BG)
        metrics.pack(fill="x", pady=(0, 16))
        metric_specs = (
            ("CAMERA", "camera_label", CYAN),
            ("ASSESSMENT", "threat_label", ACCENT),
            ("REGISTERED PEOPLE", "persons_label", TEXT),
            ("RECORDED ALERTS", "alerts_label", AMBER),
        )
        for col, (title_text, attribute, color) in enumerate(metric_specs):
            metrics.grid_columnconfigure(col, weight=1, uniform="metric")
            card = tk.Frame(
                metrics,
                bg=PANEL,
                highlightbackground=BORDER,
                highlightthickness=1,
                padx=18,
                pady=15,
            )
            card.grid(row=0, column=col, sticky="nsew", padx=(0, 8) if col < 3 else 0)
            self.label(card, title_text, 9, MUTED, True).pack(anchor="w")
            widget = self.label(card, "—", 22, color, True)
            widget.pack(anchor="w", pady=(7, 2))
            setattr(self, attribute, widget)

        body = tk.Frame(content, bg=BG)
        body.pack(fill="both", expand=True)
        feed = tk.Frame(
            body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1
        )
        feed.pack(side="left", fill="both", expand=True, padx=(0, 15))
        feed_header = tk.Frame(feed, bg=PANEL)
        feed_header.pack(fill="x", padx=17, pady=12)
        self.label(feed_header, "Live camera feed", 13, TEXT, True).pack(side="left")
        self.feed_pill = self.label(feed_header, "● NO SIGNAL", 10, MUTED, True)
        self.feed_pill.pack(side="right")
        self.video = tk.Label(
            feed,
            bg="#0b1520",
            fg=MUTED,
            text="NO LIVE CAMERA FRAME",
            font=("Segoe UI", 18, "bold"),
            anchor="center",
        )
        self.video.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.label(
            feed,
            "CAM 01   •   LOCAL WEBCAM   •   Verify all alerts manually",
            9,
            MUTED,
        ).pack(anchor="w", padx=18, pady=(0, 12))

        side = tk.Frame(
            body, bg=PANEL, width=290, highlightbackground=BORDER, highlightthickness=1
        )
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.label(side, "LIVE ANALYSIS", 12, TEXT, True).pack(
            anchor="w", padx=18, pady=(17, 12)
        )
        self.description = self.label(side, "No live assessment", 12, CYAN, True)
        self.description.configure(wraplength=245)
        self.description.pack(anchor="w", padx=18, pady=(0, 14))
        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=15)
        self.detector_label = self.label(side, "Detector: loading…", 11)
        self.detector_label.pack(anchor="w", padx=18, pady=(15, 5))
        self.faces_label = self.label(side, "Faces: —", 11)
        self.faces_label.pack(anchor="w", padx=18, pady=5)
        self.unknown_label = self.label(side, "Unknown faces: —", 11, AMBER)
        self.unknown_label.pack(anchor="w", padx=18, pady=5)
        self.hazard_label = self.label(side, "Hazards reported: —", 11, RED)
        self.hazard_label.pack(anchor="w", padx=18, pady=5)
        self.fps_label = self.label(side, "Camera FPS: —", 11, MUTED)
        self.fps_label.pack(anchor="w", padx=18, pady=5)
        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=15, pady=(14, 12))
        self.label(side, "RECENT INCIDENTS", 10, MUTED, True).pack(
            anchor="w", padx=18, pady=(0, 7)
        )
        self.log_box = tk.Listbox(
            side,
            bg="#0c1520",
            fg=TEXT,
            selectbackground="#344552",
            font=("Segoe UI", 9),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.log_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.status = self.label(self.root, "Starting…", 10, MUTED, bg=PANEL2)
        self.status.configure(wraplength=1350)
        self.status.pack(fill="x", side="bottom", padx=0, pady=0, ipadx=20, ipady=10)

    def _load_history(self):
        for row in reversed(self.service.database.get_recent_alerts(60)):
            self._add_event(
                row["timestamp"][11:19] + " UTC", row["threat_level"], row["message"]
            )

    def _add_event(self, when, level, text):
        self.log_box.insert(tk.END, f"{when}  [{level}] {text}")
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
            state = self.service.worker.state
            for number, incident in state.events_after(self._event_number):
                self._event_number = number
                self._add_event(
                    incident["timestamp"],
                    incident["threat_level"],
                    incident["description"],
                )
            snapshot = state.read()
            if snapshot.sequence != self._sequence and self.service.camera.is_available:
                self._sequence = snapshot.sequence
                if snapshot.result is not None:
                    self._last_frame = time.monotonic()
                    self._apply_result(snapshot.result)
            available = self.service.camera.is_available
            live = available and self.service.worker.is_running
            if not live:
                self._show_no_camera()
            elif time.monotonic() - self._last_frame > 5:
                self.feed_pill.config(text="● WAITING", fg=AMBER)
                self.status.config(
                    text="Camera open, but no recent analyzed frame. "
                    "Retry camera if needed.",
                    fg=AMBER,
                )
            if time.monotonic() - self._last_stats > 1:
                stats = self.service.database.get_statistics()
                self.persons_label.config(text=str(stats["registered_persons"]))
                self.alerts_label.config(text=str(stats["total_alerts"]))
                self.detector_label.config(
                    text=f"Detector: {len(self.service.objects.class_names)}/4 classes"
                )
                self._last_stats = time.monotonic()
        except Exception:
            logger.exception("Desktop dashboard update failed")
        self._schedule_poll()

    def _show_no_camera(self):
        self.camera_label.config(text="OFFLINE", fg=RED)
        self.threat_label.config(text="—", fg=MUTED)
        self.feed_pill.config(text="● NO SIGNAL", fg=RED)
        self.video.configure(image="", text="NO LIVE CAMERA FRAME")
        self.video.image = None
        self.description.config(text="No live assessment")
        self.status.config(
            text="Camera unavailable · check permissions and camera index, "
            "then Retry camera",
            fg=AMBER,
        )

    def _apply_result(self, result):
        self.camera_label.config(text="ONLINE", fg=ACCENT)
        self.feed_pill.config(text="● LIVE", fg=ACCENT)
        level = result.assessment["threat_level"]
        self.threat_label.config(text=level, fg=self.config.THREAT_LEVELS[level]["hex"])
        self.description.config(text=result.assessment["description"])
        self.faces_label.config(text=f"Faces: {len(result.faces)}")
        self.unknown_label.config(
            text=f"Unknown faces: {result.assessment['unknown_count']}"
        )
        self.hazard_label.config(
            text=f"Hazards reported: {len(result.assessment['hazardous_objects'])}"
        )
        self.fps_label.config(text=f"Camera FPS: {result.fps:.1f}")
        self.status.config(
            text="EXPERIMENTAL · "
            + (
                "; ".join(result.warnings)
                if result.warnings
                else "No verified safety assurance"
            ),
            fg=AMBER,
        )
        rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(
            (
                max(320, self.video.winfo_width() - 12),
                max(240, self.video.winfo_height() - 12),
            ),
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(image)
        self.video.configure(image=photo, text="")
        self.video.image = photo

    def retry_camera(self):
        self.retry_button.config(state="disabled")
        self.status.config(text="Retrying camera in the background…", fg=CYAN)

        def restart():
            try:
                success = self.service.retry_camera()
            except Exception:
                logger.exception("Cannot retry camera")
                success = False
            if not self._closed:
                try:
                    self.root.after(0, lambda: self._camera_restarted(success))
                except tk.TclError:
                    pass

        threading.Thread(target=restart, name="camera-retry", daemon=True).start()

    def _camera_restarted(self, success):
        if self._closed:
            return
        self.retry_button.config(state="normal")
        if not success:
            self._show_no_camera()

    def reload_faces(self):
        self.service.reload_faces()
        self.status.config(text="Face gallery queued for reload", fg=CYAN)

    def reload_detector(self):
        self.service.reload_detector()
        self.status.config(text="Detector queued for reload", fg=CYAN)

    def open_browser(self):
        if self.web_url:
            webbrowser.open(self.web_url)

    def open_registration(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Register a face · VisionGuard")
        dialog.geometry("480x285")
        dialog.configure(bg=PANEL)
        dialog.transient(self.root)
        dialog.grab_set()
        name, image = tk.StringVar(), tk.StringVar()
        self.label(dialog, "New trusted identity", 17, TEXT, True).pack(
            anchor="w", padx=22, pady=(18, 12)
        )
        self.label(dialog, "Name", 10, MUTED).pack(anchor="w", padx=22)
        tk.Entry(dialog, textvariable=name).pack(fill="x", padx=22, pady=(2, 10))
        self.label(dialog, "Photo showing exactly one face", 10, MUTED).pack(
            anchor="w", padx=22
        )
        row = tk.Frame(dialog, bg=PANEL)
        row.pack(fill="x", padx=22, pady=(3, 0))
        tk.Entry(row, textvariable=image).pack(side="left", fill="x", expand=True)

        def browse():
            path = filedialog.askopenfilename(
                parent=dialog, filetypes=[("Photos", "*.jpg *.jpeg *.png *.bmp")]
            )
            if path:
                image.set(path)

        self.button(row, "Browse", browse).pack(side="left", padx=(8, 0))

        def save():
            try:
                self.service.register_face(name.get(), image.get())
            except RegistrationError as exc:
                messagebox.showerror("Invalid registration", str(exc), parent=dialog)
                return
            except Exception:
                logger.exception("Face registration failed")
                messagebox.showerror(
                    "Failed", "Could not save this face.", parent=dialog
                )
                return
            dialog.destroy()
            self.status.config(
                text="Face registered locally · gallery reloading", fg=CYAN
            )

        self.button(dialog, "Register face", save, True).pack(pady=18)

    def manage_faces(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage registered faces")
        dialog.geometry("425x360")
        dialog.configure(bg=PANEL)
        dialog.transient(self.root)
        self.label(dialog, "Trusted identities", 17, TEXT, True).pack(
            anchor="w", padx=20, pady=16
        )
        entries = tk.Listbox(
            dialog,
            bg="#0c1520",
            fg=TEXT,
            selectbackground="#344552",
            relief="flat",
            font=("Segoe UI", 11),
        )
        entries.pack(fill="both", expand=True, padx=20)
        people = []

        def refresh():
            nonlocal people
            people = self.service.database.get_all_persons()
            entries.delete(0, tk.END)
            for person in people:
                entries.insert(tk.END, person["name"])

        def deactivate():
            selection = entries.curselection()
            if not selection:
                return
            person = people[selection[0]]
            if messagebox.askyesno(
                "Deactivate face",
                f"Stop recognizing {person['name']} and remove their photo?",
                parent=dialog,
            ):
                self.service.deactivate_face(person["id"])
                refresh()

        refresh()
        self.button(dialog, "Deactivate selected", deactivate).pack(
            anchor="e", padx=20, pady=14
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
        if self.stop_web:
            self.stop_web()
        self.service.close()
        self.root.destroy()
