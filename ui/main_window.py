import tkinter as tk
from PIL import Image, ImageTk
import cv2


class MainWindow:

    def __init__(self, root, config):

        self.root = root
        self.config = config

        self.camera = None
        self.face_module = None
        self.object_module = None
        self.threat_module = None
        self.alert_system = None
        self.database = None

        self.root.title("AI Security Monitor")
        self.root.geometry("1400x850")
        self.root.configure(bg="#111827")

        # =========================
        # Header
        # =========================

        header = tk.Frame(
            self.root,
            bg="#1f2937",
            height=60
        )
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="AI SECURITY MONITOR",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 18, "bold")
        )
        title.pack(pady=12)

        # =========================
        # Main Layout
        # =========================

        main = tk.Frame(
            self.root,
            bg="#111827"
        )
        main.pack(fill="both", expand=True)

        # Camera Area

        left_panel = tk.Frame(
            main,
            bg="#111827"
        )
        left_panel.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.video = tk.Label(
            left_panel,
            bg="black"
        )
        self.video.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # Dashboard Area

        right_panel = tk.Frame(
            main,
            bg="#1f2937",
            width=320
        )
        right_panel.pack(
            side="right",
            fill="y"
        )

        tk.Label(
            right_panel,
            text="SYSTEM STATUS",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=15)

        self.threat_label = tk.Label(
            right_panel,
            text="Threat: LOW",
            bg="#1f2937",
            fg="lime",
            font=("Segoe UI", 16, "bold")
        )
        self.threat_label.pack(pady=10)

        self.persons_label = tk.Label(
            right_panel,
            text="Registered Persons: 0",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 12)
        )
        self.persons_label.pack(pady=10)

        self.alerts_label = tk.Label(
            right_panel,
            text="Alerts: 0",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 12)
        )
        self.alerts_label.pack(pady=10)

        self.faces_label = tk.Label(
            right_panel,
            text="Faces: 0",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 12)
        )
        self.faces_label.pack(pady=10)

        self.unknown_label = tk.Label(
            right_panel,
            text="Unknown Faces: 0",
            bg="#1f2937",
            fg="orange",
            font=("Segoe UI", 12)
        )
        self.unknown_label.pack(pady=10)

        self.object_label = tk.Label(
            right_panel,
            text="Objects: 0",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 12)
        )
        self.object_label.pack(pady=10)

        self.hazard_label = tk.Label(
            right_panel,
            text="Hazardous Objects: 0",
            bg="#1f2937",
            fg="red",
            font=("Segoe UI", 12)
        )
        self.hazard_label.pack(pady=10)

        tk.Label(
            right_panel,
            text="Recent Events",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=(20, 5))

        self.log_box = tk.Listbox(
            right_panel,
            bg="#111827",
            fg="white",
            height=15
        )
        self.log_box.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.status = tk.Label(
            self.root,
            text="Monitoring Active",
            bg="#374151",
            fg="white",
            anchor="w",
            padx=10
        )
        self.status.pack(fill="x")

    def set_modules(
        self,
        camera,
        face_module,
        object_module,
        threat_module,
        alert_system,
        database
    ):
        self.camera = camera
        self.face_module = face_module
        self.object_module = object_module
        self.threat_module = threat_module
        self.alert_system = alert_system
        self.database = database

    def start(self):
        self.update_frame()

    def update_frame(self):

        if self.camera:

            frame = self.camera.get_frame()

            if frame is not None:

                faces = []
                objects = []

                try:
                    if self.face_module:
                        faces = self.face_module.detect_and_recognize(frame)
                except Exception as e:
                    print("Face error:", e)

                try:
                    if self.object_module:
                        objects = self.object_module.detect(frame)
                except Exception as e:
                    print("Object error:", e)

                # Draw Faces

                for face in faces:

                    left, top, right, bottom = face.bbox

                    color = (
                        (0, 255, 0)
                        if face.is_known
                        else (0, 0, 255)
                    )

                    cv2.rectangle(
                        frame,
                        (left, top),
                        (right, bottom),
                        color,
                        2
                    )

                    label = face.name

                    if face.confidence > 0:
                        label += f" {face.confidence:.2f}"

                    cv2.putText(
                        frame,
                        label,
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                # Draw Objects

                for obj in objects:

                    x1, y1, x2, y2 = obj.bbox

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        obj.color,
                        2
                    )

                    text = f"{obj.label} {obj.confidence:.2f}"

                    if obj.is_hazardous:
                        text += f" [{obj.threat_modifier}]"

                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        obj.color,
                        2
                    )

                known_faces = sum(
                    1 for f in faces if f.is_known
                )

                unknown_faces = sum(
                    1 for f in faces if not f.is_known
                )

                hazardous = sum(
                    1 for o in objects if o.is_hazardous
                )

                # Dashboard Updates

                self.faces_label.config(
                    text=f"Faces: {len(faces)}"
                )

                self.unknown_label.config(
                    text=f"Unknown Faces: {unknown_faces}"
                )

                self.object_label.config(
                    text=f"Objects: {len(objects)}"
                )

                self.hazard_label.config(
                    text=f"Hazardous Objects: {hazardous}"
                )

                if self.database:

                    try:
                        stats = self.database.get_statistics()

                        self.persons_label.config(
                            text=f"Registered Persons: {stats['registered_persons']}"
                        )

                        self.alerts_label.config(
                            text=f"Alerts: {stats['total_alerts']}"
                        )

                    except:
                        pass

                if hazardous > 0:
                    self.threat_label.config(
                        text="Threat: HIGH",
                        fg="red"
                    )
                elif unknown_faces > 0:
                    self.threat_label.config(
                        text="Threat: MEDIUM",
                        fg="orange"
                    )
                else:
                    self.threat_label.config(
                        text="Threat: LOW",
                        fg="lime"
                    )

                status_text = (
                    f"Faces: {len(faces)} | "
                    f"Known: {known_faces} | "
                    f"Unknown: {unknown_faces} | "
                    f"Objects: {len(objects)} | "
                    f"Hazardous: {hazardous}"
                )

                self.status.config(
                    text=status_text
                )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                img = Image.fromarray(rgb)

                photo = ImageTk.PhotoImage(img)

                self.video.configure(image=photo)
                self.video.image = photo

        self.root.after(
            30,
            self.update_frame
        )

    def show_no_camera(self):

        self.status.config(
            text="Camera Not Available"
        )

        self.video.configure(
            text="NO CAMERA DETECTED",
            font=("Arial", 24)
        )