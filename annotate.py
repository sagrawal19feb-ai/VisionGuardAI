"""Small click-and-drag hazard image labeler. Run: python annotate.py"""

import argparse
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from config import Config
from modules.training_data import read_image

COLORS = {
    "knife": "#ef4444",
    "scissors": "#fbbf24",
    "baseball bat": "#f97316",
    "gun": "#a855f7",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class Labeler:
    def __init__(self, root, images_dir, annotations):
        self.root = root
        self.images_dir = images_dir.resolve()
        self.annotations = annotations
        self.files = sorted(
            p
            for p in self.images_dir.rglob("*")
            if p.suffix.lower() in EXTENSIONS
            and p.is_file()
            and p.resolve().is_relative_to(self.images_dir)
        )
        if not self.files:
            raise ValueError(f"Put JPG/PNG/BMP images in {self.images_dir} first")
        if annotations.is_file():
            data = json.loads(annotations.read_text(encoding="utf-8"))
            if data.get("version") != 1 or not isinstance(data.get("images"), list):
                raise ValueError(
                    "Existing annotations.json is not a version-1 manifest"
                )
            self.entries = {row["file"]: row for row in data["images"]}
        else:
            self.entries = {}
        self.index = 0
        self.boxes = []
        self.image = None
        self.photo = None
        self.scale_x = self.scale_y = 1.0
        self.drag_start = None
        self.draft = None
        self.label = tk.StringVar(value="knife")

        root.title("VisionGuardAI · Hazard Labeler")
        root.geometry("1200x830")
        root.configure(bg="#111827")
        controls = tk.Frame(root, bg="#1f2937")
        controls.pack(fill="x")
        tk.Label(controls, text="Box label:", bg="#1f2937", fg="white").pack(
            side="left", padx=(15, 4)
        )
        tk.OptionMenu(controls, self.label, *Config.HAZARDOUS_OBJECTS).pack(side="left")
        for text, command in (
            ("◀ Previous (P)", lambda: self.change(-1)),
            ("Next (N) ▶", lambda: self.change(1)),
            ("Undo box (U)", self.undo),
            ("Save (Ctrl+S)", self.save),
        ):
            tk.Button(controls, text=text, command=command, padx=9).pack(
                side="left", padx=6, pady=10
            )
        self.status = tk.Label(root, bg="#111827", fg="#e2e8f0", anchor="w")
        self.status.pack(fill="x", padx=14, pady=6)
        self.canvas = tk.Canvas(
            root, bg="#020617", highlightthickness=0, width=1120, height=700
        )
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        root.bind("n", lambda event: self.change(1))
        root.bind("p", lambda event: self.change(-1))
        root.bind("u", lambda event: self.undo())
        root.bind("<Control-s>", lambda event: self.save())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_image()

    def filename(self):
        return self.files[self.index].relative_to(self.images_dir).as_posix()

    def show_image(self):
        self.image = read_image(self.files[self.index])
        key = self.filename()
        self.boxes = list(self.entries.get(key, {}).get("boxes", []))
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        real_width, real_height = image.size
        image.thumbnail((1100, 660), Image.Resampling.LANCZOS)
        self.scale_x = image.width / real_width
        self.scale_y = image.height / real_height
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.redraw_boxes()

    def redraw_boxes(self):
        self.canvas.delete("annotation")
        for box in self.boxes:
            x, y, width, height = box["bbox"]
            color = COLORS.get(box["label"], "white")
            x1, y1 = x * self.scale_x, y * self.scale_y
            x2, y2 = (x + width) * self.scale_x, (y + height) * self.scale_y
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=3, tags="annotation"
            )
            self.canvas.create_text(
                x1 + 3,
                max(10, y1 - 12),
                anchor="w",
                text=box["label"],
                fill=color,
                font=("Arial", 12, "bold"),
                tags="annotation",
            )
        self.status.config(
            text=f"{self.index + 1}/{len(self.files)} · {self.filename()} · "
            f"{len(self.boxes)} boxes · {len(self.entries)} images saved. "
            "Drag a box around each hazard. Leave empty for a background image."
        )

    def clamp(self, x, y):
        return (
            max(0, min(x, self.image.shape[1] * self.scale_x)),
            max(0, min(y, self.image.shape[0] * self.scale_y)),
        )

    def mouse_down(self, event):
        self.drag_start = self.clamp(event.x, event.y)

    def mouse_drag(self, event):
        if self.drag_start is None:
            return
        if self.draft is not None:
            self.canvas.delete(self.draft)
        x, y = self.clamp(event.x, event.y)
        self.draft = self.canvas.create_rectangle(
            *self.drag_start, x, y, outline=COLORS[self.label.get()], width=2
        )

    def mouse_up(self, event):
        if self.drag_start is None:
            return
        start_x, start_y = self.drag_start
        end_x, end_y = self.clamp(event.x, event.y)
        self.drag_start = None
        if self.draft is not None:
            self.canvas.delete(self.draft)
            self.draft = None
        x1, x2 = sorted((start_x / self.scale_x, end_x / self.scale_x))
        y1, y2 = sorted((start_y / self.scale_y, end_y / self.scale_y))
        left, top = round(x1), round(y1)
        right, bottom = round(x2), round(y2)
        if right - left >= 2 and bottom - top >= 2:
            self.boxes.append(
                {
                    "label": self.label.get(),
                    "bbox": [left, top, right - left, bottom - top],
                }
            )
            self.redraw_boxes()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.redraw_boxes()

    def save(self):
        self.entries[self.filename()] = {"file": self.filename(), "boxes": self.boxes}
        self.annotations.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "images": sorted(self.entries.values(), key=lambda row: row["file"]),
        }
        temporary = self.annotations.with_name(self.annotations.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.annotations)
        self.redraw_boxes()

    def change(self, direction):
        self.save()
        self.index = (self.index + direction) % len(self.files)
        self.show_image()

    def close(self):
        try:
            self.save()
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument(
        "--annotations", type=Path, default=Path("dataset/annotations.json")
    )
    args = parser.parse_args()
    root = tk.Tk()
    try:
        Labeler(root, args.images_dir, args.annotations)
        root.mainloop()
    except ValueError as exc:
        root.destroy()
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
