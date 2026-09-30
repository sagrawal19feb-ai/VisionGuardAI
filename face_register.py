import tkinter as tk
from tkinter import filedialog, messagebox
import sqlite3
import shutil
from pathlib import Path

DB_PATH = "data/security.db"
FACES_DIR = Path("data/faces")

FACES_DIR.mkdir(parents=True, exist_ok=True)


def browse_image():
    path = filedialog.askopenfilename(
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    if path:
        image_var.set(path)


def register_person():
    name = name_var.get().strip()
    image_path = image_var.get().strip()

    if not name:
        messagebox.showerror(
            "Error",
            "Enter a name"
        )
        return

    if not image_path:
        messagebox.showerror(
            "Error",
            "Select an image"
        )
        return

    ext = Path(image_path).suffix
    save_path = FACES_DIR / f"{name}{ext}"

    shutil.copy(image_path, save_path)

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT OR REPLACE INTO persons
        (
            name,
            face_encoding,
            engine_used,
            image_path
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            name,
            None,
            "manual",
            str(save_path)
        )
    )

    conn.commit()
    conn.close()

    messagebox.showinfo(
        "Success",
        f"{name} added successfully!"
    )

    name_var.set("")
    image_var.set("")


root = tk.Tk()
root.title("Face Registration Tool")
root.geometry("500x220")

name_var = tk.StringVar()
image_var = tk.StringVar()

tk.Label(
    root,
    text="Person Name"
).pack(pady=5)

tk.Entry(
    root,
    textvariable=name_var,
    width=40
).pack()

tk.Label(
    root,
    text="Image"
).pack(pady=5)

tk.Entry(
    root,
    textvariable=image_var,
    width=40
).pack()

tk.Button(
    root,
    text="Browse",
    command=browse_image
).pack(pady=5)

tk.Button(
    root,
    text="Register",
    command=register_person,
    bg="green",
    fg="white"
).pack(pady=10)
input("\nPress ENTER to close...")
root.mainloop()