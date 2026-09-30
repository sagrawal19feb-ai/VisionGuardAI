"""Standalone face-registration window. Run: python face_register.py"""

import logging
import tkinter as tk
from tkinter import filedialog, messagebox

from modules.registration import RegistrationError, register_person

logger = logging.getLogger(__name__)


def main():
    root = tk.Tk()
    root.title("VisionGuardAI · Face Registration")
    root.geometry("520x275")
    root.configure(bg="#111827")
    name_var = tk.StringVar()
    image_var = tk.StringVar()

    def browse_image():
        path = filedialog.askopenfilename(
            parent=root,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")],
        )
        if path:
            image_var.set(path)

    def submit():
        try:
            register_person(name_var.get(), image_var.get())
        except RegistrationError as exc:
            messagebox.showerror("Registration failed", str(exc), parent=root)
            return
        except Exception:
            logger.exception("Registration failed")
            messagebox.showerror(
                "Registration failed",
                "Could not save the registration. Check the console for details.",
                parent=root,
            )
            return
        messagebox.showinfo(
            "Registered",
            "Face registered. Click Reload Faces in the monitor to apply it.",
            parent=root,
        )
        name_var.set("")
        image_var.set("")

    tk.Label(
        root,
        text="REGISTER A PERSON",
        bg="#111827",
        fg="white",
        font=("Segoe UI", 17, "bold"),
    ).pack(pady=(18, 12))
    tk.Label(root, text="Person name", bg="#111827", fg="#cbd5e1").pack(
        anchor="w", padx=24
    )
    tk.Entry(root, textvariable=name_var, width=56).pack(padx=24, pady=(2, 10))
    tk.Label(
        root, text="Photo with exactly one visible face", bg="#111827", fg="#cbd5e1"
    ).pack(anchor="w", padx=24)
    row = tk.Frame(root, bg="#111827")
    row.pack(fill="x", padx=24)
    tk.Entry(row, textvariable=image_var).pack(side="left", fill="x", expand=True)
    tk.Button(row, text="Browse", command=browse_image).pack(side="left", padx=(8, 0))
    tk.Button(
        root,
        text="Register",
        command=submit,
        bg="#2563eb",
        fg="white",
        font=("Segoe UI", 11, "bold"),
        padx=20,
    ).pack(pady=18)
    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
