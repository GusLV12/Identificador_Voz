# ui_theme.py
from tkinter import ttk

# Paleta estilo laboratorio / dashboard técnico
COLORS = {
    "bg": "#1e1e1e",
    "panel": "#252525",
    "card": "#2c2c2c",
    "border": "#3a3a3a",
    "title": "#4fc3f7",
    "text": "#e0e0e0",
    "muted": "#9e9e9e",
    "ok": "#66bb6a",
    "warn": "#ffa726",
    "err": "#ef5350"
}

def apply_theme(root):
    root.configure(bg=COLORS["bg"])

    style = ttk.Style()
    style.theme_use("default")

    style.configure(
        "Title.TLabel",
        font=("Segoe UI", 18, "bold"),
        background=COLORS["bg"],
        foreground=COLORS["title"]
    )

    style.configure(
        "Subtitle.TLabel",
        font=("Segoe UI", 10),
        background=COLORS["bg"],
        foreground=COLORS["muted"]
    )

    style.configure(
        "Mono.TLabel",
        font=("Consolas", 10),
        background=COLORS["bg"],
        foreground=COLORS["text"]
    )

    style.configure(
        "Panel.TButton",
        font=("Segoe UI", 10),
        padding=10,
        background=COLORS["panel"],
        foreground=COLORS["text"],
        relief="flat"
    )

    style.map(
        "Panel.TButton",
        background=[("active", COLORS["border"])]
    )
