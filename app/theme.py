"""Engineering dark-neutral design system and ttk styling."""

import tkinter as tk
from tkinter import ttk


class Colors:
    """Color palette for the workbench desktop interface."""
    BG_DARK = "#181920"
    BG_PANEL = "#21222c"
    BG_CARD = "#292a36"
    BG_INPUT = "#323444"
    BG_CANVAS = "#131418"
    
    TEXT_MAIN = "#f3f4f6"
    TEXT_MUTED = "#9ca3af"
    TEXT_DISABLED = "#595d6b"
    
    ACCENT_BLUE = "#3b82f6"
    ACCENT_BLUE_HOVER = "#2563eb"
    ACCENT_GREEN = "#10b981"
    ACCENT_AMBER = "#f59e0b"
    ACCENT_RED = "#ef4444"
    
    BORDER = "#373a4d"
    BORDER_LIGHT = "#4a4e66"
    BADGE_BG = "#2a2d3d"


def apply_theme(root: tk.Tk) -> None:
    """Configure modern dark theme for Tkinter and ttk widgets."""
    root.configure(bg=Colors.BG_DARK)
    
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # Standard Frame
    style.configure("TFrame", background=Colors.BG_DARK)
    style.configure("Panel.TFrame", background=Colors.BG_PANEL)
    style.configure("Card.TFrame", background=Colors.BG_CARD, relief="flat")
    
    # Standard Label
    style.configure(
        "TLabel",
        background=Colors.BG_DARK,
        foreground=Colors.TEXT_MAIN,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Panel.TLabel",
        background=Colors.BG_PANEL,
        foreground=Colors.TEXT_MAIN,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Card.TLabel",
        background=Colors.BG_CARD,
        foreground=Colors.TEXT_MAIN,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Header.TLabel",
        background=Colors.BG_PANEL,
        foreground=Colors.TEXT_MAIN,
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "Muted.TLabel",
        background=Colors.BG_PANEL,
        foreground=Colors.TEXT_MUTED,
        font=("Segoe UI", 8),
    )
    style.configure(
        "Badge.TLabel",
        background=Colors.BADGE_BG,
        foreground=Colors.ACCENT_BLUE,
        font=("Segoe UI", 8, "bold"),
        padding=(4, 2),
    )
    style.configure(
        "SuccessBadge.TLabel",
        background=Colors.BADGE_BG,
        foreground=Colors.ACCENT_GREEN,
        font=("Segoe UI", 8, "bold"),
        padding=(4, 2),
    )

    # Buttons
    style.configure(
        "TButton",
        background=Colors.BG_INPUT,
        foreground=Colors.TEXT_MAIN,
        bordercolor=Colors.BORDER,
        darkcolor=Colors.BG_INPUT,
        lightcolor=Colors.BG_INPUT,
        font=("Segoe UI", 9),
        padding=(8, 4),
        relief="flat",
    )
    style.map(
        "TButton",
        background=[("active", Colors.BORDER_LIGHT), ("disabled", Colors.BG_CARD)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Accent Button (Blue)
    style.configure(
        "Accent.TButton",
        background=Colors.ACCENT_BLUE,
        foreground="#ffffff",
        bordercolor=Colors.ACCENT_BLUE,
        darkcolor=Colors.ACCENT_BLUE,
        lightcolor=Colors.ACCENT_BLUE,
        font=("Segoe UI", 9, "bold"),
        padding=(10, 5),
        relief="flat",
    )
    style.map(
        "Accent.TButton",
        background=[("active", Colors.ACCENT_BLUE_HOVER), ("disabled", Colors.BG_CARD)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Success Button (Green)
    style.configure(
        "Success.TButton",
        background=Colors.ACCENT_GREEN,
        foreground="#ffffff",
        bordercolor=Colors.ACCENT_GREEN,
        font=("Segoe UI", 9, "bold"),
        padding=(8, 4),
    )
    style.map(
        "Success.TButton",
        background=[("active", "#059669")],
    )

    # Combobox
    style.configure(
        "TCombobox",
        fieldbackground=Colors.BG_INPUT,
        background=Colors.BG_CARD,
        foreground=Colors.TEXT_MAIN,
        darkcolor=Colors.BORDER,
        lightcolor=Colors.BORDER,
        arrowcolor=Colors.TEXT_MAIN,
        bordercolor=Colors.BORDER,
        font=("Segoe UI", 9),
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", Colors.BG_INPUT)],
        foreground=[("readonly", Colors.TEXT_MAIN)],
    )

    # Checkbutton & Radiobutton
    style.configure(
        "TCheckbutton",
        background=Colors.BG_PANEL,
        foreground=Colors.TEXT_MAIN,
        indicatorcolor=Colors.BG_INPUT,
        indicatorrelief="flat",
        font=("Segoe UI", 9),
    )
    style.map(
        "TCheckbutton",
        indicatorcolor=[("selected", Colors.ACCENT_BLUE)],
        background=[("active", Colors.BG_PANEL)],
    )

    # Scales (Sliders)
    style.configure(
        "Horizontal.TScale",
        background=Colors.BG_PANEL,
        troughcolor=Colors.BG_INPUT,
        bordercolor=Colors.BORDER,
        lightcolor=Colors.ACCENT_BLUE,
        darkcolor=Colors.ACCENT_BLUE,
    )

    # Notebook Tabs
    style.configure(
        "TNotebook",
        background=Colors.BG_DARK,
        tabmargins=[2, 5, 2, 0],
        borderwidth=0,
    )
    style.configure(
        "TNotebook.Tab",
        background=Colors.BG_PANEL,
        foreground=Colors.TEXT_MUTED,
        padding=[10, 4],
        font=("Segoe UI", 9),
        bordercolor=Colors.BORDER,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", Colors.BG_CARD)],
        foreground=[("selected", Colors.TEXT_MAIN)],
    )

    # Treeview (for contour measurement table)
    style.configure(
        "Treeview",
        background=Colors.BG_PANEL,
        foreground=Colors.TEXT_MAIN,
        fieldbackground=Colors.BG_PANEL,
        bordercolor=Colors.BORDER,
        font=("Segoe UI", 9),
        rowheight=24,
    )
    style.map(
        "Treeview",
        background=[("selected", Colors.ACCENT_BLUE)],
        foreground=[("selected", "#ffffff")],
    )
    style.configure(
        "Treeview.Heading",
        background=Colors.BG_CARD,
        foreground=Colors.TEXT_MAIN,
        font=("Segoe UI", 8, "bold"),
        bordercolor=Colors.BORDER,
        relief="flat",
        padding=[6, 4],
    )
    style.map(
        "Treeview.Heading",
        background=[("active", Colors.BORDER_LIGHT)],
    )

    # Scrollbars
    style.configure(
        "TScrollbar",
        background=Colors.BG_INPUT,
        troughcolor=Colors.BG_DARK,
        bordercolor=Colors.BG_DARK,
        arrowcolor=Colors.TEXT_MUTED,
    )
    style.map("TScrollbar", background=[("active", Colors.BORDER_LIGHT)])
    
    # Separator
    style.configure("TSeparator", background=Colors.BORDER)
    
    # Labelframe
    style.configure(
        "TLabelframe",
        background=Colors.BG_PANEL,
        bordercolor=Colors.BORDER,
        lightcolor=Colors.BORDER,
        darkcolor=Colors.BORDER,
    )
    style.configure(
        "TLabelframe.Label",
        background=Colors.BG_PANEL,
        foreground=Colors.TEXT_MAIN,
        font=("Segoe UI", 9, "bold"),
    )
