"""Status bar component displaying image metadata, operation state, and execution time."""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from app.theme import Colors


class StatusBar(ttk.Frame):
    """Bottom status bar with multi-segmented indicators for technical feedback."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        # Left side: Status message & Pixel inspection
        left_box = ttk.Frame(self, style="Panel.TFrame")
        left_box.grid(row=0, column=0, sticky="w", padx=8, pady=3)

        self._lbl_status = ttk.Label(
            left_box,
            text="Ready. Load an image to begin.",
            style="Panel.TLabel",
            font=("Segoe UI", 9),
        )
        self._lbl_status.pack(side="left", padx=(0, 16))

        self._lbl_pixel = ttk.Label(
            left_box,
            text="Pixel: (--, --)",
            style="Muted.TLabel",
        )
        self._lbl_pixel.pack(side="left")

        # Right side: Technical metrics
        right_box = ttk.Frame(self, style="Panel.TFrame")
        right_box.grid(row=0, column=1, sticky="e", padx=8, pady=3)

        self._lbl_dim = ttk.Label(
            right_box,
            text="Dim: -- x --",
            style="Badge.TLabel",
        )
        self._lbl_dim.pack(side="left", padx=4)

        self._lbl_channels = ttk.Label(
            right_box,
            text="Ch: --",
            style="Badge.TLabel",
        )
        self._lbl_channels.pack(side="left", padx=4)

        self._lbl_op = ttk.Label(
            right_box,
            text="Op: None",
            style="Badge.TLabel",
        )
        self._lbl_op.pack(side="left", padx=4)

        self._lbl_time = ttk.Label(
            right_box,
            text="Time: -- ms",
            style="SuccessBadge.TLabel",
        )
        self._lbl_time.pack(side="left", padx=4)

    def set_status(self, text: str, is_error: bool = False) -> None:
        """Update general status message."""
        self._lbl_status.config(
            text=text,
            foreground=Colors.ACCENT_RED if is_error else Colors.TEXT_MAIN,
        )

    def set_pixel_info(self, text: str) -> None:
        """Update pixel inspection readout."""
        self._lbl_pixel.config(text=text)

    def update_metrics(
        self,
        width: int,
        height: int,
        channels: int,
        operation: str = "None",
        time_ms: float = 0.0,
    ) -> None:
        """Update technical metrics badges in status bar."""
        if width > 0 and height > 0:
            self._lbl_dim.config(text=f"Dim: {width} x {height}")
            ch_str = "Gray (1-ch)" if channels == 1 else f"Color ({channels}-ch)"
            self._lbl_channels.config(text=f"Ch: {ch_str}")
        else:
            self._lbl_dim.config(text="Dim: -- x --")
            self._lbl_channels.config(text="Ch: --")

        self._lbl_op.config(text=f"Op: {operation}")
        if time_ms > 0:
            self._lbl_time.config(text=f"Time: {time_ms:.2f} ms")
        else:
            self._lbl_time.config(text="Time: -- ms")
