"""Toolbar component providing rapid access to file and view actions."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from app.theme import Colors


class Toolbar(ttk.Frame):
    """Top toolbar with primary workflow actions and keyboard shortcut tooltips."""

    def __init__(
        self,
        parent: tk.Widget,
        on_open: Callable[[], None],
        on_save: Callable[[], None],
        on_save_as: Callable[[], None],
        on_reset: Callable[[], None],
        on_fit_view: Callable[[], None],
        on_actual_size: Callable[[], None],
        on_toggle_view: Callable[[], None],
        on_chain_source: Callable[[], None],
        on_export_csv: Callable[[], None],
    ) -> None:
        super().__init__(parent, style="Panel.TFrame")

        self.on_open = on_open
        self.on_save = on_save
        self.on_save_as = on_save_as
        self.on_reset = on_reset
        self.on_fit_view = on_fit_view
        self.on_actual_size = on_actual_size
        self.on_toggle_view = on_toggle_view
        self.on_chain_source = on_chain_source
        self.on_export_csv = on_export_csv

        self._build_ui()

    def _build_ui(self) -> None:
        # File operations group
        self.btn_open = ttk.Button(
            self,
            text="📁 Open (Ctrl+O)",
            command=self.on_open,
        )
        self.btn_open.pack(side="left", padx=(6, 2), pady=4)

        self.btn_save = ttk.Button(
            self,
            text="💾 Save (Ctrl+S)",
            command=self.on_save,
        )
        self.btn_save.pack(side="left", padx=2, pady=4)

        self.btn_save_as = ttk.Button(
            self,
            text="💾 Save As...",
            command=self.on_save_as,
        )
        self.btn_save_as.pack(side="left", padx=2, pady=4)

        # Separator
        sep1 = ttk.Separator(self, orient="vertical")
        sep1.pack(side="left", fill="y", padx=6, pady=4)

        # Pipeline actions
        self.btn_reset = ttk.Button(
            self,
            text="↺ Reset (Ctrl+R)",
            command=self.on_reset,
        )
        self.btn_reset.pack(side="left", padx=2, pady=4)

        self.btn_chain = ttk.Button(
            self,
            text="🔗 Chain as Input",
            command=self.on_chain_source,
        )
        self.btn_chain.pack(side="left", padx=2, pady=4)

        # Separator
        sep2 = ttk.Separator(self, orient="vertical")
        sep2.pack(side="left", fill="y", padx=6, pady=4)

        # View actions
        self.btn_fit = ttk.Button(
            self,
            text="⛶ Fit to View",
            command=self.on_fit_view,
        )
        self.btn_fit.pack(side="left", padx=2, pady=4)

        self.btn_1x = ttk.Button(
            self,
            text="1:1 Actual Size",
            command=self.on_actual_size,
        )
        self.btn_1x.pack(side="left", padx=2, pady=4)

        self.btn_toggle_view = ttk.Button(
            self,
            text="⇄ Toggle View",
            command=self.on_toggle_view,
        )
        self.btn_toggle_view.pack(side="left", padx=2, pady=4)

        # Separator
        sep3 = ttk.Separator(self, orient="vertical")
        sep3.pack(side="left", fill="y", padx=6, pady=4)

        # Data export
        self.btn_csv = ttk.Button(
            self,
            text="📊 Export CSV",
            command=self.on_export_csv,
        )
        self.btn_csv.pack(side="left", padx=2, pady=4)

    def set_has_image(self, has_image: bool) -> None:
        """Enable or disable buttons based on whether an image is active."""
        state = "normal" if has_image else "disabled"
        self.btn_save.config(state=state)
        self.btn_save_as.config(state=state)
        self.btn_reset.config(state=state)
        self.btn_chain.config(state=state)
        self.btn_fit.config(state=state)
        self.btn_1x.config(state=state)
        self.btn_toggle_view.config(state=state)
        self.btn_csv.config(state=state)
