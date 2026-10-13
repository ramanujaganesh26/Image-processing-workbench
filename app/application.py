"""Application initialization and lifecycle management."""

import tkinter as tk
import sys
from app.theme import apply_theme
from app.main_window import MainWindow


class Application:
    """Entry application controller initializing Tkinter, theme, and main window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self._configure_root()
        apply_theme(self.root)
        self.main_window = MainWindow(self.root)

    def _configure_root(self) -> None:
        """Configure DPI awareness and window defaults on Windows."""
        # Enable high-DPI awareness on Windows where available
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

        self.root.title("Image Processing Workbench")

    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()
