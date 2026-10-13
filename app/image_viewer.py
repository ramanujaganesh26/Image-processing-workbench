"""Dual-panel image viewer with aspect-ratio scaling, view modes, and pixel inspection."""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Tuple, Any
import cv2
import numpy as np
from PIL import Image, ImageTk

from app.theme import Colors


class ImageCanvas(tk.Canvas):
    """Specialized Canvas for rendering an image with aspect-ratio preservation and coordinate mapping."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        on_pixel_click: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        super().__init__(
            parent,
            bg=Colors.BG_CANVAS,
            highlightthickness=1,
            highlightbackground=Colors.BORDER,
            cursor="crosshair",
        )
        self.title = title
        self.on_pixel_click = on_pixel_click
        self.current_image: Optional[np.ndarray] = None
        self.view_mode: str = "fit"  # "fit" or "actual"

        # Display transformation cache: (scale, offset_x, offset_y, disp_w, disp_h)
        self._transform: Tuple[float, int, int, int, int] = (1.0, 0, 0, 0, 0)
        self._photo_ref: Optional[ImageTk.PhotoImage] = None
        self._resize_timer: Optional[str] = None

        self.bind("<Configure>", self._on_configure)
        self.bind("<Button-1>", self._on_click)

    def set_image(self, image: Optional[np.ndarray]) -> None:
        """Update active image array."""
        self.current_image = image
        self.render()

    def set_view_mode(self, mode: str) -> None:
        """Set view mode to 'fit' or 'actual'."""
        self.view_mode = mode
        self.render()

    def _on_configure(self, event: tk.Event) -> None:
        """Debounce resize redraws to maintain responsive UI."""
        if self._resize_timer is not None:
            self.after_cancel(self._resize_timer)
        self._resize_timer = self.after(50, self.render)

    def render(self) -> None:
        """Render the cached image centered with aspect-ratio preservation."""
        self.delete("all")

        cw = self.winfo_width()
        ch = self.winfo_height()

        if cw <= 10 or ch <= 10:
            return

        if self.current_image is None or self.current_image.size == 0:
            self._render_empty_state(cw, ch)
            return

        img_h, img_w = self.current_image.shape[:2]

        if self.view_mode == "fit":
            scale = min((cw - 16) / img_w, (ch - 16) / img_h)
            disp_w = max(1, int(round(img_w * scale)))
            disp_h = max(1, int(round(img_h * scale)))
        else:  # "actual" (1:1)
            scale = 1.0
            disp_w = img_w
            disp_h = img_h

        offset_x = (cw - disp_w) // 2
        offset_y = (ch - disp_h) // 2
        self._transform = (scale, offset_x, offset_y, disp_w, disp_h)

        # Convert OpenCV image (BGR or Gray) to RGB PIL Image
        if self.current_image.ndim == 2:
            pil_img = Image.fromarray(self.current_image, mode="L")
        elif self.current_image.shape[2] == 3:
            rgb_arr = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr, mode="RGB")
        elif self.current_image.shape[2] == 4:
            rgb_arr = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2RGBA)
            pil_img = Image.fromarray(rgb_arr, mode="RGBA")
        else:
            pil_img = Image.fromarray(self.current_image[:, :, 0], mode="L")

        # Resize PIL image for display
        if (disp_w, disp_h) != (img_w, img_h):
            pil_img = pil_img.resize((disp_w, disp_h), Image.Resampling.NEAREST if scale > 3 else Image.Resampling.BILINEAR)

        self._photo_ref = ImageTk.PhotoImage(pil_img)
        self.create_image(offset_x, offset_y, anchor="nw", image=self._photo_ref)

        # Draw subtle border around displayed image
        self.create_rectangle(
            offset_x,
            offset_y,
            offset_x + disp_w,
            offset_y + disp_h,
            outline=Colors.BORDER_LIGHT,
            width=1,
        )

    def _render_empty_state(self, cw: int, ch: int) -> None:
        """Render stylish engineering empty state placeholder."""
        mid_x = cw // 2
        mid_y = ch // 2

        self.create_rectangle(
            mid_x - 120, mid_y - 45, mid_x + 120, mid_y + 45,
            fill=Colors.BG_CARD,
            outline=Colors.BORDER,
            width=1,
        )
        self.create_text(
            mid_x,
            mid_y - 12,
            text=f"[{self.title.upper()}]",
            fill=Colors.ACCENT_BLUE,
            font=("Segoe UI", 9, "bold"),
        )
        self.create_text(
            mid_x,
            mid_y + 14,
            text="No image loaded\n(Open with Ctrl+O)",
            fill=Colors.TEXT_MUTED,
            font=("Segoe UI", 8),
            justify="center",
        )

    def _on_click(self, event: tk.Event) -> None:
        """Map canvas click event back to exact image array coordinates."""
        if self.current_image is None or not self.on_pixel_click:
            return

        scale, offset_x, offset_y, disp_w, disp_h = self._transform
        img_h, img_w = self.current_image.shape[:2]

        cx = event.x
        cy = event.y

        # Check if click is inside displayed image bounding box
        if offset_x <= cx < offset_x + disp_w and offset_y <= cy < offset_y + disp_h:
            ix = int((cx - offset_x) / scale)
            iy = int((cy - offset_y) / scale)
            ix = max(0, min(img_w - 1, ix))
            iy = max(0, min(img_h - 1, iy))
            self.on_pixel_click(ix, iy)


class WorkspaceImageViewer(ttk.Frame):
    """Central dual comparison viewer coordinating original and processed panels."""

    def __init__(
        self,
        parent: tk.Widget,
        on_pixel_inspected: Optional[Callable[[int, int, str, str, Optional[Tuple[int, int, int]]], None]] = None,
    ) -> None:
        super().__init__(parent, style="TFrame")
        self.on_pixel_inspected = on_pixel_inspected

        self._orig_img: Optional[np.ndarray] = None
        self._proc_img: Optional[np.ndarray] = None
        self._view_layout: str = "side_by_side"  # "side_by_side", "processed_only", "original_only"
        self._scale_mode: str = "fit"

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.paned = ttk.Panedwindow(self, orient="horizontal")
        self.paned.grid(row=0, column=0, sticky="nsew")

        # Left Panel (Original)
        self.frame_left = ttk.Frame(self.paned, style="Panel.TFrame")
        self.frame_left.columnconfigure(0, weight=1)
        self.frame_left.rowconfigure(1, weight=1)

        # Header Left
        hdr_left = ttk.Frame(self.frame_left, style="Panel.TFrame")
        hdr_left.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        ttk.Label(hdr_left, text="Original Image", style="Header.TLabel").pack(side="left")
        self.lbl_orig_info = ttk.Label(hdr_left, text="-- x -- | --", style="Badge.TLabel")
        self.lbl_orig_info.pack(side="right")

        self.canvas_orig = ImageCanvas(
            self.frame_left,
            title="Original Image",
            on_pixel_click=self._on_pixel_click,
        )
        self.canvas_orig.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Right Panel (Processed)
        self.frame_right = ttk.Frame(self.paned, style="Panel.TFrame")
        self.frame_right.columnconfigure(0, weight=1)
        self.frame_right.rowconfigure(1, weight=1)

        # Header Right
        hdr_right = ttk.Frame(self.frame_right, style="Panel.TFrame")
        hdr_right.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        ttk.Label(hdr_right, text="Processed Result", style="Header.TLabel").pack(side="left")
        self.lbl_proc_op = ttk.Label(hdr_right, text="Original Image", style="SuccessBadge.TLabel")
        self.lbl_proc_op.pack(side="right")
        self.lbl_proc_info = ttk.Label(hdr_right, text="-- x --", style="Badge.TLabel")
        self.lbl_proc_info.pack(side="right", padx=4)

        self.canvas_proc = ImageCanvas(
            self.frame_right,
            title="Processed Result",
            on_pixel_click=self._on_pixel_click,
        )
        self.canvas_proc.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        self.paned.add(self.frame_left, weight=1)
        self.paned.add(self.frame_right, weight=1)

    def set_images(
        self,
        original: Optional[np.ndarray],
        processed: Optional[np.ndarray],
        operation_name: str = "Original Image",
    ) -> None:
        """Update both original and processed image panels."""
        self._orig_img = original
        self._proc_img = processed

        if original is not None:
            oh, ow = original.shape[:2]
            oc = original.shape[2] if original.ndim == 3 else 1
            ch_desc = "Grayscale" if oc == 1 else f"{oc}-Ch BGR"
            self.lbl_orig_info.config(text=f"{ow}x{oh} | {ch_desc}")
        else:
            self.lbl_orig_info.config(text="-- x -- | --")

        if processed is not None:
            ph, pw = processed.shape[:2]
            self.lbl_proc_info.config(text=f"{pw}x{ph}")
            self.lbl_proc_op.config(text=operation_name)
        else:
            self.lbl_proc_info.config(text="-- x --")
            self.lbl_proc_op.config(text="Original Image")

        self.canvas_orig.set_image(original)
        self.canvas_proc.set_image(processed)

    def set_scale_mode(self, mode: str) -> None:
        """Switch between 'fit' and 'actual' (1:1) scale."""
        self._scale_mode = mode
        self.canvas_orig.set_view_mode(mode)
        self.canvas_proc.set_view_mode(mode)

    def cycle_view_mode(self) -> str:
        """Toggle between side_by_side, processed_only, and original_only."""
        if self._view_layout == "side_by_side":
            self._view_layout = "processed_only"
            self.paned.forget(self.frame_left)
        elif self._view_layout == "processed_only":
            self._view_layout = "original_only"
            self.paned.forget(self.frame_right)
            self.paned.add(self.frame_left, weight=1)
        else:
            self._view_layout = "side_by_side"
            self.paned.add(self.frame_right, weight=1)

        return self._view_layout

    def _on_pixel_click(self, x: int, y: int) -> None:
        """Inspect pixel values in original and processed images at (x, y)."""
        if self._orig_img is None:
            return

        # Format original pixel string
        orig_val = self._orig_img[y, x]
        color_rgb: Optional[Tuple[int, int, int]] = None

        if self._orig_img.ndim == 2:
            orig_str = f"Val: {int(orig_val)}"
            color_rgb = (int(orig_val), int(orig_val), int(orig_val))
        elif self._orig_img.ndim == 3 and self._orig_img.shape[2] >= 3:
            b, g, r = int(orig_val[0]), int(orig_val[1]), int(orig_val[2])
            orig_str = f"R:{r} G:{g} B:{b} (Hex: #{r:02x}{g:02x}{b:02x})"
            color_rgb = (r, g, b)
        else:
            orig_str = str(orig_val)

        # Format processed pixel string
        proc_str = "N/A"
        if self._proc_img is not None:
            ph, pw = self._proc_img.shape[:2]
            if 0 <= x < pw and 0 <= y < ph:
                proc_val = self._proc_img[y, x]
                if self._proc_img.ndim == 2:
                    proc_str = f"Val: {int(proc_val)}"
                elif self._proc_img.ndim == 3 and self._proc_img.shape[2] >= 3:
                    pb, pg, pr = int(proc_val[0]), int(proc_val[1]), int(proc_val[2])
                    proc_str = f"R:{pr} G:{pg} B:{pb} (#{pr:02x}{pg:02x}{pb:02x})"
                else:
                    proc_str = str(proc_val)

        if self.on_pixel_inspected:
            self.on_pixel_inspected(x, y, orig_str, proc_str, color_rgb)
