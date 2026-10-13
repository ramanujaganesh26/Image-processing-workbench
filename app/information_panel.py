"""Information panel displaying image properties, interactive pixel inspector, and contour table."""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional, Tuple, Callable
from app.theme import Colors
from core.contour_analysis import ContourData


class InformationPanel(ttk.Frame):
    """Right-side or bottom inspector panel with metadata tabs and contour table."""

    def __init__(
        self,
        parent: tk.Widget,
        on_export_csv: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self.on_export_csv = on_export_csv

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Tab 1: Image Metadata & Pixel Inspector
        self.tab_meta = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.notebook.add(self.tab_meta, text="ℹ Image Properties & Inspector")
        self._build_meta_tab()

        # Tab 2: Contour Measurements
        self.tab_contours = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.notebook.add(self.tab_contours, text="📐 Contour Analysis (0)")
        self._build_contours_tab()

    def _build_meta_tab(self) -> None:
        self.tab_meta.columnconfigure(0, weight=1)

        # 1. Properties Section
        lbl_sec1 = ttk.Label(self.tab_meta, text="IMAGE METADATA", style="Header.TLabel")
        lbl_sec1.pack(anchor="w", padx=8, pady=(8, 4))

        self.prop_box = ttk.Frame(self.tab_meta, style="Card.TFrame", padding=6)
        self.prop_box.pack(fill="x", padx=8, pady=2)
        self.prop_box.columnconfigure(1, weight=1)

        self._meta_labels: Dict[str, ttk.Label] = {}
        fields = [
            ("Filename:", "filename", "No file loaded"),
            ("Dimensions:", "dimensions", "-- x --"),
            ("Channels:", "channels", "--"),
            ("Data Type:", "dtype", "--"),
            ("Array Shape:", "shape", "--"),
            ("Intensity Range:", "range", "Min: -- | Max: --"),
            ("Memory Size:", "size", "-- KB"),
            ("Active Operation:", "operation", "None"),
            ("Processing Time:", "time", "-- ms"),
        ]

        for r, (label_text, key, default_val) in enumerate(fields):
            lbl_title = ttk.Label(self.prop_box, text=label_text, style="Card.TLabel", font=("Segoe UI", 9, "bold"))
            lbl_title.grid(row=r, column=0, sticky="w", padx=4, pady=2)

            lbl_val = ttk.Label(self.prop_box, text=default_val, style="Card.TLabel")
            lbl_val.grid(row=r, column=1, sticky="w", padx=4, pady=2)
            self._meta_labels[key] = lbl_val

        ttk.Separator(self.tab_meta, orient="horizontal").pack(fill="x", padx=8, pady=8)

        # 2. Pixel Inspector Section
        lbl_sec2 = ttk.Label(self.tab_meta, text="PIXEL INSPECTOR", style="Header.TLabel")
        lbl_sec2.pack(anchor="w", padx=8, pady=(4, 2))

        ttk.Label(
            self.tab_meta,
            text="Click anywhere on the preview canvas to inspect exact coordinates and intensity values.",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.inspect_box = ttk.Frame(self.tab_meta, style="Card.TFrame", padding=6)
        self.inspect_box.pack(fill="x", padx=8, pady=2)
        self.inspect_box.columnconfigure(1, weight=1)

        ttk.Label(self.inspect_box, text="Coordinates:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.lbl_coord = ttk.Label(self.inspect_box, text="X: --, Y: --", style="Card.TLabel")
        self.lbl_coord.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(self.inspect_box, text="Original Pixel:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.lbl_orig_px = ttk.Label(self.inspect_box, text="--", style="Card.TLabel")
        self.lbl_orig_px.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(self.inspect_box, text="Processed Pixel:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.lbl_proc_px = ttk.Label(self.inspect_box, text="--", style="Card.TLabel")
        self.lbl_proc_px.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        # Color swatch preview canvas
        swatch_frame = ttk.Frame(self.inspect_box, style="Card.TFrame")
        swatch_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2))
        ttk.Label(swatch_frame, text="Color Swatch:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.canvas_swatch = tk.Canvas(
            swatch_frame,
            width=36,
            height=20,
            bg=Colors.BG_CARD,
            highlightthickness=1,
            highlightbackground=Colors.BORDER,
        )
        self.canvas_swatch.pack(side="left", padx=8)

    def _build_contours_tab(self) -> None:
        self.tab_contours.columnconfigure(0, weight=1)
        self.tab_contours.rowconfigure(1, weight=1)

        # Header bar with export button
        top_bar = ttk.Frame(self.tab_contours, style="Panel.TFrame")
        top_bar.grid(row=0, column=0, sticky="ew", padx=6, pady=4)

        self.lbl_cnt_count = ttk.Label(top_bar, text="0 contours detected", style="Header.TLabel")
        self.lbl_cnt_count.pack(side="left")

        if self.on_export_csv:
            btn_export = ttk.Button(
                top_bar,
                text="📊 Export CSV...",
                style="Accent.TButton",
                command=self.on_export_csv,
            )
            btn_export.pack(side="right")

        # Treeview Table
        tree_container = ttk.Frame(self.tab_contours, style="Panel.TFrame")
        tree_container.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)

        cols = ("idx", "area", "perim", "bbox", "centroid", "aspect", "extent")
        self.tree = ttk.Treeview(
            tree_container,
            columns=cols,
            show="headings",
            selectmode="browse",
        )

        self.tree.heading("idx", text="#")
        self.tree.heading("area", text="Area (px²)")
        self.tree.heading("perim", text="Perimeter")
        self.tree.heading("bbox", text="Bounding Box (X,Y,W,H)")
        self.tree.heading("centroid", text="Centroid (X,Y)")
        self.tree.heading("aspect", text="Aspect Ratio")
        self.tree.heading("extent", text="Extent")

        self.tree.column("idx", width=35, anchor="center")
        self.tree.column("area", width=75, anchor="e")
        self.tree.column("perim", width=70, anchor="e")
        self.tree.column("bbox", width=140, anchor="w")
        self.tree.column("centroid", width=95, anchor="center")
        self.tree.column("aspect", width=80, anchor="e")
        self.tree.column("extent", width=70, anchor="e")

        sb_y = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        sb_x = ttk.Scrollbar(tree_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")

    def update_metadata(self, meta: Dict[str, Any], operation: str = "None", time_ms: float = 0.0) -> None:
        """Update properties table with active image metrics."""
        if not meta.get("is_valid", False):
            for k in self._meta_labels:
                self._meta_labels[k].config(text="--")
            self._meta_labels["filename"].config(text="No file loaded")
            return

        self._meta_labels["filename"].config(text=str(meta.get("filename", "N/A")))
        w, h = meta.get("width", 0), meta.get("height", 0)
        self._meta_labels["dimensions"].config(text=f"{w} × {h} px")

        ch = meta.get("channels", 1)
        ch_str = "1 (Grayscale)" if ch == 1 else f"{ch} (Color BGR)"
        self._meta_labels["channels"].config(text=ch_str)

        self._meta_labels["dtype"].config(text=str(meta.get("dtype", "uint8")))
        self._meta_labels["shape"].config(text=str(meta.get("shape", "N/A")))

        min_v = meta.get("min_val", 0)
        max_v = meta.get("max_val", 255)
        self._meta_labels["range"].config(text=f"Min: {min_v}  |  Max: {max_v}")

        size_kb = meta.get("size_kb", 0.0)
        self._meta_labels["size"].config(text=f"{size_kb} KB ({meta.get('size_bytes', 0)} bytes)")

        self._meta_labels["operation"].config(text=str(operation))
        if time_ms > 0:
            self._meta_labels["time"].config(text=f"{time_ms:.2f} ms")
        else:
            self._meta_labels["time"].config(text="-- ms")

    def set_pixel_inspection(
        self,
        x: int,
        y: int,
        orig_val: str,
        proc_val: str,
        color_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Update pixel inspection labels and color swatch."""
        self.lbl_coord.config(text=f"X: {x}, Y: {y}")
        self.lbl_orig_px.config(text=orig_val)
        self.lbl_proc_px.config(text=proc_val)

        if color_rgb:
            r, g, b = color_rgb
            hex_col = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas_swatch.config(bg=hex_col)
        else:
            self.canvas_swatch.config(bg=Colors.BG_CARD)

    def update_contours(self, contours: List[ContourData]) -> None:
        """Populate the contour measurements table."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        count = len(contours)
        self.lbl_cnt_count.config(text=f"{count} contours detected")
        self.notebook.tab(self.tab_contours, text=f"📐 Contour Analysis ({count})")

        for c in contours:
            bbox_str = f"({c.bbox_x}, {c.bbox_y}, {c.bbox_w}, {c.bbox_h})"
            centroid_str = f"({c.centroid_x}, {c.centroid_y})"
            self.tree.insert(
                "",
                "end",
                values=(
                    c.index,
                    f"{c.area:.1f}",
                    f"{c.perimeter:.1f}",
                    bbox_str,
                    centroid_str,
                    f"{c.aspect_ratio:.2f}",
                    f"{c.extent:.3f}",
                ),
            )
