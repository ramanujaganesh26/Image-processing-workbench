"""Control panel for selecting algorithms, adjusting parameters, and displaying CV concepts."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any, Optional
from app.theme import Colors


ALGORITHM_GUIDES: Dict[str, str] = {
    "Grayscale": (
        "• Formulation: Y = 0.299*R + 0.587*G + 0.114*B\n"
        "• Purpose: Reduces 3-channel color data to 1-channel luminance, drastically reducing "
        "computational load for edge detection, thresholding, and feature tracking."
    ),
    "Resize": (
        "• Spatial Scaling: Adjusts image dimensions using bilinear or area interpolation.\n"
        "• Robotics Note: Often used to downsample camera feeds before running CPU-intensive "
        "algorithms like optical flow or SLAM."
    ),
    "Flip": (
        "• Geometric Reflection: Reverses pixels across horizontal (X-axis) or vertical (Y-axis).\n"
        "• Robotics Note: Used for camera sensor inversion correction and data augmentation."
    ),
    "Rotate": (
        "• Orthogonal Rotation: Rotates 90°, 180°, or 270° without interpolation artifacts.\n"
        "• Purpose: Corrects mounting orientations for robotic vision sensors."
    ),
    "Gaussian Blur": (
        "• Formulation: 2D Gaussian convolution kernel: G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)\n"
        "• Purpose: Attenuates high-frequency Gaussian noise while preserving edges better than box blur.\n"
        "• Requirement: Kernel dimensions must be positive odd integers."
    ),
    "Median Blur": (
        "• Non-linear Filter: Replaces each center pixel with the median of its neighborhood.\n"
        "• Purpose: Highly effective at removing 'salt-and-pepper' impulse noise without blurring edges.\n"
        "• Robotics Note: Ideal for laser scanner intensity noise and camera sensor speckle."
    ),
    "Binary Threshold": (
        "• Formulation: dst(x,y) = max_val if src(x,y) > T else 0\n"
        "• Purpose: Segment foreground objects against high-contrast backgrounds."
    ),
    "Binary Inverse": (
        "• Formulation: dst(x,y) = 0 if src(x,y) > T else max_val\n"
        "• Purpose: Inverts segmentation for light objects on dark backgrounds."
    ),
    "Otsu Threshold": (
        "• Automatic Thresholding: Selects threshold T minimizing intra-class variance: σ²w(t) = w0(t)σ²0(t) + w1(t)σ²1(t)\n"
        "• Note: Optimal for bimodal histograms; threshold is calculated automatically from image statistics."
    ),
    "Otsu Inverse": (
        "• Inverted Otsu: Calculates optimal histogram separation and produces inverted binary output."
    ),
    "Adaptive Mean": (
        "• Formulation: T(x,y) = Mean of blockSize neighborhood minus Constant C.\n"
        "• Purpose: Handles non-uniform lighting and shadows across the scene where global thresholding fails."
    ),
    "Adaptive Gaussian": (
        "• Formulation: T(x,y) = Gaussian weighted sum of neighborhood minus Constant C.\n"
        "• Advantage: Yields smoother, less noisy segmentation boundaries under variable ambient lighting."
    ),
    "Canny Edge": (
        "• Multi-Stage Detection: 1. Gaussian smooth → 2. Sobel gradient magnitude/direction → "
        "3. Non-maximum suppression → 4. Hysteresis thresholding.\n"
        "• Thresholds: Pixels > upper are strong edges; pixels < lower are rejected; intermediate pixels "
        "are kept only if connected to strong edges."
    ),
    "Contour Detection": (
        "• Topological Extraction: Uses Suzuki & Abe algorithm to detect object contours and hierarchies.\n"
        "• Geometric Features: Calculates Area, Perimeter (arc length), Bounding Box, Centroid moments "
        "(m10/m00, m01/m00), Aspect Ratio, and Extent."
    ),
    "Erosion": (
        "• Morphological Shrink: Replaces pixel with local minimum in structuring element.\n"
        "• Effect: Erodes boundaries, eliminates small bright speckles, separates connected objects."
    ),
    "Dilation": (
        "• Morphological Expansion: Replaces pixel with local maximum in structuring element.\n"
        "• Effect: Expands foreground boundaries, bridges small gaps and fractures in objects."
    ),
    "Opening": (
        "• Formula: Erode followed by Dilate.\n"
        "• Effect: Removes small bright noise artifacts while preserving overall object geometry."
    ),
    "Closing": (
        "• Formula: Dilate followed by Erode.\n"
        "• Effect: Fills small dark holes and cracks inside foreground objects without altering shape."
    ),
    "Gradient": (
        "• Formula: Dilation minus Erosion.\n"
        "• Effect: Extracts structural boundaries and outlines of foreground objects."
    ),
    "Top-hat": (
        "• Formula: Image minus Opening.\n"
        "• Effect: Isolates bright elements smaller than the structuring element on uneven backgrounds."
    ),
    "Black-hat": (
        "• Formula: Closing minus Image.\n"
        "• Effect: Isolates dark elements and indentations on uneven backgrounds."
    ),
}


CATEGORIES: Dict[str, list] = {
    "1. Basic Transformations": ["Grayscale", "Resize", "Flip", "Rotate"],
    "2. Filtering": ["Gaussian Blur", "Median Blur"],
    "3. Thresholding": [
        "Binary Threshold",
        "Binary Inverse",
        "Otsu Threshold",
        "Otsu Inverse",
        "Adaptive Mean",
        "Adaptive Gaussian",
    ],
    "4. Edge Detection": ["Canny Edge"],
    "5. Contour Analysis": ["Contour Detection"],
    "6. Morphological Operations": [
        "Erosion",
        "Dilation",
        "Opening",
        "Closing",
        "Gradient",
        "Top-hat",
        "Black-hat",
    ],
}


class ControlsPanel(ttk.Frame):
    """Left control panel organizing algorithms and reactive parameters."""

    def __init__(
        self,
        parent: tk.Widget,
        on_apply: Callable[[str, str, Dict[str, Any]], None],
    ) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self.on_apply = on_apply

        self._current_category = list(CATEGORIES.keys())[0]
        self._current_operation = CATEGORIES[self._current_category][0]
        self._param_vars: Dict[str, Any] = {}
        self._auto_apply = tk.BooleanVar(value=True)
        self._debounce_timer: Optional[str] = None
        self._active_image_shape: Optional[tuple] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        # 1. Header
        header_frame = ttk.Frame(self, style="Panel.TFrame")
        header_frame.pack(fill="x", padx=10, pady=(8, 4))
        ttk.Label(header_frame, text="Processing Controls", style="Header.TLabel").pack(side="left")

        # 2. Category Selector
        cat_box = ttk.Frame(self, style="Panel.TFrame")
        cat_box.pack(fill="x", padx=10, pady=4)
        ttk.Label(cat_box, text="Category:", style="Panel.TLabel").pack(anchor="w")

        self.cb_category = ttk.Combobox(
            cat_box,
            values=list(CATEGORIES.keys()),
            state="readonly",
        )
        self.cb_category.set(self._current_category)
        self.cb_category.pack(fill="x", pady=2)
        self.cb_category.bind("<<ComboboxSelected>>", self._on_category_changed)

        # 3. Operation Selector
        op_box = ttk.Frame(self, style="Panel.TFrame")
        op_box.pack(fill="x", padx=10, pady=4)
        ttk.Label(op_box, text="Algorithm / Operation:", style="Panel.TLabel").pack(anchor="w")

        self.cb_operation = ttk.Combobox(
            op_box,
            values=CATEGORIES[self._current_category],
            state="readonly",
        )
        self.cb_operation.set(self._current_operation)
        self.cb_operation.pack(fill="x", pady=2)
        self.cb_operation.bind("<<ComboboxSelected>>", self._on_operation_changed)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=6)

        # 4. Parameters Container (rebuilt dynamically)
        self.param_frame = ttk.Frame(self, style="Panel.TFrame")
        self.param_frame.pack(fill="x", padx=10, pady=2)

        # 5. Action Buttons & Auto-apply Checkbox
        btn_box = ttk.Frame(self, style="Panel.TFrame")
        btn_box.pack(fill="x", padx=10, pady=8)

        self.chk_auto = ttk.Checkbutton(
            btn_box,
            text="Interactive Auto-Apply",
            variable=self._auto_apply,
        )
        self.chk_auto.pack(anchor="w", pady=(0, 6))

        self.btn_apply = ttk.Button(
            btn_box,
            text="Apply Operation",
            style="Accent.TButton",
            command=self.trigger_apply,
        )
        self.btn_apply.pack(fill="x")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # 6. Theory & Computer Vision Notes
        guide_box = ttk.Frame(self, style="Panel.TFrame")
        guide_box.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        ttk.Label(guide_box, text="Algorithm Guide & Principles", style="Header.TLabel").pack(anchor="w", pady=(0, 2))

        self.txt_guide = tk.Text(
            guide_box,
            wrap="word",
            bg=Colors.BG_CARD,
            fg=Colors.TEXT_MAIN,
            insertbackground=Colors.TEXT_MAIN,
            highlightthickness=1,
            highlightbackground=Colors.BORDER,
            font=("Segoe UI", 8),
            height=8,
            relief="flat",
        )
        self.txt_guide.pack(fill="both", expand=True, pady=2)

        # Initial build
        self._rebuild_parameters()
        self._update_guide()

    def set_active_image_shape(self, shape: tuple) -> None:
        """Store shape of loaded image to initialize sensible resize dimensions."""
        self._active_image_shape = shape
        if self._current_operation == "Resize":
            self._rebuild_parameters()

    def _on_category_changed(self, event=None) -> None:
        cat = self.cb_category.get()
        self._current_category = cat
        ops = CATEGORIES[cat]
        self.cb_operation.config(values=ops)
        self.cb_operation.set(ops[0])
        self._current_operation = ops[0]
        self._rebuild_parameters()
        self._update_guide()
        if self._auto_apply.get():
            self._schedule_auto_apply()

    def _on_operation_changed(self, event=None) -> None:
        self._current_operation = self.cb_operation.get()
        self._rebuild_parameters()
        self._update_guide()
        if self._auto_apply.get():
            self._schedule_auto_apply()

    def _update_guide(self) -> None:
        guide_text = ALGORITHM_GUIDES.get(
            self._current_operation,
            f"Operation: {self._current_operation}\nCategory: {self._current_category}"
        )
        self.txt_guide.config(state="normal")
        self.txt_guide.delete("1.0", "end")
        self.txt_guide.insert("1.0", guide_text)
        self.txt_guide.config(state="disabled")

    def _schedule_auto_apply(self) -> None:
        """Debounce auto-apply execution by 100ms."""
        if self._debounce_timer is not None:
            self.after_cancel(self._debounce_timer)
        self._debounce_timer = self.after(100, self.trigger_apply)

    def trigger_apply(self) -> None:
        """Collect current parameter values and invoke callback."""
        params = self.get_parameters()
        self.on_apply(self._current_category, self._current_operation, params)

    def get_parameters(self) -> Dict[str, Any]:
        """Convert UI variables into typed dictionary."""
        params: Dict[str, Any] = {}
        for k, var in self._param_vars.items():
            params[k] = var.get()
        return params

    def _rebuild_parameters(self) -> None:
        """Clear and create parameter widgets for the selected operation."""
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self._param_vars.clear()

        op = self._current_operation

        # --- Basic Transformations ---
        if op == "Grayscale":
            lbl = ttk.Label(self.param_frame, text="No parameters required.\nTransforms color BGR array to 1-channel luminance.")
            lbl.pack(anchor="w", pady=4)

        elif op == "Resize":
            default_w = 640
            default_h = 480
            if self._active_image_shape:
                default_h, default_w = self._active_image_shape[:2]

            self._create_spinbox("width", "Target Width (px):", default_w, 10, 8000, 10)
            self._create_spinbox("height", "Target Height (px):", default_h, 10, 8000, 10)
            self._create_checkbox("preserve_aspect", "Preserve Aspect Ratio", True)

        elif op == "Flip":
            self._create_combobox("direction", "Flip Direction:", ["horizontal", "vertical", "both"], "horizontal")

        elif op == "Rotate":
            self._create_combobox("angle", "Clockwise Rotation Angle:", [90, 180, 270], 90)

        # --- Filtering ---
        elif op == "Gaussian Blur":
            self._create_odd_slider("kernel_size", "Kernel Size (Odd):", 5, 1, 31)
            self._create_float_slider("sigma_x", "Sigma X (0 = Auto):", 0.0, 0.0, 10.0, 0.1)

        elif op == "Median Blur":
            self._create_odd_slider("kernel_size", "Kernel Size (Odd):", 5, 1, 31)

        # --- Thresholding ---
        elif op in ("Binary Threshold", "Binary Inverse"):
            self._create_slider("thresh", "Threshold (0-255):", 127, 0, 255)
            self._create_slider("max_val", "Max Value (0-255):", 255, 0, 255)

        elif op in ("Otsu Threshold", "Otsu Inverse"):
            lbl = ttk.Label(self.param_frame, text="Threshold calculated automatically via Otsu intra-class variance.", style="Muted.TLabel")
            lbl.pack(anchor="w", pady=(0, 4))
            self._create_slider("max_val", "Max Value (0-255):", 255, 0, 255)

        elif op in ("Adaptive Mean", "Adaptive Gaussian"):
            self._create_odd_slider("block_size", "Block Size (Odd > 1):", 11, 3, 51)
            self._create_slider("c", "Constant C:", 2, -20, 40)
            self._create_slider("max_val", "Max Value (0-255):", 255, 0, 255)

        # --- Edge Detection ---
        elif op == "Canny Edge":
            self._create_slider("lower_thresh", "Lower Hysteresis Threshold:", 100, 0, 255)
            self._create_slider("upper_thresh", "Upper Hysteresis Threshold:", 200, 0, 255)
            self._create_combobox("pre_blur_ksize", "Gaussian Pre-Filter Blur:", [0, 3, 5, 7], 0)

        # --- Contour Analysis ---
        elif op == "Contour Detection":
            self._create_combobox("retrieval_mode", "Retrieval Mode:", ["external", "all", "tree"], "external")
            self._create_combobox("approx_mode", "Approximation Mode:", ["simple", "none"], "simple")
            self._create_slider("min_area", "Min Contour Area Filter (px²):", 30, 0, 5000)
            self._create_checkbox("draw_contours", "Draw Contours (Green)", True)
            self._create_checkbox("draw_boxes", "Draw Bounding Boxes (Cyan)", True)
            self._create_checkbox("draw_centroids", "Draw Centroids & Indices (Red)", True)

        # --- Morphological Operations ---
        elif op in ("Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top-hat", "Black-hat"):
            self._create_slider("kernel_width", "Kernel Width:", 5, 1, 25)
            self._create_slider("kernel_height", "Kernel Height:", 5, 1, 25)
            self._create_combobox("kernel_shape", "Kernel Shape:", ["rect", "ellipse", "cross"], "rect")
            self._create_slider("iterations", "Iterations:", 1, 1, 10)

    # Widget creation helpers
    def _create_slider(self, name: str, label_text: str, default: int, min_val: int, max_val: int) -> None:
        box = ttk.Frame(self.param_frame, style="Panel.TFrame")
        box.pack(fill="x", pady=2)

        var = tk.IntVar(value=default)
        self._param_vars[name] = var

        val_lbl = ttk.Label(box, text=str(default), style="Badge.TLabel")
        val_lbl.pack(side="right")
        ttk.Label(box, text=label_text, style="Panel.TLabel").pack(side="left")

        def _on_move(val_str):
            v = int(float(val_str))
            var.set(v)
            val_lbl.config(text=str(v))
            if self._auto_apply.get():
                self._schedule_auto_apply()

        scale = ttk.Scale(
            self.param_frame,
            from_=min_val,
            to=max_val,
            value=default,
            command=_on_move,
        )
        scale.pack(fill="x", pady=(0, 4))

    def _create_odd_slider(self, name: str, label_text: str, default: int, min_val: int, max_val: int) -> None:
        box = ttk.Frame(self.param_frame, style="Panel.TFrame")
        box.pack(fill="x", pady=2)

        var = tk.IntVar(value=default)
        self._param_vars[name] = var

        val_lbl = ttk.Label(box, text=str(default), style="Badge.TLabel")
        val_lbl.pack(side="right")
        ttk.Label(box, text=label_text, style="Panel.TLabel").pack(side="left")

        def _on_move(val_str):
            v = int(float(val_str))
            if v % 2 == 0:
                v += 1
            var.set(v)
            val_lbl.config(text=str(v))
            if self._auto_apply.get():
                self._schedule_auto_apply()

        scale = ttk.Scale(
            self.param_frame,
            from_=min_val,
            to=max_val,
            value=default,
            command=_on_move,
        )
        scale.pack(fill="x", pady=(0, 4))

    def _create_float_slider(self, name: str, label_text: str, default: float, min_val: float, max_val: float, step: float) -> None:
        box = ttk.Frame(self.param_frame, style="Panel.TFrame")
        box.pack(fill="x", pady=2)

        var = tk.DoubleVar(value=default)
        self._param_vars[name] = var

        val_lbl = ttk.Label(box, text=f"{default:.1f}", style="Badge.TLabel")
        val_lbl.pack(side="right")
        ttk.Label(box, text=label_text, style="Panel.TLabel").pack(side="left")

        def _on_move(val_str):
            v = round(float(val_str), 1)
            var.set(v)
            val_lbl.config(text=f"{v:.1f}")
            if self._auto_apply.get():
                self._schedule_auto_apply()

        scale = ttk.Scale(
            self.param_frame,
            from_=min_val,
            to=max_val,
            value=default,
            command=_on_move,
        )
        scale.pack(fill="x", pady=(0, 4))

    def _create_combobox(self, name: str, label_text: str, values: list, default: Any) -> None:
        box = ttk.Frame(self.param_frame, style="Panel.TFrame")
        box.pack(fill="x", pady=2)

        ttk.Label(box, text=label_text, style="Panel.TLabel").pack(anchor="w")

        var = tk.StringVar(value=str(default))
        self._param_vars[name] = var

        cb = ttk.Combobox(box, values=values, textvariable=var, state="readonly")
        cb.pack(fill="x", pady=2)

        def _on_change(e):
            if self._auto_apply.get():
                self._schedule_auto_apply()

        cb.bind("<<ComboboxSelected>>", _on_change)

    def _create_spinbox(self, name: str, label_text: str, default: int, from_: int, to_: int, step: int) -> None:
        box = ttk.Frame(self.param_frame, style="Panel.TFrame")
        box.pack(fill="x", pady=2)

        ttk.Label(box, text=label_text, style="Panel.TLabel").pack(anchor="w")

        var = tk.IntVar(value=default)
        self._param_vars[name] = var

        sb = ttk.Spinbox(box, from_=from_, to=to_, increment=step, textvariable=var)
        sb.pack(fill="x", pady=2)

        def _on_change(*args):
            if self._auto_apply.get():
                self._schedule_auto_apply()

        var.trace_add("write", _on_change)

    def _create_checkbox(self, name: str, label_text: str, default: bool) -> None:
        var = tk.BooleanVar(value=default)
        self._param_vars[name] = var

        def _on_toggle():
            if self._auto_apply.get():
                self._schedule_auto_apply()

        chk = ttk.Checkbutton(self.param_frame, text=label_text, variable=var, command=_on_toggle)
        chk.pack(anchor="w", pady=3)
