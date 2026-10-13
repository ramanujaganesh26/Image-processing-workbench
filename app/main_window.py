"""Main window coordinating toolbar, image viewers, controls, inspector, and status bar."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple

from app.theme import Colors
from app.toolbar import Toolbar
from app.image_viewer import WorkspaceImageViewer
from app.controls_panel import ControlsPanel
from app.information_panel import InformationPanel
from app.status_bar import StatusBar

from core.processing_pipeline import ProcessingPipeline
from core.image_io import save_image, ImageIOError
from core.contour_analysis import export_contours_to_csv


class MainWindow:
    """Main application window managing UI layout, event coordination, and pipeline integration."""

    APP_TITLE = "Image Processing Workbench"
    APP_VERSION = "v1.0.0"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{self.APP_TITLE} - {self.APP_VERSION}")
        self.root.minsize(1050, 680)
        self.root.geometry("1280x800")

        self.pipeline = ProcessingPipeline()

        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()

        # Update initial toolbar states
        self.toolbar.set_has_image(False)
        self.status_bar.set_status("Ready. Click 'Open Image' or press Ctrl+O to begin.")

    def _build_menu(self) -> None:
        """Construct native application menu bar."""
        self.menubar = tk.Menu(self.root, bg=Colors.BG_CARD, fg=Colors.TEXT_MAIN, bd=0)

        # File Menu
        file_menu = tk.Menu(self.menubar, tearoff=0, bg=Colors.BG_CARD, fg=Colors.TEXT_MAIN)
        file_menu.add_command(label="Open Image... (Ctrl+O)", command=self.action_open_image)
        file_menu.add_command(label="Save Processed Image (Ctrl+S)", command=self.action_save_image)
        file_menu.add_command(label="Save Processed As... (Ctrl+Shift+S)", command=self.action_save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Contours to CSV...", command=self.action_export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit (Ctrl+Q)", command=self.action_exit)
        self.menubar.add_cascade(label="File", menu=file_menu)

        # Process Menu
        proc_menu = tk.Menu(self.menubar, tearoff=0, bg=Colors.BG_CARD, fg=Colors.TEXT_MAIN)
        proc_menu.add_command(label="Reset Processing (Ctrl+R)", command=self.action_reset)
        proc_menu.add_command(label="Chain as Base Input", command=self.action_chain_source)
        proc_menu.add_command(label="Re-apply Current Operation", command=self.action_reapply)
        self.menubar.add_cascade(label="Process", menu=proc_menu)

        # View Menu
        view_menu = tk.Menu(self.menubar, tearoff=0, bg=Colors.BG_CARD, fg=Colors.TEXT_MAIN)
        view_menu.add_command(label="Fit to View", command=self.action_fit_view)
        view_menu.add_command(label="1:1 Actual Size", command=self.action_actual_size)
        view_menu.add_command(label="Toggle View Layout (Side-by-Side / Single)", command=self.action_toggle_view)
        self.menubar.add_cascade(label="View", menu=view_menu)

        # Help Menu
        help_menu = tk.Menu(self.menubar, tearoff=0, bg=Colors.BG_CARD, fg=Colors.TEXT_MAIN)
        help_menu.add_command(label="Load Sample Image...", command=self.action_open_sample_image)
        help_menu.add_command(label="Algorithm Documentation", command=self.action_show_algorithm_docs)
        help_menu.add_separator()
        help_menu.add_command(label="About Image Processing Workbench", command=self.action_show_about)
        self.menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=self.menubar)

    def _build_layout(self) -> None:
        """Construct multi-panel workbench layout with resizable horizontal panes."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # 1. Top Toolbar
        self.toolbar = Toolbar(
            self.root,
            on_open=self.action_open_image,
            on_save=self.action_save_image,
            on_save_as=self.action_save_image_as,
            on_reset=self.action_reset,
            on_fit_view=self.action_fit_view,
            on_actual_size=self.action_actual_size,
            on_toggle_view=self.action_toggle_view,
            on_chain_source=self.action_chain_source,
            on_export_csv=self.action_export_csv,
        )
        self.toolbar.grid(row=0, column=0, sticky="ew")

        # 2. Main Central Horizontal Paned Window
        self.main_paned = ttk.Panedwindow(self.root, orient="horizontal")
        self.main_paned.grid(row=1, column=0, sticky="nsew")

        # Left: Controls Panel (width ~ 280)
        self.controls_panel = ControlsPanel(
            self.main_paned,
            on_apply=self.on_apply_operation,
        )
        self.main_paned.add(self.controls_panel, weight=0)

        # Center: Image Comparison Workspace (expands)
        self.viewer = WorkspaceImageViewer(
            self.main_paned,
            on_pixel_inspected=self.on_pixel_inspected,
        )
        self.main_paned.add(self.viewer, weight=3)

        # Right: Information & Contour Table Panel (width ~ 320)
        self.info_panel = InformationPanel(
            self.main_paned,
            on_export_csv=self.action_export_csv,
        )
        self.main_paned.add(self.info_panel, weight=1)

        # 3. Bottom Status Bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=2, column=0, sticky="ew")

    def _bind_shortcuts(self) -> None:
        """Bind standard desktop keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self.action_open_image())
        self.root.bind("<Control-O>", lambda e: self.action_open_image())
        self.root.bind("<Control-s>", lambda e: self.action_save_image())
        self.root.bind("<Control-S>", lambda e: self.action_save_image())
        self.root.bind("<Control-Shift-s>", lambda e: self.action_save_image_as())
        self.root.bind("<Control-Shift-S>", lambda e: self.action_save_image_as())
        self.root.bind("<Control-r>", lambda e: self.action_reset())
        self.root.bind("<Control-R>", lambda e: self.action_reset())
        self.root.bind("<Control-q>", lambda e: self.action_exit())
        self.root.bind("<Control-Q>", lambda e: self.action_exit())

    # --- Actions ---

    def action_open_image(self) -> None:
        """Open file dialog and load a new image."""
        initial_dir = os.path.abspath("images") if os.path.exists("images") else "."
        filepath = filedialog.askopenfilename(
            parent=self.root,
            title="Open Image",
            initialdir=initial_dir,
            filetypes=[
                ("Supported Images (*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif)", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif"),
                ("PNG Files (*.png)", "*.png"),
                ("JPEG Files (*.jpg;*.jpeg)", "*.jpg;*.jpeg"),
                ("BMP Files (*.bmp)", "*.bmp"),
                ("TIFF Files (*.tiff;*.tif)", "*.tiff;*.tif"),
                ("All Files (*.*)", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            self.pipeline.load_new_image(filepath)
            orig = self.pipeline.original_image
            self.viewer.set_images(orig, orig, "Original Image")
            self.controls_panel.set_active_image_shape(orig.shape)

            meta = self.pipeline.get_original_metadata()
            self.info_panel.update_metadata(meta, "Original Image", 0.0)
            self.info_panel.update_contours([])

            self.toolbar.set_has_image(True)
            self.status_bar.update_metrics(meta["width"], meta["height"], meta["channels"], "Original Image", 0.0)
            self.status_bar.set_status(f"Loaded '{os.path.basename(filepath)}' ({meta['width']}x{meta['height']}, {meta['channels']} channels).")

        except Exception as e:
            messagebox.showerror("Error Opening Image", f"Failed to open image file:\n{str(e)}", parent=self.root)
            self.status_bar.set_status(f"Failed to load image: {str(e)}", is_error=True)

    def action_open_sample_image(self) -> None:
        """Quickly load one of the built-in synthetic demonstration images."""
        sample_dir = os.path.abspath("images")
        if not os.path.exists(sample_dir):
            messagebox.showinfo("Sample Images", "Sample images directory not found.", parent=self.root)
            return

        sample_files = [f for f in os.listdir(sample_dir) if f.endswith((".png", ".jpg", ".bmp"))]
        if not sample_files:
            messagebox.showinfo("Sample Images", "No sample images available.", parent=self.root)
            return

        # Pick first sample or let user open dialog in images/
        self.action_open_image()

    def action_save_image(self) -> None:
        """Save the processed image to the default output directory or prompt save as."""
        if not self.pipeline.has_image() or self.pipeline.processed_image is None:
            messagebox.showwarning("Save Image", "No processed image available to save.", parent=self.root)
            return

        if self.pipeline.original_filepath:
            basename = os.path.basename(self.pipeline.original_filepath)
            name, ext = os.path.splitext(basename)
            os.makedirs("output", exist_ok=True)
            default_path = os.path.join("output", f"processed_{name}{ext}")
            try:
                save_image(default_path, self.pipeline.processed_image)
                self.status_bar.set_status(f"Saved processed image to: {default_path}")
                messagebox.showinfo("Image Saved", f"Successfully saved processed image to:\n{default_path}", parent=self.root)
                return
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{str(e)}", parent=self.root)
                return

        self.action_save_image_as()

    def action_save_image_as(self) -> None:
        """Prompt user for destination filepath and save processed image."""
        if not self.pipeline.has_image() or self.pipeline.processed_image is None:
            messagebox.showwarning("Save Image", "No processed image available to save.", parent=self.root)
            return

        initial_dir = os.path.abspath("output") if os.path.exists("output") else "."
        initial_file = "processed_result.png"
        if self.pipeline.original_filepath:
            name, ext = os.path.splitext(os.path.basename(self.pipeline.original_filepath))
            initial_file = f"processed_{name}{ext}"

        filepath = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Processed Image As",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".png",
            filetypes=[
                ("PNG Image (*.png)", "*.png"),
                ("JPEG Image (*.jpg;*.jpeg)", "*.jpg;*.jpeg"),
                ("BMP Image (*.bmp)", "*.bmp"),
                ("TIFF Image (*.tiff;*.tif)", "*.tiff;*.tif"),
            ],
        )
        if not filepath:
            return

        try:
            save_image(filepath, self.pipeline.processed_image)
            self.status_bar.set_status(f"Saved processed image to: {filepath}")
            messagebox.showinfo("Image Saved", f"Successfully saved processed image to:\n{filepath}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image:\n{str(e)}", parent=self.root)
            self.status_bar.set_status(f"Save failed: {str(e)}", is_error=True)

    def action_reset(self) -> None:
        """Reset processed image to the original image state."""
        if not self.pipeline.has_image():
            return

        res = self.pipeline.reset()
        self.viewer.set_images(self.pipeline.original_image, res, "Original Image")
        meta = self.pipeline.get_original_metadata()
        self.info_panel.update_metadata(meta, "Original Image", 0.0)
        self.info_panel.update_contours([])
        self.status_bar.update_metrics(meta["width"], meta["height"], meta["channels"], "Original Image", 0.0)
        self.status_bar.set_status("Reset processed image back to original.")

    def action_chain_source(self) -> None:
        """Set the current processed image as the new base original image (for pipeline chaining)."""
        if not self.pipeline.has_image() or self.pipeline.processed_image is None:
            return

        if self.pipeline.use_processed_as_source():
            orig = self.pipeline.original_image
            self.viewer.set_images(orig, orig, "Chained Base Image")
            self.controls_panel.set_active_image_shape(orig.shape)
            meta = self.pipeline.get_original_metadata()
            self.info_panel.update_metadata(meta, "Chained Base Image", 0.0)
            self.status_bar.update_metrics(meta["width"], meta["height"], meta["channels"], "Chained Base Image", 0.0)
            self.status_bar.set_status("Chained current processed image as new source input.")

    def action_reapply(self) -> None:
        """Re-run the currently selected algorithm."""
        self.controls_panel.trigger_apply()

    def action_fit_view(self) -> None:
        """Fit preview images to available canvas area preserving aspect ratio."""
        self.viewer.set_scale_mode("fit")
        self.status_bar.set_status("View Mode: Fit to Canvas (Aspect Ratio Preserved)")

    def action_actual_size(self) -> None:
        """Display preview images at 1:1 actual pixel resolution."""
        self.viewer.set_scale_mode("actual")
        self.status_bar.set_status("View Mode: 1:1 Actual Pixel Size")

    def action_toggle_view(self) -> None:
        """Cycle view layout between side-by-side, processed only, and original only."""
        mode = self.viewer.cycle_view_mode()
        mode_names = {
            "side_by_side": "Dual Comparison (Original | Processed)",
            "processed_only": "Single Panel (Processed Only)",
            "original_only": "Single Panel (Original Only)",
        }
        self.status_bar.set_status(f"Layout: {mode_names.get(mode, mode)}")

    def action_export_csv(self) -> None:
        """Export contour metrics to CSV."""
        if not self.pipeline.contour_data_list:
            messagebox.showwarning(
                "Export Contours",
                "No contours have been detected yet.\nRun 'Contour Detection' under Category 5 to detect contours.",
                parent=self.root,
            )
            return

        initial_dir = os.path.abspath("output") if os.path.exists("output") else "."
        csv_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Contour Measurements to CSV",
            initialdir=initial_dir,
            initialfile="contour_measurements.csv",
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet (*.csv)", "*.csv"), ("All Files (*.*)", "*.*")],
        )
        if not csv_path:
            return

        try:
            export_contours_to_csv(self.pipeline.contour_data_list, csv_path)
            messagebox.showinfo(
                "Export Complete",
                f"Successfully exported {len(self.pipeline.contour_data_list)} contour measurements to:\n{csv_path}",
                parent=self.root,
            )
            self.status_bar.set_status(f"Exported {len(self.pipeline.contour_data_list)} contours to {csv_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{str(e)}", parent=self.root)

    def action_show_algorithm_docs(self) -> None:
        """Display comprehensive algorithm documentation dialog."""
        doc_win = tk.Toplevel(self.root)
        doc_win.title("Algorithm Reference Guide - Image Processing Workbench")
        doc_win.geometry("750, 550")
        doc_win.configure(bg=Colors.BG_DARK)

        txt = tk.Text(
            doc_win,
            wrap="word",
            bg=Colors.BG_CARD,
            fg=Colors.TEXT_MAIN,
            font=("Segoe UI", 9),
            padx=12,
            pady=12,
        )
        txt.pack(fill="both", expand=True, padx=8, pady=8)

        content = (
            "IMAGE PROCESSING WORKBENCH - ALGORITHM REFERENCE GUIDE\n"
            "=====================================================\n\n"
            "1. BASIC TRANSFORMATIONS:\n"
            "   • Grayscale: Converts multi-channel BGR image to single-channel luminance.\n"
            "   • Resize: Scales spatial dimensions using area/bilinear interpolation.\n"
            "   • Flip: Reflects pixels horizontally (1), vertically (0), or both (-1).\n"
            "   • Rotate: 90°, 180°, or 270° orthogonal rotation.\n\n"
            "2. FILTERING:\n"
            "   • Gaussian Blur: Convolves image with a 2D Gaussian kernel to attenuate Gaussian noise.\n"
            "   • Median Blur: Replaces each pixel with the neighborhood median, removing salt-and-pepper noise.\n\n"
            "3. THRESHOLDING:\n"
            "   • Binary / Binary Inverse: Global thresholding against a constant cut-off value.\n"
            "   • Otsu Thresholding: Automatically calculates the threshold that minimizes intra-class variance.\n"
            "   • Adaptive Mean & Gaussian: Computes local thresholds across pixel neighborhoods, ideal for uneven illumination.\n\n"
            "4. EDGE DETECTION:\n"
            "   • Canny Edge Detection: Multi-stage algorithm featuring Gaussian smoothing, Sobel gradients, non-maximum suppression, and hysteresis thresholding.\n\n"
            "5. CONTOUR ANALYSIS:\n"
            "   • Suzuki-Abe Border Following: Detects connected binary components.\n"
            "   • Geometric Measurements: Computes Area, Perimeter, Bounding Box, Moments Centroid, Aspect Ratio, and Extent.\n\n"
            "6. MORPHOLOGICAL OPERATIONS:\n"
            "   • Erosion: Shrinks foreground boundaries using a structuring element.\n"
            "   • Dilation: Expands foreground boundaries to bridge gaps.\n"
            "   • Opening: Erosion followed by dilation; eliminates small speckles.\n"
            "   • Closing: Dilation followed by erosion; fills small holes and cracks.\n"
            "   • Gradient: Dilation minus erosion; extracts object boundaries.\n"
            "   • Top-Hat / Black-Hat: Isolates bright or dark structures smaller than the kernel.\n"
        )
        txt.insert("1.0", content)
        txt.config(state="disabled")

    def action_show_about(self) -> None:
        """Display application About dialog."""
        messagebox.showinfo(
            f"About {self.APP_TITLE}",
            f"{self.APP_TITLE} {self.APP_VERSION}\n\n"
            "An educational and engineering desktop workbench for mastering computer vision techniques.\n\n"
            "• Computer Vision: OpenCV & NumPy\n"
            "• GUI Framework: Python Tkinter & ttk\n"
            "• Image Rendering: Pillow (PIL)\n"
            "• Automated Tests: pytest\n\n"
            "Designed for robotics and computer vision engineers.",
            parent=self.root,
        )

    def action_exit(self) -> None:
        """Cleanly close the application."""
        self.root.quit()

    # --- Callbacks ---

    def on_apply_operation(self, category: str, operation: str, params: dict) -> None:
        """Execute selected processing algorithm and update visual state."""
        if not self.pipeline.has_image():
            self.status_bar.set_status("Please open an image first (Ctrl+O).", is_error=True)
            return

        try:
            res_img, elapsed_ms = self.pipeline.execute_operation(category, operation, params)

            # Update image viewer
            self.viewer.set_images(self.pipeline.original_image, res_img, operation)

            # Update information panel
            meta = self.pipeline.get_processed_metadata()
            self.info_panel.update_metadata(meta, operation, elapsed_ms)
            self.info_panel.update_contours(self.pipeline.contour_data_list)

            # Update status bar
            self.status_bar.update_metrics(meta["width"], meta["height"], meta["channels"], operation, elapsed_ms)
            self.status_bar.set_status(f"Completed '{operation}' in {elapsed_ms:.2f} ms.")

        except Exception as e:
            self.status_bar.set_status(f"Processing error in '{operation}': {str(e)}", is_error=True)

    def on_pixel_inspected(
        self,
        x: int,
        y: int,
        orig_val: str,
        proc_val: str,
        color_rgb: Optional[Tuple[int, int, int]],
    ) -> None:
        """Update inspection readouts when user clicks the image preview."""
        self.info_panel.set_pixel_inspection(x, y, orig_val, proc_val, color_rgb)
        self.status_bar.set_pixel_info(f"Pixel ({x}, {y}): Orig = {orig_val} | Proc = {proc_val}")
