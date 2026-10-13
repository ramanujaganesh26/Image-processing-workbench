"""Automated headless tests verifying GUI instantiation, layout, and event handling."""

import os
import pytest
import tkinter as tk
import numpy as np

from app.application import Application
from app.main_window import MainWindow
from core.image_io import save_image


def test_application_initialization():
    """Verify Tkinter root, theme, and main window initialize without error."""
    try:
        app = Application()
    except tk.TclError as e:
        pytest.skip(f"Tkinter display not available in current environment: {e}")

    assert app.root is not None
    assert app.main_window is not None
    assert isinstance(app.main_window, MainWindow)

    # Force geometry and layout update
    app.root.update_idletasks()
    app.root.update()

    # Verify initial states
    assert not app.main_window.pipeline.has_image()

    app.root.destroy()


def test_gui_image_loading_and_operation_workflow(tmp_path):
    """Test simulating image load, algorithm execution, and UI update through MainWindow."""
    try:
        app = Application()
    except tk.TclError as e:
        pytest.skip(f"Tkinter display not available in current environment: {e}")

    # Create a synthetic image on disk
    test_img = np.zeros((120, 160, 3), dtype=np.uint8)
    test_img[30:90, 40:120] = [0, 255, 128]
    img_path = str(tmp_path / "gui_test.png")
    assert save_image(img_path, test_img) is True

    # Programmatically load image into pipeline and update UI
    app.main_window.pipeline.load_new_image(img_path)
    orig = app.main_window.pipeline.original_image
    app.main_window.viewer.set_images(orig, orig, "Original Image")
    app.main_window.controls_panel.set_active_image_shape(orig.shape)
    meta = app.main_window.pipeline.get_original_metadata()
    app.main_window.info_panel.update_metadata(meta, "Original Image", 0.0)

    app.root.update_idletasks()
    app.root.update()

    assert app.main_window.pipeline.has_image()
    assert meta["width"] == 160
    assert meta["height"] == 120

    # Simulate applying Canny edge detection
    app.main_window.on_apply_operation(
        "4. Edge Detection",
        "Canny Edge",
        {"lower_thresh": 50, "upper_thresh": 150, "pre_blur_ksize": 0},
    )
    app.root.update_idletasks()
    app.root.update()

    assert app.main_window.pipeline.processed_image is not None
    assert app.main_window.pipeline.current_operation == "Canny Edge"

    # Simulate pixel click inspection
    app.main_window.on_pixel_inspected(50, 50, "R:128 G:255 B:0", "Val: 0", (128, 255, 0))
    app.root.update_idletasks()
    app.root.update()

    # Simulate contour detection
    app.main_window.on_apply_operation(
        "5. Contour Analysis",
        "Contour Detection",
        {
            "retrieval_mode": "external",
            "approx_mode": "simple",
            "min_area": 20.0,
            "draw_contours": True,
            "draw_boxes": True,
            "draw_centroids": True,
        },
    )
    app.root.update_idletasks()
    app.root.update()

    assert len(app.main_window.pipeline.contour_data_list) >= 1

    app.root.destroy()
