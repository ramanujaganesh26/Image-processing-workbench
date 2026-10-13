"""Tests for core.contour_analysis module."""

import os
import tempfile
import cv2
import numpy as np

from core.contour_analysis import (
    detect_contours,
    draw_contour_visualization,
    export_contours_to_csv,
)


def test_contour_detection_synthetic_shapes():
    """Test 9 & 10: Contour detection on synthetic shapes and metric measurements."""
    # Create black 200x200 canvas
    canvas = np.zeros((200, 200), dtype=np.uint8)

    # Draw a 40x40 white square from (30, 30) to (70, 70) -> area 1600
    canvas[30:70, 30:70] = 255

    # Draw a 20x30 white rectangle from (120, 100) to (150, 120) -> w=20, h=30, area 600
    canvas[100:130, 120:140] = 255

    results, raw_cnts, hier = detect_contours(
        canvas, retrieval_mode="external", approx_mode="simple", min_area=50.0
    )

    assert len(results) == 2
    assert len(raw_cnts) == 2

    # Verify geometric metrics
    # Contours can be returned in any order, so find the square (area ~ 1600)
    areas = [c.area for c in results]
    assert any(1500 <= a <= 1700 for a in areas)
    assert any(500 <= a <= 700 for a in areas)

    for c in results:
        assert c.perimeter > 0
        assert c.bbox_w > 0
        assert c.bbox_h > 0
        assert c.centroid_x > 0
        assert c.centroid_y > 0
        assert 0.0 < c.aspect_ratio < 5.0
        assert 0.0 < c.extent <= 1.0


def test_contour_visualization_drawing():
    """Test drawing contours, bounding boxes, and centroids."""
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas[20:60, 20:60] = 255

    results, raw_cnts, _ = detect_contours(canvas, min_area=10.0)
    assert len(results) == 1

    vis = draw_contour_visualization(
        canvas, results, raw_cnts, draw_contours=True, draw_boxes=True, draw_centroids=True
    )

    assert vis is not None
    assert vis.shape == (100, 100, 3)
    # Check that annotations introduced colored pixels (e.g. green contour or red centroid)
    # Green: (0, 255, 0)
    has_green = np.any((vis[:, :, 0] == 0) & (vis[:, :, 1] == 255) & (vis[:, :, 2] == 0))
    assert has_green


def test_export_contours_to_csv():
    """Test exporting contour measurements to CSV file."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    canvas[20:60, 20:60] = 255

    results, _, _ = detect_contours(canvas, min_area=10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "contours.csv")
        export_contours_to_csv(results, csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2  # Header + 1 record
        header = lines[0].strip().split(",")
        assert "Area" in header
        assert "Centroid_X" in header
        assert "AspectRatio" in header
