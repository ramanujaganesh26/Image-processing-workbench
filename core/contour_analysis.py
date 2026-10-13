"""Contour detection, geometric feature extraction, visualization, and CSV export."""

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
import cv2
import numpy as np


@dataclass
class ContourData:
    """Stores computed geometric properties for a single detected contour."""
    index: int
    area: float
    perimeter: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    centroid_x: int
    centroid_y: int
    aspect_ratio: float
    extent: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "area": round(self.area, 2),
            "perimeter": round(self.perimeter, 2),
            "bbox": (self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h),
            "centroid": (self.centroid_x, self.centroid_y),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "extent": round(self.extent, 3),
        }


def detect_contours(
    image: np.ndarray,
    retrieval_mode: str = "external",
    approx_mode: str = "simple",
    min_area: float = 10.0,
) -> Tuple[List[ContourData], list, Optional[np.ndarray]]:
    """Detect contours and calculate geometric properties.

    Args:
        image: Source image (binary, grayscale, or color).
        retrieval_mode: 'external', 'all', or 'tree'.
        approx_mode: 'simple' or 'none'.
        min_area: Minimum contour area threshold to filter out noise.

    Returns:
        Tuple[List[ContourData], list of raw contours, hierarchy array]
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # If the image is not strictly binary (has intermediate values), apply Otsu thresholding
    unique_vals = np.unique(gray)
    if len(unique_vals) > 2 or not np.array_equal(unique_vals, [0, 255]):
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = gray.copy()

    # Map retrieval mode
    mode_map = {
        "external": cv2.RETR_EXTERNAL,
        "all": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE,
        "ccomp": cv2.RETR_CCOMP,
    }
    retrieval = mode_map.get(retrieval_mode.lower(), cv2.RETR_EXTERNAL)

    # Map approximation mode
    approx_map = {
        "simple": cv2.CHAIN_APPROX_SIMPLE,
        "none": cv2.CHAIN_APPROX_NONE,
    }
    approx = approx_map.get(approx_mode.lower(), cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv2.findContours(binary, retrieval, approx)

    results: List[ContourData] = []
    filtered_raw_contours = []
    idx_counter = 1

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue

        perimeter = float(cv2.arcLength(cnt, closed=True))
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Centroid calculation using spatial moments
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cx = int(round(moments["m10"] / moments["m00"]))
            cy = int(round(moments["m01"] / moments["m00"]))
        else:
            cx = int(bx + bw // 2)
            cy = int(by + bh // 2)

        # Aspect ratio: width / height
        aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

        # Extent: contour area / bounding box area
        bbox_area = float(bw * bh)
        extent = area / bbox_area if bbox_area > 0 else 0.0

        item = ContourData(
            index=idx_counter,
            area=area,
            perimeter=perimeter,
            bbox_x=bx,
            bbox_y=by,
            bbox_w=bw,
            bbox_h=bh,
            centroid_x=cx,
            centroid_y=cy,
            aspect_ratio=aspect_ratio,
            extent=extent,
        )
        results.append(item)
        filtered_raw_contours.append(cnt)
        idx_counter += 1

    return results, filtered_raw_contours, hierarchy


def draw_contour_visualization(
    base_image: np.ndarray,
    contours_data: List[ContourData],
    raw_contours: list,
    draw_contours: bool = True,
    draw_boxes: bool = True,
    draw_centroids: bool = True,
) -> np.ndarray:
    """Render contours, bounding boxes, and centroids onto a copy of the image.

    Args:
        base_image: Base image to draw onto.
        contours_data: List of computed ContourData objects.
        raw_contours: Corresponding OpenCV contour point arrays.
        draw_contours: Whether to outline contours.
        draw_boxes: Whether to draw bounding boxes.
        draw_centroids: Whether to mark centroid coordinates.

    Returns:
        np.ndarray: 3-channel BGR image with drawn visual elements.
    """
    if base_image is None or base_image.size == 0:
        raise ValueError("Base image is empty or None.")

    # Convert to 3-channel BGR for colored annotations
    if base_image.ndim == 2:
        canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    elif base_image.ndim == 3 and base_image.shape[2] == 1:
        canvas = cv2.cvtColor(base_image[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_image.copy()

    # Draw contours in bright green
    if draw_contours and raw_contours:
        cv2.drawContours(canvas, raw_contours, -1, (0, 255, 0), thickness=2)

    # Draw bounding boxes and centroids
    for data in contours_data:
        if draw_boxes:
            # Yellow / Cyan box
            cv2.rectangle(
                canvas,
                (data.bbox_x, data.bbox_y),
                (data.bbox_x + data.bbox_w, data.bbox_y + data.bbox_h),
                (0, 255, 255),
                thickness=1,
            )

        if draw_centroids:
            # Red centroid circle
            cv2.circle(canvas, (data.centroid_x, data.centroid_y), 4, (0, 0, 255), -1)
            # Label
            label = f"#{data.index}"
            cv2.putText(
                canvas,
                label,
                (data.centroid_x + 6, data.centroid_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return canvas


def export_contours_to_csv(contours_data: List[ContourData], filepath: str) -> None:
    """Export contour measurements to a CSV file.

    Args:
        contours_data: List of ContourData objects.
        filepath: Destination CSV file path.
    """
    dir_path = os.path.dirname(os.path.abspath(filepath))
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    header = [
        "Index",
        "Area",
        "Perimeter",
        "BBox_X",
        "BBox_Y",
        "BBox_Width",
        "BBox_Height",
        "Centroid_X",
        "Centroid_Y",
        "AspectRatio",
        "Extent",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in contours_data:
            writer.writerow([
                c.index,
                f"{c.area:.2f}",
                f"{c.perimeter:.2f}",
                c.bbox_x,
                c.bbox_y,
                c.bbox_w,
                c.bbox_h,
                c.centroid_x,
                c.centroid_y,
                f"{c.aspect_ratio:.3f}",
                f"{c.extent:.3f}",
            ])
