"""Core modules for Image Processing Workbench."""

from core.performance import ExecutionTimer
from core.image_io import load_image, save_image, get_image_metadata, ImageIOError
from core.image_processing import (
    to_grayscale,
    resize_image,
    flip_image,
    rotate_image,
    gaussian_blur,
    median_blur,
    binary_threshold,
    adaptive_threshold,
    canny_edge_detection,
    morphological_operation,
)
from core.contour_analysis import (
    ContourData,
    detect_contours,
    draw_contour_visualization,
    export_contours_to_csv,
)
from core.processing_pipeline import ProcessingPipeline

__all__ = [
    "ExecutionTimer",
    "load_image",
    "save_image",
    "get_image_metadata",
    "ImageIOError",
    "to_grayscale",
    "resize_image",
    "flip_image",
    "rotate_image",
    "gaussian_blur",
    "median_blur",
    "binary_threshold",
    "adaptive_threshold",
    "canny_edge_detection",
    "morphological_operation",
    "ContourData",
    "detect_contours",
    "draw_contour_visualization",
    "export_contours_to_csv",
    "ProcessingPipeline",
]
