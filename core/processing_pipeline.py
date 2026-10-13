"""Processing pipeline manager coordinating image state, execution, and timing."""

from typing import Optional, Tuple, Dict, Any, List
import copy
import numpy as np

from core.performance import ExecutionTimer
from core.image_io import load_image, get_image_metadata
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
)


class ProcessingPipeline:
    """Manages original image, processed result, execution dispatch, and performance."""

    def __init__(self) -> None:
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.original_filepath: str = ""
        self.current_category: str = "None"
        self.current_operation: str = "None"
        self.current_params: Dict[str, Any] = {}
        self.last_execution_time_ms: float = 0.0
        self.contour_data_list: List[ContourData] = []
        self.raw_contours: list = []
        self.hierarchy: Optional[np.ndarray] = None

    def load_new_image(self, filepath: str) -> bool:
        """Load an image from disk and initialize pipeline state."""
        img = load_image(filepath)
        self.set_image(img, filepath=filepath)
        return True

    def set_image(self, image: np.ndarray, filepath: str = "") -> None:
        """Set a new active image, guaranteeing a deep copy of pristine data."""
        if image is None or image.size == 0:
            raise ValueError("Cannot set empty image.")
        self.original_image = image.copy()
        self.processed_image = image.copy()
        self.original_filepath = filepath
        self.current_category = "Original"
        self.current_operation = "Original Image"
        self.current_params = {}
        self.last_execution_time_ms = 0.0
        self.contour_data_list = []
        self.raw_contours = []
        self.hierarchy = None

    def has_image(self) -> bool:
        """Return True if an image is currently loaded."""
        return self.original_image is not None and self.original_image.size > 0

    def reset(self) -> Optional[np.ndarray]:
        """Reset processed image to the original image state."""
        if not self.has_image():
            return None
        self.processed_image = self.original_image.copy()
        self.current_category = "Original"
        self.current_operation = "Original Image"
        self.current_params = {}
        self.last_execution_time_ms = 0.0
        self.contour_data_list = []
        self.raw_contours = []
        self.hierarchy = None
        return self.processed_image

    def use_processed_as_source(self) -> bool:
        """Commit the current processed image as the new base original image (for chaining)."""
        if self.processed_image is None or self.processed_image.size == 0:
            return False
        self.original_image = self.processed_image.copy()
        self.current_category = "Chained Source"
        self.current_operation = "Chained Base Image"
        self.last_execution_time_ms = 0.0
        self.contour_data_list = []
        self.raw_contours = []
        return True

    def execute_operation(
        self,
        category: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Execute an operation on the pristine original image with given parameters.

        Args:
            category: Operation category name.
            operation: Specific algorithm name.
            params: Dictionary of parameters for the algorithm.

        Returns:
            Tuple[np.ndarray, float]: (processed image result, execution time in ms).
        """
        if not self.has_image():
            raise ValueError("No image loaded to process.")

        if params is None:
            params = {}

        # Always work on a fresh copy of original_image
        source = self.original_image.copy()
        result_img: Optional[np.ndarray] = None
        contours_info: List[ContourData] = []
        raw_cnts: list = []
        hier: Optional[np.ndarray] = None

        with ExecutionTimer() as timer:
            op_key = operation.lower()

            # --- 1. Basic Transformations ---
            if op_key == "grayscale":
                result_img = to_grayscale(source)

            elif op_key == "resize":
                width = int(params.get("width", source.shape[1]))
                height = int(params.get("height", source.shape[0]))
                preserve_aspect = bool(params.get("preserve_aspect", True))
                result_img = resize_image(source, width, height, preserve_aspect=preserve_aspect)

            elif op_key == "flip":
                direction = str(params.get("direction", "horizontal"))
                result_img = flip_image(source, direction=direction)

            elif op_key == "rotate":
                angle = int(params.get("angle", 90))
                result_img = rotate_image(source, angle=angle)

            # --- 2. Filtering ---
            elif op_key == "gaussian blur":
                ksize = int(params.get("kernel_size", 5))
                sigma_x = float(params.get("sigma_x", 0.0))
                result_img = gaussian_blur(source, kernel_size=ksize, sigma_x=sigma_x)

            elif op_key == "median blur":
                ksize = int(params.get("kernel_size", 5))
                result_img = median_blur(source, kernel_size=ksize)

            # --- 3. Thresholding ---
            elif op_key in ("binary threshold", "binary inverse", "otsu threshold", "otsu inverse"):
                thresh = int(params.get("thresh", 127))
                max_val = int(params.get("max_val", 255))
                mode_map = {
                    "binary threshold": "binary",
                    "binary inverse": "binary_inv",
                    "otsu threshold": "otsu",
                    "otsu inverse": "otsu_inv",
                }
                mode = mode_map.get(op_key, "binary")
                thresh_res, computed_th = binary_threshold(source, thresh=thresh, max_val=max_val, mode=mode)
                result_img = thresh_res
                # Save computed threshold back into params
                params["computed_threshold"] = computed_th

            elif op_key in ("adaptive mean", "adaptive gaussian"):
                max_val = int(params.get("max_val", 255))
                block_size = int(params.get("block_size", 11))
                c_val = int(params.get("c", 2))
                method = "mean" if "mean" in op_key else "gaussian"
                result_img = adaptive_threshold(
                    source, max_val=max_val, method=method, block_size=block_size, c=c_val
                )

            # --- 4. Edge Detection ---
            elif op_key == "canny edge":
                lower = int(params.get("lower_thresh", 100))
                upper = int(params.get("upper_thresh", 200))
                pre_blur = int(params.get("pre_blur_ksize", 0))
                result_img = canny_edge_detection(
                    source, lower_thresh=lower, upper_thresh=upper, pre_blur_ksize=pre_blur
                )

            # --- 5. Contour Analysis ---
            elif op_key == "contour detection":
                retrieval = str(params.get("retrieval_mode", "external"))
                approx = str(params.get("approx_mode", "simple"))
                min_area = float(params.get("min_area", 20.0))
                draw_cnt = bool(params.get("draw_contours", True))
                draw_box = bool(params.get("draw_boxes", True))
                draw_cent = bool(params.get("draw_centroids", True))

                contours_info, raw_cnts, hier = detect_contours(
                    source, retrieval_mode=retrieval, approx_mode=approx, min_area=min_area
                )
                result_img = draw_contour_visualization(
                    source,
                    contours_info,
                    raw_cnts,
                    draw_contours=draw_cnt,
                    draw_boxes=draw_box,
                    draw_centroids=draw_cent,
                )

            # --- 6. Morphological Operations ---
            elif op_key in ("erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat"):
                op_name_map = {
                    "erosion": "erode",
                    "dilation": "dilate",
                    "opening": "open",
                    "closing": "close",
                    "gradient": "gradient",
                    "tophat": "tophat",
                    "blackhat": "blackhat",
                }
                op_func_name = op_name_map.get(op_key, "erode")
                kw = int(params.get("kernel_width", 5))
                kh = int(params.get("kernel_height", 5))
                shape = str(params.get("kernel_shape", "rect"))
                iters = int(params.get("iterations", 1))

                result_img = morphological_operation(
                    source,
                    op=op_func_name,
                    kernel_size=(kw, kh),
                    kernel_shape=shape,
                    iterations=iters,
                )

            else:
                raise ValueError(f"Unrecognized operation '{operation}' in category '{category}'")

        self.processed_image = result_img
        self.current_category = category
        self.current_operation = operation
        self.current_params = copy.deepcopy(params)
        self.last_execution_time_ms = timer.elapsed_ms
        self.contour_data_list = contours_info
        self.raw_contours = raw_cnts
        self.hierarchy = hier

        return result_img, timer.elapsed_ms

    def get_original_metadata(self) -> Dict[str, Any]:
        """Return metadata dictionary for the original image."""
        return get_image_metadata(self.original_image, filepath=self.original_filepath)

    def get_processed_metadata(self) -> Dict[str, Any]:
        """Return metadata dictionary for the processed image."""
        meta = get_image_metadata(self.processed_image, filepath="")
        meta["operation"] = self.current_operation
        meta["execution_time_ms"] = round(self.last_execution_time_ms, 2)
        return meta
