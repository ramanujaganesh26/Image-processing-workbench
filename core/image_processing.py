"""Core computer vision algorithms implemented using OpenCV and NumPy.

Each operation is implemented as a standalone, deterministic, testable function.
"""

from typing import Tuple, Union, Optional
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel 8-bit grayscale.

    Args:
        image: Source image array (1, 3, or 4 channels).

    Returns:
        np.ndarray: Grayscale 2D array.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].copy()
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image dimensions: {image.shape}")


def resize_image(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    preserve_aspect: bool = True,
) -> np.ndarray:
    """Resize image with optional aspect-ratio preservation.

    Args:
        image: Source image array.
        target_width: Desired target width in pixels (>= 1).
        target_height: Desired target height in pixels (>= 1).
        preserve_aspect: If True, scales within target dimensions preserving aspect ratio.

    Returns:
        np.ndarray: Resized image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Dimensions must be positive: got ({target_width}, {target_height})")

    orig_h, orig_w = image.shape[:2]

    if preserve_aspect:
        scale = min(target_width / orig_w, target_height / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
    else:
        new_w = target_width
        new_h = target_height

    interp = cv2.INTER_AREA if (new_w < orig_w or new_h < orig_h) else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def flip_image(image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Flip an image horizontally, vertically, or both.

    Args:
        image: Source image array.
        direction: One of 'horizontal', 'vertical', or 'both'.

    Returns:
        np.ndarray: Flipped image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    direction_lower = direction.lower()
    if direction_lower in ("horizontal", "h"):
        flip_code = 1
    elif direction_lower in ("vertical", "v"):
        flip_code = 0
    elif direction_lower in ("both", "hv"):
        flip_code = -1
    else:
        raise ValueError(f"Unknown flip direction '{direction}'. Choose 'horizontal', 'vertical', or 'both'.")

    return cv2.flip(image, flip_code)


def rotate_image(image: np.ndarray, angle: int = 90) -> np.ndarray:
    """Rotate image by 90, 180, or 270 degrees clockwise.

    Args:
        image: Source image array.
        angle: Rotation angle in degrees (90, 180, or 270).

    Returns:
        np.ndarray: Rotated image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    angle_mod = angle % 360
    if angle_mod == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle_mod == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle_mod == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle_mod == 0:
        return image.copy()
    else:
        raise ValueError(f"Unsupported rotation angle {angle}. Supported: 90, 180, 270.")


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma_x: float = 0.0) -> np.ndarray:
    """Apply Gaussian blur with odd kernel size.

    Args:
        image: Source image array.
        kernel_size: Positive odd integer (e.g. 3, 5, 7).
        sigma_x: Gaussian kernel standard deviation in X direction (0 = auto-computed from size).

    Returns:
        np.ndarray: Blurred image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be a positive odd integer, got {kernel_size}")

    if kernel_size == 1:
        return image.copy()

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=float(sigma_x))


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter for impulse noise (salt-and-pepper) suppression.

    Args:
        image: Source image array.
        kernel_size: Positive odd integer greater than 1 (e.g. 3, 5, 7).

    Returns:
        np.ndarray: Filtered image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"Median blur kernel size must be a positive odd integer, got {kernel_size}")

    if kernel_size == 1:
        return image.copy()

    return cv2.medianBlur(image, kernel_size)


def binary_threshold(
    image: np.ndarray,
    thresh: int = 127,
    max_val: int = 255,
    mode: str = "binary",
) -> Tuple[np.ndarray, float]:
    """Apply global or Otsu thresholding to an image.

    Args:
        image: Source image (converted to grayscale if needed).
        thresh: Manual threshold value (0-255).
        max_val: Maximum value to use with THRESH_BINARY (typically 255).
        mode: 'binary', 'binary_inv', 'otsu', or 'otsu_inv'.

    Returns:
        Tuple[np.ndarray, float]: (thresholded image, computed or applied threshold value).
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    gray = to_grayscale(image)
    thresh = max(0, min(255, int(thresh)))
    max_val = max(0, min(255, int(max_val)))

    mode_lower = mode.lower()
    if mode_lower == "binary":
        val, result = cv2.threshold(gray, thresh, max_val, cv2.THRESH_BINARY)
    elif mode_lower == "binary_inv":
        val, result = cv2.threshold(gray, thresh, max_val, cv2.THRESH_BINARY_INV)
    elif mode_lower == "otsu":
        val, result = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode_lower == "otsu_inv":
        val, result = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown thresholding mode '{mode}'. Choose 'binary', 'binary_inv', 'otsu', 'otsu_inv'.")

    return result, float(val)


def adaptive_threshold(
    image: np.ndarray,
    max_val: int = 255,
    method: str = "mean",
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """Apply adaptive thresholding for illumination-invariant binarization.

    Args:
        image: Source image (converted to grayscale if needed).
        max_val: Maximum output pixel value.
        method: 'mean' or 'gaussian'.
        block_size: Size of a pixel neighborhood used to calculate threshold (odd integer > 1).
        c: Constant subtracted from the mean or weighted mean.

    Returns:
        np.ndarray: Binarized image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(f"Adaptive threshold block size must be an odd integer > 1, got {block_size}")

    gray = to_grayscale(image)
    max_val = max(0, min(255, int(max_val)))

    method_lower = method.lower()
    if method_lower == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method_lower == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError(f"Unknown adaptive method '{method}'. Choose 'mean' or 'gaussian'.")

    return cv2.adaptiveThreshold(
        gray,
        maxValue=max_val,
        adaptiveMethod=adaptive_method,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=int(c),
    )


def canny_edge_detection(
    image: np.ndarray,
    lower_thresh: int = 100,
    upper_thresh: int = 200,
    pre_blur_ksize: int = 0,
) -> np.ndarray:
    """Detect edges using Canny multi-stage algorithm.

    Args:
        image: Source image.
        lower_thresh: Lower hysteresis threshold (0-255).
        upper_thresh: Upper hysteresis threshold (0-255).
        pre_blur_ksize: Optional Gaussian pre-filter kernel size (0 to disable, or odd > 1).

    Returns:
        np.ndarray: 8-bit single channel edge map.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    gray = to_grayscale(image)

    if pre_blur_ksize > 1 and pre_blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (pre_blur_ksize, pre_blur_ksize), 0)

    # Ensure bounds and proper lower <= upper order
    lower = max(0, min(255, int(lower_thresh)))
    upper = max(0, min(255, int(upper_thresh)))
    if lower > upper:
        lower, upper = upper, lower

    return cv2.Canny(gray, threshold1=lower, threshold2=upper)


def morphological_operation(
    image: np.ndarray,
    op: str = "erode",
    kernel_size: Tuple[int, int] = (5, 5),
    kernel_shape: str = "rect",
    iterations: int = 1,
) -> np.ndarray:
    """Apply mathematical morphology operations.

    Args:
        image: Source image (grayscale or color).
        op: Operation ('erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat').
        kernel_size: (width, height) tuple of positive integers.
        kernel_shape: 'rect', 'ellipse', or 'cross'.
        iterations: Number of times operation is applied (>= 1).

    Returns:
        np.ndarray: Processed image array.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    kw, kh = kernel_size
    if kw <= 0 or kh <= 0:
        raise ValueError(f"Kernel dimensions must be positive integers: ({kw}, {kh})")
    iterations = max(1, int(iterations))

    shape_lower = kernel_shape.lower()
    if shape_lower == "rect":
        struct_shape = cv2.MORPH_RECT
    elif shape_lower == "ellipse":
        struct_shape = cv2.MORPH_ELLIPSE
    elif shape_lower == "cross":
        struct_shape = cv2.MORPH_CROSS
    else:
        raise ValueError(f"Unknown kernel shape '{kernel_shape}'. Choose 'rect', 'ellipse', or 'cross'.")

    kernel = cv2.getStructuringElement(struct_shape, (kw, kh))

    op_lower = op.lower()
    if op_lower == "erode":
        return cv2.erode(image, kernel, iterations=iterations)
    elif op_lower == "dilate":
        return cv2.dilate(image, kernel, iterations=iterations)
    elif op_lower == "open":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op_lower == "close":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op_lower == "gradient":
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif op_lower == "tophat":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif op_lower == "blackhat":
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    else:
        raise ValueError(
            f"Unknown morphological op '{op}'. Supported: erode, dilate, open, close, gradient, tophat, blackhat."
        )
