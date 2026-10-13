"""Image I/O module for robust loading, saving, and metadata inspection.

Supports Unicode paths, grayscale/color, PNG, JPEG, BMP, and TIFF formats.
"""

import os
from typing import Dict, Any, Optional
import cv2
import numpy as np


class ImageIOError(Exception):
    """Base exception for image input/output errors."""
    pass


class InvalidImageError(ImageIOError):
    """Exception raised when an image is corrupt or cannot be decoded."""
    pass


class UnsupportedFormatError(ImageIOError):
    """Exception raised when an unsupported file extension is provided."""
    pass


SUPPORTED_READ_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SUPPORTED_WRITE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_image(filepath: str) -> np.ndarray:
    """Load an image from the filesystem safely supporting Unicode paths.

    Args:
        filepath: Absolute or relative path to the image file.

    Returns:
        np.ndarray: Decoded image in BGR format (or 2D grayscale if single-channel).

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFormatError: If the file extension is not supported.
        InvalidImageError: If the image cannot be decoded or is empty.
    """
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_READ_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_READ_EXTENSIONS))}"
        )

    try:
        # Using binary read + cv2.imdecode guarantees robust Unicode path support on Windows
        with open(filepath, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

        if file_bytes.size == 0:
            raise InvalidImageError("File is empty (0 bytes).")

        # cv2.IMREAD_UNCHANGED keeps grayscale as 1-channel, color as 3/4-channel
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise InvalidImageError("Failed to decode image data. File may be corrupt.")

        # If image has 4 channels (BGRA), convert to BGR for consistency
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    except (FileNotFoundError, UnsupportedFormatError, InvalidImageError):
        raise
    except Exception as e:
        raise InvalidImageError(f"Unexpected error loading image '{filepath}': {str(e)}") from e


def save_image(filepath: str, image: np.ndarray) -> bool:
    """Save an image to disk safely supporting Unicode paths and formats.

    Args:
        filepath: Destination file path.
        image: Image array (2D grayscale or 3D BGR).

    Returns:
        bool: True if saving succeeded.

    Raises:
        ValueError: If image array is invalid or empty.
        UnsupportedFormatError: If destination format is unsupported.
        ImageIOError: If encoding or writing fails.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot save empty or invalid image array.")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_WRITE_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_WRITE_EXTENSIONS))}"
        )

    dir_path = os.path.dirname(os.path.abspath(filepath))
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    try:
        # cv2.imencode supports unicode paths when written via Python file operations
        success, encoded_buf = cv2.imencode(ext, image)
        if not success or encoded_buf is None:
            raise ImageIOError(f"Failed to encode image to format '{ext}'.")

        with open(filepath, "wb") as f:
            f.write(encoded_buf.tobytes())

        return True
    except (ValueError, UnsupportedFormatError, ImageIOError):
        raise
    except Exception as e:
        raise ImageIOError(f"Error saving image to '{filepath}': {str(e)}") from e


def get_image_metadata(image: Optional[np.ndarray], filepath: str = "") -> Dict[str, Any]:
    """Inspect and return metadata for an image array.

    Args:
        image: The image array to inspect.
        filepath: Optional path to the file.

    Returns:
        Dict[str, Any]: Metadata dictionary.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "is_valid": False,
            "filename": os.path.basename(filepath) if filepath else "None",
            "filepath": filepath,
            "width": 0,
            "height": 0,
            "channels": 0,
            "dtype": "N/A",
            "shape": "N/A",
            "min_val": 0,
            "max_val": 0,
            "size_bytes": 0,
            "size_kb": 0.0,
        }

    shape = image.shape
    height = shape[0]
    width = shape[1]
    channels = shape[2] if image.ndim == 3 else 1

    return {
        "is_valid": True,
        "filename": os.path.basename(filepath) if filepath else "Memory Array",
        "filepath": filepath,
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(image.dtype),
        "shape": str(shape),
        "min_val": int(np.min(image)),
        "max_val": int(np.max(image)),
        "size_bytes": int(image.nbytes),
        "size_kb": round(image.nbytes / 1024.0, 2),
    }
