"""Tests for core.image_processing module."""

import pytest
import numpy as np

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
)


def test_grayscale_conversion():
    """Test 3: Grayscale conversion output dimensions and type."""
    # 3-channel color image
    color_img = np.ones((80, 100, 3), dtype=np.uint8) * 128
    gray = to_grayscale(color_img)
    assert gray.ndim == 2
    assert gray.shape == (80, 100)
    assert gray.dtype == np.uint8

    # Already grayscale image
    gray_img = np.ones((50, 60), dtype=np.uint8) * 200
    gray_out = to_grayscale(gray_img)
    assert gray_out.shape == (50, 60)
    assert np.array_equal(gray_img, gray_out)


def test_gaussian_and_median_blur():
    """Test 4: Gaussian and median blur output validity."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 20:44] = 255

    # Gaussian blur valid
    blurred_g = gaussian_blur(img, kernel_size=5, sigma_x=1.5)
    assert blurred_g.shape == img.shape
    assert blurred_g.dtype == np.uint8

    # Gaussian blur invalid even kernel size
    with pytest.raises(ValueError):
        gaussian_blur(img, kernel_size=4)

    # Median blur valid
    blurred_m = median_blur(img, kernel_size=5)
    assert blurred_m.shape == img.shape
    assert blurred_m.dtype == np.uint8

    # Median blur invalid even kernel size
    with pytest.raises(ValueError):
        median_blur(img, kernel_size=6)


def test_standard_thresholding():
    """Test 5: Standard thresholding."""
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 180

    # Binary mode with threshold 100
    res, thresh_val = binary_threshold(img, thresh=100, max_val=255, mode="binary")
    assert res.shape == (50, 50)
    assert thresh_val == 100.0
    assert res[25, 25] == 255
    assert res[5, 5] == 0

    # Binary inverse mode
    res_inv, _ = binary_threshold(img, thresh=100, max_val=255, mode="binary_inv")
    assert res_inv[25, 25] == 0
    assert res_inv[5, 5] == 255


def test_otsu_thresholding():
    """Test 6: Otsu thresholding."""
    # Bimodal image: background around 50, foreground around 200
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:50, :] = 50
    img[50:, :] = 200

    res, otsu_val = binary_threshold(img, thresh=0, max_val=255, mode="otsu")
    assert res.shape == (100, 100)
    # Otsu threshold should split 50 and 200 (OpenCV returns 50 <= otsu_val < 200)
    assert 50 <= otsu_val < 200
    assert res[20, 20] == 0
    assert res[80, 80] == 255


def test_adaptive_threshold_constraints():
    """Test 7: Adaptive thresholding constraints."""
    img = np.zeros((80, 80), dtype=np.uint8)
    img[20:60, 20:60] = 150

    # Valid adaptive mean
    res_mean = adaptive_threshold(img, max_val=255, method="mean", block_size=11, c=2)
    assert res_mean.shape == (80, 80)
    assert res_mean.dtype == np.uint8

    # Valid adaptive gaussian
    res_gauss = adaptive_threshold(img, max_val=255, method="gaussian", block_size=15, c=3)
    assert res_gauss.shape == (80, 80)

    # Invalid block size: even number
    with pytest.raises(ValueError):
        adaptive_threshold(img, block_size=10)

    # Invalid block size: <= 1
    with pytest.raises(ValueError):
        adaptive_threshold(img, block_size=1)


def test_canny_edge_detection():
    """Test 8: Canny edge output."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Draw a solid white square in black background
    img[25:75, 25:75] = 255

    edges = canny_edge_detection(img, lower_thresh=50, upper_thresh=150, pre_blur_ksize=3)
    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8
    # Edge pixels should be detected along the perimeter of the square
    assert np.any(edges > 0)
    # Inner region of square should be 0 (no edges inside uniform region)
    assert edges[50, 50] == 0


def test_basic_transformations():
    """Test basic transformations: resize, flip, rotate."""
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[10:20, 10:20] = [10, 20, 30]

    # Resize preserving aspect ratio
    resized = resize_image(img, target_width=160, target_height=200, preserve_aspect=True)
    assert resized.shape == (120, 160, 3)

    # Resize non-aspect ratio
    resized_no_aspect = resize_image(img, target_width=100, target_height=50, preserve_aspect=False)
    assert resized_no_aspect.shape == (50, 100, 3)

    # Flip horizontal
    flipped_h = flip_image(img, "horizontal")
    assert flipped_h.shape == img.shape

    # Rotate 90 degrees
    rot90 = rotate_image(img, angle=90)
    assert rot90.shape == (80, 60, 3)


def test_original_image_not_modified():
    """Test 15: Ensuring the original image is never modified by processing operations."""
    original = np.zeros((50, 50, 3), dtype=np.uint8)
    original[10:30, 10:30] = [120, 150, 180]
    original_copy = original.copy()

    # Apply various operations
    _ = to_grayscale(original)
    _ = gaussian_blur(original, 5)
    _ = canny_edge_detection(original, 50, 150)
    _ = flip_image(original, "horizontal")
    _ = rotate_image(original, 90)

    # Verify original remains bit-for-bit identical
    np.testing.assert_array_equal(original, original_copy)
