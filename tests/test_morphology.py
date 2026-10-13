"""Tests for morphological operations in core.image_processing module."""

import pytest
import numpy as np

from core.image_processing import morphological_operation


def test_erosion_and_dilation():
    """Test 11: Morphological erosion and dilation."""
    # Create an image with a single bright white block
    img = np.zeros((50, 50), dtype=np.uint8)
    img[15:35, 15:35] = 255

    orig_white_pixels = np.count_nonzero(img == 255)

    # Erosion should shrink the foreground
    eroded = morphological_operation(
        img, op="erode", kernel_size=(3, 3), kernel_shape="rect", iterations=1
    )
    eroded_white_pixels = np.count_nonzero(eroded == 255)
    assert eroded_white_pixels < orig_white_pixels

    # Dilation should expand the foreground
    dilated = morphological_operation(
        img, op="dilate", kernel_size=(3, 3), kernel_shape="rect", iterations=1
    )
    dilated_white_pixels = np.count_nonzero(dilated == 255)
    assert dilated_white_pixels > orig_white_pixels


def test_opening_and_closing():
    """Test 12: Opening (noise removal) and closing (hole filling)."""
    # 1. Opening: removes small bright noise dots
    img_noise = np.zeros((60, 60), dtype=np.uint8)
    img_noise[20:40, 20:40] = 255  # Main object
    img_noise[5, 5] = 255          # Noise speckle 1
    img_noise[50, 50] = 255        # Noise speckle 2

    opened = morphological_operation(
        img_noise, op="open", kernel_size=(3, 3), kernel_shape="rect", iterations=1
    )
    # The noise speckles should be eliminated
    assert opened[5, 5] == 0
    assert opened[50, 50] == 0
    # The main object should remain
    assert opened[30, 30] == 255

    # 2. Closing: closes small dark holes inside foreground object
    img_hole = np.zeros((60, 60), dtype=np.uint8)
    img_hole[15:45, 15:45] = 255
    img_hole[30, 30] = 0  # 1-pixel dark hole

    closed = morphological_operation(
        img_hole, op="close", kernel_size=(3, 3), kernel_shape="rect", iterations=1
    )
    # The hole should be closed (filled)
    assert closed[30, 30] == 255


def test_morphology_kernel_shapes_and_ops():
    """Test various kernel shapes and operations (gradient, tophat, blackhat)."""
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 255

    for shape in ("rect", "ellipse", "cross"):
        res = morphological_operation(img, op="dilate", kernel_size=(5, 5), kernel_shape=shape)
        assert res.shape == (50, 50)

    # Gradient, tophat, blackhat
    for op in ("gradient", "tophat", "blackhat"):
        res = morphological_operation(img, op=op, kernel_size=(3, 3), kernel_shape="rect")
        assert res.shape == (50, 50)


def test_morphology_invalid_parameters():
    """Test parameter validation for morphological operations."""
    img = np.zeros((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        morphological_operation(img, op="unknown_op")

    with pytest.raises(ValueError):
        morphological_operation(img, op="erode", kernel_size=(0, 5))

    with pytest.raises(ValueError):
        morphological_operation(img, op="erode", kernel_shape="triangle")
