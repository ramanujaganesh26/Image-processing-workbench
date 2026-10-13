"""Tests for core.image_io module."""

import os
import tempfile
import pytest
import numpy as np
import cv2

from core.image_io import (
    load_image,
    save_image,
    get_image_metadata,
    ImageIOError,
    InvalidImageError,
    UnsupportedFormatError,
)


def test_save_and_load_color_image():
    """Test 1 & 13: Loading a valid generated color image and reloading saved image."""
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    img[20:80, 30:90] = [255, 128, 64]  # BGR

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_color.png")
        assert save_image(filepath, img) is True
        assert os.path.exists(filepath)

        loaded = load_image(filepath)
        assert loaded is not None
        assert loaded.shape == (100, 120, 3)
        assert loaded.dtype == np.uint8
        # Check pixel value at center
        np.testing.assert_array_equal(loaded[50, 50], [255, 128, 64])


def test_save_and_load_grayscale_image():
    """Test 14: Handling grayscale image saving and loading."""
    img = np.full((80, 90), 180, dtype=np.uint8)
    img[10:30, 10:30] = 50

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_gray.bmp")
        assert save_image(filepath, img) is True

        loaded = load_image(filepath)
        assert loaded is not None
        assert loaded.ndim == 2
        assert loaded.shape == (80, 90)
        assert loaded[20, 20] == 50
        assert loaded[50, 50] == 180


def test_load_invalid_path():
    """Test 2: Handling an invalid/non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image_file_path_xyz_123.png")


def test_load_unsupported_format():
    """Test unsupported file format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_file = os.path.join(tmpdir, "test.txt")
        with open(fake_file, "w") as f:
            f.write("not an image")
        with pytest.raises(UnsupportedFormatError):
            load_image(fake_file)


def test_load_empty_file():
    """Test loading an empty (0-byte) image file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_file = os.path.join(tmpdir, "empty.png")
        with open(empty_file, "wb") as f:
            pass
        with pytest.raises(InvalidImageError):
            load_image(empty_file)


def test_unicode_path_support():
    """Test saving and loading images with Unicode paths on Windows."""
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Include unicode characters like accents, Japanese, and spaces
        unicode_dir = os.path.join(tmpdir, "テスト_Dossier_Élève")
        os.makedirs(unicode_dir, exist_ok=True)
        filepath = os.path.join(unicode_dir, "résumé_画像.png")

        assert save_image(filepath, img) is True
        loaded = load_image(filepath)
        assert loaded is not None
        assert loaded.shape == (50, 50, 3)


def test_get_image_metadata():
    """Test image metadata extraction."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[10, 10] = [10, 20, 255]
    meta = get_image_metadata(img, "sample.png")

    assert meta["is_valid"] is True
    assert meta["width"] == 200
    assert meta["height"] == 100
    assert meta["channels"] == 3
    assert meta["min_val"] == 0
    assert meta["max_val"] == 255
    assert meta["dtype"] == "uint8"
