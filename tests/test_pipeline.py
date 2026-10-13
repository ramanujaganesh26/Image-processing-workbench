"""Tests for core.processing_pipeline module."""

import pytest
import numpy as np

from core.processing_pipeline import ProcessingPipeline


def test_pipeline_lifecycle():
    """Test full pipeline lifecycle: load array, execute op, reset, metadata."""
    pipeline = ProcessingPipeline()
    assert not pipeline.has_image()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [100, 150, 200]
    pipeline.set_image(img, filepath="test.png")
    assert pipeline.has_image()

    # Original metadata
    orig_meta = pipeline.get_original_metadata()
    assert orig_meta["width"] == 100
    assert orig_meta["height"] == 100
    assert orig_meta["channels"] == 3

    # Execute Gaussian blur
    result, elapsed = pipeline.execute_operation(
        "Filtering", "Gaussian Blur", {"kernel_size": 5, "sigma_x": 1.0}
    )
    assert result.shape == (100, 100, 3)
    assert elapsed > 0.0
    assert pipeline.current_operation == "Gaussian Blur"

    # Verify original image was not modified
    np.testing.assert_array_equal(pipeline.original_image, img)

    # Processed metadata
    proc_meta = pipeline.get_processed_metadata()
    assert proc_meta["operation"] == "Gaussian Blur"
    assert proc_meta["execution_time_ms"] > 0

    # Reset
    reset_img = pipeline.reset()
    np.testing.assert_array_equal(reset_img, img)
    assert pipeline.current_operation == "Original Image"


def test_pipeline_chaining():
    """Test using processed image as new source (chaining)."""
    pipeline = ProcessingPipeline()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    pipeline.set_image(img)

    # Operation: to grayscale
    gray, _ = pipeline.execute_operation("Basic Transformations", "Grayscale")
    assert gray.ndim == 2

    # Chain it as new source
    assert pipeline.use_processed_as_source() is True
    assert pipeline.original_image.ndim == 2
