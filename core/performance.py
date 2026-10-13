"""Performance measurement utilities for image processing algorithms."""

import time
from typing import Optional


class ExecutionTimer:
    """High-resolution execution timer context manager.

    Measures algorithm execution time in milliseconds without including UI or I/O overhead.
    """

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "ExecutionTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0


def measure_execution_time(func, *args, **kwargs) -> tuple:
    """Helper to execute a callable and return (result, elapsed_ms)."""
    with ExecutionTimer() as timer:
        result = func(*args, **kwargs)
    return result, timer.elapsed_ms
