"""Image Processing Workbench - Main Entry Point.

A modular desktop application for learning, experimenting with, and demonstrating
fundamental computer vision techniques using Python, OpenCV, and NumPy.
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.application import Application


def main() -> None:
    """Initialize and run the Image Processing Workbench desktop application."""
    try:
        app = Application()
        app.run()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error running Image Processing Workbench: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
