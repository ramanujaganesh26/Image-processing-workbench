# Image Processing Workbench

An engineering-grade desktop application for learning, experimenting with, and demonstrating fundamental computer vision algorithms using **Python**, **OpenCV**, and **NumPy**.

Built specifically for computer vision students, researchers, and robotics software engineers transitioning from foundational algorithms to advanced perception systems.

---

## 1. Overview

The **Image Processing Workbench** is a native desktop application designed to bridge theoretical computer vision concepts with interactive, real-time experimentation. Unlike black-box pipelines or command-line scripts, this workbench allows engineers to:

- Inspect pixel-level operations and color channels.
- Interactively tune hyperparameters with live visual feedback.
- Compare processed results side-by-side with original inputs.
- Measure algorithm latency in milliseconds with sub-millisecond precision.
- Extract geometric contour features and export measurements directly to CSV.

---

## 2. Key Features

- **Side-by-Side Dual Viewport**: Real-time comparison between the pristine original image and processed output with aspect-ratio preservation and letterbox handling.
- **Interactive Pixel Inspector**: Click any point on either canvas to view exact original image coordinates `(X, Y)`, color channel intensities (BGR, RGB, Hex), and a live color swatch.
- **High-Precision Performance Profiling**: Algorithm execution latency measured in milliseconds (`ms`) isolated from UI rendering and disk I/O overhead.
- **Robust Image I/O**: Full support for PNG, JPEG, BMP, and TIFF files, including full Unicode path support on Windows.
- **Pipeline State & Chaining**: Non-destructive processing ensures the original image is never overwritten. Users can optionally chain operations by setting the processed result as the new source input.
- **Contour Measurement & Export**: Calculates Area, Perimeter (arc length), Bounding Box, Centroid moments ($M_{10}/M_{00}$, $M_{01}/M_{00}$), Aspect Ratio, and Extent, with one-click export to CSV.
- **Dark Neutral Engineering Interface**: Inspired by scientific workbenches, featuring high-contrast typography, restrained color accents, and full keyboard shortcut support.

---

## 3. Supported Computer Vision Algorithms

The application organizes algorithms into 6 core categories:

### 1. Basic Transformations
* **Grayscale Conversion**: Converts 3-channel BGR images to 1-channel luminance using $Y = 0.299R + 0.587G + 0.114B$.
* **Spatial Resizing**: Scales image dimensions with an option to preserve the aspect ratio, utilizing area interpolation for downsampling and bilinear interpolation for upsampling.
* **Geometric Flipping**: Horizontal, vertical, or combined flipping.
* **Orthogonal Rotation**: 90°, 180°, and 270° clockwise rotations.

### 2. Filtering
* **Gaussian Blur**: 2D Gaussian convolution kernel ($G(x,y) = \frac{1}{2\pi\sigma^2}\exp(-\frac{x^2+y^2}{2\sigma^2})$) for high-frequency noise suppression while maintaining edge continuity.
* **Median Blur**: Non-linear filtering replacing center pixels with neighborhood medians; highly effective at removing impulse ("salt-and-pepper") noise without softening sharp edges.

### 3. Thresholding
* **Standard Binary & Binary Inverse**: Segmentation against a user-configurable cut-off threshold ($dst(x,y) = \text{max\_val}$ if $src(x,y) > T$ else $0$).
* **Otsu's Global Thresholding**: Automatically determines optimal threshold by minimizing intra-class variance ($\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)$) on bimodal histograms.
* **Adaptive Mean & Gaussian Thresholding**: Computes local thresholds across pixel neighborhoods to handle uneven scene illumination and shadows.

### 4. Edge Detection
* **Canny Edge Detection**: Multi-stage algorithm featuring Gaussian smoothing, Sobel gradient computation, non-maximum suppression (NMS), and hysteresis thresholding with optional pre-filtering.

### 5. Contour Analysis
* **Topological Extraction**: Suzuki & Abe border-following algorithm supporting External (`RETR_EXTERNAL`), All (`RETR_LIST`), and Hierarchical (`RETR_TREE`) modes.
* **Geometric Feature Extraction**:
  - **Area**: Contour area via Green's theorem (`cv2.contourArea`).
  - **Perimeter**: Closed arc length (`cv2.arcLength`).
  - **Bounding Box**: Upright bounding rectangle $(x, y, w, h)$.
  - **Centroid**: First-order spatial moments ($c_x = \frac{m_{10}}{m_{00}}$, $c_y = \frac{m_{01}}{m_{00}}$) with zero-division protection.
  - **Aspect Ratio**: Width-to-height ratio ($\frac{w}{h}$).
  - **Extent**: Contour area relative to bounding box area ($\frac{\text{Area}}{w \times h}$).
* **Visualization & CSV Export**: Customizable visualization overlays for contours (green), bounding boxes (cyan), and centroids/indices (red), with tabular viewing and CSV export.

### 6. Mathematical Morphology
* **Erosion**: Computes local neighborhood minimum; shrinks foreground structures and removes small background speckles.
* **Dilation**: Computes local neighborhood maximum; expands foreground structures and bridges micro-fractures.
* **Opening**: Erosion followed by dilation; eliminates small foreground noise artifacts.
* **Closing**: Dilation followed by erosion; closes dark holes and fissures inside foreground objects.
* **Morphological Gradient**: Difference between dilation and erosion; highlights boundary gradients.
* **Top-Hat**: Difference between input and its opening; isolates bright elements smaller than the structuring element.
* **Black-Hat**: Difference between closing and the input; isolates dark elements smaller than the structuring element.
* **Structuring Elements**: Configurable kernel dimensions (width/height) and geometries (Rectangular, Elliptical, Cross).

---

## 4. Technology Stack

- **Language**: Python 3.11+ / Python 3.12
- **Computer Vision**: OpenCV (`opencv-python` >= 4.8.0)
- **Numerical Computing**: NumPy (`numpy` >= 1.24.0)
- **Desktop GUI**: Python standard Tkinter & ttk
- **Image Rendering**: Pillow (`Pillow` >= 10.0.0)
- **Testing**: pytest (`pytest` >= 7.0.0)

---

## 5. Project Architecture

```
ImageProcessingWorkbench/
├── main.py                  # Desktop application entry point
├── requirements.txt         # Core dependencies
├── README.md                # System documentation and user guide
├── .gitignore               # Version control ignore definitions
│
├── app/                     # GUI layer (Tkinter / ttk)
│   ├── __init__.py
│   ├── application.py       # Application initialization and lifecycle
│   ├── main_window.py       # Layout coordination and keyboard shortcuts
│   ├── toolbar.py           # Quick-action top toolbar
│   ├── image_viewer.py      # Dual-canvas viewport with aspect-ratio scaling
│   ├── controls_panel.py    # Algorithm selectors and interactive parameters
│   ├── information_panel.py # Image metadata, pixel inspector, and contour table
│   ├── status_bar.py        # Real-time metrics and state indicators
│   └── theme.py             # Dark neutral design system and ttk styles
│
├── core/                    # Pure computer vision processing core
│   ├── __init__.py
│   ├── performance.py       # High-resolution execution timing utilities
│   ├── image_io.py          # Safe image loading/saving with Unicode support
│   ├── image_processing.py  # Filters, thresholds, edge detectors, and morphology
│   ├── contour_analysis.py  # Contour detection, geometric metrics, and CSV export
│   └── processing_pipeline.py # State management, execution dispatch, and history
│
├── tests/                   # Automated pytest suite
│   ├── test_image_io.py     # Image decoding, encoding, and metadata tests
│   ├── test_image_processing.py # Transform, filter, threshold, and edge tests
│   ├── test_contours.py     # Geometric features and CSV export tests
│   ├── test_morphology.py   # Morphological operations and structuring element tests
│   ├── test_pipeline.py     # Pipeline state and chaining tests
│   └── test_gui.py          # Headless GUI initialization and event tests
│
├── images/                  # Built-in synthetic test images
│   ├── geometric_shapes.png # Multi-color geometric shapes for contour analysis
│   ├── noisy_sample.png     # Salt-and-pepper noise for filter testing
│   ├── uneven_lighting_doc.png # Document with gradient shadow for adaptive thresholding
│   └── rotary_encoder_pattern.png # High-contrast radial pattern for Canny edge testing
│
└── output/                  # Default directory for saved processed images and CSVs
```

---

## 6. Installation & Execution (Windows 10/11)

### Prerequisites
- Python 3.11 or Python 3.12 installed on your system.
- `pip` package manager.

### Step 1: Create a Virtual Environment (Recommended)

Open PowerShell or Command Prompt in the project folder:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
# PowerShell:
.\venv\Scripts\Activate.ps1

# Command Prompt:
.\venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Run the Application

Launch the desktop workbench:

```powershell
python main.py
```

### Step 4: Run the Test Suite

Execute the automated test suite with pytest:

```powershell
python -m pytest -v
```

---

## 7. Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| **Ctrl + O** | Open an image file from the filesystem |
| **Ctrl + S** | Save processed image to `output/` |
| **Ctrl + Shift + S** | Save processed image as a new file |
| **Ctrl + R** | Reset processing back to original image |
| **Ctrl + Q** | Exit application |

---

## 8. Built-in Sample Images

The project includes pre-generated synthetic test images in `images/` tailored for experimentation:

1. `images/geometric_shapes.png`: Colorful rectangles, circles with holes, and triangles on a dark background. Ideal for **Contour Detection**, **Aspect Ratio**, and **Pixel Inspection**.
2. `images/noisy_sample.png`: High-density salt-and-pepper impulse noise. Demonstrates the superiority of **Median Blur** over **Gaussian Blur** for impulse noise reduction.
3. `images/uneven_lighting_doc.png`: Document text with an illumination gradient (shadow). Demonstrates why **Adaptive Thresholding** succeeds where global **Binary/Otsu Thresholding** fails.
4. `images/rotary_encoder_pattern.png`: High-contrast structural geometry with concentric rings and radial ticks. Ideal for **Canny Edge Detection** hysteresis parameter tuning.

---

## 9. Troubleshooting

* **File contains non-English characters in path**:
  The application utilizes binary stream buffer decoding (`np.frombuffer` + `cv2.imdecode`) rather than standard `cv2.imread`, fully supporting Unicode folder names and paths on Windows.
* **Image appears black or empty**:
  Verify the image file is not corrupt and has a non-zero byte size. Supported extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`.
* **Tkinter Window scaling looks blurry on High-DPI screens**:
  The workbench automatically requests per-monitor DPI awareness from the Windows shell (`ctypes.windll.shcore.SetProcessDpiAwareness(1)`).

---

## 10. Educational Roadmap & Next Steps

This workbench establishes a rigorous understanding of classical computer vision. Recommended next steps for robotics perception engineers:
1. Corner and Keypoint Detectors (Harris Corner, FAST, ORB).
2. Spatial Transformations and Homography (Perspective Warping).
3. Optical Flow (Lucas-Kanade) for robot velocity estimation.
4. Classical Object Classification via Histogram of Oriented Gradients (HOG) + Support Vector Machines (SVM).
