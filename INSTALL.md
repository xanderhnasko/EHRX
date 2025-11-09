# Installation Guide for PDF2EHR

This guide provides step-by-step instructions for setting up the PDF2EHR environment with all required dependencies.

## Prerequisites

- Anaconda or Miniconda installed
- Python 3.10+ recommended

## Installation Steps

### 1. Create Conda Environment

```bash
conda create -n pdf2ehr python=3.10
conda activate pdf2ehr
```

### 2. Install Core Dependencies (in order)

**Important**: Install dependencies in this exact order to avoid conflicts.

#### Step 2a: Install PyTorch via Conda
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

#### Step 2b: Install Detectron2 via Conda
```bash
conda install -c conda-forge detectron2
```

#### Step 2c: Install LayoutParser via Pip (environment-specific)
```bash
python -m pip install layoutparser==0.3.*
```

#### Step 2d: Install Core Dependencies via Conda
```bash
conda install -c conda-forge pandas pyyaml opencv pytesseract poppler
```

#### Step 2e: Install Remaining Dependencies via Pip
```bash
python -m pip install pdf2image pydantic typer rich pytest pytest-cov pymupdf
```

#### Step 2f: Ensure NumPy Compatibility
```bash
python -m pip install "numpy<2"
```

### 3. Verify Installation

Test that all key packages are importable:

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import detectron2; print('Detectron2: OK')"
python -c "import layoutparser; print('LayoutParser:', layoutparser.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
```

Verify poppler is installed for PDF processing:
```bash
pdftoppm -h  # Should show poppler help, not "command not found"
```

### 4. Optional: Suppress Telemetry Noise

To reduce logging noise from detectron2/iopath telemetry (optional):
```bash
export IOPATH_NO_TELEMETRY=1
```

### 5. Test Layout Detection

```bash
python test_detection_visual.py path/to/your/document.pdf
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError for installed packages**
   - Make sure you're using `python -m pip` instead of just `pip`
   - Verify conda environment is activated: `conda info --envs`

2. **NumPy compatibility errors with matplotlib**
   - Downgrade NumPy: `python -m pip install "numpy<2"`

3. **LayoutParser import fails**
   - Reinstall with environment-specific pip: `python -m pip install layoutparser==0.3.*`

4. **PyTorch/Detectron2 compatibility issues**
   - Use conda for both: `conda install pytorch detectron2 -c pytorch -c conda-forge`

5. **PDF processing fails ("poppler not found")**
   - Install poppler: `conda install -c conda-forge poppler`
   - Verify: `pdftoppm -h` should show help, not error

### Environment Verification Commands

```bash
# Check you're in the right environment
which python
python -c "import sys; print('Python path:', sys.executable)"

# List installed packages
conda list
```

## Why This Installation Method?

- **Conda for ML dependencies**: PyTorch, Detectron2, and OpenCV have complex binary dependencies that conda handles better
- **Environment-specific pip**: Using `python -m pip` ensures packages install in the current conda environment
- **Dependency order**: Installing PyTorch before Detectron2 before LayoutParser prevents version conflicts
- **NumPy version control**: Some packages aren't yet compatible with NumPy 2.x

## Alternative: Docker Setup (Future)

For a more reproducible setup, consider using Docker with a pre-configured environment. This avoids local dependency conflicts entirely.