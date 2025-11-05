"""
Setup script for ehrx package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="ehrx",
    version="0.1.0",
    description="EHR extraction tool for scanned PDFs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PDF2EHR Team",
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.3.0,<2.4.0",
        "torchvision>=0.18.0,<0.19.0",
        "layoutparser>=0.3.0",
        "detectron2==0.6",
        "pytesseract>=0.3.0",
        "opencv-python>=4.9.0",
        "pymupdf>=1.23.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.0",
        "pydantic>=2.6.0",
        "typer>=0.9.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
        ],
        "paddle": [
            "paddlepaddle",
            "layoutparser[paddledetection]",
        ],
    },
    entry_points={
        "console_scripts": [
            "ehrx=ehrx.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)

