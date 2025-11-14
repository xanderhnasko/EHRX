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
        "pytesseract>=0.3.0",
        "opencv-python>=4.9.0",
        "pymupdf>=1.23.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.0",
        "pydantic>=2.6.0",
        "numpy<2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
        ],
    },
    # entry_points will be added back when cli.py is rebuilt
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)

