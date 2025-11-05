"""
Smoke tests for ehrx extraction pipeline
"""

import pytest
from pathlib import Path


def test_import():
    """Test that ehrx can be imported"""
    import ehrx
    assert ehrx.__version__


# Additional smoke tests will be added as modules are implemented

