"""
Pytest fixtures for VLM tests.

Provides mock responses, sample images, and test configurations.
"""

import json
import pytest
import numpy as np
from PIL import Image
from datetime import datetime

from ehrx.vlm.config import VLMConfig
from ehrx.vlm.models import DocumentContext


@pytest.fixture
def vlm_config():
    """Basic VLM configuration for testing."""
    return VLMConfig(
        project_id="test-project",
        location="us-central1",
        model_name="gemini-1.5-flash",
        max_tokens=8192,
        temperature=0.1,
        enable_cost_tracking=False,  # Disable cost tracking in tests
        enable_retry=False,  # Disable retries for faster tests
    )


@pytest.fixture
def document_context():
    """Basic document context for testing."""
    return DocumentContext(
        document_type="Clinical Notes",
        section_hierarchy=["Notes", "Progress Notes"],
        page_number=0,
        total_pages=5,
    )


@pytest.fixture
def sample_image():
    """Generate sample test image (800x1000 white background with text region)."""
    # Create white image
    img = Image.new('RGB', (800, 1000), color='white')
    return img


@pytest.fixture
def sample_image_array():
    """Generate sample test image as numpy array."""
    # Create white image
    img_array = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    return img_array


@pytest.fixture
def mock_vlm_response_json():
    """Mock VLM response in expected JSON format."""
    return json.dumps({
        "elements": [
            {
                "element_id": "E_0001",
                "semantic_type": "section_header",
                "bbox": {
                    "x0": 100.0,
                    "y0": 50.0,
                    "x1": 400.0,
                    "y1": 80.0
                },
                "content": "MEDICATIONS",
                "confidence_scores": {
                    "extraction": 0.98,
                    "classification": 0.95,
                    "clinical_context": 0.90
                },
                "clinical_metadata": {
                    "temporal_qualifier": "current",
                    "clinical_domain": "pharmacology",
                    "requires_validation": False
                },
                "page": 0
            },
            {
                "element_id": "E_0002",
                "semantic_type": "medication_table",
                "bbox": {
                    "x0": 100.0,
                    "y0": 100.0,
                    "x1": 700.0,
                    "y1": 300.0
                },
                "content": "Metformin 500mg PO BID\\nLisinopril 10mg PO daily\\nAspirin 81mg PO daily",
                "confidence_scores": {
                    "extraction": 0.92,
                    "classification": 0.88,
                    "clinical_context": 0.85
                },
                "clinical_metadata": {
                    "temporal_qualifier": "current",
                    "clinical_domain": "pharmacology",
                    "requires_validation": False
                },
                "page": 0
            },
            {
                "element_id": "E_0003",
                "semantic_type": "clinical_paragraph",
                "bbox": {
                    "x0": 100.0,
                    "y0": 350.0,
                    "x1": 700.0,
                    "y1": 450.0
                },
                "content": "Patient reports good medication compliance. No reported adverse effects.",
                "confidence_scores": {
                    "extraction": 0.95,
                    "classification": 0.92,
                    "clinical_context": 0.90
                },
                "page": 0
            }
        ],
        "overall_confidence": 0.91,
        "requires_human_review": False,
        "review_reasons": []
    })


@pytest.fixture
def mock_vlm_response_low_confidence():
    """Mock VLM response with low confidence elements."""
    return json.dumps({
        "elements": [
            {
                "element_id": "E_0001",
                "semantic_type": "handwritten_annotation",
                "bbox": {
                    "x0": 500.0,
                    "y0": 100.0,
                    "x1": 600.0,
                    "y1": 150.0
                },
                "content": "Unclear handwriting",
                "confidence_scores": {
                    "extraction": 0.60,
                    "classification": 0.70,
                    "clinical_context": 0.65
                },
                "clinical_metadata": {
                    "requires_validation": True,
                    "validation_reason": "Poor handwriting quality"
                },
                "page": 0
            }
        ],
        "overall_confidence": 0.65,
        "requires_human_review": True,
        "review_reasons": ["Overall confidence 0.65 below threshold 0.85"]
    })


@pytest.fixture
def mock_vlm_response_empty():
    """Mock VLM response with no elements detected."""
    return json.dumps({
        "elements": [],
        "overall_confidence": 0.0,
        "requires_human_review": True,
        "review_reasons": ["No elements detected on page"]
    })


@pytest.fixture
def mock_vlm_response_with_markdown():
    """Mock VLM response wrapped in markdown code fence."""
    inner_json = {
        "elements": [
            {
                "element_id": "E_0001",
                "semantic_type": "section_header",
                "bbox": {"x0": 100.0, "y0": 50.0, "x1": 400.0, "y1": 80.0},
                "content": "LABS",
                "confidence_scores": {
                    "extraction": 0.98,
                    "classification": 0.96,
                    "clinical_context": 0.92
                },
                "page": 0
            }
        ],
        "overall_confidence": 0.95,
        "requires_human_review": False,
        "review_reasons": []
    }
    return f"```json\n{json.dumps(inner_json)}\n```"


@pytest.fixture
def mock_vlm_error_response():
    """Mock VLM error response (malformed JSON)."""
    return "This is not valid JSON and will cause parsing to fail"


@pytest.fixture
def sample_element_dict():
    """Sample element dictionary in pipeline format."""
    return {
        "id": "E_0001",
        "type": "medication_table",
        "page": 0,
        "bbox_px": [100.0, 100.0, 700.0, 300.0],
        "payload": {
            "text": "Metformin 500mg PO BID",
            "confidence": 0.88,
        },
        "column": 0,
        "z_order": 5,
    }
