"""
Vision-Language Model (VLM) integration for PDF2EHR.

This module provides VLM-powered layout detection and text extraction
using Google Gemini models via Vertex AI.
"""

from ehrx.vlm.client import VLMClient
from ehrx.vlm.config import VLMConfig
from ehrx.vlm.models import (
    VLMRequest,
    VLMResponse,
    ElementDetection,
    ElementType,
    ProcessingMetadata,
)

__all__ = [
    "VLMClient",
    "VLMConfig",
    "VLMRequest",
    "VLMResponse",
    "ElementDetection",
    "ElementType",
    "ProcessingMetadata",
]
