"""
Tests for VLMClient with mocked API calls.

Tests client initialization, API interaction, response parsing,
and error handling without making real API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from ehrx.vlm.client import VLMClient
from ehrx.vlm.config import VLMConfig
from ehrx.vlm.models import VLMRequest, DocumentContext


class TestVLMClientInitialization:
    """Tests for VLMClient initialization."""

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_client_initialization_success(self, mock_model, mock_init, vlm_config):
        """Test successful client initialization."""
        client = VLMClient(vlm_config)

        # Verify Vertex AI initialized
        mock_init.assert_called_once_with(
            project=vlm_config.project_id,
            location=vlm_config.location,
        )

        # Verify model created
        mock_model.assert_called_once()
        assert client.config == vlm_config
        assert client._request_count == 0
        assert client._total_cost_usd == 0.0

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_client_initialization_failure(self, mock_model, mock_init):
        """Test client initialization failure handling."""
        mock_init.side_effect = Exception("API initialization failed")

        config = VLMConfig(project_id="test-project")
        with pytest.raises(RuntimeError, match="Vertex AI initialization failed"):
            VLMClient(config)


class TestImagePreparation:
    """Tests for image preparation methods."""

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_prepare_image_from_pil(self, mock_model, mock_init, vlm_config, sample_image):
        """Test image preparation from PIL Image."""
        client = VLMClient(vlm_config)

        part = client._prepare_image(sample_image)
        assert part is not None

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_prepare_image_from_numpy(self, mock_model, mock_init, vlm_config, sample_image_array):
        """Test image preparation from numpy array."""
        client = VLMClient(vlm_config)

        part = client._prepare_image(sample_image_array)
        assert part is not None

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_prepare_image_invalid_type(self, mock_model, mock_init, vlm_config):
        """Test image preparation with invalid type."""
        client = VLMClient(vlm_config)

        with pytest.raises(TypeError, match="Unsupported image type"):
            client._prepare_image({"invalid": "type"})


class TestElementDetection:
    """Tests for element detection with mocked API."""

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_detect_elements_success(
        self,
        mock_model_class,
        mock_init,
        vlm_config,
        sample_image,
        document_context,
        mock_vlm_response_json
    ):
        """Test successful element detection."""
        # Mock the model and its response
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        # Mock the API response
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = mock_vlm_response_json

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 500

        mock_model_instance.generate_content.return_value = mock_response

        # Create client and request
        client = VLMClient(vlm_config)
        request = VLMRequest(context=document_context)

        # Detect elements
        response = client.detect_elements(sample_image, request)

        # Verify results
        assert len(response.elements) == 3
        assert response.elements[0].element_id == "E_0001"
        assert response.elements[0].semantic_type.value == "section_header"
        assert response.elements[1].element_id == "E_0002"
        assert response.overall_confidence > 0.0
        assert client._request_count == 1

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_detect_elements_with_markdown_fence(
        self,
        mock_model_class,
        mock_init,
        vlm_config,
        sample_image,
        document_context,
        mock_vlm_response_with_markdown
    ):
        """Test element detection with markdown code fence in response."""
        # Mock the model and response
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = mock_vlm_response_with_markdown

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 300

        mock_model_instance.generate_content.return_value = mock_response

        # Create client and detect
        client = VLMClient(vlm_config)
        request = VLMRequest(context=document_context)
        response = client.detect_elements(sample_image, request)

        # Should successfully parse despite markdown fence
        assert len(response.elements) == 1
        assert response.elements[0].semantic_type.value == "section_header"

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_detect_elements_low_confidence(
        self,
        mock_model_class,
        mock_init,
        vlm_config,
        sample_image,
        document_context,
        mock_vlm_response_low_confidence
    ):
        """Test element detection with low confidence response."""
        # Mock the model and response
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = mock_vlm_response_low_confidence

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 200

        mock_model_instance.generate_content.return_value = mock_response

        # Create client and detect
        client = VLMClient(vlm_config)
        request = VLMRequest(context=document_context)
        response = client.detect_elements(sample_image, request)

        # Verify low confidence handling
        assert response.requires_human_review is True
        assert len(response.review_reasons) > 0
        assert response.overall_confidence < 0.85


class TestRetryLogic:
    """Tests for retry logic and error handling."""

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    @patch('ehrx.vlm.client.time.sleep')  # Mock sleep to speed up tests
    def test_retry_on_transient_failure(
        self,
        mock_sleep,
        mock_model_class,
        mock_init,
        sample_image,
        document_context
    ):
        """Test retry logic on transient API failures."""
        # Configure retry enabled
        config = VLMConfig(
            project_id="test-project",
            enable_retry=True,
            max_retries=2
        )

        # Mock model that fails twice then succeeds
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = '{"elements": [], "overall_confidence": 0.0}'

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50

        mock_model_instance.generate_content.side_effect = [
            Exception("Transient error 1"),
            Exception("Transient error 2"),
            mock_response  # Success on third attempt
        ]

        # Create client and request
        client = VLMClient(config)
        request = VLMRequest(context=document_context)

        # Should succeed after retries
        response = client.detect_elements(sample_image, request)
        assert response is not None

        # Verify retries happened
        assert mock_model_instance.generate_content.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_retry_disabled(
        self,
        mock_model_class,
        mock_init,
        sample_image,
        document_context
    ):
        """Test that retry is disabled when configured."""
        # Configure retry disabled
        config = VLMConfig(
            project_id="test-project",
            enable_retry=False
        )

        # Mock model that fails
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance
        mock_model_instance.generate_content.side_effect = Exception("API error")

        # Create client and request
        client = VLMClient(config)
        request = VLMRequest(context=document_context)

        # Should fail immediately without retry
        with pytest.raises(RuntimeError, match="VLM API call failed"):
            client.detect_elements(sample_image, request)

        # Verify only one attempt
        assert mock_model_instance.generate_content.call_count == 1


class TestResponseParsing:
    """Tests for response parsing edge cases."""

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_parse_response_invalid_json(
        self,
        mock_model_class,
        mock_init,
        vlm_config,
        sample_image,
        document_context,
        mock_vlm_error_response
    ):
        """Test handling of invalid JSON response."""
        # Mock the model with invalid response
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = mock_vlm_error_response

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50

        mock_model_instance.generate_content.return_value = mock_response

        # Create client and detect
        client = VLMClient(vlm_config)
        request = VLMRequest(context=document_context)
        response = client.detect_elements(sample_image, request)

        # Should return error response with no elements
        assert len(response.elements) == 0
        assert response.requires_human_review is True
        assert any("parsing failed" in reason.lower() for reason in response.review_reasons)


class TestStatistics:
    """Tests for usage statistics tracking."""

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_statistics_tracking(
        self,
        mock_model_class,
        mock_init,
        sample_image,
        document_context,
        mock_vlm_response_json
    ):
        """Test that statistics are tracked correctly."""
        config = VLMConfig(
            project_id="test-project",
            enable_cost_tracking=True
        )

        # Mock successful response
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = mock_vlm_response_json

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 500

        mock_model_instance.generate_content.return_value = mock_response

        # Create client
        client = VLMClient(config)
        request = VLMRequest(context=document_context)

        # Make multiple requests
        client.detect_elements(sample_image, request)
        client.detect_elements(sample_image, request)

        # Check statistics
        stats = client.get_stats()
        assert stats['request_count'] == 2
        assert stats['total_cost_usd'] > 0
        assert stats['model_name'] == config.model_name

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_statistics_reset(self, mock_model_class, mock_init, vlm_config):
        """Test statistics reset functionality."""
        client = VLMClient(vlm_config)

        # Manually set some stats
        client._request_count = 5
        client._total_cost_usd = 0.50

        # Reset
        client.reset_stats()

        # Verify reset
        stats = client.get_stats()
        assert stats['request_count'] == 0
        assert stats['total_cost_usd'] == 0.0


class TestClientRepr:
    """Tests for client string representation."""

    @patch('ehrx.vlm.client.vertexai.init')
    @patch('ehrx.vlm.client.GenerativeModel')
    def test_client_repr(self, mock_model_class, mock_init, vlm_config):
        """Test client __repr__ method."""
        client = VLMClient(vlm_config)
        repr_str = repr(client)

        assert "VLMClient" in repr_str
        assert vlm_config.model_name in repr_str
        assert "requests=" in repr_str
        assert "cost=" in repr_str
