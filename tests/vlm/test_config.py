"""
Tests for VLM configuration.

Tests cover configuration loading, validation, and environment variable resolution.
"""

import os
import pytest
from unittest.mock import patch

from ehrx.vlm.config import VLMConfig


class TestVLMConfig:
    """Tests for VLMConfig model."""

    def test_minimal_config(self):
        """Test VLM config with minimal required fields."""
        config = VLMConfig(project_id="test-project")
        assert config.project_id == "test-project"
        assert config.location == "us-central1"
        assert config.model_name == "gemini-1.5-flash"

    def test_full_config(self):
        """Test VLM config with all fields specified."""
        config = VLMConfig(
            project_id="test-project",
            location="us-west1",
            credentials_path="/path/to/credentials.json",
            model_name="gemini-1.5-pro",
            max_tokens=4096,
            temperature=0.2,
            confidence_threshold_overall=0.90,
            enable_retry=False,
            enable_cost_tracking=False,
        )
        assert config.project_id == "test-project"
        assert config.location == "us-west1"
        assert config.model_name == "gemini-1.5-pro"
        assert config.max_tokens == 4096
        assert config.temperature == 0.2
        assert config.confidence_threshold_overall == 0.90
        assert config.enable_retry is False
        assert config.enable_cost_tracking is False

    def test_project_id_from_env(self):
        """Test project_id resolution from environment."""
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "env-project"}):
            config = VLMConfig.from_env()
            assert config.project_id == "env-project"

    def test_project_id_missing_raises_error(self):
        """Test that missing project_id raises error."""
        # Clear environment variables that provide project_id
        env_vars_to_clear = ["GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"]
        with patch.dict(os.environ, {k: "" for k in env_vars_to_clear}, clear=False):
            with pytest.raises(ValueError, match="project_id must be provided"):
                VLMConfig.from_env()

    def test_credentials_path_from_env(self):
        """Test credentials_path resolution from environment."""
        with patch.dict(os.environ, {
            "GCP_PROJECT_ID": "test-project",
            "GOOGLE_APPLICATION_CREDENTIALS": "/env/path/to/creds.json"
        }):
            config = VLMConfig.from_env()
            assert config.credentials_path == "/env/path/to/creds.json"

    def test_get_generation_config(self):
        """Test generation config dictionary creation."""
        config = VLMConfig(
            project_id="test-project",
            max_tokens=4096,
            temperature=0.2,
            top_p=0.9,
            top_k=30,
        )
        gen_config = config.get_generation_config()
        assert gen_config["max_output_tokens"] == 4096
        assert gen_config["temperature"] == 0.2
        assert gen_config["top_p"] == 0.9
        assert gen_config["top_k"] == 30

    def test_estimate_cost(self):
        """Test API cost estimation."""
        config = VLMConfig(
            project_id="test-project",
            enable_cost_tracking=True,
            cost_per_1k_input_tokens=0.00025,
            cost_per_1k_output_tokens=0.00075,
        )

        # Test with 1000 input tokens and 1000 output tokens
        cost = config.estimate_cost(input_tokens=1000, output_tokens=1000)
        expected_cost = (1000/1000 * 0.00025) + (1000/1000 * 0.00075)
        assert abs(cost - expected_cost) < 0.000001

        # Test with 5000 input tokens and 2000 output tokens
        cost = config.estimate_cost(input_tokens=5000, output_tokens=2000)
        expected_cost = (5000/1000 * 0.00025) + (2000/1000 * 0.00075)
        assert abs(cost - expected_cost) < 0.000001

    def test_estimate_cost_disabled(self):
        """Test cost estimation returns 0 when disabled."""
        config = VLMConfig(
            project_id="test-project",
            enable_cost_tracking=False,
        )
        cost = config.estimate_cost(input_tokens=1000, output_tokens=1000)
        assert cost == 0.0

    def test_config_validation_max_tokens(self):
        """Test max_tokens validation."""
        # Valid values
        config = VLMConfig(project_id="test-project", max_tokens=1024)
        assert config.max_tokens == 1024

        config = VLMConfig(project_id="test-project", max_tokens=32768)
        assert config.max_tokens == 32768

        # Invalid values
        with pytest.raises(ValueError):
            VLMConfig(project_id="test-project", max_tokens=100)  # Too low

        with pytest.raises(ValueError):
            VLMConfig(project_id="test-project", max_tokens=50000)  # Too high

    def test_config_validation_temperature(self):
        """Test temperature validation."""
        # Valid values
        config = VLMConfig(project_id="test-project", temperature=0.0)
        assert config.temperature == 0.0

        config = VLMConfig(project_id="test-project", temperature=2.0)
        assert config.temperature == 2.0

        # Invalid values
        with pytest.raises(ValueError):
            VLMConfig(project_id="test-project", temperature=-0.1)

        with pytest.raises(ValueError):
            VLMConfig(project_id="test-project", temperature=2.5)

    def test_config_validation_confidence_thresholds(self):
        """Test confidence threshold validation."""
        # Valid values
        config = VLMConfig(
            project_id="test-project",
            confidence_threshold_extraction=0.8,
            confidence_threshold_classification=0.85,
            confidence_threshold_overall=0.9,
        )
        assert config.confidence_threshold_extraction == 0.8
        assert config.confidence_threshold_classification == 0.85
        assert config.confidence_threshold_overall == 0.9

        # Invalid values
        with pytest.raises(ValueError):
            VLMConfig(project_id="test-project", confidence_threshold_extraction=1.5)

        with pytest.raises(ValueError):
            VLMConfig(project_id="test-project", confidence_threshold_overall=-0.1)

    def test_config_repr_masks_credentials(self):
        """Test that __repr__ masks credentials path."""
        config = VLMConfig(
            project_id="test-project",
            credentials_path="/secret/path/to/credentials.json"
        )
        repr_str = repr(config)
        assert "***" in repr_str
        assert "/secret/path" not in repr_str

    def test_config_repr_shows_default_for_no_credentials(self):
        """Test __repr__ shows 'default' when no credentials path."""
        config = VLMConfig(project_id="test-project")
        repr_str = repr(config)
        assert "credentials=default" in repr_str

    def test_retry_settings(self):
        """Test retry configuration settings."""
        config = VLMConfig(
            project_id="test-project",
            enable_retry=True,
            max_retries=5,
            retry_delay_seconds=2.0,
        )
        assert config.enable_retry is True
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0

    def test_caching_settings(self):
        """Test caching configuration settings."""
        config = VLMConfig(
            project_id="test-project",
            enable_caching=True,
            cache_ttl_seconds=7200,
        )
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 7200

    def test_debug_settings(self):
        """Test debug configuration settings."""
        config = VLMConfig(
            project_id="test-project",
            save_raw_responses=True,
            raw_responses_dir="/path/to/debug",
        )
        assert config.save_raw_responses is True
        assert config.raw_responses_dir == "/path/to/debug"

    def test_from_env_with_all_vars(self):
        """Test from_env with all environment variables."""
        with patch.dict(os.environ, {
            "GCP_PROJECT_ID": "env-project",
            "GCP_LOCATION": "europe-west1",
            "GOOGLE_APPLICATION_CREDENTIALS": "/env/creds.json",
            "VLM_MODEL_NAME": "gemini-1.5-pro",
            "VLM_MAX_TOKENS": "4096",
            "VLM_TEMPERATURE": "0.3",
        }):
            config = VLMConfig.from_env()
            assert config.project_id == "env-project"
            assert config.location == "europe-west1"
            assert config.credentials_path == "/env/creds.json"
            assert config.model_name == "gemini-1.5-pro"
            assert config.max_tokens == 4096
            assert config.temperature == 0.3
