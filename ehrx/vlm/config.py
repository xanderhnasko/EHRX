"""
Configuration for Vision-Language Model (VLM) processing.

Defines settings for Google Gemini API integration, processing parameters,
and quality thresholds.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class VLMConfig(BaseModel):
    """
    VLM processing configuration.

    Settings for Google Cloud Vertex AI Gemini model integration,
    including API credentials, model selection, and processing parameters.
    """

    # Google Cloud Platform settings
    project_id: str = Field(
        ...,
        description="GCP project ID (from GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID env var)"
    )
    location: str = Field(
        default="us-central1",
        description="GCP region for Vertex AI API calls"
    )
    credentials_path: Optional[str] = Field(
        default=None,
        description="Path to service account JSON (defaults to GOOGLE_APPLICATION_CREDENTIALS)"
    )

    # Model selection
    model_name: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model version (flash for speed/cost, pro for quality)"
    )

    # API parameters
    max_tokens: int = Field(
        default=8192,
        ge=1024,
        le=32768,
        description="Maximum response tokens from VLM"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Top-k sampling parameter"
    )

    # Quality thresholds
    confidence_threshold_extraction: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for text extraction quality"
    )
    confidence_threshold_classification: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for element type classification"
    )
    confidence_threshold_overall: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum overall confidence (triggers human review if below)"
    )

    # Processing behavior
    enable_retry: bool = Field(
        default=True,
        description="Enable automatic retry on transient failures"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed API calls"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial retry delay (exponential backoff applied)"
    )

    # Cost tracking
    enable_cost_tracking: bool = Field(
        default=True,
        description="Track estimated API costs per request"
    )
    cost_per_1k_input_tokens: float = Field(
        default=0.00025,
        ge=0.0,
        description="Cost per 1K input tokens (Gemini 1.5 Flash pricing)"
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.00075,
        ge=0.0,
        description="Cost per 1K output tokens (Gemini 1.5 Flash pricing)"
    )

    # Caching (optional)
    enable_caching: bool = Field(
        default=False,
        description="Enable response caching for identical requests"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds (1 hour default)"
    )

    # Timeout settings
    api_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="API request timeout in seconds"
    )

    # Debugging
    save_raw_responses: bool = Field(
        default=False,
        description="Save raw VLM responses to disk for debugging"
    )
    raw_responses_dir: Optional[str] = Field(
        default=None,
        description="Directory to save raw responses (if enabled)"
    )

    @field_validator('credentials_path', mode='before')
    @classmethod
    def resolve_credentials_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Resolve credentials path from environment if not explicitly provided.

        Falls back to GOOGLE_APPLICATION_CREDENTIALS environment variable.
        """
        if v is not None:
            return v

        # Check for explicit environment variable
        env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env_path:
            return env_path

        # Return None (let google-cloud-aiplatform use default credential chain)
        return None

    @field_validator('project_id', mode='before')
    @classmethod
    def resolve_project_id(cls, v: Optional[str]) -> str:
        """
        Resolve GCP project ID from environment if not explicitly provided.

        Checks GCP_PROJECT_ID and GOOGLE_CLOUD_PROJECT environment variables.
        """
        # Check if value is provided and non-empty
        if v is not None and v.strip():
            return v

        # Check common environment variables
        for env_var in ["GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"]:
            env_value = os.getenv(env_var)
            if env_value and env_value.strip():
                return env_value

        raise ValueError(
            "GCP project_id must be provided either in config or via "
            "GCP_PROJECT_ID/GOOGLE_CLOUD_PROJECT environment variable"
        )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )

    def get_generation_config(self) -> dict:
        """
        Get Vertex AI generation config dictionary.

        Returns configuration for GenerationConfig in Vertex AI SDK.
        """
        return {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate API cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if not self.enable_cost_tracking:
            return 0.0

        input_cost = (input_tokens / 1000.0) * self.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000.0) * self.cost_per_1k_output_tokens

        return input_cost + output_cost

    @classmethod
    def from_env(cls) -> "VLMConfig":
        """
        Create VLMConfig from environment variables.

        Useful for testing and deployment scenarios where configuration
        is provided via environment rather than config files.

        Environment variables:
            - GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT: GCP project ID
            - GCP_LOCATION: GCP region (default: us-central1)
            - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
            - VLM_MODEL_NAME: Model name (default: gemini-1.5-flash)
            - VLM_MAX_TOKENS: Max output tokens (default: 8192)
            - VLM_TEMPERATURE: Sampling temperature (default: 0.1)

        Returns:
            VLMConfig instance with values from environment
        """
        # Get project_id from environment (will trigger validator if missing)
        project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or None

        return cls(
            project_id=project_id,
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            model_name=os.getenv("VLM_MODEL_NAME", "gemini-1.5-flash"),
            max_tokens=int(os.getenv("VLM_MAX_TOKENS", "8192")),
            temperature=float(os.getenv("VLM_TEMPERATURE", "0.1")),
        )

    def __repr__(self) -> str:
        """String representation (masks credentials path for security)."""
        creds = "***" if self.credentials_path else "default"
        return (
            f"VLMConfig(project={self.project_id}, location={self.location}, "
            f"model={self.model_name}, credentials={creds})"
        )
