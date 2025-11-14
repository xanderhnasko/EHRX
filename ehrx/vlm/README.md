# VLM Module - Vision-Language Model Integration

This module provides VLM-powered layout detection and text extraction for PDF2EHR using Google Gemini models via Vertex AI.

## Overview

The VLM module replaces the previous LayoutParser-based detection system with semantic understanding powered by Google's Gemini 1.5 Flash model. It provides:

- **Full Layout Detection**: Identifies elements and their positions on document pages
- **Text Extraction**: Extracts text content directly from images
- **Semantic Classification**: Classifies elements into 15+ EHR-specific types
- **Clinical Understanding**: Recognizes medical terminology and relationships
- **Confidence Scoring**: Multi-dimensional confidence for quality assurance

## Quick Start

### 1. Setup Google Cloud Platform

Follow the detailed setup guide in `docs/GCP_SETUP.md` to:
- Create GCP project
- Enable Vertex AI API
- Create service account and credentials
- Configure environment variables

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Set up environment variables (or use `.env` file):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"
```

### 4. Basic Usage

```python
from ehrx.vlm import VLMClient, VLMConfig, VLMRequest, DocumentContext
from PIL import Image

# Initialize VLM client
config = VLMConfig.from_env()
client = VLMClient(config)

# Prepare request with context
context = DocumentContext(
    document_type="Clinical Notes",
    page_number=0,
    total_pages=5
)

request = VLMRequest(
    image_path="/path/to/page.png",
    context=context
)

# Detect elements
response = client.detect_elements(
    image=Image.open("/path/to/page.png"),
    request=request
)

# Process results
for element in response.elements:
    print(f"Type: {element.semantic_type}")
    print(f"Content: {element.content}")
    print(f"Confidence: {element.confidence_scores.overall():.2f}")
    print(f"BBox: {element.bbox.to_list()}")
    print()

# Check if human review needed
if response.requires_human_review:
    print(f"Human review required: {response.review_reasons}")

# Get usage statistics
stats = client.get_stats()
print(f"Requests: {stats['request_count']}")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
```

## Module Structure

```
ehrx/vlm/
├── __init__.py          # Module exports
├── client.py            # VLMClient - main API interface
├── config.py            # VLMConfig - configuration model
├── models.py            # Pydantic data models
├── prompts.py           # Prompt templates
└── README.md            # This file
```

## Configuration

Configuration can be provided via:

1. **Environment variables** (recommended for deployment):
   ```bash
   export GCP_PROJECT_ID="your-project"
   export GCP_LOCATION="us-central1"
   export VLM_MODEL_NAME="gemini-1.5-flash"
   ```

2. **YAML configuration** (`configs/default.yaml`):
   ```yaml
   vlm:
     project_id: "your-project"
     location: "us-central1"
     model_name: "gemini-1.5-flash"
     max_tokens: 8192
     temperature: 0.1
   ```

3. **Programmatic configuration**:
   ```python
   config = VLMConfig(
       project_id="your-project",
       location="us-central1",
       model_name="gemini-1.5-flash"
   )
   ```

### Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project_id` | *required* | GCP project ID |
| `location` | `us-central1` | GCP region |
| `model_name` | `gemini-1.5-flash` | Gemini model version |
| `max_tokens` | `8192` | Max response tokens |
| `temperature` | `0.1` | Sampling temperature (0.0-2.0) |
| `confidence_threshold_overall` | `0.85` | Minimum confidence for auto-accept |
| `enable_retry` | `true` | Enable automatic retries |
| `max_retries` | `3` | Maximum retry attempts |
| `enable_cost_tracking` | `true` | Track API costs |

## Semantic Element Types

The VLM module classifies content into 15+ semantic types:

### Document Structure
- `document_header`: Hospital/clinic identifying information
- `patient_demographics`: Name, DOB, MRN, contact info
- `page_metadata`: Page numbers, dates, document IDs
- `section_header`: Major section headings (PROBLEMS, MEDICATIONS)
- `subsection_header`: Minor headings within sections

### Clinical Content
- `clinical_paragraph`: Free-text clinical narratives
- `medication_table`: Structured medication lists
- `lab_results_table`: Laboratory values with units/ranges
- `vital_signs_table`: Temperature, BP, pulse, etc.
- `problem_list`: Diagnoses with ICD codes
- `assessment_plan`: Clinical reasoning and treatment plans
- `list_items`: Bullet/numbered lists

### Special Content
- `handwritten_annotation`: Handwritten notes
- `stamp_signature`: Official stamps or signatures
- `medical_figure`: Graphs, charts, diagrams
- `form_field_group`: Label-value pairs from forms

### Administrative
- `margin_content`: Headers, footers, confidentiality notices
- `uncategorized`: Content requiring human review

## Confidence Scores

Each element includes three confidence dimensions:

1. **Extraction Confidence**: Text extraction accuracy (OCR quality)
2. **Classification Confidence**: Semantic type assignment confidence
3. **Clinical Context Confidence**: Clinical metadata understanding

**Overall Confidence** = weighted average:
- 40% extraction
- 40% classification
- 20% clinical context

Elements below configured threshold (default 0.85) are flagged for human review.

## Cost Management

VLM API calls incur costs. The module provides cost tracking:

```python
# Get cost statistics
stats = client.get_stats()
print(f"Total requests: {stats['request_count']}")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
print(f"Avg cost per request: ${stats['average_cost_per_request']:.4f}")

# Reset statistics
client.reset_stats()
```

**Gemini 1.5 Flash Pricing** (as of 2025):
- Input: ~$0.00025 per 1K tokens
- Output: ~$0.00075 per 1K tokens
- Typical page: ~$0.05-0.15 depending on complexity

Free tier: 1,500 requests/day (check current limits)

## Error Handling

The client handles transient errors with automatic retry:

```python
config = VLMConfig(
    project_id="your-project",
    enable_retry=True,
    max_retries=3,
    retry_delay_seconds=1.0
)
```

**Retry Strategy**:
- Exponential backoff (1s, 2s, 4s, ...)
- Automatic retry on transient failures
- Configurable max attempts

**Error Responses**:
When processing fails, the client returns a VLMResponse with:
- Empty elements list
- `requires_human_review=True`
- Error details in `review_reasons`

## Integration with Existing Pipeline

VLM elements are compatible with existing pipeline components:

```python
# Convert to pipeline format
pipeline_elements = response.to_pipeline_format()

# Each element is a dict with:
# - id: Unique identifier
# - type: Semantic type string
# - page: Page number
# - bbox_px: Bounding box [x0, y0, x1, y1]
# - payload: {text, confidence, ...}
# - column: Column assignment (optional)
# - z_order: Global reading order (optional)

# Compatible with:
# - Column detection (ehrx.layout)
# - Hierarchy generation (ehrx.hierarchy)
# - Serialization (ehrx.serialize)
```

## Testing

Run VLM module tests:

```bash
# All VLM tests
pytest tests/vlm/

# Specific test file
pytest tests/vlm/test_models.py

# With coverage
pytest tests/vlm/ --cov=ehrx.vlm --cov-report=html
```

## Development

### Adding New Element Types

1. Add to `ElementType` enum in `models.py`
2. Update prompt template in `prompts.py`
3. Add examples to tests

### Customizing Prompts

Edit `ehrx/vlm/prompts.py` to modify:
- System instruction
- Element extraction guidance
- Confidence scoring criteria
- Clinical domain hints

### Debugging

Enable raw response saving for troubleshooting:

```python
config = VLMConfig(
    project_id="your-project",
    save_raw_responses=True,
    raw_responses_dir="./debug/vlm_responses"
)
```

## Troubleshooting

### "Could not determine credentials"
- Check `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- Verify credentials file exists and has correct permissions
- See `docs/GCP_SETUP.md` for detailed setup

### "Permission denied" errors
- Ensure service account has "Vertex AI User" role
- Verify Vertex AI API is enabled in GCP project

### High costs
- Review `max_tokens` setting (lower = cheaper)
- Enable caching for repeated requests
- Monitor usage with `client.get_stats()`

### Low confidence scores
- Check image quality (resolution, clarity)
- Verify document type matches clinical EHR format
- Review `save_raw_responses` output for debugging

## Next Steps

This module provides the foundation for VLM integration. Future enhancements:

1. **Flash → Pro Cascade**: Automatically escalate complex elements to Gemini Pro
2. **Cross-chunk Relationships**: Track clinical relationships across pages
3. **Table Structure Extraction**: Specialized table processing
4. **Figure Interpretation**: Medical chart/graph analysis
5. **Continuous Learning**: Feedback loop from human review

See `docs/VLM_REFACTOR.md` for complete roadmap.

## References

- [VLM Refactor PRD](../../docs/VLM_REFACTOR.md)
- [GCP Setup Guide](../../docs/GCP_SETUP.md)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Reference](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini)
