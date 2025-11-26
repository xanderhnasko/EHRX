import json
from types import SimpleNamespace

import pytest

import ehrx.agent.query as query_mod
from ehrx.agent.query import HybridQueryAgent


@pytest.fixture
def sample_schema(tmp_path):
    """Create a minimal schema file with one medication element."""
    schema = {
        "document_id": "doc_123",
        "total_pages": 1,
        "sub_documents": [
            {
                "id": "subdoc_001",
                "type": "medications",
                "title": "MEDICATIONS",
                "page_range": [1, 1],
                "page_count": 1,
                "pages": [
                    {
                        "page_number": 1,
                        "elements": [
                            {
                                "element_id": "E_0001",
                                "type": "medication_table",
                                "content": "Aspirin 81mg PO daily",
                                "bbox_pixel": [10, 20, 30, 40],
                                "bbox_pdf": [1.0, 2.0, 3.0, 4.0]
                            }
                        ]
                    }
                ],
                "element_count": 1,
                "confidence": 0.95
            }
        ],
        "processing_stats": {
            "total_elements": 1
        }
    }

    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema))
    return path, schema


def test_query_rehydrates_elements(monkeypatch, sample_schema):
    """Pro returns only IDs; agent rehydrates content/bboxes locally."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    schema_path, _ = sample_schema
    agent = HybridQueryAgent(schema_path=str(schema_path))

    # Stub Flash liberal analysis to always include medication_table
    flash_response = SimpleNamespace(text=json.dumps({
        "relevant_types": ["medication_table"],
        "relevant_subdocs": [],
        "temporal_context": "all",
        "reasoning": "meds question"
    }))
    monkeypatch.setattr(agent.flash_model, "generate_content", lambda *_, **__: flash_response)

    # Stub Pro to return only element IDs
    pro_response = SimpleNamespace(text=json.dumps({
        "elements": [
            {"element_id": "E_0001", "relevance": "Medication list"}
        ],
        "reasoning": "Selected the medication table",
        "answer_summary": "Patient takes Aspirin 81mg daily."
    }))
    monkeypatch.setattr(agent.pro_model, "generate_content", lambda *_, **__: pro_response)

    result = agent.query("What medications is the patient taking?")

    assert result["answer_summary"].startswith("Patient takes Aspirin")
    assert len(result["matched_elements"]) == 1
    elem = result["matched_elements"][0]
    assert elem["element_id"] == "E_0001"
    assert elem["content"] == "Aspirin 81mg PO daily"
    assert elem["bbox_pixel"] == [10, 20, 30, 40]
    assert elem["page_number"] == 1
    assert elem["pro_relevance"] == "Medication list"


def test_query_retries_on_truncated(monkeypatch, sample_schema):
    """If Pro truncates/returns bad JSON, we retry with compact prompt."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    schema_path, _ = sample_schema
    agent = HybridQueryAgent(schema_path=str(schema_path))

    flash_response = SimpleNamespace(text=json.dumps({
        "relevant_types": ["medication_table"],
        "relevant_subdocs": [],
        "temporal_context": "all",
        "reasoning": "meds question"
    }))
    monkeypatch.setattr(agent.flash_model, "generate_content", lambda *_, **__: flash_response)

    calls = {"count": 0}

    def pro_generate_content(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            # Truncated / invalid JSON
            return SimpleNamespace(text="{bad json", finish_reason="MAX_TOKENS")
        # Successful fallback
        return SimpleNamespace(text=json.dumps({
            "elements": [{"element_id": "E_0001"}],
            "reasoning": "Compact retry",
            "answer_summary": "Aspirin 81mg daily."
        }))

    monkeypatch.setattr(agent.pro_model, "generate_content", pro_generate_content)

    result = agent.query("What medications is the patient taking?")

    assert calls["count"] == 2
    assert result["matched_elements"][0]["element_id"] == "E_0001"
    assert "Aspirin" in result["answer_summary"]


def test_reasoning_is_trimmed(monkeypatch, sample_schema):
    """Reasoning text returned to caller is trimmed to a safe length."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    schema_path, _ = sample_schema
    agent = HybridQueryAgent(schema_path=str(schema_path))

    flash_response = SimpleNamespace(text=json.dumps({
        "relevant_types": ["medication_table"],
        "relevant_subdocs": [],
        "temporal_context": "all",
        "reasoning": "meds question"
    }))
    monkeypatch.setattr(agent.flash_model, "generate_content", lambda *_, **__: flash_response)

    long_reasoning = "R" * (query_mod.REASONING_MAX_CHARS + 200)
    pro_response = SimpleNamespace(text=json.dumps({
        "elements": [{"element_id": "E_0001"}],
        "reasoning": long_reasoning,
        "answer_summary": "ok"
    }))
    monkeypatch.setattr(agent.pro_model, "generate_content", lambda *_, **__: pro_response)

    result = agent.query("What medications is the patient taking?")
    assert len(result["reasoning"]) <= query_mod.REASONING_MAX_CHARS + 3  # allow ellipsis


def test_batches_split_when_large(monkeypatch, tmp_path):
    """Ensure multiple batches are sent when the filtered set is large."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    # Create a schema with many elements to force batching
    elements = []
    for i in range(5):
        elements.append({
            "element_id": f"E_{i:04d}",
            "type": "medication_table",
            "content": f"Medication {i}",
            "bbox_pixel": [1, 2, 3, 4],
            "bbox_pdf": [0.1, 0.2, 0.3, 0.4]
        })

    schema = {
        "document_id": "doc_batch",
        "total_pages": 1,
        "sub_documents": [
            {
                "id": "subdoc_001",
                "type": "medications",
                "title": "MEDICATIONS",
                "page_range": [1, 1],
                "page_count": 1,
                "pages": [{"page_number": 1, "elements": elements}],
                "element_count": len(elements),
                "confidence": 0.95
            }
        ],
        "processing_stats": {"total_elements": len(elements)}
    }
    schema_path = tmp_path / "schema_batch.json"
    schema_path.write_text(json.dumps(schema))

    agent = HybridQueryAgent(schema_path=str(schema_path))

    flash_response = SimpleNamespace(text=json.dumps({
        "relevant_types": ["medication_table"],
        "relevant_subdocs": [],
        "temporal_context": "all",
        "reasoning": "meds question"
    }))
    monkeypatch.setattr(agent.flash_model, "generate_content", lambda *_, **__: flash_response)

    # Force very small batches by overriding the batch builder
    monkeypatch.setattr(
        agent,
        "_build_reasoning_batches",
        lambda elements, **kwargs: [elements[i:i + 2] for i in range(0, len(elements), 2)],
    )

    calls = {"count": 0}

    def fake_invoke(question, batch, compact=False):
        calls["count"] += 1
        return {
            "elements": [{"element_id": e["element_id"]} for e in batch],
            "reasoning": f"batch {calls['count']}",
            "answer_summary": f"summary {calls['count']}"
        }, False

    monkeypatch.setattr(agent, "_invoke_pro", fake_invoke)

    result = agent.query("What medications is the patient taking?")

    assert calls["count"] >= 3  # 5 elements with batch size 2 => 3 batches
    assert len(result["matched_elements"]) == len(elements)
