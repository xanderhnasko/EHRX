"""
Tests for VLM prompt templates.

Tests cover prompt generation and context injection.
"""

import pytest

from ehrx.vlm.models import DocumentContext
from ehrx.vlm.prompts import (
    build_element_extraction_prompt,
    build_table_extraction_prompt,
    build_figure_interpretation_prompt,
    build_validation_prompt,
    get_domain_hint,
)


class TestElementExtractionPrompt:
    """Tests for element extraction prompt generation."""

    def test_minimal_context_prompt(self):
        """Test prompt with minimal document context."""
        context = DocumentContext(
            page_number=0,
            total_pages=5
        )
        prompt = build_element_extraction_prompt(context)

        assert "**Page**: 1 of 5" in prompt
        assert "Task" in prompt
        assert "Bounding Box" in prompt
        assert "Semantic Type" in prompt
        assert "Output Format" in prompt

    def test_full_context_prompt(self):
        """Test prompt with complete document context."""
        context = DocumentContext(
            document_type="Clinical Notes",
            section_hierarchy=["Notes", "Progress Notes", "Daily Assessment"],
            patient_context={"age": 65, "gender": "M"},
            page_number=4,
            total_pages=10,
            preceding_summary="Patient admitted for chest pain evaluation..."
        )
        prompt = build_element_extraction_prompt(context)

        assert "**Page**: 5 of 10" in prompt
        assert "Clinical Notes" in prompt
        assert "Notes → Progress Notes → Daily Assessment" in prompt
        assert "Patient Context" in prompt
        assert "Preceding Content" in prompt

    def test_prompt_includes_all_element_types(self):
        """Test that prompt includes all semantic element types."""
        context = DocumentContext(page_number=0, total_pages=1)
        prompt = build_element_extraction_prompt(context)

        # Document structure types
        assert "document_header" in prompt
        assert "patient_demographics" in prompt
        assert "section_header" in prompt

        # Clinical content types
        assert "clinical_paragraph" in prompt
        assert "medication_table" in prompt
        assert "lab_results_table" in prompt
        assert "vital_signs_table" in prompt

        # Special content types
        assert "handwritten_annotation" in prompt
        assert "medical_figure" in prompt

        # Administrative types
        assert "uncategorized" in prompt

    def test_prompt_with_additional_instructions(self):
        """Test prompt with additional custom instructions."""
        context = DocumentContext(page_number=0, total_pages=1)
        additional = "Pay special attention to medication dosages."

        prompt = build_element_extraction_prompt(
            context,
            additional_instructions=additional
        )

        assert additional in prompt

    def test_prompt_includes_confidence_guidance(self):
        """Test that prompt includes confidence scoring guidance."""
        context = DocumentContext(page_number=0, total_pages=1)
        prompt = build_element_extraction_prompt(context)

        assert "Confidence Scores" in prompt
        assert "extraction" in prompt
        assert "classification" in prompt
        assert "clinical_context" in prompt

    def test_prompt_includes_validation_flags_guidance(self):
        """Test that prompt includes validation flag guidance."""
        context = DocumentContext(page_number=0, total_pages=1)
        prompt = build_element_extraction_prompt(context)

        assert "requires_validation" in prompt
        assert "Unclear handwriting" in prompt  # Example validation trigger


class TestTableExtractionPrompt:
    """Tests for table extraction prompt generation."""

    def test_table_prompt_without_bbox(self):
        """Test table extraction prompt without bounding box constraint."""
        prompt = build_table_extraction_prompt()

        assert "Structured Table Extraction" in prompt
        assert "Column Headers" in prompt
        assert "Rows" in prompt
        assert "medication_table" in prompt
        assert "lab_results_table" in prompt

    def test_table_prompt_with_bbox(self):
        """Test table extraction prompt with bounding box constraint."""
        bbox = {"x0": 100.0, "y0": 200.0, "x1": 700.0, "y1": 500.0}
        prompt = build_table_extraction_prompt(table_bbox=bbox)

        assert "x0: 100.0" in prompt
        assert "y0: 200.0" in prompt
        assert "x1: 700.0" in prompt
        assert "y1: 500.0" in prompt

    def test_table_prompt_includes_structure_requirements(self):
        """Test that table prompt includes structural metadata requirements."""
        prompt = build_table_extraction_prompt()

        assert "num_columns" in prompt
        assert "num_rows" in prompt
        assert "has_header" in prompt
        assert "confidence" in prompt


class TestFigureInterpretationPrompt:
    """Tests for figure interpretation prompt generation."""

    def test_figure_prompt_structure(self):
        """Test figure interpretation prompt structure."""
        prompt = build_figure_interpretation_prompt()

        assert "Medical Figure Interpretation" in prompt
        assert "Figure Type" in prompt
        assert "Content Description" in prompt
        assert "Extracted Text" in prompt
        assert "Clinical Interpretation" in prompt

    def test_figure_prompt_includes_figure_types(self):
        """Test that figure prompt includes expected figure types."""
        prompt = build_figure_interpretation_prompt()

        assert "graph" in prompt
        assert "anatomical_diagram" in prompt
        assert "flowchart" in prompt

    def test_figure_prompt_includes_clinical_focus(self):
        """Test that figure prompt emphasizes clinical relevance."""
        prompt = build_figure_interpretation_prompt()

        assert "clinical" in prompt.lower()
        assert "significance" in prompt.lower() or "interpretation" in prompt.lower()


class TestValidationPrompt:
    """Tests for content validation prompt generation."""

    def test_validation_prompt_structure(self):
        """Test validation prompt structure."""
        content = "Metformin 500mg PO BID"
        element_type = "medication_table"

        prompt = build_validation_prompt(content, element_type)

        assert "Content Validation" in prompt
        assert content in prompt
        assert element_type in prompt
        assert "Text Accuracy" in prompt
        assert "Medical Terminology" in prompt

    def test_validation_prompt_includes_actions(self):
        """Test that validation prompt includes recommended actions."""
        prompt = build_validation_prompt("Test content", "clinical_paragraph")

        assert "recommended_action" in prompt
        assert "accept" in prompt
        assert "correct" in prompt
        assert "flag_for_review" in prompt
        assert "reject" in prompt


class TestDomainHints:
    """Tests for clinical domain hints."""

    def test_pharmacology_hint(self):
        """Test pharmacology domain hint."""
        hint = get_domain_hint("pharmacology")
        assert "medication" in hint.lower() or "dosage" in hint.lower()

    def test_laboratory_hint(self):
        """Test laboratory domain hint."""
        hint = get_domain_hint("laboratory")
        assert "test" in hint.lower() or "values" in hint.lower() or "lab" in hint.lower()

    def test_vitals_hint(self):
        """Test vitals domain hint."""
        hint = get_domain_hint("vitals")
        assert "vital" in hint.lower() or "bp" in hint.lower() or "temp" in hint.lower()

    def test_imaging_hint(self):
        """Test imaging domain hint."""
        hint = get_domain_hint("imaging")
        assert "imaging" in hint.lower() or "findings" in hint.lower()

    def test_unknown_domain_returns_empty(self):
        """Test unknown domain returns empty string."""
        hint = get_domain_hint("unknown_domain")
        assert hint == ""

    def test_none_domain_returns_empty(self):
        """Test None domain returns empty string."""
        hint = get_domain_hint(None)
        assert hint == ""

    def test_case_insensitive_domain_matching(self):
        """Test domain hints are case insensitive."""
        hint_lower = get_domain_hint("pharmacology")
        hint_upper = get_domain_hint("PHARMACOLOGY")
        hint_mixed = get_domain_hint("Pharmacology")

        # All should return the same hint (or all should be non-empty)
        assert (hint_lower == hint_upper == hint_mixed) or \
               (bool(hint_lower) and bool(hint_upper) and bool(hint_mixed))
