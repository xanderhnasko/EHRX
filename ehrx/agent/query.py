"""
Hybrid query agent with liberal filtering for EHR schema querying.

Two-stage process:
1. Flash analyzes query → identifies ALL potentially relevant fields (liberal)
2. Pro reasons over filtered context → returns matched elements with provenance
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

from ehrx.vlm.config import VLMConfig


logger = logging.getLogger(__name__)


# All available semantic element types (excluding document_summary which is not queryable)
SEMANTIC_TYPES = [
    "document_header",
    "patient_demographics",
    "page_metadata",
    "section_header",
    "subsection_header",
    "clinical_paragraph",
    "medication_table",
    "lab_results_table",
    "vital_signs_table",
    "recommendations_table",
    "general_table",
    "problem_list",
    "assessment_plan",
    "list_items",
    "handwritten_annotation",
    "stamp_signature",
    "medical_figure",
    "form_field_group",
    "margin_content",
    "uncategorized"
]

# Element types that should be excluded from querying
EXCLUDED_FROM_QUERY = {"document_summary"}

# Reasoning safeguards
MAX_ELEMENTS_PER_BATCH = 120
MAX_TOTAL_CHARS_PER_BATCH = 100_000
MAX_CONTENT_CHARS_PER_ELEMENT = 1500
COMPACT_CONTENT_CHARS_PER_ELEMENT = 600
PRO_MAX_OUTPUT_TOKENS = 4096
PRO_MAX_OUTPUT_TOKENS_COMPACT = 2048
REASONING_MAX_CHARS = 800


class HybridQueryAgent:
    """
    Query agent with liberal filtering + Pro reasoning.

    Architecture:
    1. Flash: Analyze query → identify ALL potentially relevant fields (liberal)
    2. Python: Deterministic filter → extract matching elements
    3. Pro: Reason over filtered context → answer with provenance
    """

    def __init__(
        self,
        schema_path: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        vlm_config: Optional[VLMConfig] = None
    ):
        """
        Initialize hybrid query agent.

        Args:
            schema_path: Path to schema JSON file (can be set later)
            schema: Preloaded schema dictionary (bypass file load)
            vlm_config: VLM configuration (defaults to from_env())
        """
        self.schema_path = schema_path
        self.schema = schema
        self.vlm_config = vlm_config or VLMConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Initialize Vertex AI
        vertexai.init(
            project=self.vlm_config.project_id,
            location=self.vlm_config.location
        )

        # Initialize models
        self.flash_model = GenerativeModel(model_name="gemini-2.5-flash")
        self.pro_model = GenerativeModel(model_name="gemini-2.5-pro")

        # Load schema if provided and not already set
        if schema_path and self.schema is None:
            self.load_schema(schema_path)

    def load_schema(self, schema_path: str) -> None:
        """
        Load schema JSON file.

        Args:
            schema_path: Path to schema JSON file
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        self.logger.info(f"Loading schema from: {schema_path}")

        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

        # Calculate stats
        total_elements = sum(
            len(page.get("elements", []))
            for page in self.schema.get("pages", [])
        )

        self.logger.info(
            f"Schema loaded: {self.schema.get('total_pages', 0)} pages, "
            f"{total_elements} elements"
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer question using hybrid filtering approach.

        Args:
            question: Natural language question

        Returns:
            Dictionary containing:
                - question: Original question
                - matched_elements: List of relevant elements with provenance
                - reasoning: Explanation of answer
                - filter_stats: Statistics about filtering process
        """
        if self.schema is None:
            raise ValueError("Schema not loaded. Call load_schema() first.")

        self.logger.info(f"Processing query: {question}")

        # Stage 1: Liberal field identification (Flash)
        relevant_fields = self._analyze_query_liberal(question)
        self.logger.info(f"Relevant fields identified: {relevant_fields}")

        # Stage 2: Deterministic filtering
        filtered_schema = self._filter_schema(relevant_fields)
        self.logger.info(
            f"Filtered {len(filtered_schema['elements'])} / "
            f"{self._count_total_elements()} elements"
        )

        # Precompute per-page bbox extents to help downstream scaling
        page_bbox_max = {}
        for page in self.schema.get("pages", []):
            page_num = page.get("page_number")
            if page_num is None:
                continue
            max_x = 0
            max_y = 0
            for element in page.get("elements", []):
                bbox = element.get("bbox_pixel") or element.get("bbox_norm") or element.get("bbox")
                if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    max_x = max(max_x, float(bbox[0]), float(bbox[2]))
                    max_y = max(max_y, float(bbox[1]), float(bbox[3]))
            if max_x or max_y:
                page_bbox_max[str(int(page_num))] = {"max_x_px": max_x, "max_y_px": max_y}

        # Stage 3: Reason with Pro
        answer = self._reason_with_pro(question, filtered_schema)

        reasoning_text = answer.get("reasoning", "")
        if reasoning_text and len(reasoning_text) > REASONING_MAX_CHARS:
            reasoning_text = reasoning_text[:REASONING_MAX_CHARS] + "..."

        def _coerce_bbox(bbox_val: Any) -> list:
            if isinstance(bbox_val, list):
                return bbox_val
            if isinstance(bbox_val, dict):
                ordered = [bbox_val.get(k) for k in ("x1", "y1", "x2", "y2") if k in bbox_val]
                if all(v is not None for v in ordered):
                    return ordered
            return []

        matched_elements = []
        for el in answer.get("elements", []):
            bbox_source = el.get("bbox_pixel") or el.get("bbox_norm") or el.get("bbox")
            page_num = el.get("page_number")
            page_key = str(int(page_num)) if page_num is not None else None
            page_scale = page_bbox_max.get(page_key or "")
            matched_elements.append(
                {
                    "element_id": el.get("element_id"),
                    "relevance": el.get("relevance") or el.get("justification"),
                    "text": el.get("content") or el.get("text"),
                    "page": page_num,
                    "bbox": _coerce_bbox(bbox_source),
                    "type": el.get("type"),
                    "subdoc_type": el.get("subdoc_type"),
                    "subdoc_title": el.get("subdoc_title"),
                    "page_key": page_key,
                    "bbox_norm": el.get("bbox_pdf"),
                    "page_bbox_max_x_px": page_scale.get("max_x_px") if page_scale else None,
                    "page_bbox_max_y_px": page_scale.get("max_y_px") if page_scale else None,
                }
            )

        return {
            "question": question,
            "matched_elements": matched_elements,
            "reasoning": reasoning_text,
            "answer_summary": answer.get("answer_summary", ""),
            "filter_stats": {
                "original_elements": self._count_total_elements(),
                "filtered_elements": len(filtered_schema["elements"]),
                "reduction_ratio": f"{self._count_total_elements() / max(len(filtered_schema['elements']), 1):.1f}x"
            }
        }

    def _analyze_query_liberal(self, question: str) -> Dict[str, Any]:
        """
        Use Flash to identify ALL potentially relevant schema fields (liberal).

        Args:
            question: User's natural language question

        Returns:
            Dictionary with relevant types and sub-documents
        """
        prompt = f"""Analyze this user question about an electronic health record (EHR):

QUESTION: {question}

Available semantic element types:
{', '.join(SEMANTIC_TYPES)}

Available sub-document types:
- laboratory_results: Lab tests, blood work, urinalysis
- medications: Medication lists, prescriptions, pharmacy records
- radiology_imaging: X-rays, CT scans, MRIs, ultrasounds
- vital_signs: Temperature, blood pressure, pulse, oxygen saturation
- progress_notes: Clinical notes, H&P, consultation notes
- procedures: Surgical/procedure notes
- orders: Physician orders, nursing orders
- immunizations: Vaccination records
- allergies: Allergy lists, adverse reactions
- problem_list: Diagnoses, active problems
- appointments: Visit summaries, encounters
- referrals: Specialist referrals
- patient_instructions: Discharge instructions, patient education

TASK: Identify ALL semantic types and sub-document types that are relevant to answering this question.
Err on the the slightly liberal side - better to include an extra type or two than to miss important context.

Return your answer as a JSON object with this structure:
{{
    "relevant_types": ["type1", "type2", ...],
    "relevant_subdocs": ["subdoc1", "subdoc2", ...],
    "temporal_context": "recent|historical|all|null",
    "reasoning": "Brief explanation of your selections"
}}

Only return the JSON object, nothing else."""

        # Use structured output to FORCE valid JSON
        generation_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "relevant_types": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "relevant_subdocs": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "temporal_context": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["relevant_types", "relevant_subdocs", "temporal_context", "reasoning"]
            }
        )

        response = self.flash_model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Parse JSON response (guaranteed valid due to response_schema)
        result = json.loads(response.text)

        # Heuristic backstops: ensure key element types are included when the query clearly targets them.
        q_low = question.lower()
        relevant_types = set(result.get("relevant_types", []))
        relevant_subdocs = set(result.get("relevant_subdocs", []))

        def bump(types: list[str], subdocs: list[str] | None = None):
            relevant_types.update(types)
            if subdocs:
                relevant_subdocs.update(subdocs)

        if any(k in q_low for k in ["med", "medication", "drug", "dose", "prescription"]):
            bump(["medication_table", "clinical_paragraph", "list_items"], ["medications", "problem_list"])
        if any(k in q_low for k in ["lab", "result", "cbc", "chem", "glucose", "panel"]):
            bump(["lab_results_table", "general_table", "clinical_paragraph"], ["laboratory_results"])
        if any(k in q_low for k in ["procedure", "surgery", "operation", "biopsy", "colectomy"]):
            bump(["clinical_paragraph", "general_table", "section_header"], ["procedures"])
        if any(k in q_low for k in ["imaging", "radiology", "ct", "mri", "x-ray", "ultrasound"]):
            bump(["clinical_paragraph", "general_table", "section_header"], ["radiology_imaging"])
        if any(k in q_low for k in ["problem", "diagnosis", "history", "hx"]):
            bump(["problem_list", "list_items", "clinical_paragraph"], ["problem_list", "progress_notes"])

        result["relevant_types"] = list(relevant_types)
        result["relevant_subdocs"] = list(relevant_subdocs)

        # Debug logging
        self.logger.info(f"Flash identified {len(result.get('relevant_types', []))} relevant types (after heuristics)")
        self.logger.debug(f"Relevant types: {result.get('relevant_types', [])}")

        return result

    def _filter_schema(self, relevant_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministically filter schema to include ALL matching elements.

        Liberal filtering = include anything that might be relevant.

        Args:
            relevant_fields: Output from _analyze_query_liberal

        Returns:
            Filtered schema with subset of elements
        """
        relevant_types = set(relevant_fields.get("relevant_types", []))
        relevant_subdocs = set(relevant_fields.get("relevant_subdocs", []))

        # Always include these for context
        always_include_types = {"section_header", "patient_demographics"}
        relevant_types.update(always_include_types)

        filtered_elements = []

        # IMPORTANT: Filter by ELEMENT TYPE first, subdoc type is secondary
        # This handles cases where subdoc grouping is over-generalized
        if "sub_documents" in self.schema:
            for subdoc in self.schema.get("sub_documents", []):
                subdoc_type = subdoc.get("type")

                for page in subdoc.get("pages", []):
                    for element in page.get("elements", []):
                        # Skip excluded element types (e.g., document_summary)
                        if element.get("type") in EXCLUDED_FROM_QUERY:
                            continue

                        # PRIMARY: Filter by element type (this is the key!)
                        if element.get("type") in relevant_types:
                            # Add page number and subdoc context
                            element_with_context = {
                                **element,
                                "page_number": page.get("page_number"),
                                "subdoc_type": subdoc_type,
                                "subdoc_title": subdoc.get("title")
                            }
                            filtered_elements.append(element_with_context)
                        # SECONDARY: Also check subdoc type if specified
                        elif relevant_subdocs and subdoc_type in relevant_subdocs:
                            # Include all elements from matching subdoc
                            element_with_context = {
                                **element,
                                "page_number": page.get("page_number"),
                                "subdoc_type": subdoc_type,
                                "subdoc_title": subdoc.get("title")
                            }
                            filtered_elements.append(element_with_context)

        else:
            # No sub-documents, filter pages directly
            for page in self.schema.get("pages", []):
                for element in page.get("elements", []):
                    # Skip excluded element types (e.g., document_summary)
                    if element.get("type") in EXCLUDED_FROM_QUERY:
                        continue

                    if element.get("type") in relevant_types:
                        element_with_context = {
                            **element,
                            "page_number": page.get("page_number")
                        }
                        filtered_elements.append(element_with_context)

        self.logger.info(f"Filter found {len(filtered_elements)} matching elements")
        return {"elements": filtered_elements}

    def _reason_with_pro(
        self,
        question: str,
        filtered_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Gemini Pro to reason over filtered schema and answer question.

        Args:
            question: User's question
            filtered_schema: Filtered schema with relevant elements

        Returns:
            Dictionary with answer and reasoning
        """
        # Build batches to stay under token limits
        batches = self._build_reasoning_batches(filtered_schema.get("elements", []))
        all_elements = []
        reasonings = []
        summaries = []

        for batch in batches:
            result = self._reason_with_pro_batch(question, batch, filtered_schema.get("elements", []))
            all_elements.extend(result.get("elements", []))
            if result.get("reasoning"):
                reasonings.append(result["reasoning"])
            if result.get("answer_summary"):
                summaries.append(result["answer_summary"])

        # Deduplicate by element_id while preserving order
        seen_ids = set()
        deduped_elements = []
        for elem in all_elements:
            eid = elem.get("element_id")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                deduped_elements.append(elem)

        # Choose a single summary to avoid contradictory concatenation across batches.
        # Use the longest non-empty summary to retain the most complete answer without filtering it away.
        final_summary = ""
        if summaries:
            summaries_sorted = sorted(summaries, key=lambda s: len(s or ""), reverse=True)
            final_summary = summaries_sorted[0] if summaries_sorted else ""

        return {
            "elements": deduped_elements,
            "reasoning": "\n".join(reasonings),
            "answer_summary": final_summary
        }

    def _rehydrate_elements(
        self,
        filtered_elements: List[Dict[str, Any]],
        pro_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rehydrate Pro-selected element IDs with full local metadata.

        Args:
            filtered_elements: Elements sent to Pro (with content/bboxes)
            pro_elements: Elements returned by Pro (IDs + short notes)

        Returns:
            List of enriched elements with bbox/content for provenance
        """
        element_lookup = {
            elem.get("element_id"): elem
            for elem in filtered_elements
            if elem.get("element_id")
        }

        hydrated: List[Dict[str, Any]] = []
        missing_ids: List[str] = []

        for elem in pro_elements or []:
            elem_id = elem.get("element_id")
            if not elem_id:
                continue

            base = element_lookup.get(elem_id)
            if not base:
                missing_ids.append(elem_id)
                continue

            hydrated_elem = {**base}
            if "relevance" in elem:
                hydrated_elem["pro_relevance"] = elem["relevance"]
            if "justification" in elem:
                hydrated_elem["pro_justification"] = elem["justification"]

            hydrated.append(hydrated_elem)

        if missing_ids:
            self.logger.warning(
                f"Pro returned element IDs not found in filtered context: {missing_ids}"
            )

        return hydrated

    def _build_reasoning_batches(
        self,
        elements: List[Dict[str, Any]],
        content_char_limit: int = MAX_CONTENT_CHARS_PER_ELEMENT,
        max_total_chars: int = MAX_TOTAL_CHARS_PER_BATCH,
        max_elements: int = MAX_ELEMENTS_PER_BATCH
    ) -> List[List[Dict[str, Any]]]:
        """
        Build batches of elements to keep context under token limits.

        Args:
            elements: Full filtered elements with content/bboxes
            content_char_limit: Trim each element's content to this many chars
            max_total_chars: Max total chars per batch
            max_elements: Max elements per batch
        """
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0

        for elem in elements:
            trimmed_content = (elem.get("content") or "")[:content_char_limit]
            approx_len = len(trimmed_content)
            simplified = {
                "element_id": elem.get("element_id"),
                "type": elem.get("type"),
                "content": trimmed_content,
                "page_number": elem.get("page_number")
            }

            # Decide if we need a new batch
            would_exceed = (
                len(current) >= max_elements or
                current_chars + approx_len > max_total_chars
            )
            if would_exceed and current:
                batches.append(current)
                current = []
                current_chars = 0

            current.append(simplified)
            current_chars += approx_len

        if current:
            batches.append(current)

        return batches

    def _reason_with_pro_batch(
        self,
        question: str,
        batch_elements: List[Dict[str, Any]],
        full_filtered_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run Pro on a single batch with retry using a compact prompt if needed.
        """
        result, truncated = self._invoke_pro(question, batch_elements, compact=False)
        if result is None or truncated:
            # Retry once with compact content
            compact_batch = self._build_reasoning_batches(
                batch_elements,
                content_char_limit=COMPACT_CONTENT_CHARS_PER_ELEMENT,
                max_total_chars=MAX_TOTAL_CHARS_PER_BATCH // 2,
                max_elements=max(20, MAX_ELEMENTS_PER_BATCH // 2)
            )[0]
            result, _ = self._invoke_pro(question, compact_batch, compact=True)

        if not result:
            return {
                "elements": [],
                "reasoning": "Pro reasoning failed to produce a valid response.",
                "answer_summary": ""
            }

        hydrated_elements = self._rehydrate_elements(
            full_filtered_elements,
            result.get("elements", [])
        )

        return {
            "elements": hydrated_elements,
            "reasoning": result.get("reasoning", ""),
            "answer_summary": result.get("answer_summary", "")
        }

    def _invoke_pro(
        self,
        question: str,
        batch_elements: List[Dict[str, Any]],
        compact: bool = False
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Invoke Pro on a batch. Returns (result_dict_or_None, truncated_flag).
        """
        simplified_schema = {"elements": batch_elements}
        schema_json = json.dumps(simplified_schema, indent=2)
        available_ids = [el.get("element_id") for el in batch_elements if el.get("element_id")]

        brevity_instruction = "Keep answer_summary under 80 words. Keep reasoning under 60 words." if compact else "Keep answer_summary concise. Keep reasoning under 100 words."

        prompt = f"""You are analyzing an electronic health record (EHR) to answer a specific question.

Below is a filtered subset of the EHR's structured schema containing potentially relevant elements.

CRITICAL: The actual data is in the "content" field of each element. Read the content field carefully!

Each element has:
- element_id: Unique identifier
- type: Semantic element type (e.g., medication_table, lab_results_table)
- content: THE ACTUAL EXTRACTED TEXT/DATA - THIS IS WHERE THE ANSWER IS!
- page_number: Page where element appears

FILTERED SCHEMA:
{schema_json}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. READ THE \"content\" FIELD of each element - that's where the actual data is!
2. For tables (medication_table, lab_results_table, etc.), the content field contains the full table data
3. Don't just look at element types - READ THE CONTENT!
4. Extract the actual answer from the content and present it clearly
5. OUTPUT MUST BE TINY: Only return element_id(s) for matched items. DO NOT repeat content, bbox, or page info.
6. USE ONLY element_id values from the list below. Do not invent IDs.
6. Keep any relevance note extremely short (<= 10 words).
7. {brevity_instruction}

ELEMENT IDS AVAILABLE (use these IDs exactly):
{available_ids}

TASK: Find ALL elements that answer the question and provide a clear natural language answer.

Return your answer as a JSON object:

{{
    "elements": [
        {{
            "element_id": "E_1234",
            "relevance": "Medication table listing"
        }}
    ],
    "reasoning": "Brief explanation of where the answer was found",
    "answer_summary": "Clear answer extracted from the content (concise)."
}}

If no relevant elements are found, return empty elements list with explanation in reasoning.

Only return the JSON object, nothing else."""

        generation_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=PRO_MAX_OUTPUT_TOKENS_COMPACT if compact else PRO_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "element_id": {"type": "string"},
                                "relevance": {"type": "string"},
                                "justification": {"type": "string"}
                            },
                            "required": ["element_id"]
                        }
                    },
                    "reasoning": {"type": "string"},
                    "answer_summary": {"type": "string"}
                },
                "required": ["elements", "reasoning", "answer_summary"]
            }
        )

        response = self.pro_model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Detect truncation if finish_reason indicates token limit
        finish_reason = self._extract_finish_reason(response)
        truncated = finish_reason and ("MAX_TOKENS" in finish_reason or "LENGTH" in finish_reason)

        # Parse JSON response
        try:
            raw_result = json.loads(getattr(response, "text", ""))
            return raw_result, truncated
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.error(f"Response length: {len(getattr(response, 'text', ''))} chars")
            return None, truncated

    def _extract_finish_reason(self, response: Any) -> Optional[str]:
        """Extract finish_reason from Vertex response if present."""
        if hasattr(response, "finish_reason"):
            return str(response.finish_reason)

        candidate = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

        if candidate and hasattr(candidate, "finish_reason"):
            fr = candidate.finish_reason
            if hasattr(fr, "name"):
                return fr.name
            return str(fr)

        return None

    def _count_total_elements(self) -> int:
        """Count total elements in schema."""
        if "sub_documents" in self.schema:
            return sum(
                len(page.get("elements", []))
                for subdoc in self.schema.get("sub_documents", [])
                for page in subdoc.get("pages", [])
            )
        else:
            return sum(
                len(page.get("elements", []))
                for page in self.schema.get("pages", [])
            )


def run_example_queries(agent: HybridQueryAgent) -> None:
    """
    Run example queries to demonstrate agent capabilities.

    Args:
        agent: Initialized HybridQueryAgent with loaded schema
    """
    example_queries = [
        "What were the patient's blood test results?",
        "What medications is the patient currently taking?",
        "Show me all vital signs readings",
        "What are the patient's active problems or diagnoses?",
        "Were there any radiology or imaging studies?",
        "What were the patient's most recent lab values?",
        "Is there any information about allergies?",
        "What procedures or surgeries were performed?"
    ]

    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES")
    print("=" * 80)

    for i, query in enumerate(example_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)

        try:
            result = agent.query(query)

            print(f"Matched Elements: {len(result['matched_elements'])}")
            print(f"Filter Reduction: {result['filter_stats']['reduction_ratio']}")
            print(f"\nAnswer: {result.get('reasoning', 'No answer')}")

            if result['matched_elements']:
                print(f"\nSample Elements:")
                for elem in result['matched_elements'][:2]:  # Show first 2
                    print(f"  - {elem.get('type')}: {elem.get('content', '')[:100]}...")
                    print(f"    Page {elem.get('page_number')}, Bbox: {elem.get('bbox_pixel')}")

        except Exception as e:
            print(f"Error: {e}")

        print()
