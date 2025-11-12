"""
Deterministic tree builder for hierarchical structure.

This module implements the hierarchical document structuring pipeline:
1. Document label detection (top-middle region of pages)
2. Document grouping (consecutive pages with same label)
3. Section/subsection detection (visual heuristics)
4. Category mapping (to predefined EHR categories)
5. JSON export with hierarchical structure
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Category Definitions
# ============================================================================

@dataclass
class DocumentCategory:
    """Predefined EHR document category."""
    name: str
    subcategories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


# Predefined categories from PRD
CATEGORIES = {
    "Demographics": DocumentCategory(
        name="Demographics",
        keywords=["patient information", "demographics", "patient info", "personal info"]
    ),
    "Vitals": DocumentCategory(
        name="Vitals",
        keywords=["vital signs", "vitals", "blood pressure", "temperature", "pulse", "heart rate"]
    ),
    "Orders": DocumentCategory(
        name="Orders",
        keywords=["orders", "physician orders", "prescriptions", "prescribe"]
    ),
    "Meds": DocumentCategory(
        name="Meds",
        keywords=["medications", "medicines", "drugs", "prescriptions", "pharmacy", "medication list"]
    ),
    "Notes": DocumentCategory(
        name="Notes",
        subcategories=["Visit Summaries", "Discharge Statements", "Progress Notes"],
        keywords=["clinical notes", "progress notes", "visit summary", "discharge", "notes", "assessment"]
    ),
    "Labs": DocumentCategory(
        name="Labs",
        keywords=["lab results", "laboratory", "labs", "test results", "pathology"]
    ),
    "Miscellaneous": DocumentCategory(
        name="Miscellaneous",
        keywords=[]
    )
}


# ============================================================================
# Document Label Detection
# ============================================================================

class DocumentLabelDetector:
    """
    Detects document type labels at the top-middle of pages.
    
    According to PRD assumptions:
    - Each page contains only one document type
    - Document type is indicated by a label at the top-middle of the page
    - Consecutive pages with the same label belong to the same document
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize document label detector.
        
        Args:
            config: Configuration dict or Pydantic model with detection parameters
        """
        self.config = config or {}
        
        # Helper to get config value from dict or Pydantic model
        def get_config_value(key: str, default: Any) -> Any:
            if hasattr(self.config, key):
                return getattr(self.config, key)
            elif isinstance(self.config, dict):
                return self.config.get(key, default)
            else:
                return default
        
        # Detection region (as fraction of page dimensions)
        # Top 15% of page, middle 60% horizontally
        self.top_region = get_config_value("top_region", 0.15)
        self.horizontal_start = get_config_value("horizontal_start", 0.2)
        self.horizontal_end = get_config_value("horizontal_end", 0.8)
        
        # Minimum text block height to be considered a label (pixels)
        self.min_label_height = get_config_value("min_label_height", 15)
        
        # Maximum text block height (avoid capturing large paragraphs)
        self.max_label_height = get_config_value("max_label_height", 80)
        
        # Minimum confidence for label detection
        self.min_label_confidence = get_config_value("min_label_confidence", 0.5)
        
        logger.info(
            f"DocumentLabelDetector initialized: "
            f"top_region={self.top_region}, "
            f"horizontal=[{self.horizontal_start}, {self.horizontal_end}], "
            f"label_height=[{self.min_label_height}, {self.max_label_height}]"
        )
    
    def detect_label(self, elements: List[Dict[str, Any]], page_info: Dict[str, Any]) -> Optional[str]:
        """
        Detect document type label from page elements.
        
        Args:
            elements: List of detected elements (with bbox_px, text, etc.)
            page_info: Page metadata (width_px, height_px)
        
        Returns:
            Detected label text or None if not found
        """
        page_width = page_info.get("width_px", page_info.get("page_width_px", 1700))
        page_height = page_info.get("height_px", page_info.get("page_height_px", 2200))
        
        # Define detection region
        top_boundary = page_height * self.top_region
        left_boundary = page_width * self.horizontal_start
        right_boundary = page_width * self.horizontal_end
        
        logger.debug(
            f"Label detection region: y < {top_boundary:.0f}, "
            f"x in [{left_boundary:.0f}, {right_boundary:.0f}]"
        )
        
        # Find candidate labels in the detection region
        candidates = []
        
        for elem in elements:
            # Only consider text blocks
            if elem.get("type") != "text_block":
                continue
            
            bbox_px = elem.get("bbox_px")
            if not bbox_px or len(bbox_px) < 4:
                continue
            
            x1, y1, x2, y2 = bbox_px
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            height = y2 - y1
            
            # Check if element is in the detection region
            in_top_region = center_y < top_boundary
            in_horizontal_range = left_boundary <= center_x <= right_boundary
            valid_height = self.min_label_height <= height <= self.max_label_height
            
            if in_top_region and in_horizontal_range and valid_height:
                text = elem.get("payload", {}).get("text", "").strip()
                if text:
                    candidates.append({
                        "text": text,
                        "y_position": center_y,
                        "height": height,
                        "confidence": elem.get("payload", {}).get("confidence", 0.0),
                        "element": elem
                    })
                    logger.debug(f"Label candidate: '{text}' at y={center_y:.0f}, h={height:.0f}")
        
        if not candidates:
            logger.warning("No document label candidates found in top-middle region")
            return None
        
        # Select the best candidate (highest in page, with reasonable confidence)
        # Sort by y-position (topmost first)
        candidates.sort(key=lambda c: c["y_position"])
        
        # Filter by confidence threshold
        confident_candidates = [c for c in candidates if c["confidence"] >= self.min_label_confidence]
        
        if confident_candidates:
            best_candidate = confident_candidates[0]
        else:
            # Fall back to best positioned candidate even if low confidence
            logger.warning("No high-confidence label candidates, using topmost")
            best_candidate = candidates[0]
        
        label_text = best_candidate["text"]
        logger.info(f"Detected document label: '{label_text}' (conf={best_candidate['confidence']:.2f})")
        
        return label_text
    
    def get_detection_region(self, page_info: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        Get the detection region boundaries for visualization.
        
        Args:
            page_info: Page metadata
        
        Returns:
            Tuple of (x1, y1, x2, y2) in pixels
        """
        page_width = page_info.get("width_px", page_info.get("page_width_px", 1700))
        page_height = page_info.get("height_px", page_info.get("page_height_px", 2200))
        
        x1 = page_width * self.horizontal_start
        y1 = 0
        x2 = page_width * self.horizontal_end
        y2 = page_height * self.top_region
        
        return (x1, y1, x2, y2)


# ============================================================================
# Document Grouping
# ============================================================================

@dataclass
class DocumentGroup:
    """Represents a grouped document (potentially multi-page)."""
    document_type: str
    page_range: Tuple[int, int]  # (start_page, end_page) inclusive
    pages: List[int] = field(default_factory=list)
    elements: List[Dict[str, Any]] = field(default_factory=list)
    category: Optional[str] = None
    extra_labels: List[str] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)


class DocumentGrouper:
    """
    Groups consecutive pages with the same document label.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize document grouper."""
        self.config = config or {}
        logger.info("DocumentGrouper initialized")
    
    def group_pages(
        self, 
        page_labels: List[Tuple[int, Optional[str]]]
    ) -> List[DocumentGroup]:
        """
        Group consecutive pages with identical labels.
        
        Args:
            page_labels: List of (page_num, label) tuples
        
        Returns:
            List of DocumentGroup objects
        """
        if not page_labels:
            return []
        
        groups = []
        current_label = None
        current_pages = []
        
        for page_num, label in page_labels:
            # Normalize label (None becomes "Unlabeled")
            normalized_label = label if label else "Unlabeled"
            
            if current_label is None:
                # Start first group
                current_label = normalized_label
                current_pages = [page_num]
            elif normalized_label == current_label:
                # Continue current group
                current_pages.append(page_num)
            else:
                # Label changed - finalize current group and start new one
                if current_pages:
                    group = DocumentGroup(
                        document_type=current_label,
                        page_range=(current_pages[0], current_pages[-1]),
                        pages=current_pages
                    )
                    groups.append(group)
                    logger.debug(
                        f"Created document group: '{current_label}' "
                        f"pages {current_pages[0]}-{current_pages[-1]}"
                    )
                
                # Start new group
                current_label = normalized_label
                current_pages = [page_num]
        
        # Finalize last group
        if current_pages:
            group = DocumentGroup(
                document_type=current_label,
                page_range=(current_pages[0], current_pages[-1]),
                pages=current_pages
            )
            groups.append(group)
            logger.debug(
                f"Created document group: '{current_label}' "
                f"pages {current_pages[0]}-{current_pages[-1]}"
            )
        
        logger.info(f"Grouped {len(page_labels)} pages into {len(groups)} documents")
        return groups


# ============================================================================
# Section Detection
# ============================================================================

class SectionDetector:
    """
    Detects sections and subsections within documents using visual heuristics.
    
    Heuristics:
    - Text block height (proxy for font size)
    - Content patterns (all caps, title case, short lines)
    - Position and spacing
    - Keyword matching
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize section detector."""
        self.config = config or {}
        
        # Helper to get config value from dict or Pydantic model
        def get_config_value(key: str, default: Any) -> Any:
            if hasattr(self.config, key):
                return getattr(self.config, key)
            elif isinstance(self.config, dict):
                return self.config.get(key, default)
            else:
                return default
        
        # Heading detection parameters
        self.min_heading_height = get_config_value("min_heading_height", 20)
        self.caps_ratio_threshold = get_config_value("caps_ratio_min", 0.6)
        self.gap_above_threshold = get_config_value("gap_above_px", 18)
        
        # Heading keywords from config
        heading_patterns = get_config_value("heading_regex", [])
        self.heading_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in heading_patterns]
        
        logger.info(
            f"SectionDetector initialized: "
            f"min_height={self.min_heading_height}, "
            f"caps_ratio={self.caps_ratio_threshold}, "
            f"gap_above={self.gap_above_threshold}"
        )
    
    def detect_sections(
        self, 
        elements: List[Dict[str, Any]], 
        page_nums: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Detect sections and subsections from document elements.
        
        Args:
            elements: All elements in the document (sorted by page, z_order)
            page_nums: Page numbers in this document
        
        Returns:
            List of section dictionaries with hierarchical structure
        """
        if not elements:
            return []
        
        # Identify heading candidates
        headings = []
        
        for i, elem in enumerate(elements):
            if elem.get("type") != "text_block":
                continue
            
            text = elem.get("payload", {}).get("text", "").strip()
            if not text:
                continue
            
            # Calculate heuristic scores
            is_heading, heading_level, scores = self._classify_heading(elem, elements, i)
            
            if is_heading:
                headings.append({
                    "element": elem,
                    "text": text,
                    "level": heading_level,
                    "index": i,
                    "scores": scores
                })
                logger.debug(f"Detected heading (L{heading_level}): '{text}'")
        
        # Build hierarchical structure
        sections = self._build_section_hierarchy(headings, elements)
        
        logger.info(f"Detected {len(sections)} top-level sections")
        return sections
    
    def _classify_heading(
        self, 
        elem: Dict[str, Any], 
        all_elements: List[Dict[str, Any]], 
        index: int
    ) -> Tuple[bool, int, Dict[str, float]]:
        """
        Classify if element is a heading and determine its level.
        
        Returns:
            Tuple of (is_heading, level, scores_dict)
        """
        text = elem.get("payload", {}).get("text", "").strip()
        bbox_px = elem.get("bbox_px", [0, 0, 0, 0])
        
        if len(bbox_px) < 4:
            return False, 0, {}
        
        x1, y1, x2, y2 = bbox_px
        height = y2 - y1
        width = x2 - x1
        
        # Score components
        scores = {}
        
        # 1. Height score (taller = more likely heading)
        scores["height"] = height / self.min_heading_height if height > 0 else 0
        
        # 2. Capitalization score
        if text:
            caps_count = sum(1 for c in text if c.isupper())
            alpha_count = sum(1 for c in text if c.isalpha())
            scores["caps_ratio"] = caps_count / alpha_count if alpha_count > 0 else 0
        else:
            scores["caps_ratio"] = 0
        
        # 3. Brevity score (shorter lines more likely to be headings)
        avg_height = np.mean([e["bbox_px"][3] - e["bbox_px"][1] for e in all_elements if len(e.get("bbox_px", [])) == 4])
        if width > 0 and avg_height > 0:
            aspect_ratio = width / height
            scores["brevity"] = 1.0 / (1.0 + aspect_ratio / 10.0)  # Normalize
        else:
            scores["brevity"] = 0
        
        # 4. Keyword matching
        scores["keyword_match"] = 0
        for pattern in self.heading_patterns:
            if pattern.search(text):
                scores["keyword_match"] = 1.0
                break
        
        # 5. Gap above (spacing before this element)
        if index > 0:
            prev_elem = all_elements[index - 1]
            if prev_elem.get("page") == elem.get("page"):
                prev_bbox = prev_elem.get("bbox_px", [0, 0, 0, 0])
                if len(prev_bbox) == 4:
                    gap = y1 - prev_bbox[3]
                    scores["gap_above"] = gap / self.gap_above_threshold if gap > 0 else 0
                else:
                    scores["gap_above"] = 0
            else:
                scores["gap_above"] = 1.0  # First element on page
        else:
            scores["gap_above"] = 1.0  # First element in document
        
        # Decision logic
        is_heading = False
        level = 0
        
        # Strong heading indicators
        if scores.get("keyword_match", 0) > 0 and scores.get("height", 0) >= 1.0:
            is_heading = True
            level = 1  # H1 - strong keyword + large text
        elif scores.get("caps_ratio", 0) >= self.caps_ratio_threshold and scores.get("height", 0) >= 0.8:
            is_heading = True
            level = 1 if scores.get("gap_above", 0) >= 1.0 else 2
        elif scores.get("keyword_match", 0) > 0:
            is_heading = True
            level = 2  # H2 - keyword match only
        elif scores.get("height", 0) >= 1.2 and scores.get("brevity", 0) >= 0.5:
            is_heading = True
            level = 2  # H2 - tall + brief
        
        return is_heading, level, scores
    
    def _build_section_hierarchy(
        self, 
        headings: List[Dict[str, Any]], 
        all_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build hierarchical section structure from detected headings.
        
        Args:
            headings: List of detected heading dictionaries
            all_elements: All elements in document
        
        Returns:
            List of top-level sections with nested children
        """
        if not headings:
            # No headings detected - return flat structure
            return [{
                "heading": "Content",
                "level": 1,
                "children": all_elements
            }]
        
        sections = []
        current_h1 = None
        current_h2 = None
        
        for i, heading_info in enumerate(headings):
            heading_elem = heading_info["element"]
            heading_text = heading_info["text"]
            heading_level = heading_info["level"]
            heading_index = heading_info["index"]
            
            # Determine content range (elements between this heading and next)
            if i + 1 < len(headings):
                next_heading_index = headings[i + 1]["index"]
                content_elements = all_elements[heading_index + 1:next_heading_index]
            else:
                # Last heading - include all remaining elements
                content_elements = all_elements[heading_index + 1:]
            
            section_dict = {
                "heading": heading_text,
                "level": heading_level,
                "heading_element": heading_elem,
                "children": content_elements,
                "page": heading_elem.get("page"),
                "bbox_px": heading_elem.get("bbox_px")
            }
            
            # Build hierarchy
            if heading_level == 1:
                # Top-level section
                sections.append(section_dict)
                current_h1 = section_dict
                current_h2 = None
            elif heading_level == 2:
                # Subsection
                if current_h1:
                    if "sections" not in current_h1:
                        current_h1["sections"] = []
                    current_h1["sections"].append(section_dict)
                    current_h2 = section_dict
                else:
                    # No H1 parent - treat as top-level
                    sections.append(section_dict)
            else:
                # Level 3+ - attach to H2 or H1
                if current_h2:
                    if "sections" not in current_h2:
                        current_h2["sections"] = []
                    current_h2["sections"].append(section_dict)
                elif current_h1:
                    if "sections" not in current_h1:
                        current_h1["sections"] = []
                    current_h1["sections"].append(section_dict)
                else:
                    sections.append(section_dict)
        
        return sections


# ============================================================================
# Category Mapper
# ============================================================================

class CategoryMapper:
    """
    Maps document labels to predefined EHR categories.
    """
    
    def __init__(self, categories: Optional[Dict[str, DocumentCategory]] = None):
        """Initialize category mapper."""
        self.categories = categories or CATEGORIES
        logger.info(f"CategoryMapper initialized with {len(self.categories)} categories")
    
    def map_to_category(self, document_type: str) -> str:
        """
        Map document type label to category.
        
        Args:
            document_type: Document type label
        
        Returns:
            Category name
        """
        if not document_type or document_type == "Unlabeled":
            return "Miscellaneous"
        
        # Normalize for comparison
        doc_type_lower = document_type.lower()
        
        # Check each category's keywords
        for category_name, category in self.categories.items():
            if category_name == "Miscellaneous":
                continue  # Skip default category
            
            for keyword in category.keywords:
                if keyword.lower() in doc_type_lower:
                    logger.debug(f"Mapped '{document_type}' -> '{category_name}' (keyword: '{keyword}')")
                    return category_name
        
        # No match - use Miscellaneous
        logger.debug(f"Mapped '{document_type}' -> 'Miscellaneous' (no keyword match)")
        return "Miscellaneous"


# ============================================================================
# Main Hierarchy Builder
# ============================================================================

class HierarchyBuilder:
    """
    Main orchestrator for hierarchical document structuring.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize hierarchy builder."""
        self.config = config or {}
        
        # Helper to get config value from dict or Pydantic model
        def get_config_value(key: str, default: Any) -> Any:
            if hasattr(self.config, key):
                return getattr(self.config, key)
            elif isinstance(self.config, dict):
                return self.config.get(key, default)
            else:
                return default
        
        # Initialize sub-components
        self.label_detector = DocumentLabelDetector(get_config_value("label_detection", {}))
        self.document_grouper = DocumentGrouper(get_config_value("document_grouping", {}))
        self.section_detector = SectionDetector(get_config_value("hierarchy", {}))
        self.category_mapper = CategoryMapper()
        
        logger.info("HierarchyBuilder initialized")
    
    def build_hierarchy(
        self, 
        pages_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build hierarchical structure from page data.
        
        Args:
            pages_data: List of page dictionaries, each containing:
                - page_num: Page number
                - elements: List of elements on the page
                - page_info: Page metadata (width_px, height_px, etc.)
        
        Returns:
            Hierarchical structure dictionary
        """
        logger.info(f"Building hierarchy for {len(pages_data)} pages")
        
        # Step 1: Detect document labels for each page
        page_labels = []
        for page_data in pages_data:
            page_num = page_data["page_num"]
            elements = page_data["elements"]
            page_info = page_data["page_info"]
            
            label = self.label_detector.detect_label(elements, page_info)
            page_labels.append((page_num, label))
            logger.debug(f"Page {page_num}: label='{label}'")
        
        # Step 2: Group consecutive pages with same label
        document_groups = self.document_grouper.group_pages(page_labels)
        
        # Step 3: For each document group, detect sections and map category
        for doc_group in document_groups:
            # Collect all elements for this document
            doc_elements = []
            for page_data in pages_data:
                if page_data["page_num"] in doc_group.pages:
                    doc_elements.extend(page_data["elements"])
            
            doc_group.elements = doc_elements
            
            # Detect sections
            doc_group.sections = self.section_detector.detect_sections(
                doc_elements, 
                doc_group.pages
            )
            
            # Map to category
            doc_group.category = self.category_mapper.map_to_category(doc_group.document_type)
            
            logger.info(
                f"Document '{doc_group.document_type}' "
                f"(pages {doc_group.page_range[0]}-{doc_group.page_range[1]}): "
                f"category='{doc_group.category}', "
                f"sections={len(doc_group.sections)}"
            )
        
        # Step 4: Build final hierarchy structure
        hierarchy = self._build_output_structure(document_groups)
        
        return hierarchy
    
    def _build_output_structure(self, document_groups: List[DocumentGroup]) -> Dict[str, Any]:
        """
        Build final hierarchical output structure.
        
        Args:
            document_groups: List of DocumentGroup objects
        
        Returns:
            Hierarchical structure: categories -> subcategories (if applicable) -> documents -> pages -> elements
        """
        # Group documents by category and subcategory
        by_category = defaultdict(lambda: defaultdict(list))
        
        for doc_group in document_groups:
            category = doc_group.category
            # Determine subcategory from document type for "Notes" category
            subcategory = self._get_subcategory(doc_group.document_type, category)
            
            by_category[category][subcategory].append(doc_group)
        
        # Build hierarchical output
        categories_output = {}
        
        for category_name, subcategories in by_category.items():
            category_data = {}
            
            for subcategory_name, doc_groups_list in subcategories.items():
                documents_list = []
                
                for doc_group in doc_groups_list:
                    # Group elements by page
                    pages_dict = defaultdict(list)
                    for element in doc_group.elements:
                        page_num = element.get("page", 0)
                        # Clean element - keep only essential fields
                        clean_element = self._clean_element(element)
                        pages_dict[page_num].append(clean_element)
                    
                    # Build pages array
                    pages_array = []
                    for page_num in sorted(pages_dict.keys()):
                        pages_array.append({
                            "page_num": page_num,
                            "elements": pages_dict[page_num]
                        })
                    
                    doc_dict = {
                        "document_name": doc_group.document_type,
                        "page_range": list(doc_group.page_range),
                        "pages": pages_array
                    }
                    documents_list.append(doc_dict)
                
                # Add to category structure
                if subcategory_name:
                    # Has subcategory
                    if subcategory_name not in category_data:
                        category_data[subcategory_name] = []
                    category_data[subcategory_name].extend(documents_list)
                else:
                    # No subcategory - add documents directly to category
                    if "documents" not in category_data:
                        category_data["documents"] = []
                    category_data["documents"].extend(documents_list)
            
            categories_output[category_name] = category_data
        
        return {
            "categories": categories_output,
            "total_documents": len(document_groups),
            "total_pages": sum(len(doc.pages) for doc in document_groups)
        }
    
    def _get_subcategory(self, document_type: str, category: str) -> Optional[str]:
        """
        Determine subcategory for a document based on its type and category.
        
        Args:
            document_type: Document type label
            category: Category name
        
        Returns:
            Subcategory name or None
        """
        if category != "Notes":
            return None
        
        # Check for Notes subcategories
        doc_type_lower = document_type.lower()
        
        if any(kw in doc_type_lower for kw in ["visit", "summary", "encounter"]):
            return "Visit Summaries"
        elif any(kw in doc_type_lower for kw in ["discharge", "discharge summary"]):
            return "Discharge Statements"
        elif any(kw in doc_type_lower for kw in ["progress", "progress note"]):
            return "Progress Notes"
        
        # Default: no specific subcategory
        return None
    
    def _clean_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean element to keep only essential fields.
        
        Args:
            element: Raw element dictionary
        
        Returns:
            Cleaned element with only: id, type, text, bbox_px, bbox_pdf, page, confidence
        """
        # Extract text from payload
        text = ""
        confidence = 0.0
        
        payload = element.get("payload", {})
        if isinstance(payload, dict):
            text = payload.get("text", "")
            confidence = payload.get("confidence", 0.0)
            
            # For tables, get OCR lines
            if element.get("type") == "table":
                ocr_lines = payload.get("ocr_lines", [])
                if ocr_lines:
                    text = "\n".join(ocr_lines)
        
        # Handle confidence value
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        
        return {
            "id": element.get("id", ""),
            "type": element.get("type", ""),
            "text": text,
            "bbox_px": element.get("bbox_px", []),
            "bbox_pdf": element.get("bbox_pdf", []),
            "page": element.get("page", 0),
            "confidence": float(confidence)
        }
    

