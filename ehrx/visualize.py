"""
Visual debug output for hierarchical structuring pipeline.

Generates annotated images showing:
- Document label detection regions
- Detected document labels
- Section headings with levels
- Element bounding boxes
- Category assignments
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Color scheme for visualization
COLORS = {
    "label_region": (255, 200, 100),  # Light blue for detection region
    "label_box": (0, 165, 255),  # Orange for detected label
    "text_block": (0, 255, 0),  # Green for text blocks
    "table": (255, 0, 0),  # Blue for tables
    "figure": (255, 0, 255),  # Magenta for figures
    "heading_1": (0, 0, 255),  # Red for H1
    "heading_2": (0, 140, 255),  # Dark orange for H2
    "heading_3": (0, 255, 255),  # Yellow for H3
    "section_box": (200, 200, 200),  # Light gray for section boundaries
}


class HierarchyVisualizer:
    """
    Creates visual debug output for hierarchy detection pipeline.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for output images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"HierarchyVisualizer initialized: {self.debug_dir}")
    
    def visualize_label_detection(
        self,
        page_image: np.ndarray,
        page_num: int,
        elements: List[Dict[str, Any]],
        detection_region: Tuple[float, float, float, float],
        detected_label: Optional[str],
        label_element: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Visualize document label detection on a page.
        
        Args:
            page_image: Page image (numpy array)
            page_num: Page number
            elements: All elements on page
            detection_region: (x1, y1, x2, y2) of detection region
            detected_label: Detected label text (or None)
            label_element: Element that was selected as label
        """
        # Create copy for annotation
        vis_image = page_image.copy()
        
        # Draw detection region (semi-transparent rectangle)
        x1, y1, x2, y2 = [int(coord) for coord in detection_region]
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS["label_region"], -1)
        cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)
        
        # Draw region boundary
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), COLORS["label_region"], 2)
        cv2.putText(
            vis_image,
            "Label Detection Region",
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLORS["label_region"],
            2
        )
        
        # Draw all text blocks in detection region
        for elem in elements:
            if elem.get("type") != "text_block":
                continue
            
            bbox_px = elem.get("bbox_px")
            if not bbox_px or len(bbox_px) < 4:
                continue
            
            bx1, by1, bx2, by2 = [int(c) for c in bbox_px]
            center_x = (bx1 + bx2) / 2
            center_y = (by1 + by2) / 2
            
            # Check if in detection region
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                # Draw with thin green box (candidates)
                cv2.rectangle(vis_image, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
        
        # Highlight detected label
        if label_element:
            bbox_px = label_element.get("bbox_px")
            if bbox_px and len(bbox_px) == 4:
                bx1, by1, bx2, by2 = [int(c) for c in bbox_px]
                cv2.rectangle(vis_image, (bx1, by1), (bx2, by2), COLORS["label_box"], 3)
                
                # Add label text annotation
                if detected_label:
                    label_y = by1 - 10 if by1 > 30 else by2 + 25
                    cv2.putText(
                        vis_image,
                        f"LABEL: {detected_label}",
                        (bx1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        COLORS["label_box"],
                        2
                    )
        
        # Add page header
        header_text = f"Page {page_num} - Label: {detected_label or 'NOT DETECTED'}"
        cv2.putText(
            vis_image,
            header_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3
        )
        cv2.putText(
            vis_image,
            header_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Save
        output_path = self.debug_dir / f"page_{page_num:04d}_label_detection.png"
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Saved label detection visualization: {output_path.name}")
    
    def visualize_sections(
        self,
        page_image: np.ndarray,
        page_num: int,
        elements: List[Dict[str, Any]],
        headings: List[Dict[str, Any]],
        document_type: Optional[str] = None
    ) -> None:
        """
        Visualize detected sections and headings.
        
        Args:
            page_image: Page image
            page_num: Page number
            elements: All elements on page
            headings: Detected heading elements with levels
            document_type: Document type label
        """
        vis_image = page_image.copy()
        
        # Draw all elements first
        for elem in elements:
            bbox_px = elem.get("bbox_px")
            if not bbox_px or len(bbox_px) < 4:
                continue
            
            x1, y1, x2, y2 = [int(c) for c in bbox_px]
            elem_type = elem.get("type", "text_block")
            
            # Color by type
            if elem_type == "text_block":
                color = COLORS["text_block"]
                thickness = 1
            elif elem_type == "table":
                color = COLORS["table"]
                thickness = 2
            elif elem_type == "figure":
                color = COLORS["figure"]
                thickness = 2
            else:
                color = (128, 128, 128)
                thickness = 1
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw headings on top
        for heading in headings:
            elem = heading["element"]
            if elem.get("page") != page_num:
                continue
            
            bbox_px = elem.get("bbox_px")
            if not bbox_px or len(bbox_px) < 4:
                continue
            
            x1, y1, x2, y2 = [int(c) for c in bbox_px]
            level = heading["level"]
            text = heading["text"]
            
            # Color by level
            if level == 1:
                color = COLORS["heading_1"]
                label_prefix = "H1"
            elif level == 2:
                color = COLORS["heading_2"]
                label_prefix = "H2"
            else:
                color = COLORS["heading_3"]
                label_prefix = "H3"
            
            # Draw thick box for heading
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
            
            # Add heading annotation
            annotation = f"{label_prefix}: {text[:50]}{'...' if len(text) > 50 else ''}"
            label_y = y1 - 10 if y1 > 30 else y2 + 20
            
            # Background rectangle for text
            (text_w, text_h), _ = cv2.getTextSize(
                annotation, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis_image,
                (x1, label_y - text_h - 5),
                (x1 + text_w + 10, label_y + 5),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                vis_image,
                annotation,
                (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Add page header
        header_text = f"Page {page_num} - Sections"
        if document_type:
            header_text += f" ({document_type})"
        
        cv2.putText(
            vis_image,
            header_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3
        )
        cv2.putText(
            vis_image,
            header_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Add legend
        legend_y = 80
        legend_items = [
            ("H1 (Section)", COLORS["heading_1"]),
            ("H2 (Subsection)", COLORS["heading_2"]),
            ("Text", COLORS["text_block"]),
            ("Table", COLORS["table"]),
        ]
        
        for label, color in legend_items:
            cv2.rectangle(
                vis_image,
                (20, legend_y),
                (50, legend_y + 20),
                color,
                -1
            )
            cv2.putText(
                vis_image,
                label,
                (60, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            legend_y += 30
        
        # Save
        output_path = self.debug_dir / f"page_{page_num:04d}_sections.png"
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Saved section visualization: {output_path.name}")
    
    def create_document_summary(
        self,
        hierarchy: Dict[str, Any],
        doc_id: str
    ) -> None:
        """
        Create a text summary of the hierarchical structure.
        
        Args:
            hierarchy: Hierarchical structure dictionary
            doc_id: Document ID
        """
        summary_path = self.debug_dir / f"{doc_id}_hierarchy_summary.txt"
        
        with open(summary_path, "w") as f:
            f.write(f"Hierarchical Structure Summary\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Document ID: {doc_id}\n")
            f.write(f"Total Documents: {hierarchy.get('total_documents', 0)}\n")
            f.write(f"Total Pages: {hierarchy.get('total_pages', 0)}\n")
            f.write(f"Categories: {', '.join(hierarchy.get('categories', []))}\n")
            f.write(f"\n{'=' * 80}\n\n")
            
            # Document details
            for doc in hierarchy.get("documents", []):
                f.write(f"Document: {doc['document_type']}\n")
                f.write(f"  Category: {doc['category']}\n")
                f.write(f"  Pages: {doc['page_range'][0]}-{doc['page_range'][1]} (count: {len(doc['pages'])})\n")
                f.write(f"  Sections: {len(doc.get('sections', []))}\n")
                
                # Section details
                for section in doc.get("sections", []):
                    self._write_section(f, section, indent=2)
                
                f.write(f"\n{'-' * 80}\n\n")
        
        logger.info(f"Saved hierarchy summary: {summary_path.name}")
    
    def _write_section(self, f, section: Dict[str, Any], indent: int = 0):
        """Helper to write section recursively."""
        indent_str = "  " * indent
        f.write(f"{indent_str}[L{section['level']}] {section['heading']}\n")
        f.write(f"{indent_str}    Page: {section.get('page', 'N/A')}\n")
        f.write(f"{indent_str}    Children: {len(section.get('children', []))} elements\n")
        
        # Recursively write nested sections
        for subsection in section.get("sections", []):
            self._write_section(f, subsection, indent + 1)
    
    def visualize_document_overview(
        self,
        hierarchy: Dict[str, Any],
        page_images: Dict[int, np.ndarray],
        doc_id: str
    ) -> None:
        """
        Create a multi-page overview showing document grouping.
        
        Args:
            hierarchy: Hierarchical structure
            page_images: Dict of page_num -> image
            doc_id: Document ID
        """
        if not page_images:
            return
        
        # Create thumbnails for each page with document labels
        thumbnail_size = (400, 520)  # Smaller thumbnails
        documents = hierarchy.get("documents", [])
        
        # Assign colors to each document
        doc_colors = {}
        for i, doc in enumerate(documents):
            # Generate distinct colors
            hue = int((i * 360 / len(documents)) % 360)
            color = self._hsv_to_bgr(hue, 200, 255)
            doc_colors[doc["document_type"]] = color
        
        # Create overview image
        pages_list = sorted(page_images.keys())
        grid_cols = 4
        grid_rows = (len(pages_list) + grid_cols - 1) // grid_cols
        
        overview_height = grid_rows * (thumbnail_size[1] + 100)
        overview_width = grid_cols * (thumbnail_size[0] + 50)
        overview = np.ones((overview_height, overview_width, 3), dtype=np.uint8) * 255
        
        for idx, page_num in enumerate(pages_list):
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Get page image
            page_img = page_images[page_num]
            
            # Resize to thumbnail
            h, w = page_img.shape[:2]
            scale = min(thumbnail_size[0] / w, thumbnail_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            thumbnail = cv2.resize(page_img, (new_w, new_h))
            
            # Find document for this page
            doc_type = None
            doc_category = None
            for doc in documents:
                if page_num in doc["pages"]:
                    doc_type = doc["document_type"]
                    doc_category = doc["category"]
                    break
            
            # Place thumbnail
            y_offset = row * (thumbnail_size[1] + 100) + 50
            x_offset = col * (thumbnail_size[0] + 50) + 25
            overview[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail
            
            # Add border with document color
            if doc_type and doc_type in doc_colors:
                color = doc_colors[doc_type]
                cv2.rectangle(
                    overview,
                    (x_offset - 3, y_offset - 3),
                    (x_offset + new_w + 3, y_offset + new_h + 3),
                    color,
                    6
                )
            
            # Add page label
            label_text = f"Page {page_num}"
            cv2.putText(
                overview,
                label_text,
                (x_offset, y_offset - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
            # Add document type
            if doc_type:
                doc_text = f"{doc_type[:20]}"
                cv2.putText(
                    overview,
                    doc_text,
                    (x_offset, y_offset + new_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    doc_colors[doc_type],
                    2
                )
        
        # Add legend
        legend_y = 20
        cv2.putText(
            overview,
            "Document Groups:",
            (20, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        legend_y += 30
        for doc in documents:
            doc_type = doc["document_type"]
            color = doc_colors.get(doc_type, (128, 128, 128))
            
            cv2.rectangle(
                overview,
                (20, legend_y - 15),
                (50, legend_y + 5),
                color,
                -1
            )
            
            text = f"{doc_type} (pages {doc['page_range'][0]}-{doc['page_range'][1]}, {doc['category']})"
            cv2.putText(
                overview,
                text,
                (60, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            legend_y += 25
        
        # Save
        output_path = self.debug_dir / f"{doc_id}_document_overview.png"
        cv2.imwrite(str(output_path), overview)
        logger.info(f"Saved document overview: {output_path.name}")
    
    @staticmethod
    def _hsv_to_bgr(h: int, s: int, v: int) -> Tuple[int, int, int]:
        """Convert HSV to BGR color."""
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(x) for x in bgr[0, 0])

