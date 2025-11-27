"""
Multi-page PDF processing pipeline using VLM extraction.

Orchestrates end-to-end processing of multi-page EHR PDFs with progress tracking,
error handling, and checkpointing.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from ehrx.vlm.client import VLMClient
from ehrx.vlm.config import VLMConfig
from ehrx.vlm.models import VLMRequest, VLMResponse, ElementDetection, DocumentContext
from ehrx.pdf.pager import PDFRasterizer, PageInfo, CoordinateMapper


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for document processing."""
    total_pages: int
    processed_pages: int
    failed_pages: List[int]
    total_elements: int
    total_cost_usd: float
    processing_time_seconds: float
    start_time: str
    end_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PageResult:
    """Result from processing a single page."""
    page_number: int
    elements: List[Dict[str, Any]]
    page_info: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    error: Optional[str] = None


class DocumentPipeline:
    """
    End-to-end PDF to structured schema processing pipeline.

    Processes multi-page PDFs through VLM extraction with:
    - Sequential page-by-page processing
    - Progress logging and checkpointing
    - Error handling and recovery
    - Cost tracking
    """

    def __init__(
        self,
        vlm_config: Optional[VLMConfig] = None,
        checkpoint_interval: int = 50,
        dpi: int = 200
    ):
        """
        Initialize document processing pipeline.

        Args:
            vlm_config: VLM configuration (defaults to from_env())
            checkpoint_interval: Save intermediate results every N pages
            dpi: DPI for PDF rasterization (default 200)
        """
        self.vlm_config = vlm_config or VLMConfig.from_env()
        self.vlm_client = VLMClient(self.vlm_config)
        self.checkpoint_interval = checkpoint_interval
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)

    def process_document(
        self,
        pdf_path: str,
        output_dir: str,
        page_range: Optional[str] = None,
        document_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process entire PDF through VLM extraction.

        Args:
            pdf_path: Path to input PDF file
            output_dir: Directory for output files
            page_range: Optional page range (e.g., "1-10", "all")
            document_context: Optional document-level context for VLM

        Returns:
            Dictionary containing:
                - document_id: Generated document identifier
                - total_pages: Number of pages processed
                - pages: List of page results with elements
                - processing_stats: Statistics and metadata
        """
        start_time = time.time()
        start_time_str = datetime.now().isoformat()

        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate document ID
        document_id = f"{pdf_path.stem}_{int(time.time())}"

        self.logger.info(f"Starting document processing: {pdf_path}")
        self.logger.info(f"Document ID: {document_id}")
        self.logger.info(f"Output directory: {output_dir}")

        # Initialize PDF rasterizer
        try:
            rasterizer = PDFRasterizer(pdf_path)
            total_pages = rasterizer.page_count
            self.logger.info(f"Total pages in PDF: {total_pages}")
        except Exception as e:
            self.logger.error(f"Failed to open PDF: {e}")
            raise RuntimeError(f"PDF initialization failed: {e}") from e

        # Parse page range
        if page_range and page_range != "all":
            pages_to_process = self._parse_page_range(page_range, total_pages)
        else:
            pages_to_process = list(range(total_pages))

        self.logger.info(f"Processing {len(pages_to_process)} pages")

        # Process pages sequentially
        page_results = []
        failed_pages = []
        total_elements = 0

        for idx, page_num in enumerate(pages_to_process):
            self.logger.info(f"Processing page {page_num + 1}/{total_pages} ({idx + 1}/{len(pages_to_process)})")

            try:
                page_result = self._process_page(
                    rasterizer=rasterizer,
                    page_num=page_num,
                    total_pages=total_pages,
                    document_context=document_context
                )
                page_results.append(page_result)
                total_elements += len(page_result.elements)

                self.logger.info(
                    f"Page {page_num + 1}: Extracted {len(page_result.elements)} elements"
                )

            except Exception as e:
                self.logger.error(f"Failed to process page {page_num + 1}: {e}")
                failed_pages.append(page_num)

                # Add error placeholder
                page_results.append(PageResult(
                    page_number=page_num,
                    elements=[],
                    page_info={},
                    processing_metadata={"error": str(e)},
                    error=str(e)
                ))

            # Checkpoint: Save intermediate results
            if (idx + 1) % self.checkpoint_interval == 0:
                checkpoint_path = output_dir / f"{document_id}_checkpoint_{idx + 1}.json"
                self._save_checkpoint(checkpoint_path, document_id, page_results)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Calculate final statistics
        end_time = time.time()
        processing_time = end_time - start_time

        stats = ProcessingStats(
            total_pages=total_pages,
            processed_pages=len(pages_to_process) - len(failed_pages),
            failed_pages=failed_pages,
            total_elements=total_elements,
            total_cost_usd=self.vlm_client._total_cost_usd,
            processing_time_seconds=processing_time,
            start_time=start_time_str,
            end_time=datetime.now().isoformat()
        )

        # Build final document structure
        document = {
            "document_id": document_id,
            "source_pdf": str(pdf_path),
            "total_pages": total_pages,
            "pages": [self._page_result_to_dict(pr) for pr in page_results],
            "processing_stats": stats.to_dict()
        }

        # Save final output
        output_path = output_dir / f"{document_id}_full.json"
        with open(output_path, 'w') as f:
            json.dump(document, f, indent=2)

        self.logger.info(f"Processing complete!")
        self.logger.info(f"  Total pages: {total_pages}")
        self.logger.info(f"  Successful: {stats.processed_pages}")
        self.logger.info(f"  Failed: {len(failed_pages)}")
        self.logger.info(f"  Total elements: {total_elements}")
        self.logger.info(f"  Processing time: {processing_time:.1f}s")
        self.logger.info(f"  Total cost: ${stats.total_cost_usd:.4f}")
        self.logger.info(f"  Output: {output_path}")

        if failed_pages:
            self.logger.warning(f"  Failed pages: {failed_pages}")

        return document

    def _process_page(
        self,
        rasterizer: PDFRasterizer,
        page_num: int,
        total_pages: int,
        document_context: Optional[Dict[str, Any]] = None
    ) -> PageResult:
        """
        Process a single page through VLM extraction.

        Args:
            rasterizer: PDF rasterizer instance
            page_num: Page number (0-indexed)
            total_pages: Total pages in document
            document_context: Optional document-level context

        Returns:
            PageResult with extracted elements
        """
        # Rasterize page
        page_image, page_info = rasterizer.rasterize_page(
            page_num=page_num,
            dpi=self.dpi
        )

        # Create coordinate mapper from page info
        coord_mapper = CoordinateMapper(page_info)

        # Build document context for VLM
        doc_context = DocumentContext(
            document_type=document_context.get("document_type") if document_context else "Clinical EHR",
            page_number=page_num + 1,  # 1-indexed for VLM
            total_pages=total_pages,
            section_hierarchy=document_context.get("section_hierarchy", []) if document_context else [],
            patient_context=document_context.get("patient_context") if document_context else None
        )

        # Build VLM request
        vlm_request = VLMRequest(
            context=doc_context,
            max_tokens=self.vlm_config.max_tokens,
            temperature=self.vlm_config.temperature
        )

        # Call VLM for element extraction
        vlm_response = self.vlm_client.detect_elements(
            image=page_image,
            request=vlm_request
        )

        # Convert elements to dictionaries with coordinate mapping
        elements = []
        for detection in vlm_response.elements:
            element_dict = {
                "element_id": detection.element_id,
                "type": detection.semantic_type.value,
                "content": detection.content,
                "confidence": {
                    "overall": detection.confidence_scores.overall(),
                    "extraction": detection.confidence_scores.extraction,
                    "classification": detection.confidence_scores.classification,
                    "clinical_context": detection.confidence_scores.clinical_context
                },
                "bbox_pixel": detection.bbox.to_list() if detection.bbox else None,
                "bbox_pdf": coord_mapper.pixel_to_pdf(detection.bbox.to_list()) if detection.bbox else None,
                "needs_review": detection.confidence_scores.overall() < self.vlm_config.confidence_threshold_overall
            }

            # Add clinical metadata if present
            if detection.clinical_metadata:
                element_dict["clinical_metadata"] = {
                    "temporal_qualifier": detection.clinical_metadata.temporal_qualifier,
                    "clinical_domain": detection.clinical_metadata.clinical_domain,
                    "cross_references": detection.clinical_metadata.cross_references,
                    "requires_validation": detection.clinical_metadata.requires_validation
                }

            elements.append(element_dict)

        # Build page result
        return PageResult(
            page_number=page_num,
            elements=elements,
            page_info={
                "width_pdf": page_info.width_pdf,
                "height_pdf": page_info.height_pdf,
                "width_px": page_info.width_px,
                "height_px": page_info.height_px,
                "dpi": page_info.dpi
            },
            processing_metadata={
                "model_name": vlm_response.processing_metadata.model_name,
                "processing_time_seconds": vlm_response.processing_metadata.api_latency_ms / 1000.0 if vlm_response.processing_metadata.api_latency_ms else 0.0,
                "api_cost_usd": vlm_response.processing_metadata.cost_estimate_usd or 0.0,
                "timestamp": vlm_response.processing_metadata.processing_timestamp
            }
        )

    def _page_result_to_dict(self, page_result: PageResult) -> Dict[str, Any]:
        """Convert PageResult to dictionary for JSON serialization."""
        return {
            "page_number": page_result.page_number + 1,  # 1-indexed in output
            "elements": page_result.elements,
            "page_info": page_result.page_info,
            "processing_metadata": page_result.processing_metadata,
            "error": page_result.error
        }

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        document_id: str,
        page_results: List[PageResult]
    ) -> None:
        """Save intermediate processing results."""
        checkpoint_data = {
            "document_id": document_id,
            "checkpoint_time": datetime.now().isoformat(),
            "pages_processed": len(page_results),
            "pages": [self._page_result_to_dict(pr) for pr in page_results]
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def _parse_page_range(self, page_range: str, total_pages: int) -> List[int]:
        """
        Parse page range string into list of page numbers.

        Examples:
            "1-10" -> [0, 1, 2, ..., 9]
            "1-5,8-10" -> [0, 1, 2, 3, 4, 7, 8, 9]
            "5" -> [4]

        Args:
            page_range: Page range string
            total_pages: Total pages in document

        Returns:
            List of 0-indexed page numbers
        """
        pages = []

        for part in page_range.split(','):
            part = part.strip()

            if '-' in part:
                start, end = part.split('-')
                start = int(start.strip()) - 1  # Convert to 0-indexed
                end = int(end.strip()) - 1
                pages.extend(range(start, end + 1))
            else:
                pages.append(int(part.strip()) - 1)

        # Filter to valid page numbers
        pages = [p for p in pages if 0 <= p < total_pages]

        return sorted(set(pages))  # Remove duplicates and sort
