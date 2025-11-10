"""
CLI entrypoint for ehrx extract command
"""
import logging
import time
from pathlib import Path
from typing import Optional, List, Any, Dict

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# Core imports
from .core.config import EHRXConfig
from .pdf.pager import PDFRasterizer, PageInfo
from .detect import LayoutDetector
from .ocr import OCREngine
from .route import ElementRouter
from .layout.enhanced_router import DocumentProcessor, EnhancedElementRouter
from .serialize import DocumentSerializer

# Initialize CLI app and console
app = typer.Typer(help="EHRX - Clinical EHR PDF extraction tool")
console = Console()

# Global configuration
config: Optional[EHRXConfig] = None


def load_config(config_path: Optional[Path] = None) -> EHRXConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    return EHRXConfig.from_yaml(config_path)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("iopath.common.file_io").setLevel(logging.CRITICAL)
    logging.getLogger("iopath.common.event_logger").setLevel(logging.CRITICAL)
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("layoutparser").setLevel(logging.WARNING)


def main(
    input_pdf: Path = typer.Option(..., "--in", "-i", help="Input PDF file path"),
    output_dir: Path = typer.Option(..., "--out", "-o", help="Output directory for results"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config YAML file"),
    detector: Optional[str] = typer.Option(None, help="Override detector backend (detectron2|paddle)"),
    min_conf: Optional[float] = typer.Option(None, help="Override minimum confidence threshold"),
    ocr_engine: Optional[str] = typer.Option(None, help="Override OCR engine (tesseract)"),
    pages: Optional[str] = typer.Option("all", help="Pages to process (e.g., '1-5,10,15-20' or 'all')"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG|INFO|WARNING|ERROR)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed without running")
) -> None:
    """
    Extract structured data from EHR PDF documents.
    
    This command processes scanned EHR PDFs using layout detection, OCR, and column-aware 
    ordering to produce structured JSONL output with proper document hierarchy.
    """
    start_time = time.time()
    
    # Setup
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    global config
    config = load_config(config_file)
    
    # Apply CLI overrides
    if detector:
        config.detector.backend = detector
    if min_conf is not None:
        config.detector.min_conf = min_conf
    if ocr_engine:
        config.ocr.engine = ocr_engine
    
    # Validate inputs
    if not input_pdf.exists():
        rprint(f"[red]Error: Input PDF not found: {input_pdf}[/red]")
        raise typer.Exit(1)
    
    if not input_pdf.suffix.lower() == '.pdf':
        rprint(f"[red]Error: Input file must be a PDF: {input_pdf}[/red]")
        raise typer.Exit(1)
    
    # Parse page range
    page_range = parse_page_range(pages)
    
    # Show configuration summary
    show_config_summary(input_pdf, output_dir, config, page_range, dry_run)
    
    if dry_run:
        rprint("[yellow]Dry run complete. Use --dry-run=false to process.[/yellow]")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = input_pdf.stem
    
    try:
        # Main processing pipeline
        process_document(input_pdf, output_dir, doc_id, config, page_range, logger)
        
        # Success summary
        elapsed = time.time() - start_time
        show_success_summary(output_dir, elapsed)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        rprint(f"[red]Processing failed: {e}[/red]")
        raise typer.Exit(1)


def parse_page_range(pages_str: str) -> Optional[List[int]]:
    """Parse page range string into list of page numbers."""
    if pages_str.lower() == "all":
        return None
    
    page_numbers = []
    for part in pages_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            page_numbers.extend(range(start, end + 1))
        else:
            page_numbers.append(int(part))
    
    return sorted(list(set(page_numbers)))


def show_config_summary(
    input_pdf: Path, 
    output_dir: Path, 
    config: EHRXConfig, 
    page_range: Optional[List[int]],
    dry_run: bool
) -> None:
    """Display processing configuration summary."""
    table = Table(title="EHRX Processing Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Input PDF", str(input_pdf))
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Document ID", input_pdf.stem)
    table.add_row("Layout Detector", f"{config.detector.backend} ({config.detector.model})")
    table.add_row("OCR Engine", config.ocr.engine)
    table.add_row("Min Confidence", str(config.detector.min_conf))
    
    if page_range:
        page_summary = f"{len(page_range)} pages: {page_range[:5]}{'...' if len(page_range) > 5 else ''}"
    else:
        page_summary = "All pages"
    table.add_row("Page Range", page_summary)
    table.add_row("Dry Run", str(dry_run))
    
    console.print(table)
    console.print()


def process_document(
    input_pdf: Path,
    output_dir: Path, 
    doc_id: str,
    config: EHRXConfig,
    page_range: Optional[List[int]],
    logger: logging.Logger
) -> None:
    """Main document processing pipeline."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Initialize components
        task = progress.add_task("Initializing components...", total=None)
        
        rasterizer = PDFRasterizer(input_pdf)
        detector = LayoutDetector(config.detector)
        ocr_engine = OCREngine(config.ocr)
        
        total_pages = rasterizer.page_count
        if page_range:
            pages_to_process = [p for p in page_range if 1 <= p <= total_pages]
        else:
            pages_to_process = list(range(1, total_pages + 1))
        
        logger.info(f"Processing {len(pages_to_process)} pages from {total_pages}-page document")
        
        # Step 2: Pass 1 - Layout analysis for column detection
        progress.update(task, description="Pass 1: Analyzing document layout...")
        
        all_pages_blocks = []
        page_infos = []
        
        for page_num in pages_to_process:
            page_image, page_info = rasterizer.rasterize_page(page_num, dpi=200)
            page_infos.append(page_info)
            
            layout = detector.detect_layout(page_image)
            
            # Convert layout to standard format for column detection
            page_blocks = []
            for block in layout:
                page_blocks.append({
                    "bbox_px": [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2],
                    "type": getattr(block, 'type', 1),
                    "score": getattr(block, 'score', 1.0),
                    "block": block
                })
            
            all_pages_blocks.append(page_blocks)
            progress.update(task, description=f"Pass 1: Analyzed page {page_num}/{pages_to_process[-1]}")
        
        # Column detection 
        processor = DocumentProcessor()
        column_layout = processor.analyze_document_layout(all_pages_blocks, page_width=page_infos[0].width_px)
        
        logger.info(f"Detected {column_layout.column_count} column layout")
        
        # Step 3: Pass 2 - Enhanced element processing with serialization
        progress.update(task, description="Pass 2: Processing elements with global ordering...")
        
        with DocumentSerializer(output_dir, doc_id, config.model_dump()) as serializer:
            
            # Initialize enhanced router
            enhanced_router = EnhancedElementRouter(
                config=config,
                doc_id=doc_id, 
                column_layout=column_layout
            )
            
            element_count = 0
            
            for page_idx, (page_blocks, page_info) in enumerate(zip(all_pages_blocks, page_infos)):
                page_num = pages_to_process[page_idx]
                page_image, _ = rasterizer.rasterize_page(page_num, dpi=600)  # Higher DPI for better OCR
                
                # Convert blocks back to layout format for enhanced router
                layout_blocks = [block_data["block"] for block_data in page_blocks]
                
                from .pdf.pager import CoordinateMapper
                mapper = CoordinateMapper(page_info)
                
                # Process with enhanced router
                elements = enhanced_router.process_layout_blocks_with_global_ordering(
                    layout_blocks, page_image, page_info, mapper
                )
                
                # Process each element for serialization
                for element in elements:
                    # Add basic payload and process for OCR/assets
                    crop_image = None
                    table_data = None
                    
                    # Extract text for text blocks
                    if element["type"] == "text_block":
                        try:
                            # Crop region for OCR
                            bbox_px = element["bbox_px"]
                            x0, y0, x1, y1 = [int(coord) for coord in bbox_px]
                            cropped = page_image[y0:y1, x0:x1]
                            
                            if cropped.size > 0:
                                ocr_result = ocr_engine.extract_text(cropped, "text")
                                # Handle confidence score properly
                                confidence = ocr_result.get("confidence")
                                if confidence is None:
                                    confidence = 0.0
                                elif isinstance(confidence, (int, float)):
                                    confidence = float(confidence)
                                else:
                                    confidence = 0.0
                                
                                element["payload"] = {
                                    "text": ocr_result["text"],
                                    "confidence": confidence
                                }
                            else:
                                element["payload"] = {"text": "", "confidence": 0.0}
                        except Exception as e:
                            logger.warning(f"OCR failed for element {element['id']}: {e}")
                            element["payload"] = {"text": "", "confidence": 0.0}
                    
                    # Handle visual elements
                    elif element["type"] in ["table", "figure", "handwriting"]:
                        try:
                            bbox_px = element["bbox_px"] 
                            x0, y0, x1, y1 = [int(coord) for coord in bbox_px]
                            crop_image = page_image[y0:y1, x0:x1]
                            
                            if element["type"] == "table":
                                # Basic table processing - just OCR for now
                                if crop_image.size > 0:
                                    ocr_result = ocr_engine.extract_text(crop_image, "table")
                                    lines = ocr_result["text"].split('\n') if ocr_result["text"] else []
                                    element["payload"] = {"ocr_lines": lines}
                                    
                                    # Simple table structure detection
                                    if len(lines) >= 2:
                                        table_data = {
                                            "headers": lines[0].split() if lines else [],
                                            "rows": [line.split() for line in lines[1:] if line.strip()]
                                        }
                                else:
                                    element["payload"] = {"ocr_lines": []}
                            
                            elif element["type"] == "figure":
                                element["payload"] = {"caption": ""}
                                
                            elif element["type"] == "handwriting":
                                element["payload"] = {"ocr_text": "", "ocr_confidence": 0.0}
                                
                        except Exception as e:
                            logger.warning(f"Processing failed for {element['type']} element {element['id']}: {e}")
                            element["payload"] = {}
                    
                    # Serialize element
                    serializer.serialize_element(element, crop_image, table_data)
                    element_count += 1
                
                progress.update(task, description=f"Pass 2: Processed page {page_num} ({element_count} elements)")
            
            # Finalize with empty hierarchy
            elapsed = time.time() - progress.start_time if hasattr(progress, 'start_time') else 0
            stats = serializer.finalize([], [], column_layout, elapsed)
            
            logger.info(f"Serialized {stats['total_elements']} elements")


def show_success_summary(output_dir: Path, elapsed_time: float) -> None:
    """Display processing success summary."""
    rprint(f"\n[green] Processing completed![/green]")
    rprint(f"[blue] Output dir: {output_dir}[/blue]")
    rprint(f"[blue] Processing time: {elapsed_time:.1f} seconds[/blue]")
    
    # Show output files
    files_table = Table(title=" Generated Files")
    files_table.add_column("File", style="cyan")
    files_table.add_column("Description", style="white")
    
    jsonl_file = output_dir / "document.elements.jsonl"
    index_file = output_dir / "document.index.json" 
    assets_dir = output_dir / "assets"
    
    if jsonl_file.exists():
        files_table.add_row("document.elements.jsonl", "Structured element data (one JSON per line)")
    if index_file.exists():
        files_table.add_row("document.index.json", "Document index with metadata and column layout")
    if assets_dir.exists():
        asset_count = len(list(assets_dir.glob("*")))
        files_table.add_row("assets/", f"Asset directory ({asset_count} files)")
    
    console.print(files_table)


if __name__ == "__main__":
    typer.run(main)