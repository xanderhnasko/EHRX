#!/usr/bin/env python3
"""
Debug script to investigate OCR issues
"""
import sys
import numpy as np
import cv2
from pathlib import Path

# Add ehrx to path  
sys.path.insert(0, str(Path(__file__).parent))

from ehrx.core.config import load_default_config
from ehrx.ocr import OCREngine
from ehrx.pdf.pager import Pager
from ehrx.detect import LayoutDetector

def debug_ocr_pipeline(pdf_path: str, page_num: int = 1):
    """Debug OCR pipeline step by step."""
    print("=== OCR DEBUG ANALYSIS ===")
    
    # Load config and initialize components
    config = load_default_config()
    ocr_engine = OCREngine(config.ocr)
    detector = LayoutDetector(config.detector)
    
    # Get page image
    pager = Pager(pdf_path)
    print(f"PDF has {pager.page_count} pages")
    
    # Get first page
    for i, (page_image, page_info, mapper) in enumerate(pager.pages(f"{page_num}", dpi=200)):
        if i > 0:
            break
            
        print(f"Page image shape: {page_image.shape}")
        print(f"Page image dtype: {page_image.dtype}")
        print(f"Page image range: [{page_image.min()}, {page_image.max()}]")
        
        # Run layout detection
        layout = detector.detect_layout(page_image)
        print(f"Detected {len(layout)} layout elements")
        
        # Test OCR on first few elements
        for i, block in enumerate(layout[:3]):  # Test first 3 elements
            print(f"\n--- Element {i+1} ---")
            print(f"Block type: {getattr(block, 'type', 'unknown')}")
            print(f"Block score: {getattr(block, 'score', 'unknown')}")
            
            # Get coordinates
            x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
            print(f"Block coords: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
            
            # Crop image
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Validate coordinates
            if x1 < 0 or y1 < 0 or x2 >= page_image.shape[1] or y2 >= page_image.shape[0]:
                print(f"ISSUE: Coordinates out of bounds! Image shape: {page_image.shape}")
                continue
                
            if x2 <= x1 or y2 <= y1:
                print(f"ISSUE: Invalid box dimensions: width={x2-x1}, height={y2-y1}")
                continue
            
            cropped = page_image[y1:y2, x1:x2]
            print(f"Cropped shape: {cropped.shape}")
            
            if cropped.size == 0:
                print("ISSUE: Cropped image is empty!")
                continue
            
            # Save cropped image for inspection
            crop_path = f"debug_crop_{i+1}.png"
            cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            print(f"Saved cropped image: {crop_path}")
            
            # Test OCR without preprocessing first
            print("\n1. Testing OCR WITHOUT preprocessing:")
            try:
                result_no_preprocess = ocr_engine.extract_text(cropped, "text", apply_preprocessing=False)
                print(f"   Result type: {type(result_no_preprocess)}")
                print(f"   Result keys: {result_no_preprocess.keys() if isinstance(result_no_preprocess, dict) else 'not a dict'}")
                print(f"   Text: '{result_no_preprocess.get('text', 'NO TEXT FOUND')}'")
                print(f"   Confidence: {result_no_preprocess.get('confidence', 'NO CONFIDENCE')}")
            except Exception as e:
                print(f"   ERROR: {e}")
            
            # Test OCR with preprocessing
            print("\n2. Testing OCR WITH preprocessing:")
            try:
                result_with_preprocess = ocr_engine.extract_text(cropped, "text", apply_preprocessing=True)
                print(f"   Result type: {type(result_with_preprocess)}")
                print(f"   Text: '{result_with_preprocess.get('text', 'NO TEXT FOUND')}'")
                print(f"   Confidence: {result_with_preprocess.get('confidence', 'NO CONFIDENCE')}")
            except Exception as e:
                print(f"   ERROR: {e}")
            
            # Test raw LayoutParser TesseractAgent
            print("\n3. Testing RAW LayoutParser TesseractAgent:")
            try:
                # Test different return formats
                raw_result_1 = ocr_engine.ocr_agent.detect(cropped, return_response=True, return_only_text=False)
                print(f"   Raw result 1 (return_response=True): {type(raw_result_1)}")
                if hasattr(raw_result_1, '__dict__'):
                    print(f"   Raw result 1 attributes: {raw_result_1.__dict__}")
                elif isinstance(raw_result_1, dict):
                    print(f"   Raw result 1 keys: {raw_result_1.keys()}")
                    print(f"   Raw result 1: {raw_result_1}")
                else:
                    print(f"   Raw result 1: {raw_result_1}")
                
                raw_result_2 = ocr_engine.ocr_agent.detect(cropped, return_only_text=True)
                print(f"   Raw result 2 (text only): '{raw_result_2}'")
                
            except Exception as e:
                print(f"   ERROR: {e}")
            
            print("="*50)
        
        break
    
    pager.close()

def test_simple_ocr():
    """Test OCR on a simple synthetic image."""
    print("\n=== TESTING OCR ON SYNTHETIC IMAGE ===")
    
    # Create simple text image
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(img, "Hello World 123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save for inspection
    cv2.imwrite("debug_synthetic.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("Saved synthetic image: debug_synthetic.png")
    
    # Test OCR
    config = load_default_config()
    ocr_engine = OCREngine(config.ocr)
    
    result = ocr_engine.extract_text(img, "text", apply_preprocessing=False)
    print(f"Synthetic OCR result: {result}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_ocr.py <pdf_path> [page_num]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)
    
    # Test synthetic first
    test_simple_ocr()
    
    # Then test on real PDF
    debug_ocr_pipeline(pdf_path, page_num)
    
    print("\n=== DEBUG COMPLETE ===")
    print("Check the saved cropped images to see what OCR is actually processing.")