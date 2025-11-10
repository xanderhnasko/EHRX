"""
Integration tests for serialization with enhanced router and column detection
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock

from ehrx.layout.column_detection import ColumnLayout, DocumentColumnDetector  
from ehrx.layout.global_ordering import GlobalOrderingManager
from ehrx.layout.enhanced_router import EnhancedElementRouter, DocumentProcessor
from ehrx.serialize import DocumentSerializer


class TestSerializationIntegration:
    """Integration tests combining column detection, global ordering, and serialization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "test_document"
        self.doc_id = "test_doc_001"
        self.config = {
            "detector": {"backend": "layoutparser"},
            "ocr": {"engine": "tesseract"}
        }
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_processing_with_serialization(self):
        """Test complete workflow: column detection → global ordering → enhanced routing → serialization."""
        
        # 1. Mock document blocks (simulating LayoutParser output)
        page_1_blocks = [
            {"bbox_px": [50, 100, 250, 150], "type": 1},   # Left column text
            {"bbox_px": [350, 100, 550, 150], "type": 1},  # Right column text  
            {"bbox_px": [50, 200, 250, 300], "type": 4},   # Left column table
            {"bbox_px": [350, 200, 550, 250], "type": 5}   # Right column figure
        ]
        
        page_2_blocks = [
            {"bbox_px": [50, 50, 250, 100], "type": 1},    # Left column continuation
            {"bbox_px": [350, 50, 550, 100], "type": 1},   # Right column continuation
        ]
        
        all_pages_blocks = [page_1_blocks, page_2_blocks]
        
        # 2. Document processing: Pass 1 - Column detection
        processor = DocumentProcessor()
        column_layout = processor.analyze_document_layout(all_pages_blocks, page_width=600.0)
        
        # Verify column detection worked
        assert column_layout.column_count == 2
        assert len(column_layout.boundaries) == 3
        
        # 3. Document processing: Pass 2 - Global ordering 
        processed_elements = processor.process_document_with_global_ordering(
            all_pages_blocks, column_layout
        )
        
        # Verify global ordering worked
        assert len(processed_elements) == 2  # 2 pages
        assert len(processed_elements[0]) == 4  # 4 elements on page 1
        assert len(processed_elements[1]) == 2  # 2 elements on page 2
        
        # Check z_order is global (continuous across pages)
        all_elements = []
        for page_elements in processed_elements:
            all_elements.extend(page_elements)
        
        z_orders = [elem["z_order"] for elem in all_elements]
        assert z_orders == [0, 1, 2, 3, 4, 5]  # Sequential global ordering
        
        # 4. Serialization workflow
        with DocumentSerializer(self.output_dir, self.doc_id, self.config) as serializer:
            # Serialize all elements
            for elem in all_elements:
                # Add some realistic payload data
                elem["payload"] = {"text": f"Content for {elem['id']}"} if elem["type"] == "text_block" else {}
                
                # Mock crop image for visual elements 
                crop_image = None
                table_data = None
                
                if elem["type"] == "table":
                    crop_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
                    table_data = {
                        "headers": ["Test", "Result"],
                        "rows": [["WBC", "9.8"], ["Hgb", "12.1"]]
                    }
                elif elem["type"] == "figure":
                    crop_image = np.random.randint(0, 255, (150, 250, 3), dtype=np.uint8)
                
                serializer.serialize_element(elem, crop_image, table_data)
            
            # Finalize with mock hierarchy
            hierarchy = [
                {"id": "H1_0001", "label": "SECTION A", "children": ["E_0000", "E_0002"]},
                {"id": "H1_0002", "label": "SECTION B", "children": ["E_0001", "E_0003", "E_0004", "E_0005"]}
            ]
            labels_used = ["SECTION A", "SECTION B"]
            
            stats = serializer.finalize(hierarchy, labels_used, column_layout, 15.7)
        
        # 5. Verify complete output structure
        assert (self.output_dir / "document.elements.jsonl").exists()
        assert (self.output_dir / "document.index.json").exists()
        assert (self.output_dir / "assets").exists()
        
        # 6. Verify JSONL content preserves column context
        with open(self.output_dir / "document.elements.jsonl", 'r') as f:
            jsonl_lines = f.readlines()
        
        assert len(jsonl_lines) == 6
        
        # Check that column_index is preserved in each element
        for line in jsonl_lines:
            element = json.loads(line)
            assert "column_index" in element
            assert element["column_index"] in [0, 1]  # Should be assigned to column 0 or 1
            assert "z_order" in element
            assert "id" in element
        
        # Verify table element has assets
        table_elements = [json.loads(line) for line in jsonl_lines if json.loads(line)["type"] == "table"]
        assert len(table_elements) == 1
        table_elem = table_elements[0]
        assert "image_ref" in table_elem["payload"]
        assert "csv_ref" in table_elem["payload"]
        assert table_elem["payload"]["headers"] == ["Test", "Result"]
        assert table_elem["payload"]["rows"] == [["WBC", "9.8"], ["Hgb", "12.1"]]
        
        # Verify figure element has asset
        figure_elements = [json.loads(line) for line in jsonl_lines if json.loads(line)["type"] == "figure"]
        assert len(figure_elements) == 1
        figure_elem = figure_elements[0]
        assert "image_ref" in figure_elem["payload"]
        
        # 7. Verify index content includes column layout
        with open(self.output_dir / "document.index.json", 'r') as f:
            index_data = json.load(f)
        
        assert index_data["doc_id"] == self.doc_id
        assert index_data["manifest"]["column_layout"]["column_count"] == 2
        assert index_data["manifest"]["stats"]["total_elements"] == 6
        assert index_data["hierarchy"] == hierarchy
        assert index_data["labels_used"] == labels_used
        
        # 8. Verify asset files exist
        table_id = table_elem["id"]
        figure_id = figure_elem["id"]
        
        assert (self.output_dir / "assets" / f"table_{table_id}.png").exists()
        assert (self.output_dir / "assets" / f"table_{table_id}.csv").exists()
        assert (self.output_dir / "assets" / f"figure_{figure_id}.png").exists()
        
        print(f"✅ End-to-end test passed! Generated {stats['total_elements']} elements")
    
    def test_enhanced_router_with_serialization(self):
        """Test EnhancedElementRouter integration with DocumentSerializer."""
        
        # Setup column layout 
        column_layout = ColumnLayout(
            column_count=2,
            boundaries=[0.0, 300.0, 600.0],
            page_width=600.0
        )
        
        # Mock configuration
        mock_config = Mock()
        mock_config.ocr = Mock()
        
        # Initialize enhanced router
        router = EnhancedElementRouter(
            config=mock_config,
            doc_id=self.doc_id, 
            column_layout=column_layout
        )
        
        # Mock layout blocks
        mock_blocks = [
            Mock(block=Mock(x_1=50, y_1=100, x_2=200, y_2=150), type=1, score=0.9),   # Left text
            Mock(block=Mock(x_1=350, y_1=100, x_2=500, y_2=150), type=4, score=0.85), # Right table
        ]
        
        # Mock dependencies
        mock_page_info = Mock(page_number=1)
        mock_mapper = Mock()
        mock_mapper.pixel_to_pdf.return_value = [72, 100, 540, 150]
        mock_page_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Process blocks with enhanced router
        elements = router.process_layout_blocks_with_global_ordering(
            mock_blocks, mock_page_image, mock_page_info, mock_mapper
        )
        
        # Verify enhanced router output
        assert len(elements) == 2
        for elem in elements:
            assert "column_index" in elem
            assert "z_order" in elem
            assert "id" in elem
            assert elem["doc_id"] == self.doc_id
        
        # Test serialization of enhanced router output
        with DocumentSerializer(self.output_dir, self.doc_id, self.config) as serializer:
            for element in elements:
                serializer.serialize_element(element)
            
            # Simple finalization
            stats = serializer.finalize([], [], column_layout, 5.0)
        
        # Verify serialization worked
        assert stats["total_elements"] == 2
        assert (self.output_dir / "document.elements.jsonl").exists()
        
        # Check elements preserved column data
        with open(self.output_dir / "document.elements.jsonl", 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        for line in lines:
            elem = json.loads(line)
            assert elem["column_index"] in [0, 1]
            assert elem["z_order"] in [0, 1]
            assert elem["doc_id"] == self.doc_id


if __name__ == "__main__":
    # Allow running this test file directly
    test_instance = TestSerializationIntegration()
    test_instance.setup_method()
    try:
        test_instance.test_end_to_end_processing_with_serialization()
        test_instance.test_enhanced_router_with_serialization()
        print("\n🎉 All integration tests passed!")
    finally:
        test_instance.teardown_method()