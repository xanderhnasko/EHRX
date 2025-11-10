"""
Tests for serialization components (JsonlWriter, AssetManager, IndexBuilder, DocumentSerializer)
"""
import pytest
import json
import csv
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime

from ehrx.layout.column_detection import ColumnLayout


class TestJsonlWriter:
    """Test JsonlWriter for streaming JSONL output."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_file = self.temp_dir / "test.jsonl"
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_jsonl_writer_init(self):
        """Test JsonlWriter initialization."""
        from ehrx.serialize import JsonlWriter
        
        writer = JsonlWriter(self.output_file)
        assert writer.output_path == self.output_file
        assert self.output_file.exists()
        assert writer._element_count == 0
        writer.close()
    
    def test_jsonl_writer_append_single_element(self):
        """Test appending single element."""
        from ehrx.serialize import JsonlWriter
        
        element = {
            "id": "E_0001",
            "type": "text_block", 
            "page": 1,
            "z_order": 0,
            "column_index": 0
        }
        
        with JsonlWriter(self.output_file) as writer:
            writer.append(element)
        
        # Verify file content
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed == element
    
    def test_jsonl_writer_append_multiple_elements(self):
        """Test appending multiple elements."""
        from ehrx.serialize import JsonlWriter
        
        elements = [
            {"id": "E_0001", "type": "text_block", "z_order": 0},
            {"id": "E_0002", "type": "table", "z_order": 1},
            {"id": "E_0003", "type": "figure", "z_order": 2}
        ]
        
        with JsonlWriter(self.output_file) as writer:
            for element in elements:
                writer.append(element)
        
        # Verify file content
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed == elements[i]
    
    def test_jsonl_writer_thread_safety(self):
        """Test thread safety with concurrent writes."""
        from ehrx.serialize import JsonlWriter
        import threading
        
        elements_per_thread = 10
        num_threads = 3
        
        def write_elements(writer, thread_id):
            for i in range(elements_per_thread):
                element = {
                    "id": f"T{thread_id}_E_{i:04d}",
                    "thread": thread_id,
                    "seq": i
                }
                writer.append(element)
        
        with JsonlWriter(self.output_file) as writer:
            threads = []
            for t in range(num_threads):
                thread = threading.Thread(target=write_elements, args=(writer, t))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        # Verify all elements written
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == elements_per_thread * num_threads
        
        # Verify all are valid JSON
        for line in lines:
            element = json.loads(line)
            assert "id" in element
            assert "thread" in element
            assert "seq" in element
    
    def test_jsonl_writer_context_manager(self):
        """Test context manager behavior."""
        from ehrx.serialize import JsonlWriter
        
        element = {"id": "E_0001", "test": True}
        
        # Test normal context exit
        with JsonlWriter(self.output_file) as writer:
            writer.append(element)
            count = writer._element_count
        
        assert count == 1
        assert not writer._file_handle  # Should be closed
        
        # Verify file content
        with open(self.output_file, 'r') as f:
            line = f.readline()
        parsed = json.loads(line)
        assert parsed == element


class TestAssetManager:
    """Test AssetManager for image and CSV saving."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.assets_dir = self.temp_dir / "assets"
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_asset_manager_init(self):
        """Test AssetManager initialization."""
        from ehrx.serialize import AssetManager
        
        manager = AssetManager(self.assets_dir)
        assert manager.assets_dir == self.assets_dir
        assert self.assets_dir.exists()
    
    def test_save_table_image(self):
        """Test saving table image."""
        from ehrx.serialize import AssetManager
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        element_id = "E_0045"
        
        manager = AssetManager(self.assets_dir)
        image_ref = manager.save_table_image(element_id, test_image)
        
        assert image_ref == "assets/table_E_0045.png"
        image_path = self.assets_dir / "table_E_0045.png"
        assert image_path.exists()
    
    def test_save_figure_image(self):
        """Test saving figure image."""
        from ehrx.serialize import AssetManager
        
        # Create test image (grayscale)
        test_image = np.random.randint(0, 255, (150, 300), dtype=np.uint8)
        element_id = "E_0023"
        
        manager = AssetManager(self.assets_dir)
        image_ref = manager.save_figure_image(element_id, test_image)
        
        assert image_ref == "assets/figure_E_0023.png"
        image_path = self.assets_dir / "figure_E_0023.png"
        assert image_path.exists()
    
    def test_save_handwriting_image(self):
        """Test saving handwriting image."""
        from ehrx.serialize import AssetManager
        
        # Create test image
        test_image = np.random.randint(0, 255, (80, 250, 3), dtype=np.uint8)
        element_id = "E_0156"
        
        manager = AssetManager(self.assets_dir)
        image_ref = manager.save_handwriting_image(element_id, test_image)
        
        assert image_ref == "assets/hand_E_0156.png"
        image_path = self.assets_dir / "hand_E_0156.png"
        assert image_path.exists()
    
    def test_save_table_csv_with_headers(self):
        """Test saving table CSV with headers."""
        from ehrx.serialize import AssetManager
        
        element_id = "E_0045"
        headers = ["Test", "Result", "Units", "Ref"]
        rows = [
            ["WBC", "9.8", "10^3/uL", "4.0-10.0"],
            ["Hgb", "12.1", "g/dL", "12-16"],
            ["Plt", "250", "10^3/uL", "150-400"]
        ]
        
        manager = AssetManager(self.assets_dir)
        csv_ref = manager.save_table_csv(element_id, rows, headers)
        
        assert csv_ref == "assets/table_E_0045.csv"
        csv_path = self.assets_dir / "table_E_0045.csv"
        assert csv_path.exists()
        
        # Verify CSV content
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            csv_data = list(reader)
        
        assert csv_data[0] == headers  # Headers row
        assert csv_data[1:] == rows    # Data rows
    
    def test_save_table_csv_without_headers(self):
        """Test saving table CSV without headers."""
        from ehrx.serialize import AssetManager
        
        element_id = "E_0067"
        rows = [
            ["Patient", "Age", "Gender"],
            ["John", "45", "M"],
            ["Jane", "32", "F"]
        ]
        
        manager = AssetManager(self.assets_dir)
        csv_ref = manager.save_table_csv(element_id, rows)
        
        assert csv_ref == "assets/table_E_0067.csv"
        csv_path = self.assets_dir / "table_E_0067.csv"
        assert csv_path.exists()
        
        # Verify CSV content (no header row)
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            csv_data = list(reader)
        
        assert csv_data == rows


class TestIndexBuilder:
    """Test IndexBuilder for enhanced metadata."""
    
    def test_build_manifest(self):
        """Test building manifest with column layout."""
        from ehrx.serialize import IndexBuilder
        
        config = {
            "detector": {"backend": "layoutparser"},
            "ocr": {"engine": "tesseract"}
        }
        
        stats = {
            "total_pages": 25,
            "total_elements": 567,
            "z_order_range": [0, 566],
            "elements_by_type": {"text_block": 450, "table": 67, "figure": 50},
            "processing_time": 45.7
        }
        
        column_layout = ColumnLayout(
            column_count=2,
            boundaries=[0.0, 300.0, 600.0],
            page_width=600.0
        )
        
        builder = IndexBuilder()
        manifest = builder.build_manifest(config, stats, column_layout)
        
        # Verify required fields
        assert manifest["pages"] == 25
        assert manifest["detector"] == "layoutparser"
        assert manifest["ocr"] == "tesseract"
        assert "created_at" in manifest
        assert manifest["column_layout"] == column_layout.to_dict()
        assert manifest["stats"]["total_elements"] == 567
        assert manifest["stats"]["z_order_range"] == [0, 566]
        assert manifest["stats"]["processing_time_seconds"] == 45.7
    
    def test_build_index(self):
        """Test building complete document index."""
        from ehrx.serialize import IndexBuilder
        
        manifest = {
            "doc_id": "test_doc",
            "pages": 10,
            "detector": "layoutparser",
            "column_layout": {"column_count": 1}
        }
        
        hierarchy = [
            {"id": "H1_0001", "label": "PROBLEMS", "children": ["E_0012", "E_0013"]},
            {"id": "H1_0002", "label": "MEDICATIONS", "children": ["E_0025"]}
        ]
        
        labels_used = ["PROBLEMS", "MEDICATIONS", "ALLERGIES"]
        
        builder = IndexBuilder()
        index = builder.build_index(manifest, hierarchy, labels_used)
        
        assert index["doc_id"] == "test_doc"
        assert index["manifest"] == manifest
        assert index["hierarchy"] == hierarchy
        assert index["labels_used"] == labels_used
    
    def test_write_index(self):
        """Test writing index to JSON file."""
        from ehrx.serialize import IndexBuilder
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            index_path = temp_dir / "test_index.json"
            
            index_data = {
                "doc_id": "test_doc",
                "manifest": {"pages": 5},
                "hierarchy": [],
                "labels_used": []
            }
            
            builder = IndexBuilder()
            builder.write_index(index_path, index_data)
            
            assert index_path.exists()
            
            # Verify content
            with open(index_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded == index_data
        
        finally:
            shutil.rmtree(temp_dir)


class TestDocumentSerializer:
    """Test DocumentSerializer coordination."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.doc_id = "test_document"
        self.config = {
            "detector": {"backend": "layoutparser"},
            "ocr": {"engine": "tesseract"}
        }
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_document_serializer_init(self):
        """Test DocumentSerializer initialization."""
        from ehrx.serialize import DocumentSerializer
        
        serializer = DocumentSerializer(self.output_dir, self.doc_id, self.config)
        
        assert serializer.output_dir == self.output_dir
        assert serializer.doc_id == self.doc_id
        assert serializer.config == self.config
        assert self.output_dir.exists()
        assert serializer.assets_dir.exists()
    
    def test_serialize_text_element(self):
        """Test serializing text element."""
        from ehrx.serialize import DocumentSerializer
        
        element = {
            "id": "E_0001",
            "type": "text_block",
            "page": 1,
            "z_order": 0,
            "column_index": 0,
            "payload": {"text": "Patient presents with..."}
        }
        
        with DocumentSerializer(self.output_dir, self.doc_id, self.config) as serializer:
            serializer.serialize_element(element)
        
        # Verify JSONL written
        jsonl_path = self.output_dir / "document.elements.jsonl"
        assert jsonl_path.exists()
        
        with open(jsonl_path, 'r') as f:
            line = f.readline()
        parsed = json.loads(line)
        assert parsed == element
    
    def test_serialize_table_element_with_image_and_csv(self):
        """Test serializing table element with image and CSV."""
        from ehrx.serialize import DocumentSerializer
        
        element = {
            "id": "E_0045",
            "type": "table", 
            "page": 3,
            "z_order": 9,
            "column_index": 1,
            "payload": {"ocr_lines": ["Test Result", "WBC 9.8"]}
        }
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Create test table data
        table_data = {
            "headers": ["Test", "Result"],
            "rows": [["WBC", "9.8"], ["Hgb", "12.1"]]
        }
        
        with DocumentSerializer(self.output_dir, self.doc_id, self.config) as serializer:
            serializer.serialize_element(element, test_image, table_data)
        
        # Verify JSONL updated with asset references
        jsonl_path = self.output_dir / "document.elements.jsonl"
        with open(jsonl_path, 'r') as f:
            line = f.readline()
        parsed = json.loads(line)
        
        assert parsed["payload"]["image_ref"] == "assets/table_E_0045.png"
        assert parsed["payload"]["csv_ref"] == "assets/table_E_0045.csv"
        assert parsed["payload"]["headers"] == ["Test", "Result"]
        assert parsed["payload"]["rows"] == [["WBC", "9.8"], ["Hgb", "12.1"]]
        
        # Verify assets created
        assert (self.output_dir / "assets" / "table_E_0045.png").exists()
        assert (self.output_dir / "assets" / "table_E_0045.csv").exists()
    
    def test_finalize_with_hierarchy(self):
        """Test finalizing serialization with hierarchy and index."""
        from ehrx.serialize import DocumentSerializer
        
        # Create test elements
        elements = [
            {"id": "E_0001", "type": "text_block", "z_order": 0, "column_index": 0},
            {"id": "E_0002", "type": "table", "z_order": 1, "column_index": 0},
            {"id": "E_0003", "type": "figure", "z_order": 2, "column_index": 1}
        ]
        
        hierarchy = [
            {"id": "H1_0001", "label": "PROBLEMS", "children": ["E_0001"]},
            {"id": "H1_0002", "label": "LABS", "children": ["E_0002", "E_0003"]}
        ]
        
        labels_used = ["PROBLEMS", "LABS"]
        
        column_layout = ColumnLayout(
            column_count=2,
            boundaries=[0.0, 300.0, 600.0],
            page_width=600.0
        )
        
        with DocumentSerializer(self.output_dir, self.doc_id, self.config) as serializer:
            # Serialize elements
            for element in elements:
                serializer.serialize_element(element)
            
            # Finalize
            stats = serializer.finalize(hierarchy, labels_used, column_layout, 30.5)
        
        # Verify statistics
        assert stats["total_elements"] == 3
        assert stats["z_order_range"] == [0, 2]
        assert stats["elements_by_type"]["text_block"] == 1
        assert stats["elements_by_type"]["table"] == 1 
        assert stats["elements_by_type"]["figure"] == 1
        assert stats["processing_time"] == 30.5
        
        # Verify index file created
        index_path = self.output_dir / "document.index.json"
        assert index_path.exists()
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        assert index_data["doc_id"] == self.doc_id
        assert index_data["manifest"]["stats"]["total_elements"] == 3
        assert index_data["manifest"]["column_layout"] == column_layout.to_dict()
        assert index_data["hierarchy"] == hierarchy
        assert index_data["labels_used"] == labels_used


class TestSerializationIntegration:
    """Integration tests for complete serialization workflow."""
    
    def test_complete_serialization_workflow(self):
        """Test complete workflow from elements to final output."""
        from ehrx.serialize import DocumentSerializer
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            output_dir = temp_dir / "scan_001"
            doc_id = "scan_001"
            config = {"detector": {"backend": "layoutparser"}, "ocr": {"engine": "tesseract"}}
            
            # Test data
            elements = [
                {
                    "id": "E_0001", "doc_id": doc_id, "page": 1, "type": "text_block",
                    "bbox_pdf": [72, 100, 540, 150], "bbox_px": [95, 133, 712, 200],
                    "z_order": 0, "column_index": 0, "source": "ocr",
                    "payload": {"text": "PROBLEM LIST"}
                },
                {
                    "id": "E_0002", "doc_id": doc_id, "page": 1, "type": "table", 
                    "bbox_pdf": [72, 200, 540, 400], "bbox_px": [95, 266, 712, 533],
                    "z_order": 1, "column_index": 0, "source": "ocr",
                    "payload": {"ocr_lines": ["Test Result", "WBC 9.8"]}
                }
            ]
            
            table_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            table_data = {
                "headers": ["Test", "Result"],
                "rows": [["WBC", "9.8"]]
            }
            
            hierarchy = [{"id": "H1_0001", "label": "PROBLEMS", "children": ["E_0001", "E_0002"]}]
            labels_used = ["PROBLEMS"]
            column_layout = ColumnLayout(column_count=1, boundaries=[0.0, 600.0], page_width=600.0)
            
            # Run serialization
            with DocumentSerializer(output_dir, doc_id, config) as serializer:
                serializer.serialize_element(elements[0])  # Text element
                serializer.serialize_element(elements[1], table_image, table_data)  # Table with assets
                
                stats = serializer.finalize(hierarchy, labels_used, column_layout, 25.3)
            
            # Verify complete output structure
            assert (output_dir / "document.elements.jsonl").exists()
            assert (output_dir / "document.index.json").exists()
            assert (output_dir / "assets").exists()
            assert (output_dir / "assets" / "table_E_0002.png").exists()
            assert (output_dir / "assets" / "table_E_0002.csv").exists()
            
            # Verify JSONL content
            with open(output_dir / "document.elements.jsonl", 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            
            # First element (text)
            text_elem = json.loads(lines[0])
            assert text_elem["id"] == "E_0001"
            assert text_elem["type"] == "text_block"
            
            # Second element (table with assets)
            table_elem = json.loads(lines[1])
            assert table_elem["id"] == "E_0002"
            assert table_elem["type"] == "table"
            assert table_elem["payload"]["image_ref"] == "assets/table_E_0002.png"
            assert table_elem["payload"]["csv_ref"] == "assets/table_E_0002.csv"
            
            # Verify index content
            with open(output_dir / "document.index.json", 'r') as f:
                index = json.load(f)
            
            assert index["doc_id"] == doc_id
            assert index["manifest"]["stats"]["total_elements"] == 2
            assert index["hierarchy"] == hierarchy
            assert index["labels_used"] == labels_used
            
        finally:
            shutil.rmtree(temp_dir)