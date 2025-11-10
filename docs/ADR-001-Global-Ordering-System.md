# ADR-001: Document-Wide Column Detection and Global Ordering System

## Status
Proposed

## Context

The current PDF2EHR routing system (`ehrx/route.py`) implements per-page ordering with simple (Y,X) sorting, resetting z_order to 0 on each page. This approach is insufficient for EHR reconstruction because:

1. **Multi-column layouts** are common in medical records, requiring column-aware element associations
2. **Cross-page continuity** is lost when z_order resets per page
3. **Table/figure associations** cannot be properly linked to headings in the same column
4. **Document reading order** doesn't reflect true document flow across pages and columns

EHR documents require sophisticated layout understanding to maintain clinical context during reconstruction.

## Decision

We will implement a **two-component enhancement** to the existing routing system:

### Component 1: Document-Level Column Detector
- **Class**: `DocumentColumnDetector`
- **Purpose**: Analyze layout patterns across entire document to establish consistent column boundaries
- **Algorithm**: Enhanced k-means clustering on left-edge coordinates with gap heuristics
- **Robustness**: Fallback strategies for irregular layouts, noise handling for OCR coordinate variance

### Component 2: Global Ordering Manager  
- **Class**: `GlobalOrderingManager`
- **Purpose**: Maintain document-wide state for proper element sequencing and associations
- **Features**: 
  - Global z_order counter across all pages
  - Column-aware reading order: `(column_index, y_coordinate)`
  - Active heading context tracking per column
  - Cross-page section continuity detection

## Architecture

```
Document Processing Flow:
┌─────────────────────┐    ┌─────────────────────┐
│   Pass 1: Layout    │    │  Pass 2: Element    │
│   Analysis          │───▶│  Processing         │
│   - Collect blocks  │    │  - Apply column     │
│   - Detect columns  │    │    assignments     │
│   - Establish grid  │    │  - Global ordering  │
└─────────────────────┘    │  - Context tracking │
                           └─────────────────────┘
```

### Integration Points
1. **Pre-processing**: Column analysis before element routing
2. **Enhanced Router**: Modified `ElementRouter` with global state management
3. **Serialization**: Updated output format to include column metadata
4. **Hierarchy Builder**: Receives properly ordered elements with column context

## Rationale

### Why Deviate from Single-Pass Streaming?
- **Layout Understanding**: Column detection requires global view of document structure
- **Memory Efficiency**: Two-pass approach still maintains streaming within each pass
- **Accuracy Trade-off**: Slight memory increase for significantly better reconstruction quality

### Why Enhanced Column Detection?
- **EHR Complexity**: Medical documents have irregular layouts, mixed single/multi-column pages
- **Robustness**: Need fallback strategies when simple k-means fails
- **Noise Handling**: OCR coordinate variance requires robust clustering

### Why Global State Management?
- **Document Continuity**: EHR sections span multiple pages
- **Clinical Context**: Proper associations crucial for medical interpretation
- **Reading Flow**: Global ordering preserves intended document narrative

## Implementation Strategy

### Phase 1: Column Detection Infrastructure
```python
class DocumentColumnDetector:
    def analyze_document_layout(self, all_pages_blocks) -> ColumnLayout
    def detect_column_boundaries(self, left_edges) -> List[float]  
    def assign_elements_to_columns(self, elements, boundaries) -> List[int]
```

### Phase 2: Global Ordering Manager
```python
class GlobalOrderingManager:
    def __init__(self, column_layout: ColumnLayout)
    def get_global_z_order(self) -> int
    def track_heading_context(self, element: Dict) -> None
    def find_associated_heading(self, element: Dict) -> Optional[str]
```

### Phase 3: Enhanced Element Router
- Modify existing `ElementRouter` to accept global state
- Update `_sort_blocks_reading_order` for column-aware sorting
- Add column assignment to element metadata
- Implement heading-in-column association logic

## Consequences

### Positive
- **Accurate Reconstruction**: Proper document structure preserved for downstream use
- **Clinical Context**: Tables/figures correctly associated with relevant sections
- **Scalability**: Still handles 600+ page documents efficiently
- **Robustness**: Multiple fallback strategies for edge cases

### Negative  
- **Complexity**: More sophisticated algorithm requires careful testing
- **Memory**: Slight increase due to two-pass processing
- **Development Time**: More complex than simple per-page ordering

### Mitigation
- **Incremental Implementation**: Build on existing router without breaking current functionality
- **Comprehensive Testing**: Include edge cases (irregular layouts, mixed single/multi-column)
- **Fallback Modes**: Graceful degradation to current behavior if advanced features fail

## Future Considerations

- **ML-Based Layout Detection**: Could replace rule-based column detection
- **Semantic Section Detection**: Beyond simple heading matching
- **Layout Template Recognition**: EHR-specific layout pattern recognition

## Implementation Notes

This ADR documents the architectural decision made to address limitations in the current per-page ordering system. The decision balances theoretical complexity required for accurate EHR reconstruction with practical implementation concerns, building incrementally on the existing solid foundation in `ehrx/route.py`.

The two-component approach allows for modular development and testing while maintaining backward compatibility during the transition period.

---

**Date**: 2025-11-9
**Authors**: Xander and Claude Code
**Related Files**: `ehrx/route.py`, `ehrx/hierarchy.py`