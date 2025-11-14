# VLM-First PDF2EHR Architecture: North-Star Product Requirements Document

**Version**: 1.0  
**Date**: November 14, 2025  
**Status**: Active Development Guide  
**Authors**: Xander Hnasko

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [VLM-First Vision & Architecture](#3-vlm-first-vision--architecture)
4. [Technical Specifications (MVP Focus)](#4-technical-specifications-mvp-focus)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Success Metrics & Quality Assurance](#6-success-metrics--quality-assurance)

---

## 1. Executive Summary

### 1.1 Strategic Vision

PDF2EHR is undergoing a fundamental architectural transformation from rule-based LayoutParser detection to Vision-Language Model (VLM) powered semantic understanding. This shift represents a paradigmatic advancement from spatial pattern recognition to true clinical document comprehension.

**Core Transformation**: Replace brittle, academic-document-trained LayoutParser with Google Gemini cascade architecture that understands EHR semantic context, clinical relationships, and complex multi-column medical layouts.

### 1.2 Business Impact

- **Accuracy Gains**: 40-60% improvement in content extraction completeness through semantic understanding
- **Robustness**: Graceful handling of complex, non-standard EHR layouts without manual configuration
- **Extensibility**: Rapid adaptation to new document types through prompt engineering vs model retraining
- **Clinical Context**: Preservation of medical relationships critical for downstream healthcare applications

### 1.3 Technical Foundation

**Model Cascade Strategy**: 
- **Gemini Flash**: Fast, cost-effective processing for standard text extraction and basic classification
- **Gemini Pro**: Complex semantic understanding, table processing, and clinical relationship extraction
- **Human Review**: Systematic escalation for low-confidence cases with full auditability

**Key Innovation**: Hybrid semantic chunking that maintains clinical context across document sections while preserving existing proven components (column detection, hierarchical structuring, serialization pipeline).

---

## 2. Current State Analysis

### 2.1 Existing Architecture Strengths

Based on analysis of the current codebase, several components demonstrate strong engineering and should be preserved:

#### 2.1.1 Document Processing Pipeline (`ehrx/hierarchy.py`)
- **Hierarchical Structuring**: Well-designed category mapping (Demographics, Vitals, Orders, Meds, Notes, Labs)
- **Section Detection**: Robust heuristic-based heading detection with configurable patterns
- **Document Grouping**: Effective consecutive page labeling and grouping logic
- **Deterministic Approach**: Clear, debuggable processing with confidence scoring

#### 2.1.2 Enhanced Column Detection System (`ehrx/layout/`)
- **Two-Pass Architecture**: Document-wide layout analysis followed by element processing
- **Global Ordering Manager**: Cross-page continuity with proper z-order sequencing
- **Robust Column Detection**: K-means clustering with silhouette analysis and fallback strategies
- **33 Passing Tests**: Comprehensive test coverage indicating production readiness

#### 2.1.3 Serialization Infrastructure (`ehrx/serialize.py`)
- **Streaming JSONL**: Memory-efficient element-by-element processing
- **Asset Management**: Proper handling of images, CSVs, and structured data
- **Hierarchical Index**: Clean output format suitable for downstream applications
- **Thread-Safe Operations**: Production-ready concurrent processing

### 2.2 Critical Limitations Requiring VLM Solution

#### 2.2.1 LayoutParser Fundamental Mismatch
```
Training Data: PubLayNet (academic papers)
Target Domain: Clinical EHR documents
Result: 40-60% missed content regions
```

**Specific Failure Modes**:
- Complex multi-column EHR layouts with irregular spacing
- Overlapping content (stamps, handwritten annotations over text)
- Non-standard medical forms and checklists
- Poor scan quality artifacts that confuse object detection
- Limited element taxonomy (4 types vs 15+ needed for EHRs)

#### 2.2.2 Semantic Understanding Gap
- **No Clinical Context**: Cannot distinguish medication lists from lab results based on content
- **Relationship Blindness**: Cannot associate tables with relevant section headers
- **Cross-Page Ignorance**: Cannot maintain clinical narratives spanning multiple pages
- **No Content Validation**: Cannot detect when OCR misinterprets medical terminology

#### 2.2.3 Processing Brittleness
- **Cascading Failures**: Single missed region = permanently lost clinical content
- **No Graceful Degradation**: Binary success/failure with no confidence gradation
- **Configuration Complexity**: Requires manual tuning for different document types
- **No Self-Correction**: Cannot identify and recover from its own mistakes

---

## 3. VLM-First Vision & Architecture

### 3.1 Architectural Principles

#### 3.1.1 Semantic-First Processing
**Principle**: Understand document meaning before imposing structure
- VLMs analyze clinical content semantics to inform structural decisions
- Content type drives processing approach rather than spatial heuristics
- Clinical relationships preserved through context-aware chunking

#### 3.1.2 Confidence-Driven Escalation
**Principle**: Never guess; systematic quality assurance through model cascade
- Flash handles high-confidence standard cases (80%+ of content)
- Pro processes complex semantics and low-confidence Flash outputs
- Human review for uncertain cases with full provenance tracking

#### 3.1.3 Hybrid Intelligence Architecture
**Principle**: Combine VLM semantic understanding with proven engineering
- Preserve successful components (column detection, hierarchy, serialization)
- Enhance existing pipeline with VLM semantic analysis
- Maintain deterministic behavior where possible for auditability

### 3.2 Core Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                   VLM-First PDF2EHR Pipeline                   │
└─────────────────────────────────────────────────────────────────┘

Input: EHR PDF
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│               Document Preprocessing                            │
│  • PDF to high-res images                                      │
│  • Enhanced column detection (preserve existing)               │
│  • Page-level document type classification (Flash)             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Hybrid Semantic Chunking                          │
│  • Section-based primary chunks                                │
│  • Context injection (headers, metadata)                       │
│  • Cross-chunk relationship tracking                           │
│  • Overlap buffers for boundary preservation                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              VLM Processing Cascade                             │
│                                                                 │
│  Flash (Fast/Cheap)          Pro (Complex/Expensive)           │
│  ├─ Text extraction          ├─ Table structure                 │
│  ├─ Basic classification     ├─ Figure analysis                 │
│  ├─ High-confidence cases    ├─ Clinical relationships          │
│  └─ Confidence scoring       └─ Low-confidence escalation       │
│                                                                 │
│           Human Review (Uncertain Cases)                       │
│           ├─ Bounding box provenance                           │
│           ├─ Accept/reject/edit interface                      │
│           └─ Continuous learning feedback                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│           Enhanced Hierarchy Generation                         │
│  • Clinical-aware section mapping                              │
│  • Cross-reference resolution                                  │
│  • Temporal relationship tracking                              │
│  • Global ordering with semantic context                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Auditable Output Generation                       │
│  • JSONL with provenance tracking                              │
│  • Hierarchical index (preserve existing format)               │
│  • Asset management (images, CSVs)                             │
│  • Confidence metadata and review flags                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Enhanced EHR Semantic Ontology

#### 3.3.1 Expanded Element Types (15+ vs Current 4)

**Document Structure:**
- `document_header`: Hospital/clinic identifying information
- `patient_demographics`: Name, DOB, MRN, contact information  
- `page_metadata`: Page numbers, dates, document IDs
- `section_header`: Major divisions (PROBLEMS, MEDICATIONS, LABS)
- `subsection_header`: Minor headings within sections

**Clinical Content:**
- `clinical_paragraph`: Free-text clinical narratives with semantic tagging
- `medication_table`: Structured medication lists with dosages, frequencies
- `lab_results_table`: Laboratory values with normal ranges and units
- `vital_signs_table`: Temperature, BP, pulse with temporal context
- `problem_list`: Diagnoses with ICD codes and temporal qualifiers
- `assessment_plan`: Clinical reasoning and treatment plans
- `list_items`: Bullet points, numbered lists with semantic context

**Special Content:**
- `handwritten_annotation`: Handwritten notes with clinical context
- `stamp_signature`: Official stamps, signatures with validation metadata
- `medical_figure`: Graphs, charts, anatomical diagrams with interpretation
- `form_field_group`: Label-value pairs from structured medical forms

**Administrative:**
- `margin_content`: Headers, footers, confidentiality notices
- `uncategorized`: Content requiring human review with confidence scores

#### 3.3.2 Semantic Attributes for Enhanced Processing

Each element includes rich semantic metadata:

```json
{
  "element_id": "unique_identifier",
  "semantic_type": "medication_table", 
  "content": "extracted_text_or_structured_data",
  "confidence_scores": {
    "extraction": 0.94,
    "classification": 0.88, 
    "clinical_context": 0.91
  },
  "clinical_metadata": {
    "temporal_qualifier": "current",
    "clinical_domain": "pharmacology",
    "cross_references": ["problem_list_item_123"],
    "requires_validation": false
  },
  "provenance": {
    "bbox_px": [x1, y1, x2, y2],
    "page": 5,
    "processing_model": "gemini_flash",
    "human_reviewed": false,
    "review_flags": []
  }
}
```

---

## 4. Technical Specifications (MVP Focus)

### 4.1 VLM Processing Engine

#### 4.1.1 Google Gemini Cascade Architecture

**Two-Tier Processing Strategy**:

**Tier 1: Gemini Flash** (Fast, Cost-Effective)
- Text extraction and basic OCR validation
- Document type and section classification
- High-confidence element detection (>85% certainty)
- Basic clinical term recognition
- Handles ~80% of standard content

**Tier 2: Gemini Pro** (Complex, Semantic-Heavy)
- Table structure interpretation and data extraction
- Figure/chart analysis with clinical context
- Low-confidence Flash output review
- Cross-chunk relationship analysis
- Clinical narrative coherence validation

**Escalation Triggers** (Flash → Pro):
- Confidence score < 0.85
- Complex element types (tables, figures, handwriting)
- Detected clinical relationships
- Poor OCR quality (< 0.7 score)
- Cross-chunk dependencies detected

**Tier 3: Human Review** (Uncertain Cases)
- Confidence < 0.80 after Pro analysis
- Critical clinical content with validation flags
- Conflicting outputs between models
- Novel document structures

### 4.2 Hybrid Semantic Chunking System

**Chunking Strategy** (Section-Based, Not Page-Based):
- Align chunks with natural EHR document structure (sections, not arbitrary page breaks)
- Preserve clinical narrative coherence within sections
- Dynamic chunk sizing based on content complexity
- 15% overlap between adjacent chunks to prevent boundary content loss

**Context Injection** (Each Chunk Receives):
- Document type ("Clinical Notes", "Lab Results", etc.)
- Section hierarchy (["Notes" → "Progress Notes" → "Daily Assessment"])
- Patient metadata (age, gender, relevant clinical context)
- Preceding section summaries (clinical context from earlier in document)
- Cross-references to related sections
- Temporal context (dates/times for clinical timeline)

**Cross-Chunk Relationship Management**:
- Track forward references ("see lab results below")
- Resolve backward references ("as noted in medication list above")
- Maintain pending reference registry until resolution
- Create bidirectional links between related chunks
- Provide relevant prior context to each subsequent chunk

### 4.3 Clinical Relationship Detection

**Temporal Relationships** (Examples):
- "Patient continued on previous regimen" → Link to prior medication list
- "Follow-up labs improved" → Connect to baseline lab values
- "Discharge medications same as admission" → Cross-document reference

**Causal Relationships**:
- Medication dosage changes ↔ Lab value trends
- Problem list items ↔ Assessment/plan sections
- Vital sign abnormalities ↔ Clinical interventions

**Implementation Approach**:
- Pattern matching for temporal indicators ("previous", "prior", "continued", etc.)
- VLM semantic analysis for implicit relationships
- Confidence scoring for each detected relationship (threshold: 0.7)
- Reference resolution across document sections

### 4.4 Clinical Content Validation

**Automated Validation Checks**:
- Medical terminology validation against clinical dictionaries
- Unit consistency (mg vs mcg, metric vs imperial)
- Lab value normal range validation
- Drug name and dosage format validation
- Temporal consistency across document sections

**Quality Flags** (Trigger Human Review):
- Unrecognized medical terminology
- Values outside expected ranges
- Inconsistent medication-lab relationships
- Missing expected sections for document type
- Low VLM confidence on critical content

### 4.5 Integration with Existing Strengths

**Preserve + Enhance Strategy**: Keep proven components, add VLM semantic layer

#### 4.5.1 Enhanced Column Detection (`ehrx/layout/`)

**Three-Stage Process**:
1. **Rule-Based Baseline**: Use existing k-means clustering column detection (33 passing tests)
2. **VLM Validation**: VLM validates semantic correctness of column boundaries
3. **Merge Insights**: Combine rule-based precision with VLM semantic understanding

**VLM Validation Questions**:
- Do boundaries correctly separate semantic content?
- Are there blocks spanning columns inappropriately?
- Should boundaries adjust for better clinical content grouping?
- Do clinical relationships suggest different structure?

#### 4.5.2 Enhanced Hierarchy Generation (`ehrx/hierarchy.py`)

**Three-Stage Process**:
1. **Rule-Based Baseline**: Use existing heuristic-based heading detection and category mapping
2. **VLM Semantic Enhancement**: Add clinical context and relationship understanding
3. **Cross-Reference Resolution**: Link related sections across document categories

**VLM Enhancements**:
- Validate section classifications with clinical context
- Identify cross-category relationships (e.g., medications → labs)
- Improve temporal relationship tracking across documents
- Detect semantic groupings beyond spatial heuristics

### 4.6 Provenance and Auditability

**Every Element Tracks**:

**Spatial Provenance**:
- Bounding box (pixel and PDF coordinates)
- Page number and column assignment
- Original image crop reference

**Processing Provenance**:
- Detection model used ("gemini_flash", "gemini_pro", "rule_based")
- Processing timestamp and version
- Confidence scores (extraction, classification, clinical validity)

**Semantic Provenance**:
- Original vs final classification
- Classification reasoning
- Cross-references to related elements

**Human Review Tracking** (When Applicable):
- Review status and timestamp
- Reviewer decisions (accept/reject/modify/flag)
- Original vs corrected values
- Reviewer notes and confidence adjustments

**Audit Trail** (Document-Level):
- Complete processing step history
- Human interventions log
- Quality metrics and validation results
- Full traceability for regulatory compliance

---

## 5. Implementation Roadmap

### 5.1 Phase 1: MVP Foundation (Weeks 1-8)

**Goal**: End-to-end VLM processing pipeline with basic semantic understanding

#### Week 1-2: VLM Client Infrastructure

**Deliverables**:
- Google Cloud Vertex AI client with authentication
- Gemini Flash and Pro API integration
- Prompt templates for EHR content extraction
- Error handling, retry logic, and rate limiting
- Basic cost tracking per API call

**Key Decisions**:
- Prompt structure for clinical document analysis
- Confidence threshold calibration (initial: 0.85 for Flash → Pro)
- API timeout and retry strategies

#### Week 3-4: Basic Chunking + Context Injection

**Deliverables**:
- Section-based chunking (preserve page boundaries for MVP simplicity)
- Context injection: document type, section headers, patient metadata
- 15% overlap handling between adjacent chunks
- Integration with existing PDF → image pipeline

**MVP Simplifications**:
- Static chunk sizing initially (optimize in Phase 2)
- Page-aligned chunks to avoid complex boundary handling
- Basic context only (expand in Phase 2)

#### Week 5-6: Enhanced Semantic Ontology

**Deliverables**:
- Implement 15+ semantic element types (vs current 4)
- Confidence scoring for extraction and classification
- Flash → Pro cascade for complex elements
- Backward-compatible output format (preserve existing JSONL structure)

**Element Types Priority**:
- High: clinical_paragraph, medication_table, lab_results_table, section_header
- Medium: vital_signs_table, problem_list, document_header
- Low: handwritten_annotation, medical_figure (Pro-only initially)

#### Week 7-8: Integration + End-to-End Testing

**Deliverables**:
- VLM-enhanced column detection integration
- VLM-enhanced hierarchy generation
- Basic provenance tracking (model used, confidence, timestamp)
- End-to-end pipeline testing with 10+ diverse EHR samples

**Testing Focus**:
- Completeness: Are we capturing more content than LayoutParser?
- Accuracy: Are semantic classifications correct?
- Performance: Processing time and cost per page
- Compatibility: Does existing downstream code still work?

**Phase 1 Success Criteria**:
- ✅ Process complete multi-page EHR documents through VLM pipeline
- ✅ 15+ semantic element types correctly classified (>85% accuracy)
- ✅ Flash handles >70% of content (cost optimization)
- ✅ 20-30% improvement in content extraction vs LayoutParser baseline
- ✅ Backward compatibility maintained for existing output consumers

### 5.2 Phase 2: Advanced Semantic Understanding (Weeks 9-20)

**Goal**: Production-ready semantic processing with clinical relationships

#### Week 9-11: Advanced Chunking System

**Deliverables**:
- Dynamic chunk sizing based on content complexity
- Semantic boundary detection (VLM analyzes section transitions)
- Enhanced context injection with clinical metadata
- Cross-chunk relationship tracking and resolution

**Advanced Features**:
- Chunk boundaries align with semantic sections, not pages
- Forward/backward reference resolution ("see above", "continued below")
- Pending reference registry for cross-chunk clinical relationships
- Context accumulation as document processing progresses

#### Week 12-14: Clinical Relationship Extraction

**Deliverables**:
- Temporal relationship detection ("continued from previous", "follow-up labs")
- Causal relationship mapping (medication changes ↔ lab value trends)
- Cross-reference resolution with confidence scoring
- Clinical narrative coherence validation

**Relationship Types**:
- Temporal: Links to prior/subsequent content
- Causal: Interventions → outcomes
- Associative: Related clinical findings

#### Week 15-17: Enhanced Complex Content Processing

**Tables**: Structured extraction with semantic column headers and row validation
**Figures**: Chart/graph interpretation with clinical context
**Forms**: Label-value pair extraction from structured medical forms
**Handwriting**: Specialized processing for handwritten annotations

#### Week 18-20: Production Optimization

**Performance**:
- Parallel chunk processing (5-10x throughput improvement)
- Intelligent caching for repeated patterns
- API rate limiting and cost controls
- Memory efficiency for 100+ page documents

**Reliability**:
- Comprehensive error handling and fallback chains
- VLM API failure strategies (retry, alternative provider, rule-based fallback)
- Quality regression detection and alerting
- Enhanced logging and monitoring

**Phase 2 Success Criteria**:
- ✅ Cross-chunk relationships >85% accurate
- ✅ Complex tables/figures correctly processed
- ✅ <5 min processing time per 100 pages
- ✅ <$0.20 cost per page
- ✅ <10% human review rate

### 5.3 Phase 3: Enterprise Features (Weeks 21-32)

**Focus**: Continuous improvement, advanced analytics, production scalability

**Continuous Learning** (Weeks 21-24):
- Human feedback collection and processing
- Model performance tracking and drift detection
- Automated prompt optimization
- A/B testing framework for prompts and models

**Advanced Analytics** (Weeks 25-28):
- Processing analytics dashboard
- Quality trend analysis and alerting
- Cost optimization recommendations
- Performance benchmarking

**Production Scalability** (Weeks 29-32):
- Deployment pipeline and monitoring
- Horizontal scaling (1000+ pages/day capacity)
- Advanced security and privacy controls
- Downstream integration testing

**Phase 3 Success Criteria**:
- ✅ Measurable continuous quality improvement
- ✅ Production monitoring and alerting
- ✅ Cost targets met (<$0.15/page optimized)
- ✅ Scalable to enterprise volumes

### 5.4 Migration Strategy

**Guiding Principle**: Zero disruption to existing downstream systems

#### Backward Compatibility

- Maintain existing JSONL + hierarchical JSON output format
- Preserve all existing element fields and metadata structure
- Add new semantic fields as optional extensions
- Ensure existing code continues working without changes

#### Gradual Rollout Phases

**Phase 1: Parallel Validation** (2-4 weeks)
- Run LayoutParser and VLM pipelines in parallel
- Compare outputs for quality and completeness
- Human validation of discrepancies
- Build confidence before switching

**Phase 2: Selective Deployment** (4-6 weeks)
- VLM for complex documents (LayoutParser struggles)
- LayoutParser for simple, well-structured documents
- Gradual expansion based on performance data

**Phase 3: Full Migration** (Ongoing)
- VLM as primary pipeline
- Automatic fallback to LayoutParser for VLM failures
- Comprehensive monitoring and alerting
- Human review for uncertain cases

---

## 6. Success Metrics & Quality Assurance

### 6.1 Primary Success Metrics

#### 6.1.1 Accuracy Metrics

**Content Extraction Completeness**
- **Target**: 95%+ of document content successfully extracted and classified
- **Measurement**: Manual review of random document samples with ground truth annotation
- **Baseline**: Current LayoutParser achieves ~60-70% completeness on complex EHRs
- **Milestone Targets**:
  - Phase 1: 80% completeness (20% improvement over baseline)
  - Phase 2: 90% completeness (40% improvement over baseline) 
  - Phase 3: 95%+ completeness (50%+ improvement over baseline)

**Semantic Classification Accuracy**
- **Target**: 92%+ elements correctly classified by semantic type
- **Measurement**: Precision, recall, and F1-score per semantic element type
- **Focus Areas**: Complex elements (tables, figures, handwriting, clinical forms)
- **Quality Gates**: No semantic type below 85% accuracy in production

**Clinical Relationship Accuracy**
- **Target**: 85%+ clinical relationships correctly identified and linked
- **Measurement**: Evaluation against clinical expert annotations
- **Examples**: Medication-lab relationships, temporal references, causal links
- **Progressive Improvement**: Start at 70% in Phase 1, reach 85% by Phase 3

#### 6.1.2 Efficiency Metrics

**Processing Performance**
- **Speed Target**: Complete document processing in <5 minutes per 100 pages
- **Cost Target**: <$0.20 total processing cost per page (including VLM API costs)
- **Scalability**: Linear scaling to 1000+ pages per day with horizontal infrastructure

**Automation Rate**
- **Target**: 90%+ documents processed without human intervention
- **Measurement**: Percentage of elements passing confidence thresholds
- **Quality Gate**: Human review rate <10% for production deployment

**Model Efficiency**
- **Flash Utilization**: 85%+ of elements processed successfully by Flash (cheaper model)
- **Pro Escalation**: <20% of elements require Pro model analysis
- **Human Escalation**: <5% of elements require human review

---

## Conclusion

This MVP-focused PRD establishes the foundation for PDF2EHR's transformation into a VLM-powered, semantically-aware EHR processing system. The document provides:

1. **Clear Strategic Vision**: VLM-first architecture with semantic understanding as the core differentiator
2. **Pragmatic Technical Architecture**: High-level design preserving existing strengths while adding VLM capabilities
3. **Phased Roadmap**: 8-week MVP (Phase 1), followed by advanced features (Phases 2-3)
4. **Measurable Success Criteria**: Concrete accuracy, performance, and cost targets

**Key Success Factors**:
- Systematic VLM cascade (Flash → Pro → Human) for cost-effective processing
- Hybrid approach preserving proven components (column detection, hierarchy generation)
- Section-based semantic chunking maintaining clinical context
- Comprehensive provenance tracking for auditability
- Backward compatibility ensuring zero disruption to existing systems

This document serves as an accessible guide for AI agents and development teams, balancing technical detail with clarity to ensure consistent progress toward a robust, accurate EHR processing system.

---

*Document Version: 1.0 (Streamlined)*
*Date: November 14, 2025*
*Focus: MVP Development Guide*
