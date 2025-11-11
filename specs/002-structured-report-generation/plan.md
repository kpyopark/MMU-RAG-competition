# Implementation Plan: Structured Report Generation with Iterative Section Synthesis

**Feature**: Structured Report Generation
**Branch**: `002-structured-report-generation`
**Created**: 2025-11-11
**Input Spec**: `/home/postgres/devel/MMU-RAG-competition/specs/002-structured-report-generation/spec.md`

---

## Executive Summary

This plan details the implementation of a comprehensive structured report generation system that transforms the current single-pass report generation into a multi-chapter, iterative section synthesis approach. The system will generate 2,500-4,000 word reports with 4-6 analytical perspectives, managing output token limits through iterative generation and progressive context summarization.

**Key Changes**:
- Replace single-pass final report generation with structured chapter/section framework
- Implement iterative section generation with context propagation
- Add multi-perspective analysis engine
- Introduce progressive context summarization system
- Enhance citation tracking with section-level attribution

**Impact**:
- Report depth increases 5-8× (from ~500 to 2,500-4,000 words)
- Multi-perspective coverage (4-6 distinct viewpoints)
- Zero output token truncation through iterative generation
- Professional report structure suitable for stakeholder presentation

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TTD_DR_Pipeline (Modified)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 1: Research & Planning (Existing)                      │  │
│  │  - generate_research_plan()                                  │  │
│  │  - generate_initial_draft()                                  │  │
│  │  - perform_iterative_search_and_synthesis()                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ NEW: Phase 2: Report Structure Generation                    │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────┐      │  │
│  │  │ ReportStructureGenerator                           │      │  │
│  │  │  - analyze_query_perspectives()                    │      │  │
│  │  │  - generate_chapter_outline()                      │      │  │
│  │  │  - create_section_specifications()                 │      │  │
│  │  │  Input: query, plan, draft, Q&A history           │      │  │
│  │  │  Output: ReportStructure                           │      │  │
│  │  └────────────────────────────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ NEW: Phase 3: Iterative Section Generation                   │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────┐      │  │
│  │  │ SectionGenerator                                   │      │  │
│  │  │  - generate_executive_summary()                    │      │  │
│  │  │  - generate_chapter_section()                      │      │  │
│  │  │  - generate_conclusion()                           │      │  │
│  │  │  Input: section_spec, context_summary             │      │  │
│  │  │  Output: section_content (≤2048 tokens)           │      │  │
│  │  └────────────────────────────────────────────────────┘      │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────┐      │  │
│  │  │ ContextManager                                     │      │  │
│  │  │  - compress_section_to_summary()                   │      │  │
│  │  │  - build_generation_context()                      │      │  │
│  │  │  - maintain_sliding_window()                       │      │  │
│  │  │  Context compression: 60% token reduction          │      │  │
│  │  └────────────────────────────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ NEW: Phase 4: Quality Validation & Assembly                  │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────┐      │  │
│  │  │ QualityValidator                                   │      │  │
│  │  │  - validate_section_depth()      (≥300 words)      │      │  │
│  │  │  - check_citation_density()      (≥1 per 150 words)│      │  │
│  │  │  - detect_redundancy()           (≤15% overlap)    │      │  │
│  │  │  - measure_coherence()           (≥0.8 similarity) │      │  │
│  │  └────────────────────────────────────────────────────┘      │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────┐      │  │
│  │  │ ReportAssembler                                    │      │  │
│  │  │  - assemble_final_report()                         │      │  │
│  │  │  - organize_citations_by_chapter()                 │      │  │
│  │  │  - format_markdown_structure()                     │      │  │
│  │  └────────────────────────────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Models

```python
@dataclass
class ReportStructure:
    """Defines the chapter/section hierarchy for a report."""
    executive_summary: SectionSpec
    chapters: List[Chapter]
    conclusion: SectionSpec
    estimated_word_count: int
    estimated_sections: int

@dataclass
class Chapter:
    """Represents a report chapter with multiple sections."""
    title: str
    perspective: str  # Financial, Technical, Regulatory, etc.
    sections: List[SectionSpec]
    chapter_number: int

@dataclass
class SectionSpec:
    """Specification for generating a single section."""
    title: str
    chapter_number: int
    section_number: int
    perspective: str
    guidance: str  # What to cover in this section
    target_word_count: int  # 300-500 words
    max_output_tokens: int  # 2048

@dataclass
class GeneratedSection:
    """Result of section generation."""
    spec: SectionSpec
    content: str  # Markdown content
    word_count: int
    citations_used: List[str]  # URLs referenced
    generation_time: float
    summary: str  # Compressed for context (≤200 tokens)

@dataclass
class ContextSummary:
    """Compressed context for section generation."""
    key_insights: List[str]  # Max 10 insights
    previous_sections: List[str]  # Compressed summaries
    research_highlights: str  # From Q&A history
    total_tokens: int  # Should be ≤ 40% of context window
```

### Prompt Engineering

```python
STRUCTURE_GENERATION_PROMPT = """
You are a research report structuring expert. Analyze the user's query and research context to create a comprehensive report structure.

**User's Original Query:**
{query}

**Research Plan:**
{plan}

**Research Findings Summary:**
{research_summary}

**Task:** Create a detailed chapter-and-section outline for a comprehensive research report.

**Requirements:**
1. Identify 4-6 distinct analytical perspectives relevant to the query
2. Create 4-6 chapters, each addressing one major perspective
3. Within each chapter, define 3-5 specific sections
4. Each section should be focused and substantive (300-500 words target)
5. Avoid redundancy - each section should cover unique aspects

**Perspective Examples:**
- Financial/Economic Analysis
- Technical/Operational Analysis
- Regulatory/Legal Analysis
- Strategic/Competitive Analysis
- Risk/Challenge Analysis
- Market/Industry Analysis
- Social/Ethical Analysis
- Historical/Trend Analysis

**Output Format:**
```json
{
  "executive_summary": {
    "title": "Executive Summary",
    "guidance": "High-level synthesis...",
    "target_word_count": 400
  },
  "chapters": [
    {
      "title": "Chapter Title",
      "perspective": "Financial Analysis",
      "sections": [
        {
          "title": "Section Title",
          "guidance": "Specific aspects to cover...",
          "target_word_count": 350
        }
      ]
    }
  ],
  "conclusion": {
    "title": "Conclusion and Implications",
    "guidance": "Forward-looking synthesis...",
    "target_word_count": 400
  }
}
```
"""

SECTION_GENERATION_PROMPT = """
You are writing a specific section of a comprehensive research report.

**Overall Query:** {query}

**Current Section:** {section_title}
**Chapter:** {chapter_title} (Perspective: {perspective})
**Section Guidance:** {section_guidance}
**Target Length:** {target_word_count} words

**Context from Previous Sections:**
{context_summary}

**Research Data Available:**
{research_data}

**Instructions:**
1. Write a detailed, substantive section of {target_word_count} words
2. Maintain coherence with previous sections (reference them where appropriate)
3. Use inline citations [Source N] for all factual claims
4. Provide new insights - do not repeat content from previous sections
5. Follow the perspective: {perspective}

**Output Format:**
Return ONLY the markdown-formatted section content with:
- Clear paragraph structure
- Inline citations [Source N]
- Cross-references to previous sections where relevant
- No section heading (will be added during assembly)
"""

CONTEXT_COMPRESSION_PROMPT = """
Compress the following section into a concise summary for context propagation.

**Section Title:** {section_title}
**Full Content:**
{section_content}

**Task:** Create a summary that:
1. Captures 3-5 key insights or findings
2. Is ≤200 tokens
3. Enables coherent reference in future sections
4. Maintains critical facts and numbers

**Output Format:**
Return ONLY the compressed summary as a single paragraph.
"""
```

---

## Implementation Phases

### Phase 0: Research & Analysis
**Goal**: Understand current pipeline architecture and identify integration points

**Tasks**:
1. ✅ Review `src/pipeline.py` structure and prompts
2. ✅ Analyze current `generate_final_report()` implementation
3. ✅ Map SSE streaming callback integration points
4. ✅ Identify Q&A history and citation tracking mechanisms
5. ✅ Document Gemini API token limits and constraints

**Deliverables**:
- Architecture analysis document
- Integration point specifications
- Risk assessment for backward compatibility

### Phase 1: Core Data Models & Structure Generation
**Goal**: Implement report structure generation and data models

**Tasks**:
1. Create data models (`src/report_structure.py`):
   - `ReportStructure`, `Chapter`, `SectionSpec`
   - `GeneratedSection`, `ContextSummary`

2. Implement `ReportStructureGenerator` (`src/structure_generator.py`):
   - `analyze_query_perspectives()`: Extract perspectives from query
   - `generate_chapter_outline()`: Create chapter structure with LLM
   - `create_section_specifications()`: Define section-level specs

3. Add prompt templates:
   - `STRUCTURE_GENERATION_PROMPT` with JSON schema validation

4. Unit tests:
   - Test perspective analysis with various query types
   - Validate JSON structure parsing
   - Test adaptive chapter count (2-3 for simple, 5-7 for complex)

**Acceptance Criteria**:
- Structure generation completes in ≤10 seconds
- Output contains 4-6 chapters for complex queries
- Each chapter has 3-5 section specs
- All section specs have valid target word counts (300-500)

### Phase 2: Context Management System
**Goal**: Implement progressive context summarization for iterative generation

**Tasks**:
1. Create `ContextManager` (`src/context_manager.py`):
   - `compress_section_to_summary()`: Compress to ≤200 tokens
   - `build_generation_context()`: Assemble context for section generation
   - `maintain_sliding_window()`: Keep recent 3-5 sections in full detail

2. Implement summarization:
   - Use Gemini API with `CONTEXT_COMPRESSION_PROMPT`
   - Extract 3-5 key insights per section
   - Target 60% token reduction vs. full content

3. Add context window tracking:
   - Monitor cumulative token usage
   - Trigger compression when approaching limits
   - Log compression ratios for optimization

4. Unit tests:
   - Verify 60% compression ratio
   - Test sliding window maintenance
   - Validate insight extraction quality

**Acceptance Criteria**:
- Section summaries ≤200 tokens
- Compression achieves ≥60% token reduction
- Sliding window maintains 3-5 most recent sections
- No context overflow errors in 50+ section reports

### Phase 3: Iterative Section Generator
**Goal**: Implement per-section generation with context propagation

**Tasks**:
1. Create `SectionGenerator` (`src/section_generator.py`):
   - `generate_section()`: Generate single section with context
   - `generate_executive_summary()`: Special handling for exec summary
   - `generate_conclusion()`: Synthesize all chapters

2. Implement generation loop:
   - Iterate through section specs in order
   - Build context from previous section summaries
   - Generate section with `SECTION_GENERATION_PROMPT`
   - Validate output ≤2048 tokens
   - Compress to summary for next iteration

3. Add retry logic:
   - Retry up to 3 times on API errors
   - Implement exponential backoff
   - Save partial report state on final failure

4. SSE streaming integration:
   - Send progress updates per section
   - Include section number, chapter, progress percentage
   - Stream partial sections for user feedback

5. Unit tests:
   - Test single section generation
   - Verify context propagation across sections
   - Test retry mechanism
   - Validate output token limits

**Acceptance Criteria**:
- Each section generates in ≤30 seconds
- No sections exceed 2048 output tokens
- Context successfully propagates across 20+ sections
- Retry succeeds ≥90% for transient errors
- Progress updates sent via SSE every 30 seconds

### Phase 4: Quality Validation System
**Goal**: Ensure generated sections meet quality standards

**Tasks**:
1. Create `QualityValidator` (`src/quality_validator.py`):
   - `validate_section_depth()`: Check word count ≥300
   - `check_citation_density()`: Verify ≥1 citation per 150 words
   - `detect_redundancy()`: Measure overlap between sections
   - `measure_coherence()`: Semantic similarity of transitions

2. Implement redundancy detection:
   - Use sentence embeddings for similarity comparison
   - Flag sections with >70% similarity
   - Trigger regeneration with diversification prompt

3. Add citation tracking:
   - Extract inline citations [Source N] from sections
   - Map citations to source URLs from Q&A history
   - Validate citation references exist

4. Implement regeneration triggers:
   - Auto-regenerate if word count <250
   - Auto-regenerate if citation density <0.5 per 150 words
   - Auto-regenerate if redundancy >70%
   - Max 2 regeneration attempts per section

5. Unit tests:
   - Test word count validation
   - Test citation extraction and density calculation
   - Test redundancy detection with similar content
   - Verify regeneration logic

**Acceptance Criteria**:
- Word count validation catches sections <300 words
- Citation density correctly calculated
- Redundancy detection identifies >70% overlap
- Regeneration improves quality on 2nd attempt
- Validation completes in ≤2 seconds per section

### Phase 5: Report Assembly & Formatting
**Goal**: Assemble sections into final structured report with citations

**Tasks**:
1. Create `ReportAssembler` (`src/report_assembler.py`):
   - `assemble_final_report()`: Combine all sections
   - `organize_citations_by_chapter()`: Group citations
   - `format_markdown_structure()`: Apply consistent formatting

2. Implement citation organization:
   - Extract all unique citations from sections
   - Group by chapter
   - Generate citations section with URLs and titles
   - Add inline reference mapping

3. Add markdown formatting:
   - H1 for report title
   - H2 for chapters
   - H3 for sections
   - Consistent spacing and structure

4. Generate metadata:
   - Total word count
   - Section count
   - Chapter count
   - Citation count
   - Generation time
   - Perspective coverage

5. Unit tests:
   - Test section assembly order
   - Verify citation deduplication
   - Test markdown structure validity
   - Validate metadata accuracy

**Acceptance Criteria**:
- Final report has correct chapter/section hierarchy
- Citations organized by chapter
- No duplicate citations
- Markdown structure valid
- Metadata accurately reflects report stats

### Phase 6: Pipeline Integration
**Goal**: Integrate new system into TTD_DR_Pipeline

**Tasks**:
1. Modify `src/pipeline.py`:
   - Add `generate_report_structure()` after research phase
   - Replace `generate_final_report()` with `generate_structured_report()`
   - Integrate SSE progress updates
   - Add fallback to legacy single-pass generation

2. Update callback system:
   - Send structure generation update
   - Send per-section progress (e.g., "Section 5/20")
   - Send chapter completion updates
   - Send final assembly update

3. Add configuration:
   - `ENABLE_STRUCTURED_REPORTS` flag (default: True)
   - `MAX_SECTIONS_PER_CHAPTER` (default: 5)
   - `TARGET_SECTION_WORDS` (default: 350)
   - `MAX_OUTPUT_TOKENS_PER_SECTION` (default: 2048)

4. Backward compatibility:
   - Preserve existing `/run` and `/evaluate` endpoints
   - Add `?structured=false` query param for legacy mode
   - Maintain SSE event format compatibility

5. Integration tests:
   - Test full pipeline end-to-end
   - Verify SSE streaming works
   - Test fallback to legacy mode
   - Validate `/run` and `/evaluate` compliance

**Acceptance Criteria**:
- Pipeline generates structured reports by default
- SSE streaming includes all progress updates
- Legacy mode works with `?structured=false`
- Existing endpoints remain compatible
- End-to-end test passes with competition queries

### Phase 7: Testing & Optimization
**Goal**: Validate system performance and optimize for production

**Tasks**:
1. Performance testing:
   - Test with 10 diverse queries (simple → complex)
   - Measure generation time per section
   - Measure total report generation time
   - Profile token usage and compression efficiency

2. Quality validation:
   - Evaluate report depth (word count)
   - Assess perspective coverage
   - Verify citation density
   - Measure section coherence

3. Edge case testing:
   - Test with vague queries (ambiguity handling)
   - Test with limited research data (<3 sources)
   - Test with 30+ section reports (context overflow)
   - Test API failure mid-generation (resume capability)
   - Test redundant content detection

4. Optimization:
   - Tune context compression for balance (speed vs. quality)
   - Optimize prompt templates for clarity
   - Implement caching for repeated summarizations
   - Add parallel section generation for independent sections

5. Load testing:
   - Test concurrent report generation (3-5 simultaneous)
   - Verify API rate limit handling
   - Test memory usage with large reports

**Acceptance Criteria**:
- 10-minute target met for 4000-word reports
- All edge cases handled gracefully
- Quality metrics meet spec thresholds
- System stable under concurrent load
- Token compression achieves 60% reduction

### Phase 8: Documentation & Deployment
**Goal**: Document system and prepare for production deployment

**Tasks**:
1. Code documentation:
   - Add docstrings to all classes and methods
   - Document data model fields
   - Add inline comments for complex logic

2. API documentation:
   - Update endpoint documentation
   - Document new query parameters
   - Provide example requests/responses

3. User guide:
   - Explain structured vs. legacy reports
   - Provide query optimization tips
   - Document perspective categories

4. Deployment preparation:
   - Update Docker configuration
   - Add environment variables for config
   - Update `local_test.py` with structured report tests
   - Prepare rollback plan

5. Monitoring setup:
   - Add logging for generation metrics
   - Track section generation times
   - Monitor API usage and costs
   - Alert on quality threshold violations

**Acceptance Criteria**:
- All code documented with docstrings
- API documentation updated
- User guide completed
- Docker build succeeds
- Monitoring dashboards configured

---

## File Structure

```
src/
├── pipeline.py                    # Modified: Integrate structured generation
├── generator.py                   # Existing: LLM client
├── retriever.py                   # Existing: Search & retrieval
├── chunker.py                     # Existing: Text processing
│
├── report_structure.py            # NEW: Data models
├── structure_generator.py         # NEW: Report structure generation
├── section_generator.py           # NEW: Iterative section generation
├── context_manager.py             # NEW: Context compression & propagation
├── quality_validator.py           # NEW: Quality validation
└── report_assembler.py            # NEW: Final report assembly

specs/002-structured-report-generation/
├── spec.md                        # Feature specification
├── plan.md                        # This implementation plan
├── architecture.md                # Technical architecture details
├── research.md                    # Research findings
└── tasks.md                       # Detailed task breakdown (generated by /tasks)
```

---

## Risk Management

### Risk 1: Generation Time Exceeds 10-Minute Target
**Mitigation**:
- Implement parallel generation for independent sections
- Aggressive context compression (target 70% reduction)
- Cache frequently used summaries
- Use smaller model for summarization tasks

**Contingency**:
- Reduce target section word count to 300
- Limit max sections per chapter to 4
- Implement adaptive timeout based on query complexity

### Risk 2: Context Compression Loses Critical Information
**Mitigation**:
- A/B test compression strategies
- Validate information retention with embeddings
- Preserve full context for critical sections (exec summary, conclusion)
- Implement quality-aware compression (preserve high-value insights)

**Contingency**:
- Fall back to larger context summaries (300 tokens)
- Reduce max sections if context window insufficient
- Implement hierarchical summarization (chapter-level + section-level)

### Risk 3: API Cost Increase
**Mitigation**:
- Context compression reduces average prompt size by 60%
- Use smaller models for non-critical tasks (summarization)
- Implement request batching where possible
- Monitor and alert on cost thresholds

**Contingency**:
- Add cost-based generation limits
- Implement tiered report quality (basic/standard/comprehensive)
- Cache and reuse summaries across similar queries

### Risk 4: Quality Variance Across Sections
**Mitigation**:
- Quality validation after each section
- Auto-regeneration for low-quality sections
- Preserve full context for critical sections
- A/B test prompt templates

**Contingency**:
- Manual review and regeneration for critical reports
- Implement section-level quality scoring
- Add human-in-the-loop for high-stakes reports

---

## Success Criteria

### Technical Metrics
- ✅ All unit tests pass (>95% coverage)
- ✅ Integration tests pass with competition queries
- ✅ Generation time ≤10 minutes for 4000-word reports
- ✅ Zero output token truncation
- ✅ Context compression achieves ≥60% reduction
- ✅ API retry success rate ≥90%

### Quality Metrics
- ✅ Average section word count ≥350 words
- ✅ Perspective coverage ≥4 distinct viewpoints for complex queries
- ✅ Citation density ≥1 per 150 words
- ✅ Section uniqueness ≥85% (redundancy ≤15%)
- ✅ Inter-section coherence ≥0.8

### User Experience Metrics
- ✅ Progress updates sent every 30 seconds
- ✅ SSE streaming remains responsive
- ✅ Report structure easy to navigate
- ✅ Citations verifiable and well-organized

---

## Timeline Estimate

**Phase 0**: Research & Analysis - 0.5 days
**Phase 1**: Data Models & Structure Generation - 1.5 days
**Phase 2**: Context Management - 1.5 days
**Phase 3**: Iterative Section Generator - 2 days
**Phase 4**: Quality Validation - 1.5 days
**Phase 5**: Report Assembly - 1 day
**Phase 6**: Pipeline Integration - 1.5 days
**Phase 7**: Testing & Optimization - 2 days
**Phase 8**: Documentation & Deployment - 1 day

**Total**: ~12.5 days (2.5 weeks)

**Parallel Work Opportunities**:
- Phases 1-2 can be developed in parallel (data models + context management)
- Phases 4-5 can be developed in parallel (validation + assembly)
- Testing can begin during Phase 6 integration

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Run `/tasks` command** to generate detailed task breakdown with dependencies
3. **Begin Phase 0** research and architecture analysis
4. **Set up development branch** with initial file structure
5. **Implement incrementally** following phase order
6. **Test continuously** with competition queries
7. **Iterate based on feedback** from validation testing

---

## Appendix A: Prompt Templates

### Structure Generation Prompt
See "Prompt Engineering" section above for full template.

### Section Generation Prompt
See "Prompt Engineering" section above for full template.

### Context Compression Prompt
See "Prompt Engineering" section above for full template.

## Appendix B: Data Flow Diagram

```
User Query
    │
    ▼
Research Phase (Existing)
  ├─ generate_research_plan()
  ├─ generate_initial_draft()
  └─ perform_iterative_search_and_synthesis()
    │
    ▼
Structure Generation (NEW)
  ├─ analyze_query_perspectives()
  ├─ generate_chapter_outline()
  └─ create_section_specifications()
    │
    ├─── ReportStructure
    │      │
    │      ▼
    │   Iterative Section Generation Loop (NEW)
    │      │
    │      ├─ For each section in structure:
    │      │   │
    │      │   ├─ Build context from previous summaries
    │      │   ├─ Generate section (≤2048 tokens)
    │      │   ├─ Validate quality
    │      │   ├─ Compress to summary (≤200 tokens)
    │      │   └─ Add to generated sections
    │      │
    │      ▼
    │   Quality Validation (NEW)
    │      │
    │      ├─ validate_section_depth()
    │      ├─ check_citation_density()
    │      ├─ detect_redundancy()
    │      └─ measure_coherence()
    │      │
    │      ▼
    │   Report Assembly (NEW)
    │      │
    │      ├─ assemble_final_report()
    │      ├─ organize_citations_by_chapter()
    │      └─ format_markdown_structure()
    │      │
    │      ▼
    └──── Final Structured Report
           │
           ▼
        SSE Stream to User
```

## Appendix C: Configuration Reference

```python
# Configuration constants (src/pipeline.py)
ENABLE_STRUCTURED_REPORTS = True  # Feature flag
MAX_SECTIONS_PER_CHAPTER = 5      # Limit sections per chapter
TARGET_SECTION_WORDS = 350        # Target word count per section
MAX_OUTPUT_TOKENS_PER_SECTION = 2048  # Token limit per generation
CONTEXT_SUMMARY_MAX_TOKENS = 200  # Max tokens for section summary
SLIDING_WINDOW_SIZE = 5           # Number of recent sections in full detail
MAX_REGENERATION_ATTEMPTS = 2     # Retry attempts for low-quality sections
REDUNDANCY_THRESHOLD = 0.70       # Similarity threshold for redundancy
CITATION_DENSITY_THRESHOLD = 0.0067  # 1 citation per 150 words
MIN_SECTION_WORDS = 300           # Minimum acceptable section length
COHERENCE_THRESHOLD = 0.80        # Minimum inter-section coherence
```
