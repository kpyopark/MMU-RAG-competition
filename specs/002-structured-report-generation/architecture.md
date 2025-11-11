# Technical Architecture: Structured Report Generation

**Feature**: Structured Report Generation with Iterative Section Synthesis
**Version**: 1.0
**Last Updated**: 2025-11-11

---

## System Overview

The Structured Report Generation system extends the existing TTD-DR (Test-Time Diffusion Deep Researcher) pipeline with a multi-phase report generation architecture that produces comprehensive, multi-perspective research reports through iterative section synthesis.

### Architecture Principles

1. **Iterative Generation**: Break large reports into manageable sections (≤2048 tokens each)
2. **Progressive Contextualization**: Propagate insights across sections through compressed summaries
3. **Quality-Driven**: Validate each section before proceeding to next
4. **Backward Compatible**: Maintain existing API contracts and SSE streaming
5. **Fail-Safe**: Graceful degradation with fallback to legacy single-pass generation

---

## Component Architecture

### Layer 1: Pipeline Orchestration

**Component**: `TTD_DR_Pipeline` (Modified)
**Responsibility**: Orchestrate end-to-end report generation workflow
**Location**: `src/pipeline.py`

```python
class TTD_DR_Pipeline:
    """
    Extended pipeline with structured report generation capability.

    Workflow:
    1. Research Phase (existing)
       - generate_research_plan()
       - generate_initial_draft()
       - perform_iterative_search_and_synthesis()

    2. Structure Generation Phase (new)
       - generate_report_structure()

    3. Iterative Section Generation Phase (new)
       - generate_structured_report()
         ├─ generate_executive_summary()
         ├─ for each chapter:
         │   ├─ for each section:
         │   │   ├─ generate_section()
         │   │   ├─ validate_section_quality()
         │   │   └─ compress_to_summary()
         │   └─ send_chapter_complete_update()
         └─ generate_conclusion()

    4. Assembly Phase (new)
       - assemble_final_report()
       - organize_citations()
    """

    def __init__(self, callback, enable_structured=True):
        self.callback = callback
        self.enable_structured = enable_structured

        # Existing attributes
        self.plan = ""
        self.draft = ""
        self.q_a_history = []
        self.intermediate_log = []
        self.citations = []

        # New attributes for structured generation
        self.report_structure: Optional[ReportStructure] = None
        self.generated_sections: List[GeneratedSection] = []
        self.context_manager = ContextManager()
        self.structure_generator = ReportStructureGenerator()
        self.section_generator = SectionGenerator()
        self.quality_validator = QualityValidator()
        self.report_assembler = ReportAssembler()
```

**Integration Points**:
- **Input**: Receives query, callback from FastAPI endpoints
- **Output**: Streams SSE updates, returns final report
- **Dependencies**: All Layer 2 components

---

### Layer 2: Core Generation Components

#### Component 2.1: Report Structure Generator

**Responsibility**: Analyze query and research to create chapter/section hierarchy
**Location**: `src/structure_generator.py`

```python
class ReportStructureGenerator:
    """
    Generates adaptive report structure based on query complexity.

    Key Methods:
    - analyze_query_perspectives(query) -> List[str]
      Identifies relevant analytical perspectives (Financial, Technical, etc.)

    - generate_chapter_outline(query, plan, research_summary) -> List[Chapter]
      Creates chapter structure with LLM using STRUCTURE_GENERATION_PROMPT

    - create_section_specifications(chapters) -> ReportStructure
      Defines detailed section specs for each chapter
    """

    def __init__(self):
        self.default_perspectives = [
            "Financial/Economic Analysis",
            "Technical/Operational Analysis",
            "Regulatory/Legal Analysis",
            "Strategic/Competitive Analysis",
            "Risk/Challenge Analysis",
            "Market/Industry Analysis"
        ]

    def analyze_query_perspectives(self, query: str) -> List[str]:
        """
        Identify relevant perspectives for query using keyword matching + LLM.

        Returns: 4-6 perspectives ranked by relevance
        """
        pass

    def generate_chapter_outline(
        self,
        query: str,
        plan: str,
        research_summary: str
    ) -> List[Chapter]:
        """
        Generate chapter structure with LLM.

        Prompt includes:
        - Query and research context
        - Perspective guidance
        - JSON schema for structure

        Returns: 4-6 chapters with section specs
        """
        pass
```

**Algorithm**: Query Complexity Adaptive Structuring
```
Input: query, research_plan, q_a_history

1. Extract key concepts from query (entities, topics, actions)
2. Identify explicit perspective requests (e.g., "financial impact")
3. Infer implicit perspectives from query type:
   - News verification → Verification + Impact + Context
   - Company analysis → Financial + Strategic + Market + Risk
   - Technology review → Technical + Market + Use Cases + Limitations

4. Score perspective relevance (0-1):
   - Explicit mention: 1.0
   - Implicit inference: 0.7
   - Research findings support: +0.2
   - Generic default: 0.3

5. Select top 4-6 perspectives (threshold: >0.4)

6. Generate chapter outline with LLM:
   - Prompt: STRUCTURE_GENERATION_PROMPT
   - Include selected perspectives
   - Request 3-5 sections per chapter
   - Enforce JSON schema validation

7. Post-process structure:
   - Validate section count (3-5 per chapter)
   - Validate target word counts (300-500)
   - Estimate total sections (15-30 range)
   - Calculate estimated generation time

Output: ReportStructure with chapters and section specs
```

**Performance**:
- Target latency: ≤10 seconds
- LLM call: 1 (structure generation)
- Output tokens: ~1500 (JSON structure)

---

#### Component 2.2: Section Generator

**Responsibility**: Generate individual sections with context propagation
**Location**: `src/section_generator.py`

```python
class SectionGenerator:
    """
    Generates sections iteratively with context from previous sections.

    Key Methods:
    - generate_section(spec, context, research_data) -> GeneratedSection
      Core section generation with SECTION_GENERATION_PROMPT

    - generate_executive_summary(structure, research) -> GeneratedSection
      Special handling for executive summary (synthesize all)

    - generate_conclusion(structure, sections) -> GeneratedSection
      Forward-looking synthesis of all findings
    """

    def __init__(self):
        self.gemini_client = GeminiClient()
        self.max_output_tokens = 2048
        self.target_word_count = 350

    def generate_section(
        self,
        spec: SectionSpec,
        context_summary: ContextSummary,
        research_data: str
    ) -> GeneratedSection:
        """
        Generate single section with context.

        Steps:
        1. Build prompt with context + research + spec
        2. Call Gemini API (max_output_tokens=2048)
        3. Parse response and extract content
        4. Extract citations
        5. Calculate word count
        6. Compress to summary for future context

        Returns: GeneratedSection with content and metadata
        """
        pass
```

**Algorithm**: Context-Aware Section Generation
```
Input: section_spec, context_summary, research_data

1. Build generation context:
   - Recent sections (full detail): Last 3-5 sections
   - Older sections: Compressed summaries (≤200 tokens each)
   - Research highlights: Relevant Q&A from history
   - Key insights: Top 10 insights from previous sections

2. Construct prompt:
   - Template: SECTION_GENERATION_PROMPT
   - Fill placeholders: query, section_title, guidance, etc.
   - Append context summary
   - Append relevant research data
   - Total prompt size: Monitor to stay <80% context window

3. Generate with Gemini:
   - max_output_tokens: 2048
   - temperature: 0.7 (balance creativity and consistency)
   - system_prompt: Role as research report writer

4. Post-process response:
   - Extract markdown content
   - Parse inline citations [Source N]
   - Calculate word count
   - Validate no truncation (check for incomplete sentences)

5. Create summary for context:
   - Extract 3-5 key insights
   - Compress to ≤200 tokens
   - Use CONTEXT_COMPRESSION_PROMPT

6. Retry logic:
   - If API error: Retry with exponential backoff (3 attempts)
   - If truncation detected: Reduce target word count by 20%, retry
   - If quality validation fails: Regenerate with diversification prompt

Output: GeneratedSection with content, citations, summary
```

**Performance**:
- Target latency: ≤30 seconds per section
- LLM calls per section: 2 (generation + compression)
- Output tokens: ~1500-2000 (content) + ~300 (compression)

---

#### Component 2.3: Context Manager

**Responsibility**: Manage context window through progressive summarization
**Location**: `src/context_manager.py`

```python
class ContextManager:
    """
    Manages context propagation across sections with compression.

    Key Methods:
    - compress_section_to_summary(section) -> str
      Compress section to ≤200 tokens preserving key insights

    - build_generation_context(sections, research) -> ContextSummary
      Assemble context for next section generation

    - maintain_sliding_window(sections, window_size=5) -> Dict
      Keep recent sections in full detail, compress older ones
    """

    def __init__(self):
        self.gemini_client = GeminiClient()
        self.sliding_window_size = 5
        self.summary_max_tokens = 200
        self.compression_cache = {}  # Cache summaries

    def compress_section_to_summary(
        self,
        section: GeneratedSection
    ) -> str:
        """
        Compress section to concise summary.

        Approach:
        1. Extract 3-5 key insights
        2. Preserve critical facts and numbers
        3. Target ≤200 tokens
        4. Cache result for reuse

        Returns: Compressed summary string
        """
        pass
```

**Algorithm**: Progressive Context Compression
```
Input: generated_sections (list), current_section_index

1. Identify recent vs. old sections:
   recent_sections = sections[-5:]  # Last 5 in full detail
   old_sections = sections[:-5]      # Compress these

2. For each old section:
   - Check cache for existing summary
   - If not cached:
     * Call Gemini with CONTEXT_COMPRESSION_PROMPT
     * Extract 3-5 key insights
     * Target ≤200 tokens
     * Cache result (key: section title + content hash)

3. Build context summary object:
   ContextSummary(
     key_insights=[...]  # Top 10 from all sections
     previous_sections=[...]  # Recent full + old compressed
     research_highlights=...  # Relevant Q&A
     total_tokens=...  # Tracked cumulatively
   )

4. Validate token budget:
   - Total context should be ≤40% of model context window
   - If exceeding: Further compress old sections (summary of summaries)

5. Quality check:
   - Verify key facts preserved (compare embeddings)
   - Ensure no critical data loss
   - Target ≥90% information retention score

Output: ContextSummary for next generation
```

**Token Economics**:
- Full section: ~1500 tokens
- Compressed summary: ~200 tokens
- Compression ratio: 87% reduction
- Cumulative savings: For 20 sections, saves ~26,000 tokens

---

#### Component 2.4: Quality Validator

**Responsibility**: Validate section quality and trigger regeneration
**Location**: `src/quality_validator.py`

```python
class QualityValidator:
    """
    Validates generated sections against quality thresholds.

    Key Methods:
    - validate_section_depth(section) -> ValidationResult
      Check word count ≥300

    - check_citation_density(section) -> ValidationResult
      Verify ≥1 citation per 150 words

    - detect_redundancy(section, previous_sections) -> float
      Measure overlap with previous sections

    - measure_coherence(section, previous_section) -> float
      Semantic similarity of transitions
    """

    def __init__(self):
        self.min_words = 300
        self.max_words = 600
        self.citation_density_threshold = 1.0 / 150  # 1 per 150 words
        self.redundancy_threshold = 0.70
        self.coherence_threshold = 0.80

    def validate_section(
        self,
        section: GeneratedSection,
        previous_sections: List[GeneratedSection]
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive section validation.

        Returns: (is_valid, list_of_issues)
        """
        pass
```

**Validation Rules**:

1. **Depth Validation**:
   - Word count ≥300 and ≤600
   - If <250: FAIL (regenerate required)
   - If 250-299: WARN (acceptable but flagged)
   - If >600: WARN (may be too verbose)

2. **Citation Density**:
   - Extract [Source N] references
   - Calculate: citations / (word_count / 150)
   - Threshold: ≥1.0 (at least 1 citation per 150 words)
   - If <0.5: FAIL (insufficient sourcing)
   - If 0.5-0.99: WARN (below target)

3. **Redundancy Detection**:
   - Compare section to each previous section
   - Use sentence embeddings (cosine similarity)
   - If any similarity >0.70: FAIL (too similar)
   - Trigger regeneration with diversification prompt

4. **Coherence Measurement**:
   - Compare last paragraph of previous section to first of current
   - Semantic similarity should be 0.6-0.9 (balanced)
   - If <0.4: WARN (weak transition)
   - If >0.95: WARN (too repetitive)

**Regeneration Logic**:
```
If validation fails:
  1. Log failure reason
  2. If attempt < MAX_REGENERATION_ATTEMPTS (2):
     - Modify prompt based on failure:
       * Low depth → Increase target word count by 20%
       * Low citations → Emphasize sourcing in prompt
       * High redundancy → Add diversification instruction
     - Retry generation
  3. If still failing after max attempts:
     - Log quality issue
     - Proceed with best attempt (don't block pipeline)
     - Flag section in final report metadata
```

---

#### Component 2.5: Report Assembler

**Responsibility**: Assemble sections into final structured markdown report
**Location**: `src/report_assembler.py`

```python
class ReportAssembler:
    """
    Assembles generated sections into final report with citations.

    Key Methods:
    - assemble_final_report(structure, sections) -> str
      Combine all sections with formatting

    - organize_citations_by_chapter(sections) -> Dict
      Group citations by chapter

    - format_markdown_structure(report) -> str
      Apply consistent markdown formatting
    """

    def assemble_final_report(
        self,
        structure: ReportStructure,
        sections: List[GeneratedSection]
    ) -> str:
        """
        Assemble complete report.

        Structure:
        # [Report Title]

        ## Executive Summary
        [Executive summary section]

        ## Chapter 1: [Title]
        ### Section 1.1: [Title]
        [Content]

        ### Section 1.2: [Title]
        [Content]

        ...

        ## Conclusion and Implications
        [Conclusion section]

        ## Citations
        ### Chapter 1: [Title]
        1. [Source 1] - [URL]
        2. [Source 2] - [URL]

        ...
        """
        pass
```

**Assembly Algorithm**:
```
Input: report_structure, generated_sections

1. Initialize report markdown string
2. Add title and metadata header
3. Add executive summary section
4. For each chapter:
   - Add chapter heading (H2)
   - For each section in chapter:
     * Add section heading (H3)
     * Add section content
     * Track citations used
5. Add conclusion section
6. Build citations section:
   - Group by chapter
   - Deduplicate URLs
   - Format as numbered list
   - Include URL and title
7. Add metadata footer:
   - Total word count
   - Section count
   - Citation count
   - Generation time
   - Perspectives covered
8. Return formatted markdown

Output: Complete markdown report string
```

---

## Data Flow Architecture

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User submits query via /run endpoint                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. TTD_DR_Pipeline.run(query)                                   │
│    - Research Phase (existing)                                  │
│      * generate_research_plan()                                 │
│      * generate_initial_draft()                                 │
│      * perform_iterative_search_and_synthesis()                 │
│      → Output: plan, draft, q_a_history, citations             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Structure Generation Phase (NEW)                             │
│    - ReportStructureGenerator.generate_chapter_outline()        │
│      * Input: query, plan, research_summary                     │
│      * LLM call: STRUCTURE_GENERATION_PROMPT                    │
│      * Parse JSON structure                                     │
│      → Output: ReportStructure (4-6 chapters, 15-30 sections)  │
│    - Send SSE update: "Structure generated (20 sections)"       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Iterative Section Generation Phase (NEW)                     │
│                                                                  │
│    ┌─ For section in [1..N]:                                   │
│    │                                                             │
│    │  ┌────────────────────────────────────────────────────┐  │
│    │  │ 4a. Build Context                                   │  │
│    │  │  - ContextManager.build_generation_context()        │  │
│    │  │    * Recent sections (full): Last 5               │  │
│    │  │    * Old sections (compressed): Summaries         │  │
│    │  │    * Research data: Relevant Q&A                  │  │
│    │  │    → ContextSummary                                │  │
│    │  └────────────────────────────────────────────────────┘  │
│    │                    │                                       │
│    │                    ▼                                       │
│    │  ┌────────────────────────────────────────────────────┐  │
│    │  │ 4b. Generate Section                                │  │
│    │  │  - SectionGenerator.generate_section()              │  │
│    │  │    * Prompt: SECTION_GENERATION_PROMPT             │  │
│    │  │    * Context + Research + Spec                     │  │
│    │  │    * LLM call (max 2048 tokens)                    │  │
│    │  │    → GeneratedSection (content + citations)       │  │
│    │  └────────────────────────────────────────────────────┘  │
│    │                    │                                       │
│    │                    ▼                                       │
│    │  ┌────────────────────────────────────────────────────┐  │
│    │  │ 4c. Validate Quality                                │  │
│    │  │  - QualityValidator.validate_section()              │  │
│    │  │    * Check depth (≥300 words)                      │  │
│    │  │    * Check citations (≥1 per 150 words)            │  │
│    │  │    * Check redundancy (<70% overlap)               │  │
│    │  │    * Check coherence (≥0.8)                        │  │
│    │  │  - If fails: Regenerate (max 2 attempts)           │  │
│    │  │    → ValidationResult                               │  │
│    │  └────────────────────────────────────────────────────┘  │
│    │                    │                                       │
│    │                    ▼                                       │
│    │  ┌────────────────────────────────────────────────────┐  │
│    │  │ 4d. Compress to Summary                             │  │
│    │  │  - ContextManager.compress_section_to_summary()     │  │
│    │  │    * LLM call: CONTEXT_COMPRESSION_PROMPT          │  │
│    │  │    * Extract key insights (3-5)                    │  │
│    │  │    * Target ≤200 tokens                            │  │
│    │  │    → Summary (cached for future use)              │  │
│    │  └────────────────────────────────────────────────────┘  │
│    │                    │                                       │
│    │                    ▼                                       │
│    │  ┌────────────────────────────────────────────────────┐  │
│    │  │ 4e. Send Progress Update                            │  │
│    │  │  - SSE event: "Section 5/20 complete"              │  │
│    │  │  - Include: section title, chapter, progress %    │  │
│    │  └────────────────────────────────────────────────────┘  │
│    │                                                             │
│    └─ End Loop                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Report Assembly Phase (NEW)                                  │
│    - ReportAssembler.assemble_final_report()                    │
│      * Combine all sections in order                            │
│      * Format markdown structure (H1, H2, H3)                   │
│      * Organize citations by chapter                            │
│      * Add metadata footer                                      │
│      → Final markdown report (2,500-4,000 words)               │
│    - Send SSE update: "Report assembly complete"                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Final callback with complete=True                            │
│    - SSE event: final_report + citations + complete=true        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Latency Budget (4000-word report, 20 sections)

| Phase | Component | Time | LLM Calls |
|-------|-----------|------|-----------|
| Structure Generation | ReportStructureGenerator | 10s | 1 |
| Section 1-20 Generation | SectionGenerator | 600s (30s×20) | 20 |
| Section Compression | ContextManager | 100s (5s×20) | 20 |
| Quality Validation | QualityValidator | 40s (2s×20) | 0 |
| Report Assembly | ReportAssembler | 5s | 0 |
| **Total** | | **755s (~12.5 min)** | **41** |

**Optimization Opportunities**:
- Parallel section generation for independent sections: -30%
- Cached summaries for repeated concepts: -15%
- Smaller model for compression: -20%
- **Optimized Total**: ~420s (7 minutes)

### Token Usage (4000-word report)

| Operation | Tokens/Call | Calls | Total |
|-----------|-------------|-------|-------|
| Structure Generation | 3,000 | 1 | 3,000 |
| Section Generation (input) | 5,000 | 20 | 100,000 |
| Section Generation (output) | 1,800 | 20 | 36,000 |
| Compression (input) | 1,500 | 20 | 30,000 |
| Compression (output) | 250 | 20 | 5,000 |
| **Total** | | | **174,000** |

**Cost Estimate** (Gemini Flash):
- Input: 144,000 tokens × $0.075/1M = $0.011
- Output: 41,250 tokens × $0.30/1M = $0.012
- **Total per report**: $0.023 (~2.3 cents)

---

## Integration with Existing System

### Backward Compatibility

1. **Feature Flag**: `ENABLE_STRUCTURED_REPORTS = True`
   - Default: Structured generation enabled
   - Fallback: `?structured=false` query param for legacy mode

2. **SSE Event Format**: Extended, not changed
   ```python
   # Existing events still work:
   {
     "intermediate_steps": "...",
     "final_report": "...",
     "citations": [...],
     "complete": true
   }

   # New events added:
   {
     "structure_generated": true,
     "total_sections": 20,
     "section_progress": "5/20",
     "current_section": "Financial Analysis - Deal Structure"
   }
   ```

3. **API Endpoints**: Unchanged
   - `/run` - SSE streaming (structured by default)
   - `/evaluate` - Static response (structured by default)
   - Both accept `?structured=false` for legacy

### Migration Strategy

**Phase 1**: Parallel operation (Week 1)
- Deploy structured generation as opt-in (`?structured=true`)
- Monitor performance and quality metrics
- Collect feedback from test queries

**Phase 2**: Default enabled (Week 2)
- Make structured generation default
- Legacy available via `?structured=false`
- Monitor error rates and fallback usage

**Phase 3**: Full transition (Week 3)
- Remove legacy code path
- Optimize performance based on metrics
- Finalize configuration tuning

---

## Scalability Considerations

### Horizontal Scaling

**Current bottleneck**: Sequential section generation
**Solution**: Parallel generation for independent sections

```python
# Identify independent sections (different chapters, no cross-references)
independent_sections = identify_parallel_candidates(structure)

# Generate in parallel (max 3 concurrent to respect API limits)
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(generate_section, spec, context, research)
        for spec in independent_sections
    ]
    results = [f.result() for f in futures]
```

**Benefit**: 30-40% reduction in total generation time

### API Rate Limit Management

- **Gemini API limit**: 60 requests/minute (free tier) or 1000/minute (paid)
- **Current usage**: ~40 requests per report
- **Throttling strategy**:
  ```python
  from ratelimit import limits, sleep_and_retry

  @sleep_and_retry
  @limits(calls=50, period=60)  # 50 calls per minute
  def call_gemini_api(prompt):
      return gemini_client.complete(prompt)
  ```

### Caching Strategy

**Cache summaries** for repeated concepts:
```python
summary_cache = {
    hash(section_content): compressed_summary
}

# Before compression:
content_hash = hash(section.content)
if content_hash in summary_cache:
    return summary_cache[content_hash]
```

**Benefit**: 15-20% reduction in compression LLM calls

---

## Error Handling & Resilience

### Failure Modes

1. **API Error During Section Generation**
   - **Detection**: Exception from Gemini client
   - **Response**: Retry with exponential backoff (3 attempts)
   - **Fallback**: Save partial report, notify user of incomplete generation
   - **Recovery**: Resume from last successful section

2. **Context Window Overflow**
   - **Detection**: Token count exceeds threshold
   - **Response**: Aggressive compression of older sections (summary of summaries)
   - **Fallback**: Reduce sliding window size from 5 to 3
   - **Prevention**: Monitor cumulative tokens after each section

3. **Quality Validation Failure**
   - **Detection**: Section fails depth/citation/redundancy checks
   - **Response**: Regenerate with modified prompt (max 2 attempts)
   - **Fallback**: Proceed with best attempt, flag in metadata
   - **Logging**: Record failure reason for analysis

4. **SSE Stream Interruption**
   - **Detection**: Client disconnection
   - **Response**: Continue generation, save final report
   - **Recovery**: Client can poll `/status/{job_id}` for completion

### Partial Report Recovery

```python
@dataclass
class GenerationState:
    """Serializable state for resume capability."""
    report_structure: ReportStructure
    completed_sections: List[GeneratedSection]
    current_section_index: int
    context_summaries: List[str]
    timestamp: datetime

def save_generation_state(state: GenerationState, job_id: str):
    """Persist state for resume."""
    with open(f"/tmp/generation_{job_id}.json", "w") as f:
        json.dump(asdict(state), f)

def resume_generation(job_id: str) -> GenerationState:
    """Resume from saved state."""
    with open(f"/tmp/generation_{job_id}.json") as f:
        return GenerationState(**json.load(f))
```

---

## Security & Privacy

### Data Handling

1. **No persistent storage**: Generated reports not saved to disk (except temp state)
2. **API key security**: Gemini API key from environment variables
3. **Input validation**: Sanitize user queries before LLM calls
4. **Output sanitization**: Remove potential PII from generated reports

### Rate Limiting

- Per-user rate limiting: 10 reports/hour
- Global rate limiting: 100 reports/hour
- Queue overflow: Return 429 with retry-after header

---

## Monitoring & Observability

### Metrics to Track

1. **Performance Metrics**:
   - Total generation time (p50, p95, p99)
   - Section generation time distribution
   - Context compression time
   - LLM API latency

2. **Quality Metrics**:
   - Average report word count
   - Average section word count
   - Citation density distribution
   - Validation failure rate by type

3. **Cost Metrics**:
   - Tokens used per report (input/output)
   - Cost per report
   - Cache hit rate for summaries

4. **Error Metrics**:
   - API error rate
   - Retry success rate
   - Partial report rate
   - Validation failure rate

### Logging Structure

```python
logger.info("Section generated", extra={
    "section_title": section.spec.title,
    "chapter_number": section.spec.chapter_number,
    "section_number": section.spec.section_number,
    "word_count": section.word_count,
    "citations_count": len(section.citations_used),
    "generation_time_ms": section.generation_time * 1000,
    "validation_passed": validation_result.is_valid
})
```

---

## Testing Strategy

### Unit Tests

- `test_structure_generator.py`: Structure generation logic
- `test_section_generator.py`: Section generation and retry logic
- `test_context_manager.py`: Compression and sliding window
- `test_quality_validator.py`: All validation rules
- `test_report_assembler.py`: Assembly and formatting

### Integration Tests

- `test_pipeline_integration.py`: End-to-end with mock Gemini
- `test_sse_streaming.py`: SSE event flow
- `test_backward_compatibility.py`: Legacy mode still works

### Performance Tests

- `test_generation_time.py`: Verify <10 minute target
- `test_token_usage.py`: Verify compression efficiency
- `test_parallel_generation.py`: Validate parallel speedup

### Quality Tests

- `test_report_quality.py`: Validate depth, citations, coherence
- `test_diverse_queries.py`: 10 diverse query types
- `test_edge_cases.py`: Ambiguous queries, limited data, overflow

---

## Future Enhancements

1. **Parallel Section Generation**: Implement concurrent generation for independent sections
2. **Adaptive Quality Thresholds**: Adjust validation rules based on query complexity
3. **User Customization**: Allow users to specify report structure preferences
4. **Visual Elements**: Add support for embedding charts and diagrams
5. **Multi-Language Support**: Generate reports in multiple languages
6. **Export Formats**: PDF, DOCX export capabilities
7. **Interactive Refinement**: Allow users to request section regeneration
8. **Template Library**: Pre-defined report templates for common query types
