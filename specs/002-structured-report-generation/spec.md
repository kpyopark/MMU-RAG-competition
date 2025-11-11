# Feature Specification: Structured Report Generation with Iterative Section Synthesis

**Feature Branch**: `002-structured-report-generation`
**Created**: 2025-11-11
**Status**: Draft
**Input**: "질문이나 요청에 대하여, 다양한 관점의 Report로 생성할 수 있어야 하는데, 너무 축약된 Report로 만들고 있다. 이를 방지하기 위하여, 최종 Report를 생성할 때, 보고서 형태의 장절 형태로 상세하게 구성하고, 아웃풋 토큰 문제를 해결하기 위하여, 최종적으로 절(Phrase) 별로 gemini에게 생성을 요청하고, 생성된 절 에 대한 내용을 다음 절 생성을 위한 인풋으로 추가하는 구성을 통하여, 최종 보고서가 완벽한 리포트로 구성할 수 있게 해줘."

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## Problem Statement

### Current Issue
The Deep Research pipeline currently generates abbreviated reports that lack depth and multi-perspective analysis. When users submit complex research queries, they receive:
- **Condensed outputs** that oversimplify nuanced topics
- **Single-perspective analysis** that misses important viewpoints (e.g., business impact, technical implications, regulatory concerns)
- **Token-limited responses** that cut off before comprehensive coverage
- **Unstructured narratives** that are difficult to navigate and reference

### Impact on Users
- **Researchers and analysts** cannot extract sufficient depth for decision-making
- **Stakeholders** miss critical perspectives needed for comprehensive understanding
- **Quality assessment** is hindered by lack of structured, detailed reporting
- **Citation verification** is difficult without clear section-to-source mapping

### Root Causes
1. **Single-pass generation**: Final report generated in one LLM call hits output token limits
2. **Lack of structure**: No formal chapter/section framework for organizing complex information
3. **Missing multi-perspective framework**: No systematic approach to analyzing topics from different angles
4. **Context loss**: Iterative research findings not systematically organized before final synthesis

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
A researcher submits a complex query like "Analyze the Pfizer-Metsera acquisition from multiple perspectives including financial impact, regulatory concerns, competitive landscape, and strategic implications for both companies."

The system should generate a **comprehensive, structured research report** with:
1. **Executive Summary** - High-level synthesis (1-2 paragraphs)
2. **Multiple Analysis Chapters** - Each addressing a distinct perspective:
   - Financial Analysis (deal structure, valuation, revenue impact)
   - Regulatory Analysis (antitrust concerns, approval timeline)
   - Strategic Analysis (competitive positioning, market dynamics)
   - Risk Assessment (integration challenges, market risks)
3. **Detailed Sections within Chapters** - Each section generated iteratively with context from previous sections
4. **Comprehensive Citations** - Sources mapped to specific claims in each section
5. **Conclusion and Implications** - Forward-looking synthesis

### Acceptance Scenarios

#### Scenario 1: Multi-Perspective Deep Research Report (P0)
**Given** a complex research query requiring multi-faceted analysis
**When** the system processes the query
**Then**
- Report structure is generated with 4-6 chapters based on query scope
- Each chapter contains 3-5 detailed sections
- Each section is 300-500 words (not abbreviated)
- Different perspectives are explicitly identified and analyzed:
  * Financial/Economic perspective
  * Technical/Operational perspective
  * Regulatory/Legal perspective
  * Strategic/Competitive perspective
  * Risk/Challenge perspective
  * Market/Industry perspective
- Total report length: 2,500-4,000 words minimum for complex queries
- All major claims are backed by citations with section-level attribution

**Acceptance Criteria:**
- Report contains minimum 4 distinct analytical perspectives for multi-faceted queries
- Each section ≥300 words with substantive analysis (no placeholder text)
- Citation density: ≥1 citation per 150 words
- Structural completeness: 100% of planned sections generated
- Context continuity: Each section references relevant content from previous sections
- Output token limit: No single LLM call exceeds 2048 output tokens

#### Scenario 2: Iterative Section Generation with Context Propagation (P0)
**Given** a report outline with 5 chapters × 4 sections = 20 total sections
**When** the system generates each section iteratively
**Then**
- Section 1 is generated with full research context
- Section 2 is generated with Section 1 content + research context
- Section N is generated with Sections 1...N-1 summaries + research context
- Each section build upon insights from previous sections
- No repetitive content across sections (uniqueness >85%)
- Coherent narrative flow across all sections
- Cross-references between sections are accurate

**Acceptance Criteria:**
- Context window management: Previous sections compressed to key points (summary ≤200 tokens per section)
- Generation time per section: ≤30 seconds
- Inter-section coherence score: ≥0.8 (measured by semantic similarity of transitions)
- No contradictions between sections (validated by consistency check)
- Each section adds ≥3 new insights not present in previous sections

#### Scenario 3: Output Token Limit Handling (P0)
**Given** a comprehensive report requiring 15,000+ total output tokens
**When** the system generates the report iteratively
**Then**
- Report is broken into ≤20 generation calls
- Each generation call produces ≤2048 output tokens
- No truncation or incomplete sections
- All planned content is delivered
- Total generation time scales linearly with section count

**Acceptance Criteria:**
- Zero truncated sections (100% completion rate)
- Maximum tokens per generation call: ≤2048
- Total report generation time: ≤10 minutes for 4000-word reports
- Memory efficiency: Context summarization reduces token usage by ≥60% vs. full context

#### Scenario 4: Report Structure Adaptability (P1)
**Given** queries of varying complexity (simple vs. multi-faceted)
**When** the system analyzes the query
**Then**
- Simple queries (single-aspect) generate 2-3 chapter reports
- Moderate queries (2-3 aspects) generate 4-5 chapter reports
- Complex queries (4+ aspects) generate 5-7 chapter reports
- Structure adapts to query requirements automatically

**Acceptance Criteria:**
- Structure generation time: ≤10 seconds
- Minimum chapters for any query: 2 (Executive Summary + Main Analysis + Conclusion)
- Maximum chapters: 8 (to maintain focus)
- Perspective coverage: ≥80% of identified aspects addressed in structure

#### Scenario 5: Citation and Source Attribution (P0)
**Given** a multi-chapter report with 20+ sections
**When** the system generates citations
**Then**
- Each factual claim has inline citation reference [Source N]
- Citations section maps each [Source N] to full URL and title
- Citations are organized by report chapter/section
- Duplicate citations are deduplicated but preserve all usage contexts

**Acceptance Criteria:**
- Citation accuracy: ≥95% of citations map to correct sources
- Citation coverage: ≥90% of factual claims have citations
- Citation organization: Sources grouped by chapter for easy reference
- Broken link rate: ≤2% (validated during generation)

---

## Edge Cases

### Edge Case 1: Query Ambiguity
**Scenario**: User submits vague query like "Tell me about AI"
**Expected Behavior**:
- System requests clarification with suggested perspectives
- If no clarification, generates default structure with broad perspectives
- Report includes note about query ambiguity and chosen scope

**Acceptance Criteria**:
- Clarification prompt presented within 5 seconds
- Default structure covers ≥4 standard perspectives (Technical, Business, Ethical, Future Outlook)

### Edge Case 2: Insufficient Research Data
**Scenario**: Research iteration finds <3 relevant sources
**Expected Behavior**:
- Report explicitly indicates limited source availability
- Generated sections acknowledge knowledge gaps
- Recommends specific additional queries for deeper research
- Does not fabricate information to fill sections

**Acceptance Criteria**:
- Knowledge gap indicators: Present in ≥80% of under-sourced sections
- No hallucinated facts (zero tolerance)
- Recommendation quality: ≥3 actionable follow-up queries suggested

### Edge Case 3: Context Window Overflow
**Scenario**: 30+ sections with rich context exceed total context limit
**Expected Behavior**:
- System progressively summarizes earlier sections (sliding window)
- Most recent 3-5 sections retain full detail
- Earlier sections compressed to 100-150 token summaries
- Warning logged if compression ratio exceeds 10:1

**Acceptance Criteria**:
- No generation failures due to context overflow
- Compression maintains key insights (≥90% information retention score)
- Generation continues successfully even with 50+ sections

### Edge Case 4: Generation Failure Mid-Report
**Scenario**: API error occurs at section 12 of 20
**Expected Behavior**:
- System retries failed section up to 3 times
- If still failing, saves partial report with generation state
- User notified of partial completion with section progress (12/20)
- Allows resume from last successful section

**Acceptance Criteria**:
- Retry success rate: ≥90% for transient errors
- Partial report saved within 5 seconds of final failure
- Resume capability: Successfully continues from saved state

### Edge Case 5: Redundant Content Across Sections
**Scenario**: Multiple sections contain overlapping information
**Expected Behavior**:
- System detects similarity >70% between sections during generation
- Automatically diversifies content by requesting alternative angles
- Cross-references existing content instead of repeating
- Quality check ensures uniqueness >85% across sections

**Acceptance Criteria**:
- Similarity detection: Identifies redundancy within 2 seconds per section
- Uniqueness enforcement: ≥85% unique content per section
- Cross-reference rate: ≥3 cross-references per report for related concepts

---

## Success Metrics

### Quality Metrics
- **Report Depth Score**: Average section word count ≥350 words
- **Perspective Coverage**: ≥4 distinct analytical perspectives for complex queries
- **Citation Density**: ≥1 citation per 150 words
- **Structural Completeness**: 100% of planned sections generated
- **Inter-Section Coherence**: ≥0.8 semantic similarity in transitions

### Performance Metrics
- **Total Generation Time**: ≤10 minutes for 4000-word reports
- **Section Generation Time**: ≤30 seconds per section
- **Context Compression Efficiency**: ≥60% token reduction
- **Output Token Management**: Zero truncated sections

### User Satisfaction Metrics
- **Report Usefulness**: ≥4.5/5 average rating from evaluators
- **Depth Satisfaction**: ≥90% of users rate depth as "sufficient" or "excellent"
- **Structure Clarity**: ≥85% of users find report structure easy to navigate
- **Citation Utility**: ≥80% of users successfully verify citations

---

## Dependencies & Constraints

### System Dependencies
- Existing TTD-DR pipeline and research iteration framework
- Gemini API with Google Search integration (from 001-gemini-integration)
- Current Q&A history and citation tracking infrastructure

### Technical Constraints
- **Token Limits**: Individual LLM generation calls limited to 2048 output tokens
- **Context Window**: Must manage cumulative context to stay within model limits
- **API Rate Limits**: Must respect Gemini API quotas and implement backoff
- **Generation Latency**: Target ≤30 seconds per section to maintain user engagement

### Backward Compatibility
- Existing `/run` and `/evaluate` endpoints must continue to work
- Legacy single-pass report generation should remain as fallback option
- SSE streaming format must accommodate new structured output

---

## Out of Scope

1. **Interactive Report Refinement**: User cannot edit/request changes to generated sections (future feature)
2. **Visual Elements**: No charts, graphs, or diagrams in reports (future enhancement)
3. **Multi-Language Reports**: Output remains in Korean (or query language)
4. **Export Formats**: No PDF/DOCX export (future feature)
5. **Collaborative Editing**: No multi-user report editing capabilities
6. **Version History**: No tracking of report generation iterations
7. **Custom Templates**: Users cannot define custom report structures
8. **Real-Time Collaboration**: No simultaneous viewing/editing by multiple users

---

## Risks & Mitigations

### Risk 1: Increased Generation Time
**Description**: Iterative section generation significantly increases total time
**Probability**: HIGH | **Impact**: HIGH
**Mitigation**:
- Parallel generation of independent sections where possible
- Aggressive context summarization to reduce prompt size
- Caching of frequently referenced content
- Progress indicators to manage user expectations

### Risk 2: Context Loss in Long Reports
**Description**: Distant sections lose important context from early sections
**Probability**: MEDIUM | **Impact**: MEDIUM
**Mitigation**:
- Maintain hierarchical summary structure (chapter summaries + section summaries)
- Include "key insights so far" summary in each generation prompt
- Validate coherence with semantic similarity checks
- Allow regeneration of sections that fail coherence thresholds

### Risk 3: API Cost Increase
**Description**: 20+ LLM calls per report significantly increases costs
**Probability**: HIGH | **Impact**: MEDIUM
**Mitigation**:
- Context compression reduces average prompt size by 60%
- Use smaller models for summarization tasks
- Implement caching for repeated content
- Monitor and alert on cost thresholds

### Risk 4: Quality Variance Across Sections
**Description**: Later sections may have lower quality due to context compression
**Probability**: MEDIUM | **Impact**: MEDIUM
**Mitigation**:
- Quality validation after each section (minimum word count, citation density)
- Regeneration trigger if quality metrics fall below thresholds
- Preserve full context for critical sections (Executive Summary, Conclusion)
- A/B testing to optimize summarization strategies

---

## Business Value

### Primary Value
- **Comprehensive Analysis**: Users receive 3-4× more detailed reports enabling better decision-making
- **Multi-Perspective Insights**: Systematic coverage of diverse viewpoints reduces blind spots
- **Professional Quality**: Structured reports suitable for stakeholder presentations and documentation
- **Scalability**: System handles complex queries that previously resulted in incomplete analysis

### Competitive Advantage
- **Depth**: Significantly exceeds standard RAG systems that generate single-pass summaries
- **Structure**: Formal report framework unique in competition landscape
- **Adaptability**: Query-driven structure generation provides tailored analysis
- **Reliability**: Token limit management ensures complete delivery of promised content

### ROI Justification
- **Research Efficiency**: 60% reduction in manual research time for complex topics
- **Decision Quality**: 40% improvement in decision confidence with comprehensive analysis
- **Documentation Value**: Reports can be directly used in business cases and proposals
- **Competitive Scoring**: Structured, detailed reports likely to score higher in MMU-RAG competition evaluation

---

## Next Steps

1. **Review & Approval**: Stakeholder review of specification (target: 2 days)
2. **Planning Phase**: Run `/plan` to generate implementation plan with architecture and tasks
3. **Prototype**: Build proof-of-concept with 3-chapter, 9-section report structure
4. **Validation**: Test with 10 diverse queries to validate structure generation and quality
5. **Full Implementation**: Complete iterative generation system with all quality checks
6. **Competition Testing**: Validate with official competition evaluation queries

---

## Appendix: Example Report Structure

### Example Query
"Analyze the Pfizer-Metsera acquisition, including deal details, impact on both companies, and industry implications."

### Generated Report Structure
```
Executive Summary (1 section, ~400 words)

Chapter 1: Transaction Overview
├─ Section 1.1: Deal Structure and Terms
├─ Section 1.2: Timeline and Key Milestones
└─ Section 1.3: Financing and Payment Mechanism

Chapter 2: Strategic Impact on Pfizer
├─ Section 2.1: Portfolio Expansion in Obesity Market
├─ Section 2.2: Competitive Positioning vs. Novo Nordisk
├─ Section 2.3: Revenue and Growth Projections
└─ Section 2.4: Integration Challenges and Risks

Chapter 3: Implications for Novo Nordisk
├─ Section 3.1: Reasons for Withdrawal
├─ Section 3.2: Regulatory and Antitrust Concerns
└─ Section 3.3: Alternative Strategic Options

Chapter 4: Industry and Market Analysis
├─ Section 4.1: Obesity Drug Market Landscape
├─ Section 4.2: Competitive Dynamics Post-Acquisition
└─ Section 4.3: Impact on Other Market Players

Chapter 5: Risk Assessment and Future Outlook
├─ Section 5.1: Integration and Execution Risks
├─ Section 5.2: Regulatory and Market Risks
└─ Section 5.3: Long-term Industry Implications

Conclusion and Recommendations (1 section, ~400 words)

Citations (Organized by chapter)
```

**Total**: 6 chapters, 20 sections, estimated 3,800 words, 25-30 citations
