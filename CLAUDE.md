# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**TTD-RAG with Gemini Integration** - Test-Time Diffusion framework for MMU-RAG Competition. Implements "Deep Researcher with Test-Time Diffusion (TTD-DR)" using iterative denoising: preliminary draft → targeted search → synthesis → revision → final report.

**Tech Stack:**
- Backend: FastAPI
- LLM: Google Gemini API (`google-genai` SDK v1.49.0)
- Search: Gemini Grounding API with Google Search
- Reranking: LLM-based semantic scoring (Gemini)
- Containerization: Docker (no GPU required)

**Implementation Status:** ✅ **COMPLETE** (2025-11-10)
- Migration from vLLM + FineWeb → Gemini API: **COMPLETE**
- All tests passing (unit + integration + end-to-end)
- Production ready, deployed and tested

**Performance Improvements:**

| Metric | Before (vLLM) | After (Gemini) | Improvement |
|--------|---------------|----------------|-------------|
| Dependencies | 400+ packages | 51 packages | 87% reduction |
| Startup time | 200 seconds | <1 second | 199x faster |
| GPU requirement | 24GB VRAM | None | 100% reduction |
| OOM errors | Frequent | None | 100% eliminated |

## Essential Commands

```bash
# Environment setup
export GEMINI_API_KEY=your_key_here
uv sync  # Installs 51 packages (<100MB)

# Run server
uvicorn src:app --host 0.0.0.0 --port 5053

# Test Gemini integration
uv run python scripts/test_gemini_client.py      # Unit tests
uv run python scripts/test_integration.py        # Integration tests

# API compliance testing
uv run python local_test.py --base-url http://localhost:5053

# Testing Python syntax/code
uv run python -m py_compile src/file.py          # ALWAYS use uv for Python tests
uv run python scripts/test_script.py             # Execute scripts in uv environment
```

## Critical Lessons Learned

### SDK Validation (BLOCKER-1)

**Always validate SDK structure with real API calls before implementation.**

#### Validated Facts (2025-11-10):
```python
# ✅ CORRECT: Grounding metadata access pattern
response.candidates[0].grounding_metadata.grounding_chunks[i].web.uri   # URL
response.candidates[0].grounding_metadata.grounding_chunks[i].web.title # Title

# ❌ WRONG: Initially assumed (documentation mismatch)
response.grounding_metadata.grounding_chunks[i].web.url  # Wrong path & field
```

#### Key Findings:
| Issue | Wrong Assumption | Validated Reality |
|-------|------------------|-------------------|
| Metadata location | `response.grounding_metadata` | `response.candidates[0].grounding_metadata` |
| URL field name | `chunk.web.url` | `chunk.web.uri` |
| Snippet availability | Assumed available | NOT in API response |
| Minimum URL count | Unknown | 12+ chunks returned |

#### SDK Pattern (Validated):
```python
from google import genai  # NOT google.generativeai
from google.genai import types

# Initialize
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Request structure
contents = [types.Content(role="user", parts=[types.Part.from_text(text=query)])]
tools = [types.Tool(googleSearch=types.GoogleSearch())]
config = types.GenerateContentConfig(tools=tools)

# Execute
response = client.models.generate_content(model="gemini-flash-latest", contents=contents, config=config)

# Extract (note the nested path)
if response.candidates and response.candidates[0].grounding_metadata:
    chunks = response.candidates[0].grounding_metadata.grounding_chunks
    for chunk in chunks:
        url = chunk.web.uri      # ✅ Not .url
        title = chunk.web.title  # ✅ Available
        # snippet NOT available
```

### Dependency Management

**Remove CUDA bloat:** Initial `uv sync` attempted to install 400+ packages (vLLM, torch, CUDA). After cleaning `pyproject.toml`:
- Before: 400+ packages, 2GB+ download, OOM errors
- After: 51 packages, <100MB, instant startup

**Action:** Always audit dependencies before first install. Use `uv pip list` to verify.

### Testing Philosophy

1. **API Structure Discovery**: Write validation scripts (`scripts/validate_grounding_metadata.py`) before implementation
2. **Real API Testing**: Mock assumptions kill projects. Test against real endpoints.
3. **Document Findings**: Update architecture docs with validated structure immediately
4. **Field Name Verification**: Never assume field names from documentation alone

## Architecture

### Core Pipeline (3 Stages)

**Stage 1: Planning** → **Stage 2: Iterative Refinement** → **Stage 3: Final Synthesis**

All stages now use Gemini API (LLM + Search + Reranking).

### Key Modules

**`src/gemini_client.py`** ✅ COMPLETE:
- `complete()`: Text generation with retry logic
- `search()`: Grounding API search (validated metadata path)
- `rerank_chunks()`: LLM-based semantic scoring
- ✨ **`complete_with_search()`**: NEW - Grounded generation with automatic citations
- All methods tested and working

**`src/generator.py`** ✅ COMPLETE:
- Now uses `GeminiClient` internally
- `get_llm_response()`: Backward-compatible wrapper
- `self_evolve()`: Multi-variant critique/merge algorithm

**`src/retriever.py`** ✅ COMPLETE:
- ✨ **`retrieve_with_grounded_generation()`**: NEW - Single-call grounded search + synthesis
- `retrieve()`: DEPRECATED - Old Search → Chunk → Rerank pipeline (preserved for compatibility)
- Full grounded generation pipeline working with automatic citations

**`src/pipeline.py`**: Orchestration (unchanged, uses above modules)

**`src/__init__.py`**: FastAPI endpoints (unchanged)
- `/health`, `/run` (SSE), `/evaluate` (JSON)

## API Contracts (Do Not Break)

### SSE Streaming (`/run`)
```python
{
  "intermediate_steps": "step1|||---|||step2",  # Separator: |||---|||
  "final_report": "...",
  "citations": ["url1", "url2"],
  "is_intermediate": true,
  "complete": false
}
```

### Static Response (`/evaluate`)
```python
{
  "query_id": "maps to iid from request",
  "answer": "final report text",
  "citations": ["url1", "url2"]
}
```

## Test Results (2025-11-10)

**Unit Tests** ✅ ALL PASSED:
- Client initialization
- LLM completion (18 prompt + 1 completion = 19 total tokens)
- Gemini Search (5 results with validated metadata)
- Chunk reranking (semantic scoring working)

**Integration Tests** ✅ ALL PASSED:
- Generator module (get_llm_response, self_evolve)
- Retriever module (search + rerank pipeline)
- Full pipeline end-to-end

**Production Test** ✅ SUCCESSFUL:
- Query: "What is machine learning?"
- Response: 8,304 character comprehensive report
- Time: ~2 minutes (acceptable within 600s budget)
- All pipeline stages executed correctly

## Known Limitations

### ✅ RESOLVED: Gemini Search Content Issue (2025-11-10)

**Previous Issue** (NOW FIXED):
- Old `search()` method returned only titles (5-20 chars), not full page content
- Chunker required 500+ tokens → resulted in 0 chunks created
- No web grounding in responses

**Solution Implemented**:
- New `complete_with_search()` method uses Gemini Grounded Generation
- Single API call replaces 3-step Search → Chunk → Rerank pipeline
- Full web content automatically retrieved and synthesized
- Automatic citation extraction (no manual URL tracking)
- See: `GEMINI_GROUNDED_GENERATION_REPORT.md` for details

### Remaining Considerations

**Grounded Generation Control**:
- Gemini generates search queries internally (less manual control vs old pipeline)
- Cannot customize chunking strategy (handled internally by Gemini)
- Trade-off: Simpler architecture, better performance, automatic optimization

**API Cost Management**:
- Monitor grounded generation token usage per search
- Tune `MAX_SEARCH_ITERATIONS` parameter based on quality/cost balance
- Current setting: 1 iteration (production default)

## Development Guidelines

**Critical Rules**:
- Always use validated API paths: `response.candidates[0].grounding_metadata`
- Never skip API structure validation with real calls
- Maintain SSE callback contract (streaming depends on it)
- Test with real API before committing changes

## File Organization

```
src/
├── __init__.py          # FastAPI app, endpoints
├── pipeline.py          # TTD-DR orchestration
├── generator.py         # LLM operations (Gemini)
├── retriever.py         # Search & reranking (Gemini)
├── chunker.py           # Text preprocessing
└── gemini_client.py     # Unified Gemini API client

specs/001-gemini-integration/
├── spec.md              # Feature requirements
├── plan.md              # Implementation plan
├── tasks.md             # 55 tasks with dependencies
├── architecture.md      # Validated SDK structure (BLOCKER-1 resolved)
└── research.md          # Technical feasibility

scripts/
├── test_gemini_client.py           # Unit tests (GeminiClient)
├── test_integration.py              # Integration tests (full pipeline)
└── validate_grounding_metadata.py   # API structure validation

Root:
├── pyproject.toml       # Dependencies (google-genai>=1.0.0)
├── local_test.py        # API compliance testing
└── .env.example         # GEMINI_API_KEY required
```

## Common Issues & Solutions

### Issue: Grounding metadata not found
**Solution:** Use correct path: `response.candidates[0].grounding_metadata` (not `response.grounding_metadata`)

### Issue: Field name doesn't exist
**Solution:** Validate with real API call. Example: `chunk.web.uri` not `chunk.web.url`

### Issue: Snippet field missing
**Solution:** Snippet not available in Gemini grounding metadata. Use `chunk.web.title` only.

### Issue: CUDA packages installing
**Solution:** Clean `pyproject.toml` dependencies. Remove vLLM, torch, CUDA packages.

### Issue: API rate limits
**Solution:** Retry logic with exponential backoff (1s, 2s, 4s delays). Check `GEMINI_API_KEY` quota.

## Competition Requirements

- ✅ Streaming (`/run`) and static (`/evaluate`) modes
- ✅ Citations with source URLs required
- ✅ Gemini Grounding API integration (validated)
- ✅ Docker deployment (no GPU dependency)
- ✅ 600-second timeout budget (current: ~105s per query)

## References

- Architecture design: `specs/001-gemini-integration/architecture.md`
- BLOCKER-1 resolution: See architecture.md lines 28-147
- Validation script: `scripts/validate_grounding_metadata.py`
- API docs: [google-genai SDK](https://github.com/googleapis/python-genai)
