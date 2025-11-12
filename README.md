# TTD-RAG: A Test-Time Diffusion Framework for the MMU-RAG Competition

This repository contains our submission for the **[MMU-RAG Competition](https://agi-lti.github.io/MMU-RAGent/)**, a deep research agent named TTD-RAG. Our system is a faithful implementation of the framework proposed in the paper *"[Deep Researcher with Test-Time Diffusion (TTD-DR)](https://research.google/blog/deep-researcher-with-test-time-diffusion/)"*.

It conceptualizes report generation as an iterative "denoising" process, starting with a preliminary draft and progressively refining it through cycles of targeted search, synthesis, and revision. This approach is designed to excel at complex, multi-hop reasoning tasks that require coherent, long-form answers.

## 🆕 Recent Updates

### Version 2.0 - Gemini Integration & Structured Reports (2025-11-12)

**Major Features Added:**

1. **Gemini Grounded Generation** (Feature 001)
   - Replaced FineWeb Search + Chunk + Rerank pipeline with Gemini's native grounded generation
   - Single API call performs: web search → answer generation → citation extraction
   - Automatic citation tracking from authoritative web sources
   - Significant reduction in token usage and API complexity

2. **Structured Report Generation** (Feature 002)
   - Multi-chapter report generation with 4-6 perspectives
   - Section-by-section synthesis (300-500 words per section)
   - Progressive context management with sliding window (≤40% of context window)
   - Quality validation: depth (≥300 words), citations (≥1 per 150 words), redundancy (<70%), coherence (≥0.8)
   - Automatic regeneration for quality improvements
   - Professional markdown output with organized citations

**New Modules:**
- `src/gemini_client.py` - Gemini API client with grounded generation
- `src/structure_generator.py` - Multi-chapter report structure planning
- `src/section_generator.py` - Section-by-section generation with token limits
- `src/context_manager.py` - Progressive context summarization (87% reduction)
- `src/quality_validator.py` - Section quality validation and regeneration
- `src/report_assembler.py` - Final markdown report assembly
- `src/report_structure.py` - Data structures for reports

## 🎯 Key Features

* **Test-Time Diffusion Framework**: Models research report generation as an iterative process of refining a "noisy" draft with external information, ensuring coherence and reducing information loss.
* **Gemini Grounded Generation**: Native web search integration with automatic citation extraction (replaces traditional RAG pipeline).
* **Structured Multi-Chapter Reports**: Generate comprehensive reports with 4-6 perspectives, 15+ sections, executive summary, and conclusion.
* **Progressive Context Management**: Sliding window with compression (87% token reduction) to maintain rich context within budget.
* **Quality Validation**: Automatic section validation and regeneration for depth, citations, redundancy, and coherence.
* **Component-wise Self-Evolution**: Enhances the quality of each step in the workflow (planning, synthesis) by generating diverse variants, critiquing them, and merging them into a superior output.
* **Competition Compliant**: Fully supports both dynamic (streaming) and static evaluation endpoints as required by the competition rules, validated with the provided `local_test.py` script.

## ⚙️ System Architecture & Workflow

The agent operates in a structured, multi-stage process orchestrated by `src/pipeline.py`:

### 1. Stage 1: Planning & Initial Drafting
* An initial **Research Plan** is generated to outline the key areas of investigation.
* A preliminary **Noisy Draft** is created based on the LLM's internal knowledge, serving as the starting point for the diffusion process.

### 2. Stage 2: Iterative Search & Denoising
The system enters a loop, where for each iteration:
1. A **new search query** is generated, informed by the current draft's deficiencies and the overall plan.
2. **Gemini grounded generation** performs web search and answer synthesis in a single API call.
3. **Citations are automatically extracted** from web sources and tracked.
4. The draft is **revised ("denoised")** by integrating this new information.

### 3. Stage 3: Report Generation

**Legacy Mode (Single-Pass):**
- Synthesizes the final draft, plan, and Q&A history into a comprehensive report

**Structured Mode (Multi-Chapter):**
1. **Structure Generation**: Creates multi-chapter outline with 4-6 perspectives
2. **Executive Summary**: High-level synthesis covering all major findings
3. **Iterative Section Generation**: Generate 300-500 word sections with:
   - Progressive context compression (recent 5 sections in full, older compressed)
   - Quality validation (depth, citations, redundancy, coherence)
   - Automatic regeneration if quality thresholds not met
4. **Conclusion**: Forward-looking synthesis with recommendations
5. **Assembly**: Professional markdown with organized citations by chapter

## 🛠️ Technology Stack

* **Backend Framework**: FastAPI
* **LLM Integration**: Google Gemini API (grounded generation)
* **Generative LLM**: Gemini Flash (via API)
* **Containerization**: Docker
* **GPU Acceleration**: Optional (legacy vLLM mode)

## 🚀 Getting Started

### Prerequisites

**For Gemini Integration (Recommended):**
- Python 3.10+
- Google Gemini API key
- No GPU required

**For Legacy vLLM Mode:**
- Docker and Docker Compose
- NVIDIA GPU with 24GB+ VRAM
- NVIDIA Container Toolkit

### 1. Configure Environment

Create environment file with your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required for Gemini integration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Specify Gemini model (default: gemini-flash-latest)
GEMINI_MODEL=gemini-flash-latest

# Legacy: FineWeb API (only needed for legacy mode)
FINEWEB_API_KEY=your_fineweb_api_key_here

# Legacy: OpenRouter fallback (only needed for legacy mode)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Get Gemini API Key:**
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Copy to `.env` file

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Feature Configuration

Edit `src/pipeline.py` to configure system behavior:

```python
# Line 19: Enable/disable structured report generation
ENABLE_STRUCTURED_REPORTS = True  # Multi-chapter reports
# ENABLE_STRUCTURED_REPORTS = False  # Legacy single-pass

# Lines 13-16: Performance tuning
NUM_VARIANTS = 1          # Self-evolution variants (↑ = better quality, slower)
EVOLUTION_STEPS = 1       # Critique-revise cycles (↑ = better quality, slower)
MAX_SEARCH_ITERATIONS = 1 # Search-synthesis loops (1-3 recommended)
SEARCH_TOP_K = 50        # Web sources per search
```

**Recommended Configurations:**

```python
# Fast testing (1-2 minutes)
ENABLE_STRUCTURED_REPORTS = False
MAX_SEARCH_ITERATIONS = 1

# Production quality (5-10 minutes)
ENABLE_STRUCTURED_REPORTS = True
NUM_VARIANTS = 2
EVOLUTION_STEPS = 2
MAX_SEARCH_ITERATIONS = 3
```

## ✅ Testing Your Implementation

### Method 1: Direct Python API (Fastest - No Server Required)

**Simple Static Test:**
```bash
uv run python -c "from src.pipeline import run_rag_static; print(run_rag_static('What is quantum computing?'))"
```

**Interactive Test Script:**
```python
# test_direct.py
from src.pipeline import run_rag_static, run_rag_dynamic

# Method 1: Static (simplest - returns final report)
print("Testing static pipeline...")
report = run_rag_static("Analyze the impact of AI on healthcare")
print(report)

# Save to file
with open("report.md", "w") as f:
    f.write(report)

# Method 2: Dynamic (with progress updates)
def progress_callback(data):
    if data["is_intermediate"]:
        print(f"Progress: {data['intermediate_steps']}")
    if data["complete"]:
        print(f"\n\nFinal Report:\n{data['final_report']}")
        if data.get("citations"):
            print(f"\nCitations: {data['citations']}")

print("\nTesting dynamic pipeline...")
run_rag_dynamic("What are the latest developments in renewable energy?", progress_callback)
```

Run:
```bash
uv run python test_direct.py
```

### Method 2: Using Existing Test Scripts

```bash
# Test Gemini integration
uv run python scripts/test_gemini_client.py

# Test grounded generation
uv run python scripts/test_grounded_generation.py

# Test full pipeline with grounding
uv run python scripts/test_pipeline_grounded.py

# Test integration
uv run python scripts/test_integration.py
```

### Method 3: Server-Based Testing (Full API Compliance)

**Start the server:**
```bash
# Option 1: Direct uvicorn
uv run uvicorn src:app --host 0.0.0.0 --port 5053

# Option 2: Docker Compose (includes vLLM for legacy mode)
docker compose up --build
```

**Test with curl:**
```bash
# Dynamic evaluation (streaming)
curl -X POST http://localhost:5053/run \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

# Static evaluation
curl -X POST http://localhost:5053/evaluate \
  -H "Content-Type: application/json" \
  -d '{"iid": "test-001", "question": "Explain quantum computing"}'
```

**Competition compliance test:**
```bash
uv sync
source venv/bin/activate

# Test both endpoints (full test)
python local_test.py --base-url http://localhost:5053

# Test only dynamic /run endpoint
python local_test.py --base-url http://localhost:5053 --test-mode run

# Test only static /evaluate endpoint
python local_test.py --base-url http://localhost:5053 --test-mode evaluate
```

### Method 4: Quick One-Liner Tests

```bash
# Simple test
uv run python -c "from src.pipeline import run_rag_static; print(run_rag_static('What is AI?'))"

# With file output
uv run python -c "from src.pipeline import run_rag_static; open('test_report.md', 'w').write(run_rag_static('Explain quantum computing'))"

# Korean query test
uv run python -c "from src.pipeline import run_rag_static; print(run_rag_static('양자 컴퓨팅에 대해 설명해줘'))"
```

## 📋 API Endpoints

* **Health Check**: `GET /health`
  * Confirms the service is running. Returns `{"status": "ok"}`.

* **Dynamic Evaluation**: `POST /run`
  * **Input**: `{"question": "string"}`
  * **Output**: Server-Sent Events (SSE) stream with real-time updates:
    - `intermediate_steps`: Progress updates
    - `citations`: Web sources used
    - `final_report`: Complete markdown report
    - `complete`: Completion flag

* **Static Evaluation**: `POST /evaluate`
  * **Input**: `{"iid": "string", "question": "string"}`
  * **Output**: `{"query_id": "string", "result": "string"}`

## 📊 Performance & Quality Metrics

### Structured Report Output (ENABLE_STRUCTURED_REPORTS = True)

**Example Report Statistics:**
- **Total Sections**: 17 (1 executive summary + 15 main + 1 conclusion)
- **Total Word Count**: ~6,000 words
- **Chapters**: 4-6 multi-perspective chapters
- **Citations**: Organized by chapter with source URLs
- **Generation Time**: 5-10 minutes (depends on iterations)

**Quality Metrics:**
- Depth: ≥300 words per section (target: 350)
- Citations: ≥1 per 150 words
- Redundancy: <70% similarity with previous sections
- Coherence: ≥0.8 score (no placeholders/errors)

### Context Management Efficiency

- **Full Section**: ~500 tokens (350 words)
- **Compressed Summary**: ≤200 tokens (87% reduction)
- **Context Budget**: ≤8K tokens (40% of 20K Gemini window)
- **Sliding Window**: Recent 5 sections in full, older compressed

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Gemini API Configuration
GEMINI_API_KEY=your_key          # Required
GEMINI_MODEL=gemini-flash-latest # Optional (default shown)

# Legacy Configuration (vLLM mode)
FINEWEB_API_KEY=your_key         # Only for legacy mode
OPENROUTER_API_KEY=your_key      # Only for legacy fallback
OPENROUTER_MODEL=...             # Optional fallback model
```

### Feature Flags (src/pipeline.py)

```python
# Line 19: Report generation mode
ENABLE_STRUCTURED_REPORTS = True   # Multi-chapter structured reports
ENABLE_STRUCTURED_REPORTS = False  # Legacy single-pass generation

# Lines 13-16: Performance tuning
NUM_VARIANTS = 1          # Parallel variants for self-evolution
EVOLUTION_STEPS = 1       # Critique-revise cycles per component
MAX_SEARCH_ITERATIONS = 1 # Number of search-synthesis-revision loops
SEARCH_TOP_K = 50        # Documents to retrieve per search

# Quality thresholds (src/quality_validator.py)
MIN_WORD_COUNT = 300             # Minimum words per section
TARGET_WORD_COUNT = 350          # Target words per section
MIN_CITATION_DENSITY = 1.0/150   # Citations per 150 words
MAX_REDUNDANCY_SIMILARITY = 0.70 # Maximum section similarity
MIN_COHERENCE_SCORE = 0.8        # Minimum coherence score

# Context management (src/context_manager.py)
MAX_CONTEXT_BUDGET = 8000        # Target context tokens (40% of window)
SUMMARY_TARGET_TOKENS = 200      # Tokens per compressed section
SLIDING_WINDOW_SIZE = 5          # Recent sections kept in full
```

## 🏗️ Project Structure

```
MMU-RAG-competition/
├── src/
│   ├── __init__.py              # FastAPI app and API endpoints
│   ├── pipeline.py              # Main TTD-DR pipeline orchestration
│   ├── generator.py             # LLM client management
│   ├── retriever.py             # Document retrieval (legacy)
│   ├── chunker.py               # Text preprocessing (legacy)
│   │
│   ├── gemini_client.py         # Gemini API integration (NEW)
│   ├── structure_generator.py   # Multi-chapter planning (NEW)
│   ├── section_generator.py     # Section synthesis (NEW)
│   ├── context_manager.py       # Context compression (NEW)
│   ├── quality_validator.py     # Quality validation (NEW)
│   ├── report_assembler.py      # Final assembly (NEW)
│   └── report_structure.py      # Data structures (NEW)
│
├── scripts/
│   ├── test_gemini_client.py          # Test Gemini integration
│   ├── test_grounded_generation.py    # Test grounded generation
│   ├── test_pipeline_grounded.py      # Test full pipeline
│   └── test_integration.py            # Integration tests
│
├── specs/                         # Feature specifications
│   ├── 001-gemini-integration/    # Gemini integration docs
│   └── 002-structured-report-generation/  # Structured reports docs
│
├── .env.example                   # Environment template
├── compose.yml                    # Docker Compose configuration
├── Dockerfile                     # Container image definition
├── pyproject.toml                 # Python dependencies (uv)
├── local_test.py                  # Competition compliance test
└── README.md                      # This file
```

## 🚢 Competition Submission

### Build for Deployment

```bash
# Build for linux/amd64 platform
docker build --platform linux/amd64 -t ttt-dr:latest .
```

### Push to AWS ECR

1. **Sign in to AWS ECR**

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com
```

2. **Tag the Image**

```bash
docker tag ttt-dr:latest \
  <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/neurips2025text/ttt-dr:latest
```

3. **Push to Registry**

```bash
docker push \
  <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/neurips2025text/ttt-dr:latest
```

## 🤝 Contributing

See `CLAUDE.md` for development guidelines and project conventions.

## 📄 License

This project is submitted for the MMU-RAG Competition. Please refer to the competition rules for usage terms.

## 🙏 Acknowledgments

This implementation is based on the "Deep Researcher with Test-Time Diffusion" framework from Google Research. We acknowledge the original authors and the MMU-RAG Competition organizers.
