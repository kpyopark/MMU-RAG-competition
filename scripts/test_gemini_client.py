"""
Quick validation test for GeminiClient implementation.

Tests all three core methods:
1. complete() - LLM text generation
2. search() - Grounding API search
3. rerank_chunks() - Semantic relevance scoring
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gemini_client import GeminiClient


def test_client_initialization():
    """Test client initialization with env var."""
    print("=" * 70)
    print("Test 1: Client Initialization")
    print("=" * 70)

    try:
        client = GeminiClient()
        print(f"✅ Client initialized with model: {client.model_name}")
        return client
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        sys.exit(1)


def test_complete(client):
    """Test LLM completion."""
    print("\n" + "=" * 70)
    print("Test 2: LLM Completion")
    print("=" * 70)

    try:
        response = client.complete(
            prompt="What is 2+2?",
            system_prompt="You are a helpful assistant. Respond concisely.",
            max_tokens=50
        )
        print(f"✅ Completion response: {response[:100]}")
        return True
    except Exception as e:
        print(f"❌ Completion failed: {e}")
        return False


def test_search(client):
    """Test Gemini search with grounding."""
    print("\n" + "=" * 70)
    print("Test 3: Gemini Search (Grounding API)")
    print("=" * 70)

    try:
        results = client.search(
            query="What are the latest developments in AI as of 2024?",
            top_k=5
        )
        print(f"✅ Search returned {len(results)} results")

        if results:
            print("\nFirst result:")
            print(f"  URL: {results[0].url}")
            print(f"  Title: {results[0].title}")
            print(f"  Metadata: {results[0].metadata}")

        return True
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False


def test_rerank(client):
    """Test chunk reranking."""
    print("\n" + "=" * 70)
    print("Test 4: Chunk Reranking")
    print("=" * 70)

    # Sample chunks for reranking
    chunks = [
        {
            "chunk_id": 0,
            "text": "Python is a high-level programming language known for its simplicity.",
            "url": "https://example.com/python",
            "token_count": 12
        },
        {
            "chunk_id": 1,
            "text": "Machine learning is a subset of artificial intelligence.",
            "url": "https://example.com/ml",
            "token_count": 10
        },
    ]

    try:
        ranked = client.rerank_chunks(
            query="What is Python?",
            chunks=chunks,
            top_k=2
        )
        print(f"✅ Reranking returned {len(ranked)} chunks")

        for i, chunk in enumerate(ranked):
            print(f"\nRank {i+1}:")
            print(f"  Chunk ID: {chunk.chunk_id}")
            print(f"  Score: {chunk.score:.3f}")
            print(f"  Text: {chunk.text[:60]}...")

        return True
    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        return False


if __name__ == "__main__":
    print("GeminiClient Validation Test")
    print("=" * 70)
    print()

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set in environment")
        sys.exit(1)

    # Run tests
    client = test_client_initialization()

    results = {
        "complete": test_complete(client),
        "search": test_search(client),
        "rerank": test_rerank(client),
    }

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - GeminiClient is ready for use")
    else:
        print("⚠️ SOME TESTS FAILED - Review errors above")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)
