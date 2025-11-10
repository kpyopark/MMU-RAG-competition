"""
Integration test for Gemini-based pipeline.

Tests the complete flow:
1. GeminiClient initialization
2. LLM completion via generator.py
3. Document retrieval via retriever.py (Gemini Search + reranking)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generator import get_llm_response, self_evolve
from src.retriever import retrieve


def test_generator():
    """Test LLM completion through generator.py wrapper."""
    print("=" * 70)
    print("Test 1: Generator (LLM Completion)")
    print("=" * 70)

    try:
        response = get_llm_response(
            "What is 2+2?",
            "You are a helpful assistant. Be concise."
        )
        print(f"✅ Generator response: {response[:100]}")
        return True
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_self_evolve():
    """Test self-evolution algorithm."""
    print("\n" + "=" * 70)
    print("Test 2: Self-Evolution Algorithm")
    print("=" * 70)

    try:
        final_text, variants = self_evolve(
            initial_prompt="Explain what Python is in one sentence.",
            system_prompt="You are a technical writer.",
            num_variants=1,  # Use 1 to save API calls
            evolution_steps=1,
        )
        print(f"✅ Self-evolution completed")
        print(f"   Final text: {final_text[:100]}...")
        print(f"   Variants generated: {len(variants)}")
        return True
    except Exception as e:
        print(f"❌ Self-evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retriever():
    """Test retrieval with Gemini Search and reranking."""
    print("\n" + "=" * 70)
    print("Test 3: Retriever (Gemini Search + Reranking)")
    print("=" * 70)

    try:
        chunks = retrieve(
            query="What are the latest developments in AI?",
            top_k=3,
            search_top_k=5,
        )
        print(f"✅ Retrieval completed")
        print(f"   Chunks returned: {len(chunks)}")

        if chunks:
            print("\n   Top chunk:")
            print(f"     URL: {chunks[0].get('url', 'N/A')}")
            print(f"     Score: {chunks[0].get('rerank_score', 'N/A')}")
            print(f"     Text: {chunks[0].get('text', '')[:80]}...")

        return True
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Gemini Integration Test")
    print("=" * 70)
    print()

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set in environment")
        sys.exit(1)

    # Run tests
    results = {
        "generator": test_generator(),
        "self_evolve": test_self_evolve(),
        "retriever": test_retriever(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("   The Gemini pipeline is ready for production use")
    else:
        print("⚠️ SOME TESTS FAILED - Review errors above")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)
