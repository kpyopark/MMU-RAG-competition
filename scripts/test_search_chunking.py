"""
Diagnostic script to test Gemini Search and chunking pipeline.

This script tests why chunks are not being created from search results.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gemini_client import GeminiClient
from src.chunker import chunk_document

def test_search_and_chunking():
    """Test the complete search → chunk pipeline."""
    print("=" * 70)
    print("Gemini Search and Chunking Diagnostic")
    print("=" * 70)

    # Initialize client
    client = GeminiClient()

    # Test search
    query = "What is machine learning?"
    print(f"\n1. Testing Gemini Search with query: '{query}'")
    print("-" * 70)

    search_results = client.search(query, top_k=5)
    print(f"✅ Search returned {len(search_results)} results")

    # Inspect first result structure
    if search_results:
        print(f"\n2. First Search Result Structure:")
        print("-" * 70)
        first = search_results[0]
        print(f"URL: {first.url}")
        print(f"Title: {first.title}")
        print(f"Text: {first.text}")
        print(f"Text length: {len(first.text)} characters")
        print(f"Metadata: {first.metadata}")

        # Convert to document format (as retriever does)
        print(f"\n3. Document Format (as retriever.py creates):")
        print("-" * 70)
        doc = {
            "url": first.url,
            "content": first.text,
            "title": first.title,
            "metadata": first.metadata,
        }
        print(f"Document keys: {doc.keys()}")
        print(f"'content' value: {doc['content']}")
        print(f"'content' length: {len(doc['content'])} characters")

        # Test chunking
        print(f"\n4. Testing chunk_document():")
        print("-" * 70)
        print(f"Chunker expects doc.get('text', '')")
        print(f"But document has: {list(doc.keys())}")
        print(f"doc.get('text', '') = '{doc.get('text', '')}'")
        print(f"doc.get('content', '') = '{doc.get('content', '')}'")

        # Try chunking with current format
        chunks = chunk_document(doc)
        print(f"\n❌ Chunks created with current format: {len(chunks)}")

        # Fix: add 'text' key
        doc['text'] = doc['content']
        chunks_fixed = chunk_document(doc)
        print(f"✅ Chunks created after adding 'text' key: {len(chunks_fixed)}")

        if chunks_fixed:
            print(f"\nFirst chunk preview:")
            print(f"  Text: {chunks_fixed[0]['text'][:100]}...")
            print(f"  Tokens: {chunks_fixed[0]['token_count']}")
            print(f"  URL: {chunks_fixed[0]['url']}")

    print("\n" + "=" * 70)
    print("ROOT CAUSE IDENTIFIED")
    print("=" * 70)
    print("❌ retriever.py creates document with 'content' key")
    print("❌ chunker.py expects document with 'text' key")
    print("❌ Mismatch causes chunker to receive empty string")
    print("")
    print("✅ FIX: Change retriever.py line 44 from:")
    print("   'content': result.text")
    print("   TO:")
    print("   'text': result.text")
    print("=" * 70)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set in environment")
        sys.exit(1)

    test_search_and_chunking()
