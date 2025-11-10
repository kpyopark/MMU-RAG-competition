"""
Test Gemini's native grounded generation with Google Search.

This tests if we can replace the entire Search→Chunk→Rerank pipeline
with Gemini's built-in Google Search grounding.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from google import genai
from google.genai import types

def test_grounded_generation():
    """Test Gemini with Google Search tool enabled."""
    print("=" * 70)
    print("Gemini Grounded Generation Test")
    print("=" * 70)

    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return

    client = genai.Client(api_key=api_key)

    # Test query
    query = "Latest developments in AI 2024"
    print(f"\n1. Query: '{query}'")
    print("-" * 70)

    # Create prompt
    prompt = f"""Based on current web information, answer this question comprehensively:

{query}

Provide a detailed answer with specific facts and citations."""

    # Configure with Google Search tool
    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)]
    )]

    tools = [types.Tool(google_search=types.GoogleSearch())]

    config = types.GenerateContentConfig(
        tools=tools,
        temperature=0.7,
        max_output_tokens=2048,
    )

    print("\n2. Executing Gemini with Google Search enabled...")
    print("-" * 70)

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",  # Using stable model instead of experimental
            contents=contents,
            config=config,
        )

        print("\n3. Response Structure Analysis:")
        print("-" * 70)

        # Check response structure
        print(f"Has candidates: {bool(response.candidates)}")
        if response.candidates:
            candidate = response.candidates[0]
            print(f"Has content: {bool(candidate.content)}")
            print(f"Has grounding_metadata: {bool(candidate.grounding_metadata)}")

            # Check grounding metadata
            if candidate.grounding_metadata:
                gm = candidate.grounding_metadata
                print(f"\nGrounding Metadata:")
                print(f"  grounding_support: {gm.grounding_support if hasattr(gm, 'grounding_support') else 'N/A'}")
                print(f"  web_search_queries: {gm.web_search_queries if hasattr(gm, 'web_search_queries') else 'N/A'}")
                print(f"  search_entry_point: {gm.search_entry_point if hasattr(gm, 'search_entry_point') else 'N/A'}")

                # Check grounding chunks
                if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                    print(f"\nGrounding Chunks: {len(gm.grounding_chunks)} chunks")
                    for i, chunk in enumerate(gm.grounding_chunks[:3]):
                        print(f"\n  Chunk {i+1}:")
                        if hasattr(chunk, 'web'):
                            print(f"    URL: {chunk.web.uri}")
                            print(f"    Title: {chunk.web.title}")
                        else:
                            print(f"    Structure: {dir(chunk)}")
                else:
                    print(f"  No grounding_chunks found")

                # Check grounding supports (citations in text)
                if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
                    print(f"\nGrounding Supports: {len(gm.grounding_supports)} citations")
                    for i, support in enumerate(gm.grounding_supports[:3]):
                        print(f"\n  Support {i+1}:")
                        print(f"    segment: {support.segment if hasattr(support, 'segment') else 'N/A'}")
                        print(f"    grounding_chunk_indices: {support.grounding_chunk_indices if hasattr(support, 'grounding_chunk_indices') else 'N/A'}")
                        print(f"    confidence_scores: {support.confidence_scores if hasattr(support, 'confidence_scores') else 'N/A'}")

        # Get response text
        response_text = response.text
        print(f"\n4. Generated Response:")
        print("-" * 70)
        print(f"Length: {len(response_text)} characters")
        print(f"\nFirst 500 characters:")
        print(response_text[:500])
        print("...")

        # Extract citations
        print(f"\n5. Citation Extraction:")
        print("-" * 70)

        citations = []
        if response.candidates and response.candidates[0].grounding_metadata:
            gm = response.candidates[0].grounding_metadata
            if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if hasattr(chunk, 'web'):
                        citations.append({
                            'url': chunk.web.uri,
                            'title': chunk.web.title
                        })

        print(f"Total citations: {len(citations)}")
        for i, citation in enumerate(citations[:5]):
            print(f"{i+1}. {citation['title']}")
            print(f"   {citation['url']}")

        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)

        has_grounding = bool(response.candidates and
                            response.candidates[0].grounding_metadata and
                            hasattr(response.candidates[0].grounding_metadata, 'grounding_chunks') and
                            response.candidates[0].grounding_metadata.grounding_chunks)

        if has_grounding:
            print("✅ Grounded generation WORKS")
            print("✅ Citations automatically included")
            print("✅ Can replace Search→Chunk→Rerank pipeline")
            print("\nBenefits:")
            print("  - Simpler architecture (1 API call vs 3)")
            print("  - Automatic relevance (Gemini selects best sources)")
            print("  - Integrated citations (no manual tracking)")
            print("  - Lower latency (no separate reranking)")
            print("\nTrade-offs:")
            print("  - Less control over chunking strategy")
            print("  - Cannot inspect/modify search results before synthesis")
            print("  - Depends on Gemini's search quality")
        else:
            print("❌ No grounding metadata found")
            print("⚠️  May need different model or configuration")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_grounded_generation()
