"""
Grounding metadata structure validation for BLOCKER-1.

This script validates the grounding metadata structure returned by the new
google-genai SDK to resolve BLOCKER-1 before T020 implementation.

Usage:
    export GEMINI_API_KEY=your_key_here
    python scripts/validate_grounding_metadata.py
"""

import os
import sys
import json
from google import genai
from google.genai import types

def test_grounding_metadata_extraction():
    """Test grounding metadata structure with sample query."""

    # Initialize client (new SDK pattern)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Test query designed to return web sources
    test_query = "What are the latest developments in Large Language Models as of 2024?"

    print(f"Testing grounding metadata extraction...")
    print(f"Query: {test_query}\n")

    try:
        # Use new SDK tool pattern (from user example)
        model = "gemini-flash-latest"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=test_query),
                ],
            ),
        ]
        tools = [
            types.Tool(googleSearch=types.GoogleSearch()),
        ]
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
        )

        # Use non-streaming for easier response inspection
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        # Introspect response structure
        print("=== Response Structure ===")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}\n")

        # Check for candidates and grounding metadata (correct location)
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            print(f"✅ candidates found: {len(response.candidates)} candidates")
            print(f"Candidate type: {type(candidate)}\n")

            if hasattr(candidate, 'grounding_metadata'):
                print("✅ grounding_metadata attribute found in candidate")
                print(f"Metadata type: {type(candidate.grounding_metadata)}")
                print(f"Metadata attributes: {dir(candidate.grounding_metadata)}\n")

                # Check for grounding chunks
                if hasattr(candidate.grounding_metadata, 'grounding_chunks'):
                    chunks = candidate.grounding_metadata.grounding_chunks
                    print(f"✅ grounding_chunks found: {len(chunks)} chunks\n")

                    # Inspect first chunk structure
                    if chunks:
                        chunk = chunks[0]
                        print("=== First Chunk Structure ===")
                        print(f"Chunk type: {type(chunk)}")
                        print(f"Chunk attributes: {dir(chunk)}\n")

                        # Check for web attribute
                        if hasattr(chunk, 'web'):
                            print("✅ web attribute found")
                            print(f"Web type: {type(chunk.web)}")
                            print(f"Web attributes: {dir(chunk.web)}\n")

                            # Extract URL (test multiple field names)
                            url = None
                            for field in ['uri', 'url', 'link', 'href']:
                                if hasattr(chunk.web, field):
                                    url = getattr(chunk.web, field)
                                    print(f"✅ URL found at: chunk.web.{field}")
                                    print(f"   URL: {url}\n")
                                    break

                            if not url:
                                print("❌ URL field not found in chunk.web")
                                print(f"   Available fields: {[a for a in dir(chunk.web) if not a.startswith('_')]}\n")

                            # Extract title
                            if hasattr(chunk.web, 'title'):
                                print(f"✅ Title found: {chunk.web.title}\n")
                            else:
                                print("⚠️ title field not found\n")

                            # Extract snippet
                            if hasattr(chunk.web, 'snippet'):
                                print(f"✅ Snippet found: {chunk.web.snippet[:100]}...\n")
                            else:
                                print("⚠️ snippet field not found\n")
                        else:
                            print("❌ web attribute not found in chunk")
                            print(f"   Available attributes: {[a for a in dir(chunk) if not a.startswith('_')]}\n")

                    # Validate URL count
                    urls_found = 0
                    for chunk in chunks:
                        if hasattr(chunk, 'web'):
                            for field in ['uri', 'url', 'link', 'href']:
                                if hasattr(chunk.web, field):
                                    urls_found += 1
                                    break

                    print(f"=== Validation Summary ===")
                    print(f"Total chunks: {len(chunks)}")
                    print(f"URLs extracted: {urls_found}")

                    if urls_found >= 3:
                        print("✅ Meets minimum URL requirement (≥3)")
                        print("\nBLOCKER-1 RESOLUTION: SUCCESS")
                        print("Grounding metadata structure validated.")
                        print("Ready to proceed with T020 implementation.")
                    else:
                        print(f"❌ FAILS minimum URL requirement: {urls_found} < 3")
                        print("   This violates spec.md Edge Case 5\n")
                        print("\nBLOCKER-1 RESOLUTION: PARTIAL")
                        print("Need to test with additional queries or adjust implementation.")
                else:
                    print("❌ grounding_chunks attribute not found")
                    print("   Available attributes:", [a for a in dir(candidate.grounding_metadata) if not a.startswith('_')])
            else:
                print("❌ grounding_metadata attribute not found in candidate")
                print("   Available candidate attributes:", [a for a in dir(candidate) if not a.startswith('_')])
        else:
            print("❌ candidates not found in response")
            print("   Available response attributes:", [a for a in dir(response) if not a.startswith('_')])

        # Print raw response for manual inspection
        print("\n=== Raw Response (for manual inspection) ===")
        try:
            print(json.dumps(response.__dict__, indent=2, default=str))
        except:
            print("Could not serialize response to JSON. Printing repr:")
            print(repr(response))

    except Exception as e:
        print(f"❌ Error during grounding test: {e}")
        import traceback
        traceback.print_exc()
        print("\nBLOCKER-1 RESOLUTION: FAILED")
        print("Error occurred during validation. Check API key and SDK installation.")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 70)
    print("BLOCKER-1 Resolution: Grounding Metadata Structure Validation")
    print("=" * 70)
    print()
    test_grounding_metadata_extraction()
    print()
    print("=" * 70)
