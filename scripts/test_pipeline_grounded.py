#!/usr/bin/env python3
"""
Test the simplified pipeline with Gemini grounded generation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import TTD_DR_Pipeline

def test_callback(data):
    """Callback to print pipeline updates."""
    if data.get("intermediate_steps"):
        print("\n" + "="*70)
        print("INTERMEDIATE UPDATE:")
        print("="*70)
        steps = data["intermediate_steps"].split("|||---|||")
        for step in steps[-3:]:
            print(step)

    if data.get("citations"):
        print(f"\nCitations: {len(data['citations'])} sources")

    if data.get("final_report"):
        print("\n" + "="*70)
        print("FINAL REPORT:")
        print("="*70)
        print(data["final_report"])

    if data.get("complete"):
        print("\nPipeline complete!")

def main():
    """Run test of the grounded generation pipeline."""
    print("="*70)
    print("Testing Gemini Grounded Generation Pipeline")
    print("="*70)

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return 1

    pipeline = TTD_DR_Pipeline(callback=test_callback)

    query = "What are the latest developments in AI for 2024?"
    print(f"\nQuery: {query}")
    print(f"Iterations: 1 (simplified)")
    print("\n" + "-"*70 + "\n")

    try:
        pipeline.run(query, max_iterations=1)
        print("\nTest passed!")
        return 0
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
