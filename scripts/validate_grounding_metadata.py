#!/usr/bin/env python3
"""
Validate Gemini grounded generation for Pfizer-Metsera news.

This script directly tests whether Gemini with Google Search can find
the real news: "Pfizer wins $10 billion bidding war for Metsera as Novo Nordisk exits"
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gemini_client import GeminiClient


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def test_direct_search(query: str):
    """Test direct search with Gemini grounded generation."""
    print_section("DIRECT SEARCH TEST")
    print(f"Query: {query}\n")

    try:
        client = GeminiClient()

        response_text, citations = client.complete_with_search(
            prompt=query,
            system_prompt="You are a factual news verification assistant. Search for and verify this news, providing specific sources.",
            temperature=0.3,  # Lower temperature for factual queries
        )

        print_section("RESPONSE TEXT")
        print(response_text)

        print_section("CITATIONS FOUND")
        if citations:
            print(f"Total citations: {len(citations)}\n")
            for idx, citation in enumerate(citations, 1):
                print(f"{idx}. {citation.get('title', 'No title')}")
                print(f"   URL: {citation.get('url', 'No URL')}\n")
        else:
            print("⚠️  WARNING: No citations found!")
            print("This indicates Gemini did not use Google Search or found no results.")

        return response_text, citations

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_context_aware_search(query: str):
    """Test search with context similar to pipeline's usage."""
    print_section("CONTEXT-AWARE SEARCH TEST (Pipeline Simulation)")

    context_prompt = f"""You are researching to answer this query: {query}

Provide a comprehensive, well-researched answer based on current web information.
Focus on specific facts, data, and details from authoritative sources.

Include:
1. Verification of whether this news is real
2. Key facts about the deal (amount, companies involved)
3. Date of the announcement
4. Sources confirming this information"""

    print(f"Query: {query}")
    print(f"\nContext Prompt:\n{context_prompt}\n")

    try:
        client = GeminiClient()

        response_text, citations = client.complete_with_search(
            prompt=query,
            system_prompt=context_prompt,
            temperature=0.3,
        )

        print_section("RESPONSE TEXT")
        print(response_text)

        print_section("CITATIONS FOUND")
        if citations:
            print(f"Total citations: {len(citations)}\n")
            for idx, citation in enumerate(citations, 1):
                print(f"{idx}. {citation.get('title', 'No title')}")
                print(f"   URL: {citation.get('url', 'No URL')}\n")
        else:
            print("⚠️  WARNING: No citations found!")

        return response_text, citations

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_variations():
    """Test multiple query variations to identify search issues."""
    print_section("QUERY VARIATION TESTS")

    queries = [
        "Pfizer wins $10 billion bidding war for Metsera as Novo Nordisk exits",
        "Pfizer Metsera acquisition 10 billion",
        "Pfizer Metsera deal Novo Nordisk",
        "Metsera obesity drug company acquisition",
    ]

    client = GeminiClient()
    results = []

    for query in queries:
        print(f"\n{'─' * 80}")
        print(f"Testing: {query}")
        print('─' * 80)

        try:
            response_text, citations = client.complete_with_search(
                prompt=query,
                system_prompt="Verify this news and provide sources.",
                temperature=0.3,
            )

            citation_count = len(citations) if citations else 0
            print(f"✓ Citations found: {citation_count}")
            print(f"  Response preview: {response_text[:150]}...")

            results.append({
                "query": query,
                "citation_count": citation_count,
                "citations": citations,
                "response_preview": response_text[:200],
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "query": query,
                "citation_count": 0,
                "error": str(e),
            })

    print_section("VARIATION TEST SUMMARY")
    for result in results:
        print(f"Query: {result['query']}")
        print(f"Citations: {result.get('citation_count', 0)}")
        if result.get('error'):
            print(f"Error: {result['error']}")
        print()


def analyze_root_cause(response: str, citations: list):
    """Analyze potential root causes for missing search results."""
    print_section("ROOT CAUSE ANALYSIS")

    issues = []

    # Check 1: No citations at all
    if not citations or len(citations) == 0:
        issues.append({
            "issue": "No citations found",
            "severity": "CRITICAL",
            "possible_causes": [
                "Google Search tool not being invoked by Gemini",
                "News too recent for Google indexing",
                "Query not matching indexed content",
                "API rate limiting or quota issues",
            ],
        })

    # Check 2: Response indicates uncertainty
    uncertainty_markers = [
        "cannot verify",
        "no information",
        "unable to confirm",
        "I don't have",
        "not available",
    ]

    if any(marker in response.lower() for marker in uncertainty_markers):
        issues.append({
            "issue": "Response indicates uncertainty or lack of information",
            "severity": "HIGH",
            "possible_causes": [
                "Gemini not finding relevant search results",
                "Search results not matching the query",
                "News not indexed or recent",
            ],
        })

    # Check 3: Few citations (less than 3)
    if citations and len(citations) < 3:
        issues.append({
            "issue": f"Low citation count ({len(citations)})",
            "severity": "MEDIUM",
            "possible_causes": [
                "Limited search results available",
                "Query too specific or narrow",
                "Recent news with limited coverage",
            ],
        })

    if not issues:
        print("✅ No obvious issues detected!")
        print(f"   Citations found: {len(citations)}")
        print("   Response appears factual and grounded.")
    else:
        print(f"⚠️  {len(issues)} potential issue(s) detected:\n")
        for idx, issue in enumerate(issues, 1):
            print(f"{idx}. [{issue['severity']}] {issue['issue']}")
            print("   Possible causes:")
            for cause in issue['possible_causes']:
                print(f"   - {cause}")
            print()


def main():
    """Run all validation tests."""
    print("=" * 80)
    print(" GEMINI GROUNDED GENERATION VALIDATION")
    print(" Testing: Pfizer-Metsera News Search")
    print("=" * 80)

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n❌ ERROR: GEMINI_API_KEY not set")
        print("Please set the API key in your environment or .env file")
        return 1

    print("\n✓ GEMINI_API_KEY found")

    # Primary query
    primary_query = "Pfizer wins $10 billion bidding war for Metsera as Novo Nordisk exits"

    # Test 1: Direct search
    print("\n\n")
    response1, citations1 = test_direct_search(primary_query)

    # Test 2: Context-aware search (simulating pipeline behavior)
    print("\n\n")
    response2, citations2 = test_context_aware_search(primary_query)

    # Test 3: Query variations
    print("\n\n")
    test_variations()

    # Root cause analysis
    if response2 is not None:
        print("\n\n")
        analyze_root_cause(response2, citations2)

    print_section("VALIDATION COMPLETE")
    print("\nNext steps:")
    print("1. Review the citations found (if any)")
    print("2. Check if Gemini's response matches reality")
    print("3. Investigate root causes identified above")
    print("4. Consider adjusting pipeline prompts or search strategy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
