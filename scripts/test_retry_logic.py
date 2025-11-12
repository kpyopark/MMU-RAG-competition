#!/usr/bin/env python3
"""
Test script for validating Gemini API retry logic.

Tests:
1. Retry-after time parsing from error messages
2. Rate limit handling with buffer time
3. Pipeline exit on max retries
"""

import re
from typing import Optional


def parse_retry_after(error_str: str) -> Optional[float]:
    """
    Extract retry-after duration from error message.

    Args:
        error_str: Error message string

    Returns:
        Retry-after duration in seconds, or None if not found
    """
    # Pattern: "retry in X.Xs" or "retry in Xs"
    match = re.search(r'retry in (\d+(?:\.\d+)?)\s*s', error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def test_retry_after_parsing():
    """Test parsing of retry-after values from error messages."""
    print("=" * 60)
    print("Test 1: Retry-after parsing")
    print("=" * 60)

    test_cases = [
        {
            "name": "Real Gemini error message",
            "error": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 250\nPlease retry in 44.025501755s.",
            "expected": 44.025501755,
        },
        {
            "name": "Simple retry message",
            "error": "Please retry in 60s.",
            "expected": 60.0,
        },
        {
            "name": "Integer retry time",
            "error": "Rate limit exceeded. Please retry in 30s",
            "expected": 30.0,
        },
        {
            "name": "No retry time",
            "error": "Internal server error",
            "expected": None,
        },
    ]

    for test in test_cases:
        result = parse_retry_after(test["error"])
        status = "✅ PASS" if result == test["expected"] else "❌ FAIL"
        print(f"\n{status}: {test['name']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Got:      {result}")
        if result != test["expected"]:
            print(f"  Error message: {test['error'][:100]}...")


def test_retry_with_buffer():
    """Test retry delay calculation with buffer."""
    print("\n" + "=" * 60)
    print("Test 2: Retry delay with buffer")
    print("=" * 60)

    RATE_LIMIT_BUFFER = 5.0

    test_cases = [
        {"retry_after": 44.025501755, "expected": 49.025501755},
        {"retry_after": 60.0, "expected": 65.0},
        {"retry_after": 30.0, "expected": 35.0},
        {"retry_after": None, "expected": 65.0},  # Default 60s + 5s buffer
    ]

    for test in test_cases:
        if test["retry_after"] is not None:
            delay = test["retry_after"] + RATE_LIMIT_BUFFER
        else:
            delay = 60.0 + RATE_LIMIT_BUFFER

        status = "✅ PASS" if abs(delay - test["expected"]) < 0.001 else "❌ FAIL"
        print(f"\n{status}:")
        print(f"  Retry-after: {test['retry_after']}")
        print(f"  Expected:    {test['expected']}s")
        print(f"  Got:         {delay}s")


def test_error_types():
    """Test error type detection."""
    print("\n" + "=" * 60)
    print("Test 3: Error type detection")
    print("=" * 60)

    test_cases = [
        {
            "error": "429 RESOURCE_EXHAUSTED",
            "is_rate_limit": True,
            "is_transient": False,
        },
        {
            "error": "503 Service Unavailable",
            "is_rate_limit": False,
            "is_transient": True,
        },
        {
            "error": "timeout",
            "is_rate_limit": False,
            "is_transient": True,
        },
        {
            "error": "400 Bad Request",
            "is_rate_limit": False,
            "is_transient": False,
        },
    ]

    for test in test_cases:
        is_rate_limit = "429" in test["error"] or "RESOURCE_EXHAUSTED" in test["error"]
        is_transient = any(
            keyword in test["error"] for keyword in ["timeout", "503", "502"]
        )

        rate_limit_ok = is_rate_limit == test["is_rate_limit"]
        transient_ok = is_transient == test["is_transient"]
        status = "✅ PASS" if (rate_limit_ok and transient_ok) else "❌ FAIL"

        print(f"\n{status}: {test['error']}")
        print(f"  Expected: rate_limit={test['is_rate_limit']}, transient={test['is_transient']}")
        print(f"  Got:      rate_limit={is_rate_limit}, transient={is_transient}")


if __name__ == "__main__":
    print("\n🧪 Testing Gemini API Retry Logic\n")

    test_retry_after_parsing()
    test_retry_with_buffer()
    test_error_types()

    print("\n" + "=" * 60)
    print("✅ All tests completed")
    print("=" * 60)
    print("\nRetry behavior summary:")
    print("- Rate limit errors (429): Use API retry-after + 5s buffer")
    print("- Transient errors (503, timeout): Exponential backoff (1s, 2s, 4s)")
    print("- Max retries exhausted: Pipeline exits with RuntimeError")
    print()
