"""
Gemini API client for LLM, Search, and Reranking operations.

This module implements the unified GeminiClient using the validated google-genai SDK
patterns from BLOCKER-1 resolution (2025-11-10).

Key Validated Patterns:
- Client initialization: genai.Client(api_key=...)
- Request structure: types.Content with types.Part.from_text()
- Grounding metadata path: response.candidates[0].grounding_metadata
- URL field: chunk.web.uri (not .url)
- Title field: chunk.web.title (available)
- Snippet field: NOT available in API response
"""

import os
import time
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from google import genai
from google.genai import types
from loguru import logger


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Standardized search result format."""
    url: str
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Note: snippet field not available in google-genai SDK grounding metadata


@dataclass
class RerankedChunk:
    """Standardized reranked chunk format."""
    chunk_id: int
    text: str
    score: float
    url: Optional[str] = None
    token_count: int = 0


class GeminiClient:
    """
    Unified Gemini API client for LLM, Search, and Reranking operations.

    Features:
    - Single source of truth for all Gemini API interactions
    - Exponential backoff retry (max 3 attempts)
    - Fail-fast with clear error messages
    - No fallback mechanisms

    Validated SDK patterns (BLOCKER-1 resolution 2025-11-10):
    - Client init: genai.Client(api_key=...)
    - Content structure: types.Content(role="user", parts=[types.Part.from_text(text=...)])
    - Grounding metadata: response.candidates[0].grounding_metadata.grounding_chunks
    - URL extraction: chunk.web.uri (not .url)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-flash-latest",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delays: List[float] = [1.0, 2.0, 4.0],
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model identifier (gemini-flash-latest, gemini-1.5-pro, etc.)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient failures
            retry_delays: Exponential backoff delays (seconds)

        Raises:
            ValueError: If API key not found in env or parameter
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set the environment variable:\n"
                "export GEMINI_API_KEY=your_api_key_here"
            )

        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delays = retry_delays

        # Initialize client with new SDK pattern (VALIDATED)
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"Initialized GeminiClient with model: {model}")

    def _retry_with_backoff(self, func, operation_name: str):
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            operation_name: Human-readable operation name for error messages

        Returns:
            Function result

        Raises:
            RuntimeError: After max retries exhausted with clear error message
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                error_str = str(e)

                # Check if retryable error
                is_retryable = any(
                    keyword in error_str
                    for keyword in ["429", "RESOURCE_EXHAUSTED", "timeout", "503"]
                )

                if is_retryable and attempt < self.max_retries - 1:
                    delay = self.retry_delays[attempt] if attempt < len(self.retry_delays) else self.retry_delays[-1]
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries}): {error_str}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue

                # Non-retryable error or max retries reached
                logger.error(
                    f"{operation_name} failed after {attempt + 1} attempts: {error_str}"
                )
                raise RuntimeError(
                    f"Gemini API {operation_name} failed: {error_str}\n"
                    f"Attempts: {attempt + 1}/{self.max_retries}\n"
                    f"Please check:\n"
                    f"1. API key is valid\n"
                    f"2. Rate limits not exceeded\n"
                    f"3. Network connectivity\n"
                    f"4. Gemini API status: https://status.cloud.google.com/"
                ) from e

        # Should never reach here, but just in case
        raise RuntimeError(
            f"Gemini API {operation_name} failed after {self.max_retries} retries"
        ) from last_exception

    def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a world-class research assistant.",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion using Gemini API.

        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature (0-1)
            max_tokens: Max output tokens (overrides default)

        Returns:
            Generated text content

        Raises:
            RuntimeError: If API call fails after retries
        """

        def _generate():
            # Prepend system prompt to user message
            full_prompt = f"{system_prompt}\n\n{prompt}"

            # Prepare request with new SDK pattern (VALIDATED)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)],
                ),
            ]

            # Build generation config
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens or 8192,
            )

            # Execute generation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generation_config,
            )

            # Extract text from response
            content = response.text

            # Log usage metadata
            if hasattr(response, "usage_metadata"):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
                logger.debug(f"LLM usage: {usage}")

            logger.debug(f"Gemini response (200 chars): {content[:200]}")

            return content

        return self._retry_with_backoff(_generate, "LLM completion")

    def complete_with_search(
        self,
        prompt: str,
        system_prompt: str = "You are a world-class research assistant.",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, List[Dict[str, str]]]:
        """
        Generate text completion with Google Search grounding.

        This method enables Gemini's native grounded generation capability,
        which automatically searches the web, retrieves relevant content,
        and generates a response with embedded citations.

        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature (0-1)
            max_tokens: Max output tokens (overrides default)

        Returns:
            Tuple of (response_text, citations)
            - response_text: Generated text content with grounded information
            - citations: List of dicts with 'url' and 'title' keys

        Raises:
            RuntimeError: If API call fails after retries
        """

        def _complete_with_search():
            # Prepend system prompt to user message
            full_prompt = f"{system_prompt}\n\n{prompt}"

            # Prepare request with new SDK pattern
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)],
                ),
            ]

            # Configure with Google Search tool for grounding
            tools = [types.Tool(google_search=types.GoogleSearch())]

            generation_config = types.GenerateContentConfig(
                tools=tools,
                temperature=temperature,
                max_output_tokens=max_tokens or 8192,
            )

            # Execute grounded generation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generation_config,
            )

            # Extract text from response
            response_text = response.text

            # Extract citations from grounding metadata
            citations = []
            if (
                response.candidates
                and response.candidates[0].grounding_metadata
                and hasattr(response.candidates[0].grounding_metadata, "grounding_chunks")
            ):
                chunks = response.candidates[0].grounding_metadata.grounding_chunks
                for chunk in chunks:
                    if hasattr(chunk, "web"):
                        citations.append({
                            "url": chunk.web.uri,
                            "title": chunk.web.title,
                        })

            # Log metadata
            if hasattr(response, "usage_metadata"):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
                logger.debug(f"Grounded generation usage: {usage}")

            logger.info(
                f"Grounded generation complete: {len(response_text)} chars, "
                f"{len(citations)} citations"
            )
            logger.debug(f"Response preview (200 chars): {response_text[:200]}")

            return response_text, citations

        return self._retry_with_backoff(_complete_with_search, "Grounded generation")

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[SearchResult]:
        """
        Search using Gemini Grounding API.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of SearchResult objects with url, text, metadata

        Raises:
            RuntimeError: If API call fails after retries
        """

        def _search():
            # Prepare request with new SDK pattern (VALIDATED)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=query)],
                ),
            ]

            # Configure tools for grounding/search
            tools = [types.Tool(googleSearch=types.GoogleSearch())]

            config = types.GenerateContentConfig(
                tools=tools,
                max_output_tokens=2048,
            )

            # Execute search
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            results = []

            # Extract grounding metadata (VALIDATED PATH)
            # CRITICAL: Metadata is in candidates[0], not at response root level
            if response.candidates and response.candidates[0].grounding_metadata:
                chunks = response.candidates[0].grounding_metadata.grounding_chunks

                for chunk in chunks:
                    # Parse web source (VALIDATED FIELDS)
                    if hasattr(chunk, "web"):
                        # VALIDATED: Use chunk.web.uri (not .url)
                        # VALIDATED: chunk.web.title is available
                        # VALIDATED: snippet field NOT available
                        results.append(
                            SearchResult(
                                url=chunk.web.uri,  # ✅ Correct field name
                                text=chunk.web.title,  # Title only (no snippet)
                                title=chunk.web.title,
                                metadata={
                                    "source": "gemini_search",
                                    "chunk_id": len(results),
                                },
                            )
                        )

            # Limit to top_k
            results = results[:top_k]

            logger.debug(f"Gemini Search returned {len(results)} results for query: {query}")

            if len(results) == 0:
                logger.warning(f"Gemini Search returned no results for query: {query}")

            return results

        return self._retry_with_backoff(_search, "Search")

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> List[RerankedChunk]:
        """
        Rerank chunks using Gemini LLM for semantic relevance scoring.

        Args:
            query: Search query
            chunks: List of chunk dicts with keys: chunk_id, text, url, token_count
            top_k: Number of top-ranked chunks to return

        Returns:
            List of RerankedChunk objects sorted by relevance score (descending)

        Raises:
            RuntimeError: If API call fails after retries for all chunks
        """

        def _score_chunk(chunk: Dict[str, Any]) -> float:
            """Score a single chunk for relevance to query."""
            scoring_prompt = f"""
Rate the relevance of the following text chunk to the query on a scale of 0-10.

Query: {query}

Text Chunk:
{chunk['text'][:1000]}

Provide ONLY a numeric score (0-10) where:
- 0 = Completely irrelevant
- 5 = Somewhat relevant
- 10 = Highly relevant and directly answers the query

Score:"""

            try:
                response = self.complete(
                    prompt=scoring_prompt,
                    system_prompt="You are a relevance scoring system. Provide only numeric scores.",
                    temperature=0.0,  # Deterministic scoring
                    max_tokens=10,
                )

                # Parse numeric score from response
                score_text = response.strip()
                # Extract first number found
                match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                if match:
                    score = float(match.group(1))
                    # Normalize to 0-1 range
                    return min(max(score / 10.0, 0.0), 1.0)
                else:
                    logger.warning(f"Could not parse score from response: {score_text}")
                    return 0.0

            except Exception as e:
                logger.warning(f"Failed to score chunk {chunk.get('chunk_id', '?')}: {e}")
                return 0.0  # Assign lowest score on error

        # Score all chunks
        scored_chunks = []
        for chunk in chunks:
            score = _score_chunk(chunk)
            scored_chunks.append(
                RerankedChunk(
                    chunk_id=chunk.get("chunk_id", 0),
                    text=chunk.get("text", ""),
                    score=score,
                    url=chunk.get("url"),
                    token_count=chunk.get("token_count", 0),
                )
            )

        # Sort by score (descending) and return top_k
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k]
