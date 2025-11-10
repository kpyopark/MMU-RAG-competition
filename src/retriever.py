"""
Document retrieval and reranking using Gemini API.

This module replaces FineWeb API with Gemini Search (Grounding API)
and vLLM reranker with GeminiClient.rerank_chunks().
"""

import os
from typing import List, Dict, Any
from loguru import logger
from .chunker import chunk_document
from .gemini_client import GeminiClient

# Initialize Gemini client for search and reranking
gemini_client = GeminiClient(
    api_key=os.getenv("GEMINI_API_KEY"),
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
)


def retrieve_gemini_search(query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve documents using Gemini Search (Grounding API).

    This replaces the FineWeb API with Gemini's built-in search functionality.

    Args:
        query: Search query string
        top_k: Number of search results to retrieve

    Returns:
        List of document dictionaries with 'url' and 'content' keys
    """
    try:
        # Use GeminiClient.search() method (validated BLOCKER-1 resolution)
        search_results = gemini_client.search(query, top_k=top_k)

        # Convert SearchResult objects to document format
        documents = []
        for result in search_results:
            # Create document dict compatible with existing pipeline
            # IMPORTANT: chunker.py expects 'text' key, not 'content'
            documents.append({
                "url": result.url,
                "text": result.text,  # Title from grounding metadata (chunker expects 'text')
                "title": result.title,
                "metadata": result.metadata,
            })

        logger.info(f"Gemini Search retrieved {len(documents)} documents for query: {query[:50]}...")
        return documents

    except Exception as e:
        logger.error(f"Error in Gemini search: {e}")
        raise


def retrieve_with_grounded_generation(
    query: str,
    context_prompt: str = "",
) -> tuple[str, List[Dict[str, str]]]:
    """
    Retrieve and synthesize answer using Gemini's grounded generation.

    This replaces the entire Search → Chunk → Rerank → Synthesize pipeline
    with a single Gemini API call that:
    1. Automatically searches Google for relevant information
    2. Generates a comprehensive answer grounded in web content
    3. Returns citations automatically

    Args:
        query: Search/research query
        context_prompt: Optional context to guide the search and synthesis

    Returns:
        Tuple of (synthesized_answer, citations)
        - synthesized_answer: Comprehensive answer grounded in web sources
        - citations: List of dicts with 'url' and 'title' keys

    Raises:
        RuntimeError: If API call fails
    """
    try:
        # Prepare prompt with context if provided
        if context_prompt:
            full_prompt = f"{context_prompt}\n\n{query}"
        else:
            full_prompt = query

        logger.info(f"Grounded generation query: {query[:100]}...")

        # Use Gemini's grounded generation
        answer, citations = gemini_client.complete_with_search(
            prompt=full_prompt,
            system_prompt="You are a world-class research analyst. Provide comprehensive, well-researched answers with specific facts and details from web sources.",
            temperature=0.7,
            max_tokens=8192,
        )

        logger.info(
            f"Grounded generation complete: {len(answer)} chars, "
            f"{len(citations)} citations"
        )

        return answer, citations

    except Exception as e:
        logger.error(f"Error in grounded generation: {e}")
        raise


def retrieve(query: str, top_k: int = 5, search_top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Main retrieval function using Gemini Search and reranking.

    DEPRECATED: This function uses the old Search → Chunk → Rerank pipeline.
    Use retrieve_with_grounded_generation() for better performance and accuracy.

    Pipeline:
    1. Retrieve documents using Gemini Search (Grounding API)
    2. Chunk documents for processing
    3. Rerank chunks using GeminiClient.rerank_chunks()
    4. Return top_k best chunks

    Args:
        query: Search query string
        top_k: Number of top chunks to return after reranking
        search_top_k: Number of documents to retrieve from search

    Returns:
        List of reranked chunk dictionaries sorted by relevance
    """
    try:
        logger.debug(f"Gemini Search query: {query[:100]}...")

        # Step 1: Retrieve documents using Gemini Search
        docs = retrieve_gemini_search(query, top_k=search_top_k)
        logger.debug(f"Retrieved {len(docs)} documents from Gemini Search")

        # Step 2: Chunk documents
        chunks = []
        for doc in docs:
            doc_chunks = chunk_document(doc)
            chunks.extend(doc_chunks)

        logger.debug(f"Created {len(chunks)} chunks from documents")

        if not chunks:
            logger.warning("No chunks created from retrieved documents")
            return []

        # Step 3: Rerank chunks using Gemini LLM
        try:
            reranked_chunks = gemini_client.rerank_chunks(
                query=query,
                chunks=chunks,
                top_k=top_k,
            )

            # Convert RerankedChunk objects back to dict format for pipeline
            reordered = []
            for ranked_chunk in reranked_chunks:
                # Find original chunk by chunk_id
                original_chunk = chunks[ranked_chunk.chunk_id] if ranked_chunk.chunk_id < len(chunks) else None
                if original_chunk:
                    # Add score to chunk
                    chunk_dict = original_chunk.copy()
                    chunk_dict["rerank_score"] = ranked_chunk.score
                    reordered.append(chunk_dict)

            logger.debug(f"Reranked {len(reordered)} chunks, top score: {reordered[0]['rerank_score']:.3f if reordered else 0}")
            return reordered

        except Exception as e:
            logger.error(f"Error in Gemini reranking: {e}, returning unranked chunks")
            return chunks[:top_k]

    except Exception as e:
        logger.error(f"Error in retrieve: {e}")
        return []
