import os
import json
import base64
import requests
from typing import List, Dict, Any


def retrieve_fineweb(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve documents from FineWeb Search API.

    Args:
        query: Search query string
        top_k: Number of documents to return

    Returns:
        List of document dictionaries with 'url' and 'content' keys
    """
    api_key = os.getenv("FINEWEB_API_KEY")
    if not api_key:
        raise ValueError("FINEWEB_API_KEY environment variable not set")

    base_url = "https://clueweb22.us/fineweb/search"

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    params = {"query": query, "k": top_k}

    try:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(
                f"FineWeb API error {response.status_code}: {response.text}"
            )

        result = response.json()

        documents = []
        for doc in result.get("results", []):
            try:
                # Decode base64 JSON document
                decoded_data = base64.b64decode(doc).decode("utf-8")
                document = json.loads(decoded_data)

                # Extract relevant information
                doc_info = {
                    "url": document.get("url", ""),
                    "content": document.get("text", "")[:1000],  # Limit content length
                    "title": document.get("title", ""),
                }
                documents.append(doc_info)

            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                print(f"Error processing document: {e}")
                continue

        return documents

    except requests.RequestException as e:
        raise Exception(f"Network error connecting to FineWeb API: {e}")


def retrieve(
    query: str, index_path: str | None = None, top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant documents for a given query using FineWeb Search API.

    Args:
        query: User query to search for
        index_path: Not used for FineWeb search (kept for compatibility)
        top_k: Number of top documents to retrieve

    Returns:
        List of document dictionaries with 'url' and 'content' keys
    """
    # TODO: Implement retrieval logic
    # - Load the saved FAISS index
    # - Generate query embedding using same model as indexing
    # - Search index for top_k most similar chunks
    # - Return retrieved text chunks for generation
    # cohere rerank, graph retreival
    try:
        return retrieve_fineweb(query, top_k)
    except Exception as e:
        print(f"Error in retrieve: {e}")
        # Return empty list as fallback
        return []
