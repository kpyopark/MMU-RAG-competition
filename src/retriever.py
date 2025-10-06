import os
import json
import base64
import requests
from typing import List, Dict, Any
from loguru import logger
from .chunker import chunk_document


def launch_rerank_server():
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server

    # This is equivalent to running the following command in your terminal
    # python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0
    cmd = """python3 -m sglang.launch_server \
  --model-path BAAI/bge-reranker-v2-m3 \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend triton \
  --is-embedding"""
    server_process, port = launch_server_cmd(cmd)

    wait_for_server(f"http://localhost:{port}", timeout=120)
    return server_process, port


def call_rerank_api(texts: List[str], port: int = 3001) -> List[Dict[str, Any]]:
    url = f"http://127.0.0.1:{port}/v1/rerank"

    payload = {
        # TODO: use qwen 3 model for better performance
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "what is panda?",
        "documents": texts,
    }

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Rerank API error {response.status_code}: {response.text}")
    response_json = response.json()

    for item in response_json:
        print(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
    return response_json


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
                decoded_data = base64.b64decode(doc).decode("utf-8")
                document = json.loads(decoded_data)
                documents.append(document)

            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                logger.error(f"Error processing document: {e}")
                continue

        return documents

    except requests.RequestException as e:
        raise Exception(f"Network error connecting to FineWeb API: {e}")


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        docs = retrieve_fineweb(query, top_k)
        chunks = []
        for doc in docs:
            doc_chunks = chunk_document(doc)
            chunks.extend(doc_chunks)
        texts = [chunk["text"] for chunk in chunks]
        try:
            results = call_rerank_api(texts, port=3001)
            idxs = [x["index"] for x in results]
            reordered = [chunks[i] for i in idxs]
            return reordered[:top_k]
        except Exception as e:
            logger.error(f"Error calling rerank API: {e}")
            return chunks

    except Exception as e:
        logger.error(f"Error in retrieve: {e}")
        return []
