import os
import json
import base64
import requests
from typing import List, Dict, Any
from loguru import logger
from .chunker import chunk_document

# try:
from vllm import LLM

model = LLM(
    model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    task="score",
    # TODO: handle proper chunking
    max_model_len=1024,
    gpu_memory_utilization=0.1,
)
# except Exception as e:
#     print(f"Couldn't creat vllm ranker {e}")


def call_rerank_api(texts: List[str], port: int = 3001) -> List[Dict[str, Any]]:
    url = f"http://127.0.0.1:{port}/v1/rerank"

    payload = {
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
            # results = call_rerank_api(texts, port=3001)
            # idxs = [x["index"] for x in results]
            outputs = model.score(query, texts)
            scores = [output.outputs.score for output in outputs]
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)
            reordered = [chunks[i] for i in idxs]
            return reordered[:top_k]
        except Exception as e:
            logger.error(f"Error calling rerank API: {e}")
            return chunks

    except Exception as e:
        logger.error(f"Error in retrieve: {e}")
        return []
