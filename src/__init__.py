# FILE: Text-to-Text/api_server.py
import os
import sys
import json
from dotenv import load_dotenv

# Add src directory to path to import pipeline modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from fastapi import FastAPI, Request
from typing import Dict, Any
from src.pipeline import run_rag_dynamic, run_rag_static
from loguru import logger

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="MMU-RAG TTD-DR Implementation",
    description="An API server for the Test-Time Diffusion Deep Researcher.",
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate(request: Request):
    """
    Static evaluation endpoint.
    Receives a query and iid, runs the full RAG pipeline, and returns the final response.
    """
    data = await request.json()
    query = data.get("query")
    iid = data.get("iid")

    if not query or not iid:
        return {"error": "Missing 'query' or 'iid' in request."}, 400
    logger.debug(f"query: {query}")
    generated_response = run_rag_static(query)

    return {
        "query_id": iid,
        "generated_response": generated_response,
    }


@app.post("/run")
async def run_endpoint(request: Request):
    """
    Dynamic evaluation endpoint.
    Receives a question, runs the RAG pipeline, and returns the final response.
    """
    data = await request.json()
    question = data.get("question")

    if not question:
        return {"error": "Missing 'question' in request."}, 400

    # Run the pipeline and return the final result
    generated_response = run_rag_static(question)

    return {
        "question": question,
        "generated_response": generated_response,
    }


if __name__ == "__main__":
    import uvicorn

    # To run: python Text-to-Text/api_server.py
    # The competition might specify a different port, which can be changed here.
    port = int(os.getenv("PORT", 5053))
    uvicorn.run(app, host="0.0.0.0", port=port)
