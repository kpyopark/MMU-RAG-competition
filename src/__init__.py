import asyncio
import contextlib
import json
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.pipeline import run_rag_dynamic, run_rag_static

load_dotenv()

app = FastAPI(
    title="MMU-RAG TTD-DR Implementation",
    description="API server exposing static and streaming RAG endpoints.",
)


class EvaluateRequest(BaseModel):
    query: str
    iid: str


class EvaluateResponse(BaseModel):
    query_id: str
    generated_response: str


class RunRequest(BaseModel):
    question: str


@app.get("/health")
def health_check() -> dict[str, str]:
    """Lightweight health check endpoint."""
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(payload: EvaluateRequest) -> EvaluateResponse:
    """Static evaluation endpoint returning a single JSON response."""
    try:
        generated_response = await asyncio.to_thread(run_rag_static, payload.query)
    except Exception as e:
        logger.exception(f"Static evaluation failed {e}")
        raise HTTPException(status_code=500, detail="Static evaluation failed")

    return EvaluateResponse(query_id=payload.iid, generated_response=generated_response)


@app.post("/run")
async def run_endpoint(payload: RunRequest) -> EventSourceResponse:
    """Streaming endpoint that emits SSE updates for the research pipeline."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def pipeline_callback(update: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, update)

    def run_pipeline() -> None:
        try:
            run_rag_dynamic(payload.question, pipeline_callback)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive: upstream services may fail
            logger.exception("Dynamic pipeline failed")
            error_payload = {"error": str(exc), "complete": True}
            loop.call_soon_threadsafe(queue.put_nowait, error_payload)

    pipeline_task = asyncio.create_task(asyncio.to_thread(run_pipeline))

    async def event_publisher() -> AsyncGenerator[Dict[str, str], None]:
        try:
            while True:
                update = await queue.get()
                payload = json.dumps(update)
                yield {"data": payload}
                if update.get("complete") is True:
                    break
        except asyncio.CancelledError:
            pipeline_task.cancel()
            raise
        finally:
            with contextlib.suppress(Exception):
                await pipeline_task

    return EventSourceResponse(event_publisher(), media_type="text/event-stream")


__all__ = ["app"]
