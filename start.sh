#!/bin/bash
set -e

echo "Starting SGLang..."
uv run python3 -m sglang.launch_server \
  --model-path BAAI/bge-reranker-v2-m3 \
  --host 0.0.0.0 \
  --port 3001 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend triton \
  --is-embedding &
SGLANG_PID=$!


sleep 20

if ps -p $SGLANG_PID > /dev/null; then
    echo "✅ SGLang started successfully (PID: $SGLANG_PID)"
else
    echo "⚠️  SGLang failed to start. Continuing with FastAPI anyway..."
fi


echo "Starting SGLang llm..."
uv run python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 3002 & 
SGLANG_LLM_PID=$!
sleep 20
if ps -p $SGLANG_LLM_PID > /dev/null; then
    echo "✅ SGLang llm started successfully (PID: $SGLANG_LLM_PID)"
else
    echo "⚠️  SGLang llm failed to start. Continuing with FastAPI anyway..."
fi

echo "Starting FastAPI..."
fastapi run --host 0.0.0.0 --port 5053 --reload src
