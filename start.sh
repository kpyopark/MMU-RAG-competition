#!/bin/bash
set -e

echo "Starting llm..."
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16k \
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
