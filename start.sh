#!/bin/bash
set -e
echo "Starting llm..."
nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16k --gpu-memory-utilization 0.75 --enforce-eager \
    --host 0.0.0.0 \
    --port 3002 & 
LOCAL_LLM_PID=$!
# had to put sleep due to KV cache overload and OOM
sleep 200
if ps -p $LOCAL_LLM_PID > /dev/null; then
    echo "✅ LOCAL llm started successfully (PID: $LOCAL_LLM_PID)"
else
    echo "⚠️  LOCAL llm failed to start. Continuing with FastAPI anyway..."
fi


echo "Starting FastAPI..."
fastapi run --host 0.0.0.0 --port 5053 src
