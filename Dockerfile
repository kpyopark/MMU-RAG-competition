FROM vllm/vllm-openai:latest

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install .

COPY . .

EXPOSE 5053

ENTRYPOINT []
CMD ["bash", "start.sh"]