FROM vllm/vllm-openai:latest

WORKDIR /app

COPY . .

RUN pip install .

EXPOSE 5053

CMD ["bash", "start.sh"]