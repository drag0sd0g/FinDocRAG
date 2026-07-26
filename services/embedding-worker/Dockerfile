FROM python:3.12-slim AS base

WORKDIR /app

# System deps for confluent-kafka (librdkafka), psycopg2, + curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc librdkafka-dev libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Pre-download the embedding model so it's baked into the image (no slow
# first-boot download). trust_remote_code loads nomic's custom nomic-bert.
ARG EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', trust_remote_code=True)"

EXPOSE 8002

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]