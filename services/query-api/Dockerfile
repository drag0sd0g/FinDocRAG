FROM python:3.12-slim AS base

WORKDIR /app

# System deps for psycopg2 + curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Pre-download the embedding model so it's baked into the image (no slow
# first-boot download). trust_remote_code loads nomic's custom nomic-bert.
ARG EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', trust_remote_code=True)"

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]