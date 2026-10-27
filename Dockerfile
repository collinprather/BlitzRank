FROM python:3.11-slim-trixie

# uv for fast, cache-friendly installs
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

# JDK 21 for Pyserini (pyjnius), gcc for pytrec-eval's sdist
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PYTHONUNBUFFERED=1 \
    UV_TORCH_BACKEND=auto

WORKDIR /app
COPY pyproject.toml README.md ./
COPY blitzrank ./blitzrank
RUN uv pip install --system -e ".[all]" # torch installs CPU-only into linux via auto backend

COPY configs ./configs
COPY scripts ./scripts

# Mount for resumable downloads/results:
#   ./.cache/datasets -> /app/.cache/datasets        (BM25 hit cache)
#   ~/.cache/pyserini -> /root/.cache/pyserini       (Lucene index, ~2.6 GB)
#   ./blitzrank_outputs -> /app/blitzrank_outputs    (results + summary.jsonl)
CMD ["bash"]
