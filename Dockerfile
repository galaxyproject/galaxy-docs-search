# Hosted galaxy-docs search service: MCP at /mcp, JSON at /v1/*, /healthz.
# The Chroma snapshot is mounted at /data/embeddings; the embedder is reached
# over EMBEDDINGS_BASE_URL and must match the snapshot's manifest.json.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra mcp --no-install-project

COPY pipelines ./pipelines
COPY retrieval ./retrieval
COPY mcp_server ./mcp_server
RUN uv sync --locked --no-dev --extra mcp

ENV GALAXY_DOCS_EMBEDDINGS_DIR=/data/embeddings \
    GALAXY_DOCS_TRANSPORT=streamable-http \
    GALAXY_DOCS_HOST=0.0.0.0 \
    GALAXY_DOCS_PORT=8000
EXPOSE 8000

RUN useradd --system --no-create-home docs
USER docs

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["/app/.venv/bin/galaxy-docs-mcp"]
