"""
Plain HTTP search API, served from the same process as the MCP endpoint.

Consumers that don't speak MCP (GalaxyAI inside Galaxy, scripts, the help-forum
agent) get the same retrieval over two JSON routes:

    GET  /v1/collections  -- what this snapshot holds, and which model built it
    POST /v1/search       -- {query, collections?, k?} -> {index_version, hits}

Query text is never logged. On a shared service a query like "featureCounts
GTF parse error" says what went wrong for someone.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from retrieval import GalaxyDocsRetriever

MAX_QUERY_CHARS = 2000
MAX_K = 20
DEFAULT_K = 5

PUBLIC_PATHS = frozenset({"/healthz"})


class SearchRequestError(ValueError):
    pass


def parse_search_request(
    body: Any, available: set[str]
) -> tuple[str, list[str], int]:
    """Validate a /v1/search body; returns (query, collection keys, k)."""
    if not isinstance(body, dict):
        raise SearchRequestError("body must be a JSON object")

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise SearchRequestError("`query` must be a non-empty string")
    if len(query) > MAX_QUERY_CHARS:
        raise SearchRequestError(f"`query` is limited to {MAX_QUERY_CHARS} characters")

    k = body.get("k", DEFAULT_K)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise SearchRequestError(f"`k` must be an integer from 1 to {MAX_K}")

    collections = body.get("collections")
    if collections is None:
        keys = sorted(available)
    elif isinstance(collections, list) and collections and all(isinstance(c, str) for c in collections):
        unknown = [c for c in collections if c not in available]
        if unknown:
            raise SearchRequestError(
                f"unknown collection(s) {unknown}; available: {sorted(available)}"
            )
        keys = list(dict.fromkeys(collections))
    else:
        raise SearchRequestError("`collections` must be a non-empty list of collection keys")

    return query.strip(), keys, k


def index_info(retriever: GalaxyDocsRetriever) -> dict[str, Any]:
    manifest = retriever.manifest
    counts = manifest.collections if manifest else {}
    return {
        "index_version": manifest.built_at if manifest else None,
        "model": retriever.model,
        "collections": [
            {
                "key": info.key,
                "title": info.title,
                "description": info.description,
                **counts.get(info.chroma_name, {}),
            }
            for info in retriever.collections
        ],
    }


def register_http_routes(mcp: FastMCP, get_retriever: Callable[[], GalaxyDocsRetriever]) -> None:
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/v1/collections", methods=["GET"])
    async def list_collections(request: Request) -> Response:
        return JSONResponse(index_info(get_retriever()))

    @mcp.custom_route("/v1/search", methods=["POST"])
    async def search(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "body must be valid JSON"}, status_code=400)

        retriever = get_retriever()
        try:
            query, keys, k = parse_search_request(body, {c.key for c in retriever.collections})
        except SearchRequestError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        # Embedding the query and the Chroma lookup both block.
        hits = await run_in_threadpool(
            lambda: [h.to_dict() for key in keys for h in retriever.search(query, key, k=k)]
        )
        manifest = retriever.manifest
        return JSONResponse(
            {"index_version": manifest.built_at if manifest else None, "hits": hits}
        )


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Bearer-token gate over everything but /healthz, including /mcp.

    Only installed when GALAXY_DOCS_API_KEYS is set; an open deployment relies
    on the reverse proxy for rate limiting.
    """

    def __init__(self, app, api_keys: list[str]):
        super().__init__(app)
        self.api_keys = [k.encode() for k in api_keys]

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not any(
            hmac.compare_digest(token.encode(), key) for key in self.api_keys
        ):
            return JSONResponse({"error": "missing or invalid API key"}, status_code=401)
        return await call_next(request)
