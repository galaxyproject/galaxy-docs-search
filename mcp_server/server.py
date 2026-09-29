"""
FastMCP app for galaxy-docs-mcp.

Exposes a single tool, `search_galaxy_docs`, that semantic-searches across
four Galaxy documentation collections (FAQs, GTN tutorials, Discourse forum,
IWC workflows). The collection enum and per-collection guidance live in
retrieval.collections so this file stays focused on MCP plumbing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastmcp import FastMCP
from starlette.middleware import Middleware

from retrieval import GalaxyDocsRetriever
from retrieval.collections import COLLECTIONS

from mcp_server.http_api import ApiKeyMiddleware, register_http_routes

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDINGS_DIR = Path("./data/embeddings")

CollectionArg = Literal["faqs", "tutorials", "discourse", "workflows", "tools", "all"]


def _build_tool_description() -> str:
    """Compose the tool docstring from per-collection metadata.

    The LLM picks `collection` based on what it reads here, so the
    description stays close to the actual content of each corpus.
    """
    lines = [
        "Semantic search over Galaxy Project documentation.",
        "",
        "Pick `collection` based on what kind of answer you need:",
        "",
    ]
    for info in COLLECTIONS.values():
        lines.append(f"- `{info.key}`: {info.description}")
    lines.extend(
        [
            "- `all`: Search every collection in this snapshot; returns top-k from each, "
            "flattened into one list (each Hit carries its `collection` field). "
            "No cross-collection reranking -- raw scores aren't comparable across "
            "corpora, and the caller can filter.",
            "",
            "Args:",
            "    query: Natural-language search query.",
            "    collection: Which corpus to search. Defaults to 'all'.",
            "    k: Number of results per collection (default 5).",
            "",
            "Returns: list of Hit dicts with `text`, `score`, `collection`, "
            "`source_url` (when available), and `metadata`.",
        ]
    )
    return "\n".join(lines)


def build_server(
    embeddings_dir: Path | None = None,
    retriever: GalaxyDocsRetriever | None = None,
) -> FastMCP:
    """Construct the FastMCP app.

    The retriever can be injected for tests; otherwise it's built lazily on
    first tool call so the server starts even if the embedder is briefly
    unreachable.
    """
    load_dotenv()
    embeddings_dir = embeddings_dir or Path(
        os.getenv("GALAXY_DOCS_EMBEDDINGS_DIR", str(DEFAULT_EMBEDDINGS_DIR))
    )

    mcp: FastMCP = FastMCP("galaxy-docs")

    _retriever_box: dict[str, GalaxyDocsRetriever] = {}
    if retriever is not None:
        _retriever_box["instance"] = retriever

    def _get_retriever() -> GalaxyDocsRetriever:
        if "instance" not in _retriever_box:
            _retriever_box["instance"] = GalaxyDocsRetriever(embeddings_dir=embeddings_dir)
        return _retriever_box["instance"]

    @mcp.tool(description=_build_tool_description())
    def search_galaxy_docs(
        query: str,
        collection: CollectionArg = "all",
        k: int = 5,
    ) -> list[dict[str, Any]]:
        r = _get_retriever()
        if collection == "all":
            hits = r.search_all(query, per_collection_k=k)
        elif collection not in {c.key for c in r.collections}:
            raise ValueError(f"Collection {collection!r} isn't in this snapshot.")
        else:
            hits = r.search(query, collection, k=k)
        return [h.to_dict() for h in hits]

    register_http_routes(mcp, _get_retriever)

    return mcp


def http_middleware() -> list[Middleware]:
    keys = [k.strip() for k in os.getenv("GALAXY_DOCS_API_KEYS", "").split(",") if k.strip()]
    return [Middleware(ApiKeyMiddleware, api_keys=keys)] if keys else []


def run_http(mcp: FastMCP, transport: str, host: str | None, port: int | None) -> None:
    """Run the server on streamable-http or sse, alongside the /v1 JSON routes."""
    kwargs: dict[str, Any] = {"transport": transport, "middleware": http_middleware()}
    if host:
        kwargs["host"] = host
    if port:
        kwargs["port"] = port
    mcp.run(**kwargs)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: pick transport, build the server, run it."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run galaxy-docs-mcp.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=os.getenv("GALAXY_DOCS_TRANSPORT", "stdio"),
        help="MCP transport (default: stdio).",
    )
    parser.add_argument("--host", default=os.getenv("GALAXY_DOCS_HOST"))
    parser.add_argument("--port", type=int, default=_int_env("GALAXY_DOCS_PORT"))
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Override GALAXY_DOCS_EMBEDDINGS_DIR.",
    )
    args = parser.parse_args(argv)

    mcp = build_server(embeddings_dir=args.embeddings_dir)

    if args.transport in {"streamable-http", "sse"}:
        run_http(mcp, args.transport, args.host, args.port)
    else:
        mcp.run()


def _int_env(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


if __name__ == "__main__":
    main()
