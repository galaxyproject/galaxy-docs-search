#!/usr/bin/env python
"""
Write data/embeddings/manifest.json describing the current Chroma snapshot.

Reads embedder config from EMBEDDINGS_MODEL / EMBEDDINGS_BASE_URL (env or .env)
and queries the on-disk Chroma store for per-collection counts. The retrieval
module (and the galaxy-docs-mcp server on top of it) reads this file at startup
to refuse a model/data mismatch and to expose collection stats to consumers.

Run after `make update-all`, or invoke directly:
    uv run python -m scripts.write_manifest
    uv run python -m scripts.write_manifest --embeddings-dir ./data/embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDINGS_DIR = Path("data/embeddings")
MANIFEST_FILENAME = "manifest.json"


DOCUMENT_KEY_FALLBACKS = [
    ("source", "faq_index"),  # FAQ pipeline glues every Q&A under one source file
    ("source",),
    ("path",),
    ("url",),
]


def count_documents(metadatas: list[dict | None]) -> int:
    """Count distinct documents using the most-specific available key set."""
    for keys in DOCUMENT_KEY_FALLBACKS:
        distinct: set[tuple] = set()
        for meta in metadatas:
            if not meta:
                continue
            values = tuple(meta.get(k) for k in keys)
            if all(v is not None for v in values):
                distinct.add(values)
        if distinct:
            return len(distinct)
    return 0


def collect_stats(client: chromadb.api.ClientAPI) -> dict[str, dict[str, int]]:
    """Return {collection_name: {chunks, documents}} for every collection."""
    stats: dict[str, dict[str, int]] = {}
    for col in client.list_collections():
        # chromadb >=0.6 returns names; <0.6 returns Collection objects.
        name = col if isinstance(col, str) else col.name
        collection = client.get_collection(name)
        chunks = collection.count()
        documents = 0
        if chunks:
            page = collection.get(include=["metadatas"], limit=chunks)
            documents = count_documents(page.get("metadatas") or [])
        stats[name] = {"chunks": chunks, "documents": documents}
    return stats


def build_manifest(embeddings_dir: Path) -> dict:
    client = chromadb.PersistentClient(path=str(embeddings_dir))
    return {
        "model": os.getenv("EMBEDDINGS_MODEL", ""),
        "embeddings_base_url": os.getenv("EMBEDDINGS_BASE_URL", ""),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collections": collect_stats(client),
    }


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Path to the Chroma persist dir (default: {DEFAULT_EMBEDDINGS_DIR})",
    )
    args = parser.parse_args()

    embeddings_dir: Path = args.embeddings_dir
    if not embeddings_dir.exists():
        logger.error("Embeddings dir not found: %s", embeddings_dir)
        return 1

    manifest = build_manifest(embeddings_dir)
    out = embeddings_dir / MANIFEST_FILENAME
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote %s", out)
    for name, info in manifest["collections"].items():
        logger.info("  %s: %d chunks across %d documents", name, info["chunks"], info["documents"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
