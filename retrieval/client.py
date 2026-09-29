"""
GalaxyDocsRetriever -- semantic search over the four chatgxy-materials
Chroma collections. Lazy-loads each collection on first use.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from retrieval import manifest as manifest_mod
from retrieval.collections import (
    COLLECTIONS,
    CollectionInfo,
    CollectionKey,
    available_collections,
)

logger = logging.getLogger(__name__)

@dataclass
class Hit:
    text: str
    score: float
    collection: CollectionKey
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_url(self) -> str | None:
        """Best-guess URL/path for the source document."""
        for key in ("url", "source_url", "source", "path"):
            value = self.metadata.get(key)
            if value:
                return str(value)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "collection": self.collection,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }


class GalaxyDocsRetriever:
    """
    Semantic retriever over the four chatgxy-materials collections.

    Construct once per process, call .search(query, collection) or
    .search_all(query) for each query. Each Chroma collection is loaded on
    first use and cached.
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        embeddings_model: str | None = None,
        embeddings_base_url: str | None = None,
        embeddings_api_key: str | None = None,
        validate_manifest: bool = True,
        strict_manifest: bool = False,
    ):
        self.embeddings_dir = Path(embeddings_dir)

        self.model = embeddings_model or os.getenv("EMBEDDINGS_MODEL")
        self.base_url = embeddings_base_url or os.getenv("EMBEDDINGS_BASE_URL")
        self.api_key = embeddings_api_key or os.getenv("EMBEDDINGS_API_KEY")

        if not self.model:
            raise ValueError("embeddings_model not set (pass it or set EMBEDDINGS_MODEL)")
        if not self.base_url:
            raise ValueError("embeddings_base_url not set (pass it or set EMBEDDINGS_BASE_URL)")
        if not self.api_key:
            raise ValueError("embeddings_api_key not set (pass it or set EMBEDDINGS_API_KEY)")

        self.manifest = manifest_mod.load(self.embeddings_dir)
        if validate_manifest:
            manifest_mod.validate(self.manifest, self.model, strict=strict_manifest)
        self.collections = available_collections(
            self.manifest.collections if self.manifest else None
        )

        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )

        self._stores: dict[CollectionKey, Chroma] = {}

    def _get_store(self, info: CollectionInfo) -> Chroma:
        store = self._stores.get(info.key)
        if store is None:
            store = Chroma(
                collection_name=info.chroma_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.embeddings_dir),
            )
            self._stores[info.key] = store
        return store

    def search(self, query: str, collection: CollectionKey, k: int = 5) -> list[Hit]:
        info = COLLECTIONS.get(collection)
        if info is None:
            raise ValueError(
                f"Unknown collection {collection!r}. Valid keys: {list(COLLECTIONS)}"
            )
        store = self._get_store(info)
        results = store.similarity_search_with_relevance_scores(query, k=k)
        return [
            Hit(
                text=doc.page_content,
                score=score,
                collection=info.key,
                metadata=dict(doc.metadata or {}),
            )
            for doc, score in results
        ]

    def search_all(self, query: str, per_collection_k: int = 3) -> list[Hit]:
        """Top-k from each collection in the snapshot, flattened. No cross-collection reranking."""
        hits: list[Hit] = []
        for info in self.collections:
            hits.extend(self.search(query, info.key, k=per_collection_k))
        return hits
