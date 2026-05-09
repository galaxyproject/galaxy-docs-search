"""
Smoke tests for retrieval.GalaxyDocsRetriever against the on-disk Chroma
snapshot. Skipped when EMBEDDINGS_* env vars or the snapshot are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_DIR = Path("./data/embeddings")

pytestmark = pytest.mark.skipif(
    not (
        EMBEDDINGS_DIR.exists()
        and os.getenv("EMBEDDINGS_API_KEY")
        and os.getenv("EMBEDDINGS_BASE_URL")
        and os.getenv("EMBEDDINGS_MODEL")
    ),
    reason="Needs data/embeddings + EMBEDDINGS_* env vars to run",
)


@pytest.fixture(scope="module")
def retriever():
    from retrieval import GalaxyDocsRetriever

    return GalaxyDocsRetriever(embeddings_dir=EMBEDDINGS_DIR)


def test_each_collection_returns_hits(retriever):
    queries = {
        "faqs": "how do I upload a file",
        "tutorials": "rna-seq differential expression",
        "discourse": "fastq upload error",
        "workflows": "rna-seq differential expression",
    }
    for collection, query in queries.items():
        hits = retriever.search(query, collection, k=3)
        assert hits, f"No hits for {collection}"
        for hit in hits:
            assert hit.text
            assert hit.collection == collection
            assert isinstance(hit.score, float)


def test_search_all_returns_hits_from_all_collections(retriever):
    hits = retriever.search_all("how do I upload a fastq file", per_collection_k=2)
    seen = {h.collection for h in hits}
    assert seen == {"faqs", "tutorials", "discourse", "workflows"}


def test_unknown_collection_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("anything", "nonexistent", k=1)


def test_manifest_mismatch_blocks_init():
    """Constructing with a different model than the manifest should refuse."""
    from retrieval.client import GalaxyDocsRetriever
    from retrieval.manifest import ManifestMismatch

    with pytest.raises(ManifestMismatch):
        GalaxyDocsRetriever(
            embeddings_dir=EMBEDDINGS_DIR,
            embeddings_model="not-the-real-model",
            strict_manifest=True,
        )
