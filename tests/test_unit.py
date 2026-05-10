"""
Unit tests for pure-logic helpers that don't need TACC creds or the embeddings
snapshot. These are the tests CI runs on every push.
"""

from __future__ import annotations

import pytest
from langchain.schema import Document

from pipelines.common import EmbeddingPipeline, strip_yaml_frontmatter
from retrieval.collections import COLLECTIONS, collection_by_chroma_name
from retrieval.manifest import Manifest, ManifestMismatch, validate as validate_manifest
from scripts.write_manifest import count_documents


class TestStripYamlFrontmatter:
    def test_strips_simple_block(self):
        text = "---\nfoo: bar\n---\nhello\n"
        assert strip_yaml_frontmatter(text) == "hello\n"

    def test_passthrough_when_no_frontmatter(self):
        text = "# Header\n\nbody"
        assert strip_yaml_frontmatter(text) == text

    def test_only_strips_leading_block(self):
        # A `---` later in the document (e.g. used as a horizontal rule) must
        # be left alone -- the regex ate too much in an earlier draft.
        text = "---\nfoo: bar\n---\nbody1\n\n---\n\nbody2\n"
        assert strip_yaml_frontmatter(text) == "body1\n\n---\n\nbody2\n"

    def test_handles_crlf(self):
        text = "---\r\nfoo: bar\r\n---\r\nbody"
        assert strip_yaml_frontmatter(text) == "body"


@pytest.fixture
def make_pipeline(monkeypatch, tmp_path):
    """Factory for an EmbeddingPipeline with stub creds and the requested splitter."""
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "stub")
    monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://stub")

    def _make(splitter: str = "recursive") -> EmbeddingPipeline:
        return EmbeddingPipeline(
            chroma_directory=str(tmp_path),
            collection_name="x",
            tracking_file=str(tmp_path / "t.json"),
            splitter=splitter,
        )

    return _make


class TestEmbeddingPipelineConstruction:
    def test_rejects_unknown_splitter(self, make_pipeline):
        with pytest.raises(ValueError, match="Unknown splitter"):
            make_pipeline(splitter="not-a-real-strategy")

    def test_requires_api_creds(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDINGS_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="EMBEDDINGS_API_KEY"):
            EmbeddingPipeline(
                chroma_directory=str(tmp_path),
                collection_name="x",
                tracking_file=str(tmp_path / "t.json"),
            )


class TestMarkdownAwareSplitting:
    """Behavioral contract for split_documents(splitter='markdown').

    Asserts what the splitter is for -- frontmatter stripped, header chain
    in metadata, parent metadata preserved -- rather than just "constructor
    accepts the name." The actual UnstructuredMarkdownLoader regression
    was upstream of this point (loader flattened the markdown before it
    arrived), so these tests don't catch loader-shape mistakes; they do
    catch silent splitter-dispatch breakage.
    """

    SAMPLE = """---
layout: tutorial_hands_on
title: Test
---

# Introduction

Hello world.

## Step 1

Click upload.

## Step 2

Pick the file.
"""

    def test_strips_yaml_frontmatter(self, make_pipeline):
        pipeline = make_pipeline(splitter="markdown")
        chunks = pipeline.split_documents([Document(page_content=self.SAMPLE, metadata={"source": "x.md"})])
        for chunk in chunks:
            assert "layout: tutorial_hands_on" not in chunk.page_content
            assert "title: Test" not in chunk.page_content

    def test_attaches_header_chain_to_metadata(self, make_pipeline):
        pipeline = make_pipeline(splitter="markdown")
        chunks = pipeline.split_documents([Document(page_content=self.SAMPLE, metadata={"source": "x.md"})])

        h1_values = {c.metadata.get("h1") for c in chunks if c.metadata.get("h1")}
        assert "Introduction" in h1_values, f"expected h1='Introduction' somewhere; got {h1_values}"

        h2_values = {c.metadata.get("h2") for c in chunks if c.metadata.get("h2")}
        assert {"Step 1", "Step 2"}.issubset(h2_values), f"expected both h2 sections; got {h2_values}"

    def test_preserves_parent_metadata(self, make_pipeline):
        pipeline = make_pipeline(splitter="markdown")
        chunks = pipeline.split_documents([
            Document(page_content=self.SAMPLE, metadata={"source": "x.md", "topic": "intro"})
        ])
        for chunk in chunks:
            assert chunk.metadata["source"] == "x.md"
            assert chunk.metadata["topic"] == "intro"

    def test_recursive_splitter_does_not_add_header_metadata(self, make_pipeline):
        # The dispatch is real: recursive path leaves header keys absent.
        pipeline = make_pipeline(splitter="recursive")
        chunks = pipeline.split_documents([Document(page_content=self.SAMPLE, metadata={"source": "x.md"})])
        for chunk in chunks:
            assert "h1" not in chunk.metadata
            assert "h2" not in chunk.metadata


class TestCollectionRegistry:
    def test_reverse_lookup_round_trips(self):
        for key, info in COLLECTIONS.items():
            assert collection_by_chroma_name(info.chroma_name).key == key

    def test_reverse_lookup_unknown_returns_none(self):
        assert collection_by_chroma_name("not-a-collection") is None


class TestManifestValidation:
    """The manifest exists to prevent the silent garbage failure mode where a
    retriever embeds queries with model B against a Chroma index built by
    model A. These four cases cover the full decision matrix."""

    def test_missing_manifest_warns_when_not_strict(self, caplog):
        # Assert level rather than message text -- the wording can churn
        # without changing the actual contract.
        import logging
        with caplog.at_level(logging.WARNING, logger="retrieval.manifest"):
            validate_manifest(None, "any-model", strict=False)
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_missing_manifest_raises_when_strict(self):
        with pytest.raises(ManifestMismatch):
            validate_manifest(None, "any-model", strict=True)

    def test_matching_model_passes(self):
        m = Manifest(model="model-A", embeddings_base_url="", built_at="", collections={})
        validate_manifest(m, "model-A", strict=False)
        validate_manifest(m, "model-A", strict=True)

    def test_mismatched_model_raises_regardless_of_strict(self):
        m = Manifest(model="model-A", embeddings_base_url="", built_at="", collections={})
        with pytest.raises(ManifestMismatch, match="Embedder mismatch"):
            validate_manifest(m, "model-B", strict=False)
        with pytest.raises(ManifestMismatch, match="Embedder mismatch"):
            validate_manifest(m, "model-B", strict=True)


class TestCountDocuments:
    def test_uses_source_when_available(self):
        metas = [{"source": "/a.md"}, {"source": "/b.md"}, {"source": "/a.md"}]
        assert count_documents(metas) == 2

    def test_prefers_source_plus_faq_index_when_present(self):
        # Real bug from development: FAQ chunks share one source file but have
        # distinct faq_index values, so source-only counting reported 1 doc
        # for 25 distinct FAQs. The two-key fallback fixes this.
        metas = [
            {"source": "/faq.md", "faq_index": 0},
            {"source": "/faq.md", "faq_index": 1},
            {"source": "/faq.md", "faq_index": 2},
        ]
        assert count_documents(metas) == 3

    def test_handles_none_entries(self):
        # chromadb sometimes returns None metadata for chunks; don't crash.
        assert count_documents([None, {"source": "/a.md"}, None]) == 1
