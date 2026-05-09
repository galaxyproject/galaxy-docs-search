"""
Unit tests for pure-logic helpers that don't need TACC creds or the embeddings
snapshot. These are the tests CI runs on every push.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.common import (
    YAML_FRONTMATTER_RE,
    EmbeddingPipeline,
    strip_yaml_frontmatter,
)
from retrieval.collections import COLLECTIONS, collection_by_chroma_name
from retrieval.manifest import Manifest, ManifestMismatch, validate as validate_manifest
from scripts.write_manifest import count_documents


class TestStripYamlFrontmatter:
    def test_strips_simple_block(self):
        text = "---\nfoo: bar\n---\nhello\n"
        assert strip_yaml_frontmatter(text) == "hello\n"

    def test_strips_multiline_block(self):
        text = "---\nfoo: bar\nbaz: qux\nlist:\n  - a\n  - b\n---\nbody\n"
        assert strip_yaml_frontmatter(text) == "body\n"

    def test_passthrough_when_no_frontmatter(self):
        text = "# Header\n\nbody"
        assert strip_yaml_frontmatter(text) == text

    def test_only_strips_leading_block(self):
        # A `---` later in the document (e.g. used as a horizontal rule) must
        # be left alone.
        text = "---\nfoo: bar\n---\nbody1\n\n---\n\nbody2\n"
        assert strip_yaml_frontmatter(text) == "body1\n\n---\n\nbody2\n"

    def test_handles_crlf(self):
        text = "---\r\nfoo: bar\r\n---\r\nbody"
        assert strip_yaml_frontmatter(text) == "body"

    def test_does_not_match_partial_delimiter(self):
        # Triple-dash inside a code fence is not a frontmatter delimiter.
        text = "```\n---\n```\nbody"
        assert strip_yaml_frontmatter(text) == text

    def test_regex_is_anchored(self):
        # Sanity check that the constant we expose really is anchored at start
        # of string (`^` with re.DOTALL).
        assert YAML_FRONTMATTER_RE.match("body\n---\nfoo: bar\n---\n") is None


class TestEmbeddingPipelineConstruction:
    """Pure-logic checks; we don't actually wire up Chroma/OpenAI here."""

    def test_rejects_unknown_splitter(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "x")
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://x")
        with pytest.raises(ValueError, match="Unknown splitter"):
            EmbeddingPipeline(
                chroma_directory=str(tmp_path),
                collection_name="x",
                tracking_file=str(tmp_path / "t.json"),
                splitter="not-a-real-strategy",
            )

    def test_accepts_recursive_and_markdown(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "x")
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://x")
        for splitter in ("recursive", "markdown"):
            EmbeddingPipeline(
                chroma_directory=str(tmp_path),
                collection_name="x",
                tracking_file=str(tmp_path / "t.json"),
                splitter=splitter,
            )

    def test_requires_api_creds(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDINGS_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="EMBEDDINGS_API_KEY"):
            EmbeddingPipeline(
                chroma_directory=str(tmp_path),
                collection_name="x",
                tracking_file=str(tmp_path / "t.json"),
            )


class TestCollectionRegistry:
    def test_keys_match_chroma_names_exactly_once(self):
        chroma_names = {info.chroma_name for info in COLLECTIONS.values()}
        assert len(chroma_names) == len(COLLECTIONS), "duplicate chroma_name"

    def test_reverse_lookup_round_trips(self):
        for key, info in COLLECTIONS.items():
            assert collection_by_chroma_name(info.chroma_name).key == key

    def test_reverse_lookup_unknown_returns_none(self):
        assert collection_by_chroma_name("not-a-collection") is None


class TestManifestValidation:
    def test_missing_manifest_warns_when_not_strict(self, caplog):
        validate_manifest(None, "any-model", strict=False)
        assert any("manifest.json" in r.message for r in caplog.records)

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
        metas = [
            {"source": "/a.md"},
            {"source": "/b.md"},
            {"source": "/a.md"},
        ]
        assert count_documents(metas) == 2

    def test_prefers_source_plus_faq_index_when_present(self):
        # FAQ chunks share one source file but have distinct faq_index values;
        # the two-key fallback should produce 3 not 1.
        metas = [
            {"source": "/faq.md", "faq_index": 0},
            {"source": "/faq.md", "faq_index": 1},
            {"source": "/faq.md", "faq_index": 2},
        ]
        assert count_documents(metas) == 3

    def test_falls_back_to_path_or_url(self):
        assert count_documents([{"path": "x"}, {"path": "y"}]) == 2
        assert count_documents([{"url": "x"}, {"url": "y"}]) == 2

    def test_returns_zero_when_no_recognized_keys(self):
        assert count_documents([{"unrelated": "x"}]) == 0

    def test_handles_none_entries(self):
        assert count_documents([None, {"source": "/a.md"}, None]) == 1
