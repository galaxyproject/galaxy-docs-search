"""
HTTP API tests against a fake retriever -- no snapshot or embedder needed, so
these run in CI.
"""

from __future__ import annotations

import pytest
from starlette.middleware import Middleware
from starlette.testclient import TestClient

from mcp_server.http_api import ApiKeyMiddleware, SearchRequestError, parse_search_request
from mcp_server.server import build_server
from retrieval.client import Hit, public_url
from retrieval.collections import available_collections
from retrieval.manifest import Manifest


class FakeRetriever:
    model = "fake-embedder"

    def __init__(self):
        self.manifest = Manifest(
            model="fake-embedder",
            embeddings_base_url="http://stub",
            built_at="2026-09-29T00:00:00Z",
            collections={
                "galaxy_faqs": {"chunks": 3, "documents": 2},
                "gtn_tutorials": {"chunks": 10, "documents": 1},
            },
        )
        self.collections = available_collections(self.manifest.collections)
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query, collection, k=5):
        self.calls.append((query, collection, k))
        return [
            Hit(
                text=f"{collection} hit",
                score=0.5,
                collection=collection,
                metadata={"url": f"https://example.org/{collection}"},
            )
        ]

    def search_all(self, query, per_collection_k=3):
        return [h for c in self.collections for h in self.search(query, c.key, per_collection_k)]


@pytest.fixture
def retriever():
    return FakeRetriever()


def make_client(retriever, api_keys=None):
    mcp = build_server(retriever=retriever)
    middleware = [Middleware(ApiKeyMiddleware, api_keys=api_keys)] if api_keys else None
    return TestClient(mcp.http_app(middleware=middleware))


class TestAvailableCollections:
    def test_only_collections_in_the_manifest(self):
        keys = [c.key for c in available_collections({"galaxy_tools": {}, "galaxy_faqs": {}})]
        assert keys == ["faqs", "tools"]

    def test_no_manifest_means_everything(self):
        assert len(available_collections(None)) == 5


class TestParseSearchRequest:
    available = {"faqs", "tutorials"}

    def test_defaults_to_every_available_collection(self):
        assert parse_search_request({"query": " trim reads "}, self.available) == (
            "trim reads",
            ["faqs", "tutorials"],
            5,
        )

    @pytest.mark.parametrize(
        "body",
        [
            [],
            {},
            {"query": "   "},
            {"query": "x" * 2001},
            {"query": "q", "k": 0},
            {"query": "q", "k": 21},
            {"query": "q", "k": True},
            {"query": "q", "collections": []},
            {"query": "q", "collections": "faqs"},
            {"query": "q", "collections": ["tools"]},
        ],
    )
    def test_rejects(self, body):
        with pytest.raises(SearchRequestError):
            parse_search_request(body, self.available)


class TestRoutes:
    def test_collections_reports_snapshot(self, retriever):
        body = make_client(retriever).get("/v1/collections").json()
        assert body["index_version"] == "2026-09-29T00:00:00Z"
        assert body["model"] == "fake-embedder"
        assert [(c["key"], c["chunks"]) for c in body["collections"]] == [
            ("faqs", 3),
            ("tutorials", 10),
        ]

    def test_search(self, retriever):
        resp = make_client(retriever).post(
            "/v1/search", json={"query": "bowtie2 index", "collections": ["faqs"], "k": 2}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["index_version"] == "2026-09-29T00:00:00Z"
        assert body["hits"][0]["source_url"] == "https://example.org/faqs"
        assert retriever.calls == [("bowtie2 index", "faqs", 2)]

    def test_search_rejects_collection_missing_from_snapshot(self, retriever):
        resp = make_client(retriever).post("/v1/search", json={"query": "q", "collections": ["tools"]})
        assert resp.status_code == 400
        assert retriever.calls == []

    def test_search_rejects_invalid_json(self, retriever):
        resp = make_client(retriever).post(
            "/v1/search", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert resp.status_code == 400


class TestApiKeys:
    def test_healthz_is_open(self, retriever):
        assert make_client(retriever, api_keys=["s3cret"]).get("/healthz").status_code == 200

    @pytest.mark.parametrize("path", ["/v1/collections", "/mcp"])
    def test_rejects_missing_key(self, retriever, path):
        assert make_client(retriever, api_keys=["s3cret"]).get(path).status_code == 401

    def test_accepts_valid_key(self, retriever):
        client = make_client(retriever, api_keys=["other", "s3cret"])
        resp = client.get("/v1/collections", headers={"Authorization": "Bearer s3cret"})
        assert resp.status_code == 200

    def test_rejects_wrong_key(self, retriever):
        client = make_client(retriever, api_keys=["s3cret"])
        resp = client.get("/v1/collections", headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401


class TestPublicUrl:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (
                "/Users/someone/work/training-material/topics/x/tutorials/y/tutorial.md",
                "https://training.galaxyproject.org/training-material/topics/x/tutorials/y/tutorial.html",
            ),
            (
                "/srv/build/training-material/faqs/galaxy/upload.md",
                "https://training.galaxyproject.org/training-material/faqs/galaxy/upload.html",
            ),
            ("https://help.galaxyproject.org/t/x/1", "https://help.galaxyproject.org/t/x/1"),
            ("/home/builder/private/notes.md", None),
        ],
    )
    def test_public_url(self, source, expected):
        assert public_url(source) == expected

    def test_hit_never_exposes_a_local_path(self):
        hit = Hit(text="", score=0.0, collection="faqs", metadata={"source": "/tmp/x.txt"})
        assert hit.source_url is None
