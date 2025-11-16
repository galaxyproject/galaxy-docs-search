"""
Test search quality across all Galaxy documentation collections.

Tests semantic search accuracy, metadata completeness, and retrieval strategies.
"""

import pytest
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


@pytest.fixture(scope="module")
def embeddings():
    """Initialize embeddings for all tests."""
    return OpenAIEmbeddings(
        model="text-embedding-nomic-embed-text-v2-moe",
        openai_api_key=os.getenv("EMBEDDINGS_API_KEY"),
        openai_api_base=os.getenv("EMBEDDINGS_BASE_URL"),
        tiktoken_enabled=False,
        check_embedding_ctx_length=False
    )


@pytest.fixture(scope="module")
def faq_store(embeddings):
    """Load FAQ collection."""
    return Chroma(
        collection_name="galaxy_faqs",
        embedding_function=embeddings,
        persist_directory="./data/embeddings"
    )


@pytest.fixture(scope="module")
def iwc_store(embeddings):
    """Load IWC workflows collection."""
    return Chroma(
        collection_name="iwc_workflows",
        embedding_function=embeddings,
        persist_directory="./data/embeddings"
    )


@pytest.fixture(scope="module")
def gtn_store(embeddings):
    """Load GTN tutorials collection."""
    return Chroma(
        collection_name="gtn_tutorials",
        embedding_function=embeddings,
        persist_directory="./data/embeddings"
    )


@pytest.fixture(scope="module")
def discourse_store(embeddings):
    """Load Discourse posts collection."""
    return Chroma(
        collection_name="discourse_posts",
        embedding_function=embeddings,
        persist_directory="./data/embeddings"
    )


class TestFAQCollection:
    """Test FAQ collection search quality."""

    def test_faq_collection_exists(self, faq_store):
        """Verify FAQ collection is loaded."""
        assert faq_store is not None

    def test_faq_metadata_structure(self, faq_store):
        """Verify FAQ documents have correct metadata."""
        results = faq_store.similarity_search("workflow", k=1)
        assert len(results) > 0

        doc = results[0]
        assert "question" in doc.metadata
        assert "category" in doc.metadata
        assert "data_source" in doc.metadata
        assert doc.metadata["data_source"] == "gtn_faq"
        assert doc.metadata["content_type"] == "faq"

    def test_faq_question_format(self, faq_store):
        """Verify FAQ documents have Q&A format."""
        results = faq_store.similarity_search("how do I", k=1)
        assert len(results) > 0

        content = results[0].page_content
        assert "Q:" in content or "A:" in content

    @pytest.mark.parametrize("query,expected_keyword", [
        ("how do I use workflows", "workflow"),
        ("getting started with Galaxy", "Galaxy"),
        ("where can I find tutorials", "tutorial"),
    ])
    def test_faq_question_search(self, faq_store, query, expected_keyword):
        """Test FAQ search for common questions."""
        results = faq_store.similarity_search(query, k=3)
        assert len(results) > 0

        # At least one result should contain the expected keyword
        found = any(expected_keyword.lower() in doc.page_content.lower()
                   for doc in results)
        assert found, f"Expected keyword '{expected_keyword}' not found in top 3 results"


class TestIWCCollection:
    """Test IWC workflows collection search quality."""

    def test_iwc_collection_exists(self, iwc_store):
        """Verify IWC collection is loaded."""
        assert iwc_store is not None

    def test_iwc_metadata_structure(self, iwc_store):
        """Verify IWC documents have correct metadata."""
        results = iwc_store.similarity_search("workflow", k=1)
        assert len(results) > 0

        doc = results[0]
        assert "workflow_name" in doc.metadata
        assert "workflow_id" in doc.metadata
        assert "topic" in doc.metadata
        assert "url" in doc.metadata
        assert "data_source" in doc.metadata
        assert doc.metadata["data_source"] == "iwc"
        assert doc.metadata["content_type"] == "workflow"

    @pytest.mark.parametrize("query,expected_topic", [
        ("RNA-seq analysis", "transcriptomics"),
        ("variant calling", ["variant-calling", "sars-cov-2-variant-calling", "virology"]),
        ("genome assembly", ["genome-assembly", "VGP-assembly-v2"]),
        ("ChIP-seq", "epigenetics"),
        ("proteomics", "proteomics"),
    ])
    def test_iwc_analysis_type_search(self, iwc_store, query, expected_topic):
        """Test IWC search for different analysis types."""
        results = iwc_store.similarity_search(query, k=5)
        assert len(results) > 0

        # Check if any result has the expected topic
        topics = [doc.metadata.get("topic", "") for doc in results]
        if isinstance(expected_topic, list):
            found = any(t in expected_topic for t in topics)
        else:
            found = expected_topic in topics

        assert found, f"Expected topic(s) {expected_topic} not found in results: {topics}"

    def test_iwc_covid_category(self, iwc_store):
        """Test IWC search for COVID-19 workflows."""
        results = iwc_store.similarity_search("COVID-19", k=5)
        assert len(results) > 0

        # At least one result should have COVID-19 category
        categories = [doc.metadata.get("categories", "") for doc in results]
        found = any("COVID-19" in cat for cat in categories)
        assert found, f"COVID-19 category not found in results: {categories}"

    def test_iwc_workflow_has_readme(self, iwc_store):
        """Verify IWC workflows contain README content."""
        results = iwc_store.similarity_search("workflow", k=10)
        assert len(results) > 0

        # At least some chunks should have substantial content
        substantial_chunks = [r for r in results if len(r.page_content) > 100]
        assert len(substantial_chunks) > 0, "No workflow chunks with substantial content found"

    def test_iwc_metadata_completeness(self, iwc_store):
        """Test that IWC workflows have rich metadata."""
        results = iwc_store.similarity_search("RNA-seq", k=1)
        assert len(results) > 0

        doc = results[0]
        # Check for optional but expected metadata
        assert doc.metadata.get("url", "").startswith("https://iwc.galaxyproject.org")


class TestGTNCollection:
    """Test GTN tutorials collection search quality."""

    def test_gtn_collection_exists(self, gtn_store):
        """Verify GTN collection is loaded."""
        assert gtn_store is not None

    def test_gtn_metadata_structure(self, gtn_store):
        """Verify GTN documents have correct metadata."""
        results = gtn_store.similarity_search("tutorial", k=10)
        assert len(results) > 0

        doc = results[0]
        # All docs should have these fields
        assert "data_source" in doc.metadata
        assert doc.metadata["data_source"] == "gtn"
        assert doc.metadata["content_type"] == "tutorial"
        assert "source" in doc.metadata
        # Note: title and topic may not be in all documents due to chunking

    @pytest.mark.parametrize("query,expected_keywords", [
        ("RNA-seq tutorial", ["rna", "transcriptom"]),
        ("variant calling guide", ["variant"]),
        ("proteomics tutorial", ["proteom"]),
    ])
    def test_gtn_tutorial_search(self, gtn_store, query, expected_keywords):
        """Test GTN search for tutorials."""
        results = gtn_store.similarity_search(query, k=10)
        assert len(results) > 0

        # Check that relevant content is found (check source path or content)
        combined_text = " ".join([
            doc.page_content.lower() + " " + doc.metadata.get('source', '').lower()
            for doc in results
        ])
        found = any(keyword.lower() in combined_text for keyword in expected_keywords)
        assert found, f"Expected keywords {expected_keywords} not found in top 10 results"


class TestDiscourseCollection:
    """Test Discourse posts collection search quality."""

    def test_discourse_collection_exists(self, discourse_store):
        """Verify Discourse collection is loaded."""
        assert discourse_store is not None

    def test_discourse_metadata_structure(self, discourse_store):
        """Verify Discourse documents have correct metadata."""
        results = discourse_store.similarity_search("workflow", k=1)
        assert len(results) > 0

        doc = results[0]
        assert "title" in doc.metadata
        assert "url" in doc.metadata
        assert "category" in doc.metadata
        assert "data_source" in doc.metadata
        assert doc.metadata["data_source"] == "discourse"
        assert doc.metadata["content_type"] == "forum_post"

    def test_discourse_url_format(self, discourse_store):
        """Verify Discourse URLs are valid."""
        results = discourse_store.similarity_search("workflow", k=1)
        assert len(results) > 0

        url = results[0].metadata.get("url", "")
        assert url.startswith("https://help.galaxyproject.org")

    @pytest.mark.parametrize("query", [
        "workflow error",
        "installation problem",
        "how do I upload data",
    ])
    def test_discourse_problem_search(self, discourse_store, query):
        """Test Discourse search for common problems."""
        results = discourse_store.similarity_search(query, k=5)
        assert len(results) > 0

        # Results should have titles
        for doc in results:
            assert len(doc.metadata.get("title", "")) > 0


class TestCrossCollectionSearch:
    """Test searching across multiple collections."""

    def test_all_collections_loaded(self, faq_store, iwc_store, gtn_store, discourse_store):
        """Verify all collections are accessible."""
        assert faq_store is not None
        assert iwc_store is not None
        assert gtn_store is not None
        assert discourse_store is not None

    def test_workflow_search_across_collections(self, faq_store, iwc_store, gtn_store, discourse_store):
        """Test workflow-related search finds relevant results in each collection."""
        query = "RNA-seq workflow"

        faq_results = faq_store.similarity_search(query, k=1)
        iwc_results = iwc_store.similarity_search(query, k=1)
        gtn_results = gtn_store.similarity_search(query, k=1)
        discourse_results = discourse_store.similarity_search(query, k=1)

        # Should get results from at least IWC and GTN
        assert len(iwc_results) > 0, "IWC should have RNA-seq workflows"
        assert len(gtn_results) > 0, "GTN should have RNA-seq tutorials"

    def test_different_sources_different_metadata(self, iwc_store, gtn_store):
        """Verify different collections have distinct metadata."""
        iwc_results = iwc_store.similarity_search("RNA-seq", k=1)
        gtn_results = gtn_store.similarity_search("RNA-seq", k=1)

        assert iwc_results[0].metadata["data_source"] == "iwc"
        assert gtn_results[0].metadata["data_source"] == "gtn"

        # IWC has workflow_name, GTN has source path
        assert "workflow_name" in iwc_results[0].metadata
        assert "source" in gtn_results[0].metadata


class TestRetrievalStrategies:
    """Test different retrieval strategies."""

    def test_prioritized_search(self, faq_store, iwc_store, gtn_store, discourse_store):
        """Test prioritized search strategy (FAQ > IWC > GTN > Discourse)."""
        query = "How do I analyze RNA-seq data?"

        results = {
            'faq': faq_store.similarity_search(query, k=2),
            'workflows': iwc_store.similarity_search(query, k=2),
            'tutorials': gtn_store.similarity_search(query, k=2),
            'community': discourse_store.similarity_search(query, k=2)
        }

        # Should get results from multiple sources
        total_results = sum(len(r) for r in results.values())
        assert total_results > 0

    def test_merged_search_with_scores(self, faq_store, iwc_store, gtn_store, discourse_store):
        """Test merged search with relevance scores."""
        query = "ChIP-seq analysis"

        faq_results = faq_store.similarity_search_with_score(query, k=2)
        iwc_results = iwc_store.similarity_search_with_score(query, k=2)
        gtn_results = gtn_store.similarity_search_with_score(query, k=2)
        discourse_results = discourse_store.similarity_search_with_score(query, k=2)

        all_results = []
        for doc, score in faq_results:
            all_results.append((doc, score, 'faq'))
        for doc, score in iwc_results:
            all_results.append((doc, score, 'workflow'))
        for doc, score in gtn_results:
            all_results.append((doc, score, 'tutorial'))
        for doc, score in discourse_results:
            all_results.append((doc, score, 'community'))

        # Sort by score (lower is better)
        all_results.sort(key=lambda x: x[1])

        # Top result should have a reasonable score (< 1.0 typically)
        if all_results:
            top_score = all_results[0][1]
            assert top_score < 2.0, f"Top result score seems too high: {top_score}"


class TestSearchEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self, iwc_store):
        """Test handling of empty query."""
        # Should not crash, though results may be arbitrary
        results = iwc_store.similarity_search("", k=1)
        # Just verify it doesn't crash
        assert isinstance(results, list)

    def test_very_specific_query(self, iwc_store):
        """Test very specific technical query."""
        results = iwc_store.similarity_search(
            "GROMACS molecular dynamics protein-ligand binding free energy",
            k=5
        )
        assert len(results) > 0

    def test_large_k_value(self, discourse_store):
        """Test requesting many results."""
        results = discourse_store.similarity_search("workflow", k=100)
        # Should get results up to the limit of available docs
        assert len(results) > 0
        assert len(results) <= 100

    def test_metadata_types(self, iwc_store):
        """Verify metadata values are correct types."""
        results = iwc_store.similarity_search("workflow", k=1)
        assert len(results) > 0

        doc = results[0]
        # All metadata values should be strings, ints, floats, or bools
        for key, value in doc.metadata.items():
            assert isinstance(value, (str, int, float, bool)), \
                f"Metadata field '{key}' has invalid type: {type(value)}"


class TestCollectionStats:
    """Test collection statistics and coverage."""

    def test_faq_collection_size(self, faq_store):
        """Verify FAQ collection has expected number of documents."""
        # We expect around 25 FAQs
        # Query with high k to get many results
        results = faq_store.similarity_search("Galaxy", k=100)
        assert len(results) >= 20, "FAQ collection seems too small"

    def test_iwc_collection_size(self, iwc_store):
        """Verify IWC collection has expected number of workflows."""
        # We expect around 105 workflows
        results = iwc_store.similarity_search("workflow", k=200)
        assert len(results) >= 80, "IWC collection seems too small"

    def test_gtn_collection_size(self, gtn_store):
        """Verify GTN collection has expected number of tutorials."""
        # We expect around 168 tutorials
        results = gtn_store.similarity_search("tutorial", k=300)
        assert len(results) >= 150, "GTN collection seems too small"

    def test_discourse_collection_size(self, discourse_store):
        """Verify Discourse collection has substantial content."""
        # We expect thousands of topics
        results = discourse_store.similarity_search("Galaxy", k=100)
        assert len(results) >= 100, "Discourse collection seems too small"
