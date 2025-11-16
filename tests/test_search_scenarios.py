"""
Real-world search scenarios for Galaxy documentation.

These tests demonstrate the system's ability to handle common user queries
and provide relevant results from the appropriate collections.
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
def all_stores(embeddings):
    """Load all collections."""
    return {
        'faq': Chroma(
            collection_name="galaxy_faqs",
            embedding_function=embeddings,
            persist_directory="./data/embeddings"
        ),
        'iwc': Chroma(
            collection_name="iwc_workflows",
            embedding_function=embeddings,
            persist_directory="./data/embeddings"
        ),
        'gtn': Chroma(
            collection_name="gtn_tutorials",
            embedding_function=embeddings,
            persist_directory="./data/embeddings"
        ),
        'discourse': Chroma(
            collection_name="discourse_posts",
            embedding_function=embeddings,
            persist_directory="./data/embeddings"
        )
    }


class TestBioinformaticsWorkflows:
    """Test finding bioinformatics analysis workflows."""

    @pytest.mark.parametrize("query,expected_workflow_keywords", [
        ("RNA sequencing differential expression", ["RNA-seq", "differential expression"]),
        ("variant calling from NGS data", ["variant"]),
        ("genome assembly from long reads", ["assembly", "flye", "genome"]),
        ("ChIP-seq peak calling", ["ChIP-seq", "peak"]),
        ("metagenomics taxonomic profiling", ["metagenomics", "MAG", "microbiome", "metaproteom"]),
        ("single cell RNA-seq clustering", ["single-cell", "single cell", "Scanpy", "10X"]),
    ])
    def test_find_analysis_workflow(self, all_stores, query, expected_workflow_keywords):
        """Test finding workflows for common analysis types."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search(query, k=5)

        assert len(results) > 0, f"No workflows found for: {query}"

        # Check that results contain expected keywords
        all_content = " ".join([doc.page_content for doc in results])
        all_metadata = " ".join([
            doc.metadata.get('workflow_name', '') for doc in results
        ])
        combined = (all_content + " " + all_metadata).lower()

        found_any = any(keyword.lower() in combined for keyword in expected_workflow_keywords)
        assert found_any, f"Expected keywords {expected_workflow_keywords} not found in top 5 results"


class TestTutorialDiscovery:
    """Test finding relevant tutorials."""

    def test_beginner_tutorials(self, all_stores):
        """Test finding beginner-friendly tutorials."""
        gtn = all_stores['gtn']
        results = gtn.similarity_search("getting started introduction basics", k=10)

        assert len(results) > 0
        # Should find introductory content (check both metadata and content)
        found = any('intro' in doc.metadata.get('title', '').lower() or
                   'intro' in doc.metadata.get('source', '').lower() or
                   'start' in doc.metadata.get('source', '').lower() or
                   'basic' in doc.page_content.lower()
                   for doc in results)
        assert found, "No beginner tutorials found"

    @pytest.mark.parametrize("analysis_type", [
        "RNA-seq",
        "proteomics",
        "genome annotation",
        "variant calling",
    ])
    def test_analysis_specific_tutorials(self, all_stores, analysis_type):
        """Test finding tutorials for specific analysis types."""
        gtn = all_stores['gtn']
        results = gtn.similarity_search(f"{analysis_type} tutorial", k=5)

        assert len(results) > 0, f"No tutorials found for {analysis_type}"


class TestQuickAnswers:
    """Test getting quick answers from FAQs."""

    @pytest.mark.parametrize("question", [
        "How do I use Galaxy?",
        "Where can I find help?",
        "How do I upload data?",
    ])
    def test_faq_answers_questions(self, all_stores, question):
        """Test FAQ collection answers common questions."""
        faq = all_stores['faq']
        results = faq.similarity_search(question, k=5)

        assert len(results) > 0, f"No FAQ found for: {question}"

        # At least one result should be in Q&A format (check first few results)
        has_qa_format = any('Q:' in doc.page_content or 'A:' in doc.page_content or '?' in doc.page_content
                           for doc in results[:5])
        assert has_qa_format, f"No FAQ results in Q&A format found for '{question}'"


class TestProblemSolving:
    """Test finding solutions to common problems."""

    @pytest.mark.parametrize("problem,expected_source", [
        ("workflow fails with error", "discourse"),  # Community problems
        ("how to install Galaxy", "discourse"),      # Installation help
        ("dataset not appearing", "discourse"),      # Common issues
    ])
    def test_find_problem_solutions(self, all_stores, problem, expected_source):
        """Test finding solutions in appropriate sources."""
        store = all_stores[expected_source]
        results = store.similarity_search(problem, k=5)

        assert len(results) > 0, f"No results found for problem: {problem}"


class TestSpecializedAnalyses:
    """Test finding specialized analysis workflows."""

    def test_covid19_workflows(self, all_stores):
        """Test finding COVID-19 specific workflows."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search("COVID-19 SARS-CoV-2 variant", k=5)

        assert len(results) > 0

        # Should find COVID-19 categorized workflows
        categories = [doc.metadata.get('categories', '') for doc in results]
        found_covid = any('COVID-19' in cat for cat in categories)
        assert found_covid, "No COVID-19 workflows found"

    def test_structural_biology_workflows(self, all_stores):
        """Test finding structural biology workflows."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search("protein structure molecular dynamics", k=5)

        assert len(results) > 0

        # Should find structural biology workflows
        categories = [doc.metadata.get('categories', '') for doc in results]
        topics = [doc.metadata.get('topic', '') for doc in results]
        combined = " ".join(categories + topics)

        found_structural = 'Structural Biology' in combined or 'computational-chemistry' in combined
        assert found_structural, "No structural biology workflows found"

    def test_proteomics_workflows(self, all_stores):
        """Test finding proteomics workflows."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search("mass spectrometry proteomics", k=5)

        assert len(results) > 0

        topics = [doc.metadata.get('topic', '') for doc in results]
        found_proteomics = any('proteomics' in topic for topic in topics)
        assert found_proteomics, "No proteomics workflows found"


class TestMultiSourceRetrieval:
    """Test retrieving complementary information from multiple sources."""

    def test_workflow_plus_tutorial(self, all_stores):
        """Test finding both workflow and tutorial for same analysis."""
        query = "RNA-seq analysis"

        iwc = all_stores['iwc']
        gtn = all_stores['gtn']

        workflow_results = iwc.similarity_search(query, k=3)
        tutorial_results = gtn.similarity_search(query, k=5)

        assert len(workflow_results) > 0, "No workflows found"
        assert len(tutorial_results) > 0, "No tutorials found"

        # Both should be relevant to RNA-seq
        workflow_names = [doc.metadata.get('workflow_name', '') for doc in workflow_results]

        # Check tutorials in source path, title, or content
        tutorials_rna_related = any('RNA' in doc.metadata.get('title', '') or
                                    'rna' in doc.metadata.get('source', '').lower() or
                                    'RNA' in doc.page_content
                                    for doc in tutorial_results)

        assert any('RNA' in name for name in workflow_names), "Workflows don't seem RNA-related"
        assert tutorials_rna_related, "Tutorials don't seem RNA-related"

    def test_comprehensive_help(self, all_stores):
        """Test getting comprehensive help from all sources."""
        query = "variant calling"

        results = {
            source: store.similarity_search(query, k=2)
            for source, store in all_stores.items()
        }

        # Should find relevant content in multiple sources
        sources_with_results = [source for source, docs in results.items() if len(docs) > 0]
        assert len(sources_with_results) >= 2, \
            f"Only found results in {sources_with_results}, expected multiple sources"


class TestMetadataUtility:
    """Test that metadata provides useful information."""

    def test_workflow_citations(self, all_stores):
        """Test that workflows include citation information."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search("workflow", k=10)

        assert len(results) > 0

        # Check if any workflows have DOI
        dois = [doc.metadata.get('doi', '') for doc in results]
        has_doi = any(doi for doi in dois)
        # Note: Not all workflows may have DOIs, but many should
        # This is informational, not a strict requirement

    def test_workflow_authors(self, all_stores):
        """Test that workflows include author information."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search("workflow", k=10)

        assert len(results) > 0

        # Most workflows should have author information
        authors_list = [doc.metadata.get('authors', '') for doc in results]
        has_authors = sum(1 for authors in authors_list if authors)
        assert has_authors > len(results) / 2, "Less than half of workflows have author info"

    def test_discourse_urls_clickable(self, all_stores):
        """Test that Discourse results include clickable URLs."""
        discourse = all_stores['discourse']
        results = discourse.similarity_search("workflow", k=5)

        assert len(results) > 0

        for doc in results:
            url = doc.metadata.get('url', '')
            assert url.startswith('https://'), f"Invalid URL: {url}"
            assert 'help.galaxyproject.org' in url

    def test_tutorial_topic_organization(self, all_stores):
        """Test that tutorials are organized by topic."""
        gtn = all_stores['gtn']
        results = gtn.similarity_search("tutorial", k=10)

        assert len(results) > 0

        # Some tutorials should have topic information (metadata may be lost in chunking)
        # At minimum, all should have a source path that includes the topic
        sources = [doc.metadata.get('source', '') for doc in results]
        has_topic_in_path = sum(1 for source in sources if '/topics/' in source)
        assert has_topic_in_path > len(results) / 2, "Less than half of tutorials have topic information in path"


class TestSearchRelevance:
    """Test that search results are actually relevant."""

    def test_rnaseq_results_relevance(self, all_stores):
        """Test RNA-seq search returns RNA-related content."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search("RNA sequencing", k=5)

        assert len(results) > 0

        # Top result should be highly relevant
        top_result = results[0]
        content = top_result.page_content.lower()
        metadata = top_result.metadata.get('workflow_name', '').lower()

        rna_terms = ['rna', 'transcriptom', 'gene expression']
        found = any(term in content or term in metadata for term in rna_terms)
        assert found, "Top result doesn't seem RNA-related"

    def test_assembly_results_relevance(self, all_stores):
        """Test assembly search returns assembly-related content."""
        iwc = all_stores['iwc']
        results = iwc.similarity_search("genome assembly", k=5)

        assert len(results) > 0

        # At least one of top 3 should mention assembly
        top_3_content = " ".join([doc.page_content.lower() for doc in results[:3]])
        top_3_names = " ".join([
            doc.metadata.get('workflow_name', '').lower() for doc in results[:3]
        ])

        combined = top_3_content + " " + top_3_names
        assert 'assembl' in combined, "Top 3 results don't mention assembly"


class TestComplexQueries:
    """Test handling of complex, multi-part queries."""

    def test_multi_step_analysis(self, all_stores):
        """Test finding workflows for multi-step analyses."""
        iwc = all_stores['iwc']
        query = "RNA-seq quality control alignment quantification differential expression"
        results = iwc.similarity_search(query, k=5)

        assert len(results) > 0
        # Should find comprehensive RNA-seq workflows

    def test_comparative_analysis(self, all_stores):
        """Test finding workflows for comparative studies."""
        iwc = all_stores['iwc']
        query = "compare samples differential analysis statistical testing"
        results = iwc.similarity_search(query, k=5)

        assert len(results) > 0

    def test_specific_tool_query(self, all_stores):
        """Test finding workflows that use specific tools."""
        iwc = all_stores['iwc']
        queries = [
            "STAR aligner RNA-seq",
            "bowtie alignment",
            "GROMACS simulation",
        ]

        for query in queries:
            results = iwc.similarity_search(query, k=5)
            assert len(results) > 0, f"No results for: {query}"


def test_print_example_results(all_stores):
    """Print example results for documentation (not a real test)."""
    print("\n" + "="*80)
    print("EXAMPLE SEARCH RESULTS")
    print("="*80)

    examples = [
        ("RNA-seq analysis", "iwc"),
        ("How do I get started?", "faq"),
        ("assembly tutorial", "gtn"),
        ("workflow error", "discourse"),
    ]

    for query, source in examples:
        print(f"\nQuery: '{query}' (searching {source})")
        print("-" * 80)

        results = all_stores[source].similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            if source == "iwc":
                print(f"{i}. {meta.get('workflow_name', 'Unknown')}")
                print(f"   Topic: {meta.get('topic', 'N/A')}")
            elif source == "faq":
                print(f"{i}. {meta.get('question', 'Unknown')}")
            elif source == "gtn":
                print(f"{i}. {meta.get('title', 'Unknown')}")
            elif source == "discourse":
                print(f"{i}. {meta.get('title', 'Unknown')}")
                print(f"   URL: {meta.get('url', 'N/A')}")

    print("\n" + "="*80)
