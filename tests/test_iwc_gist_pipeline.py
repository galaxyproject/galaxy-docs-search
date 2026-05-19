"""
Unit tests for the IWC workflow summaries embedding pipeline.

These tests cover the new `iwc_workflows_gist` collection path without
calling the live IWC manifest, Chroma, embeddings, or a chat API.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyEmbeddingPipeline:
    """Minimal stand-in for EmbeddingPipeline used by IWCPipeline."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.run_calls = []
        DummyEmbeddingPipeline.instances.append(self)

    def run(self, documents, force_rebuild=False):
        self.run_calls.append((documents, force_rebuild))
        return {"documents": documents, "force_rebuild": force_rebuild}


common_stub = types.ModuleType("pipelines.common")
common_stub.EmbeddingPipeline = DummyEmbeddingPipeline
sys.modules.setdefault("pipelines.common", common_stub)

import pipelines.iwc_gist as iwc_gist
from pipelines.iwc_gist import IWCPipeline


def make_pipeline(monkeypatch, tmp_path, **kwargs) -> IWCPipeline:
    """Create an IWCPipeline with the embedding layer mocked out."""
    DummyEmbeddingPipeline.instances.clear()
    monkeypatch.setattr(iwc_gist, "EmbeddingPipeline", DummyEmbeddingPipeline)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_BASE_URL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    return IWCPipeline(chroma_directory=str(tmp_path), **kwargs)


def sample_workflow() -> dict:
    return {
        "iwcID": "rna-seq-main",
        "name": "Fallback Name",
        "readme": "This workflow quantifies paired-end RNA-seq reads.",
        "categories": ["RNA-seq", "Transcriptomics"],
        "collections": ["Single-cell and bulk RNA"],
        "authors": [
            {"name": "Ada Lovelace", "orcid": "0000-0001"},
            {"name": "Grace Hopper"},
        ],
        "doi": "10.5281/zenodo.123",
        "updated": "2026-01-02",
        "definition": {
            "name": "RNA-seq workflow",
            "annotation": "Analyze paired-end RNA-seq reads.",
            "license": "MIT",
            "release": "1.0",
            "steps": {
                "2": {
                    "id": 2,
                    "order_index": 2,
                    "name": "featureCounts",
                    "annotation": "Count reads over genes.",
                    "input_connections": {
                        "input": {"datatype": "bam"},
                    },
                    "outputs": [
                        {"name": "counts", "format": "tabular"},
                    ],
                },
                "1": {
                    "id": 1,
                    "order_index": 1,
                    "tool_id": "fastqc",
                    "label": "Quality control reads.",
                    "input_connections": {
                        "reads": [{"extension": "fastq"}],
                    },
                    "outputs": [
                        {"name": "html", "extension": "html"},
                        {"name": "data", "datatype": "txt"},
                    ],
                },
            },
        },
    }


class TestIWCGistPipelineInitialization:
    def test_initializes_new_collection_defaults(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(monkeypatch, tmp_path)

        assert pipeline.collection_name == "iwc_workflows_gist"
        assert pipeline.summaries_file.name == "iwc_workflow_summaries.json"
        assert pipeline.summarize_workflows is True
        assert pipeline.summary_model == "gpt-4o-mini"

        embedding_pipeline = DummyEmbeddingPipeline.instances[0]
        assert embedding_pipeline.kwargs["collection_name"] == "iwc_workflows_gist"
        assert embedding_pipeline.kwargs["tracking_file"].endswith(
            ".iwc_workflows_gist_tracking.json"
        )
        assert embedding_pipeline.kwargs["incremental"] is True

    def test_initializes_custom_summary_configuration(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(
            monkeypatch,
            tmp_path,
            collection_name="custom_iwc_gist",
            summarize_workflows=False,
            summary_model="summary-model",
            summary_base_url="https://chat.example/v1",
            summary_api_key="secret",
            summaries_file=tmp_path / "summaries.json",
        )

        assert pipeline.collection_name == "custom_iwc_gist"
        assert pipeline.summarize_workflows is False
        assert pipeline.summary_model == "summary-model"
        assert pipeline.summary_base_url == "https://chat.example/v1"
        assert pipeline.summary_api_key == "secret"
        assert pipeline.summaries_file == tmp_path / "summaries.json"


class TestIWCGistSummaryHelpers:
    def test_extracts_ordered_step_context_and_datatypes(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(monkeypatch, tmp_path)

        steps = pipeline.extract_ordered_step_context(
            sample_workflow()["definition"]["steps"]
        )

        assert [step["step_id"] for step in steps] == ["1", "2"]
        assert steps[0]["tool_name"] == "fastqc"
        assert steps[0]["step_purpose"] == "Quality control reads."
        assert steps[0]["input_datatypes"] == ["fastq"]
        assert steps[0]["output_datatypes"] == ["html", "txt"]
        assert steps[1]["tool_name"] == "featureCounts"
        assert steps[1]["input_datatypes"] == ["bam"]
        assert steps[1]["output_datatypes"] == ["tabular"]

    def test_local_summary_uses_workflow_context(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(monkeypatch, tmp_path)
        context = pipeline.workflow_summary_context(
            sample_workflow(),
            "./workflows/transcriptomics/rna-seq",
        )

        summary = pipeline.summarize_workflow_locally(context)

        assert summary["workflow_name"] == "RNA-seq workflow"
        assert summary["workflow_purpose"] == "Analyze paired-end RNA-seq reads."
        assert summary["scientific_domain"] == "transcriptomics"
        assert summary["main_analysis_task"] == "RNA-seq workflow"
        assert len(summary["ordered_steps"]) == 2
        assert summary["summary"]["categories"] == ["RNA-seq", "Transcriptomics"]

    def test_summary_to_text_includes_step_fields(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(monkeypatch, tmp_path)
        summary = {
            "workflow_name": "RNA-seq workflow",
            "workflow_purpose": "Analyze paired-end RNA-seq reads.",
            "scientific_domain": "transcriptomics",
            "main_analysis_task": "quantification",
            "ordered_steps": [
                {
                    "step_id": "1",
                    "tool_name": "fastqc",
                    "step_purpose": "Quality control reads.",
                    "input_datatypes": ["fastq"],
                    "output_datatypes": ["html"],
                }
            ],
            "summary": {"source_path": "./workflows/transcriptomics/rna-seq"},
        }

        text = pipeline.summary_to_text(summary)

        assert "Workflow summary name: RNA-seq workflow" in text
        assert "Scientific domain: transcriptomics" in text
        assert "tool_name=fastqc" in text
        assert "input_datatypes: fastq" in text
        assert "source_path: ./workflows/transcriptomics/rna-seq" in text


class TestIWCGistDocuments:
    def test_parse_workflow_to_document_adds_summary_content_and_metadata(
        self, monkeypatch, tmp_path
    ):
        pipeline = make_pipeline(monkeypatch, tmp_path)

        doc = pipeline.parse_workflow_to_document(
            sample_workflow(),
            "./workflows/transcriptomics/rna-seq",
        )

        assert "# RNA-seq workflow" in doc.page_content
        assert "## Structured Workflow Summary" in doc.page_content
        assert "Workflow purpose: Analyze paired-end RNA-seq reads." in doc.page_content
        assert "This workflow quantifies paired-end RNA-seq reads." in doc.page_content

        assert doc.metadata["source"] == (
            "iwc://./workflows/transcriptomics/rna-seq#rna-seq-main"
        )
        assert doc.metadata["data_source"] == "iwc"
        assert doc.metadata["content_type"] == "workflow"
        assert doc.metadata["workflow_name"] == "RNA-seq workflow"
        assert doc.metadata["workflow_id"] == "rna-seq-main"
        assert doc.metadata["topic"] == "transcriptomics"
        assert doc.metadata["categories"] == "RNA-seq, Transcriptomics"
        assert doc.metadata["authors"] == "Ada Lovelace, Grace Hopper"
        assert doc.metadata["orcids"] == "0000-0001"
        assert doc.metadata["summary_model"] == "gpt-4o-mini"

        workflow_summary = json.loads(doc.metadata["workflow_summary"])
        assert workflow_summary["workflow_name"] == "RNA-seq workflow"
        assert workflow_summary["ordered_steps"][0]["tool_name"] == "fastqc"

    def test_parse_workflow_to_document_can_skip_summaries(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(monkeypatch, tmp_path, summarize_workflows=False)

        doc = pipeline.parse_workflow_to_document(
            sample_workflow(),
            "./workflows/transcriptomics/rna-seq",
        )

        assert "## Structured Workflow Summary" not in doc.page_content
        assert doc.metadata["workflow_summary"] == ""
        assert doc.metadata["summary_model"] == ""

    def test_document_to_summary_record(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(monkeypatch, tmp_path)
        doc = pipeline.parse_workflow_to_document(
            sample_workflow(),
            "./workflows/transcriptomics/rna-seq",
        )

        record = pipeline.document_to_summary_record(doc)

        assert record["workflow_id"] == "rna-seq-main"
        assert record["name"] == "RNA-seq workflow"
        assert (
            record["source"] == "iwc://./workflows/transcriptomics/rna-seq#rna-seq-main"
        )
        assert record["source_file"] == "./workflows/transcriptomics/rna-seq"
        assert record["topic"] == "transcriptomics"
        assert record["tags"] == ["RNA-seq", "Transcriptomics"]
        assert record["status"] == "summarised"
        assert record["summary"]["workflow_name"] == "RNA-seq workflow"
        assert "Workflow summary name: RNA-seq workflow" in record["text_summary"]

    def test_write_summary_records_persists_json(self, monkeypatch, tmp_path):
        summaries_file = tmp_path / "iwc_workflow_summaries.json"
        pipeline = make_pipeline(
            monkeypatch,
            tmp_path,
            summaries_file=summaries_file,
        )
        doc = pipeline.parse_workflow_to_document(
            sample_workflow(),
            "./workflows/transcriptomics/rna-seq",
        )

        pipeline.write_summary_records([doc])

        records = json.loads(summaries_file.read_text(encoding="utf-8"))
        assert len(records) == 1
        assert records[0]["workflow_id"] == "rna-seq-main"
        assert records[0]["summary"]["ordered_steps"][0]["tool_name"] == "fastqc"


class TestIWCGistRun:
    def test_run_passes_documents_to_embedding_pipeline(self, monkeypatch, tmp_path):
        pipeline = make_pipeline(monkeypatch, tmp_path)
        docs = [
            Document(
                page_content="summary",
                metadata={"source": "iwc://workflow", "workflow_name": "Workflow"},
            )
        ]
        monkeypatch.setattr(pipeline, "load_documents", lambda: docs)

        vectorstore = pipeline.run(force_rebuild=True)

        embedding_pipeline = DummyEmbeddingPipeline.instances[0]
        assert embedding_pipeline.run_calls == [(docs, True)]
        assert vectorstore == {"documents": docs, "force_rebuild": True}
