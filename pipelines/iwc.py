"""
IWC (Intergalactic Workflow Commission) workflow embedding pipeline.

Fetches workflow manifest from iwc.galaxyproject.org and creates embeddings
for workflow documentation (README, annotations, metadata).
"""

import os
import json
import logging
import argparse
import requests
from pathlib import Path
from typing import Any, List, Dict
from langchain_core.documents import Document
from dotenv import load_dotenv
from tqdm import tqdm

from pipelines.common import EmbeddingPipeline

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SUMMARIES_FILE = "data/iwc_workflow_summaries.json"


SUMMARY_PROMPT = """You are an expert bioinformatics and Galaxy workflow analyst.

Your task is to analyse the provided Galaxy workflow JSON and produce a structured summary for downstream tool recommendation for each workflow step.

IMPORTANT RULES:
- Use ONLY the provided workflow context.
- Do NOT invent tools, datatypes, or steps.
- If information is missing, return null or "unknown".
- Be concise and precise.

OUTPUT FORMAT (valid JSON only, with no Markdown code fences):

{
  "workflow_name": "",
  "workflow_purpose": "",
  "scientific_domain": "",
  "main_analysis_task": "",
  "ordered_steps": [
    {
      "step_id": "",
      "tool_name": "",
      "step_purpose": "",
      "input_datatypes": [],
      "output_datatypes": []
    }
  ],
  "summary": {}
}
"""


class IWCPipeline:
    """Pipeline for processing IWC workflow documentation."""

    def __init__(
        self,
        manifest_url: str = "https://iwc.galaxyproject.org/workflow_manifest.json",
        chroma_directory: str = "./data/embeddings",
        collection_name: str = "iwc_workflows",
        incremental: bool = True,
        summarize_workflows: bool = True,
        summary_model: str | None = None,
        summary_base_url: str | None = None,
        summary_api_key: str | None = None,
        summaries_file: str | Path = DEFAULT_SUMMARIES_FILE,
    ):
        """
        Initialize IWC pipeline.

        Args:
            manifest_url: URL to workflow manifest JSON
            chroma_directory: Path to Chroma database
            collection_name: Name of Chroma collection
            incremental: Enable incremental updates
            summarize_workflows: Add a structured workflow summary to each embedded document
            summary_model: Chat model used for LLM summaries
            summary_base_url: OpenAI-compatible chat API base URL
            summary_api_key: API key for summary generation
            summaries_file: JSON file where generated workflow summaries are written
        """
        self.manifest_url = manifest_url
        self.chroma_directory = Path(
            os.getenv("CHROMA_DIRECTORY", str(chroma_directory))
        )
        self.collection_name = collection_name
        self.summaries_file = Path(summaries_file)
        self.summarize_workflows = summarize_workflows
        self.summary_model = summary_model or os.getenv(
            "CHAT_MODEL",
            os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        )
        self.summary_api_key = summary_api_key or os.getenv(
            "CHAT_API_KEY",
            os.getenv("CHAT_API_KEY"),
        )
        self.summary_base_url = summary_base_url or os.getenv(
            "CHAT_BASE_URL",
            os.getenv("CHAT_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        )

        # Initialize embedding pipeline
        tracking_file = self.chroma_directory / f".{collection_name}_tracking.json"
        self.pipeline = EmbeddingPipeline(
            chroma_directory=str(self.chroma_directory),
            collection_name=collection_name,
            tracking_file=str(tracking_file),
            incremental=incremental
        )

        logger.info("IWC Pipeline initialized")
        logger.info(f"Manifest URL: {manifest_url}")
        logger.info(f"Workflow summaries: {'enabled' if summarize_workflows else 'disabled'}")
        logger.info(f"Summaries file: {self.summaries_file}")

    def fetch_manifest(self) -> Dict:
        """
        Fetch workflow manifest from IWC website.

        Returns:
            Parsed manifest dictionary
        """
        logger.info(f"Fetching manifest from {self.manifest_url}")

        try:
            response = requests.get(self.manifest_url, timeout=30)
            response.raise_for_status()
            manifest = response.json()

            logger.info(f"Fetched manifest with {len(manifest)} workflow directories")
            return manifest

        except requests.RequestException as e:
            logger.error(f"Failed to fetch manifest: {e}")
            raise

    def parse_workflow_to_document(
        self,
        workflow: Dict,
        path: str
    ) -> Document:
        """
        Convert workflow metadata to LangChain Document.

        Args:
            workflow: Workflow object from manifest
            path: Directory path (provides category context)

        Returns:
            LangChain Document with workflow documentation
        """
        definition = workflow.get('definition', {})

        # Extract metadata
        workflow_name = definition.get('name', workflow.get('name', 'Unnamed'))
        workflow_id = workflow.get('iwcID', '')
        annotation = definition.get('annotation', '')
        readme = workflow.get('readme', '')
        workflow_summary = self.summarize_workflow(workflow, path) if self.summarize_workflows else {}

        # Build content from structured summary, README, and annotation.
        content_parts = []

        # Title
        content_parts.append(f"# {workflow_name}")

        if workflow_summary:
            content_parts.append("\n## Structured Workflow Summary\n")
            content_parts.append(self.summary_to_text(workflow_summary))

        # Description from annotation
        if annotation:
            content_parts.append(f"\n{annotation}")

        # README content
        if readme:
            content_parts.append(f"\n{readme}")

        page_content = "\n".join(content_parts)

        # Build metadata
        authors = workflow.get('authors', [])
        author_names = [a.get('name', '') for a in authors if a.get('name')]
        author_orcids = [a.get('orcid', '') for a in authors if a.get('orcid')]

        # Extract topic from path (e.g., "computational-chemistry")
        path_parts = path.strip('./').split('/')
        topic = path_parts[1] if len(path_parts) > 1 else 'unknown'
        workflow_source = f"iwc://{path}#{workflow_id or workflow_name}"

        metadata = {
            "source": workflow_source,
            "data_source": "iwc",
            "content_type": "workflow",
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "topic": topic,
            "path": path,
            "categories": ", ".join(workflow.get('categories', [])),
            "collections": ", ".join(workflow.get('collections', [])),
            "authors": ", ".join(author_names),
            "orcids": ", ".join(author_orcids),
            "license": definition.get('license', ''),
            "release": definition.get('release', ''),
            "doi": workflow.get('doi', ''),
            "updated": workflow.get('updated', ''),
            "summary_model": self.summary_model if workflow_summary else "",
            "workflow_summary": json.dumps(workflow_summary, ensure_ascii=False) if workflow_summary else "",
            "url": f"https://iwc.galaxyproject.org/workflow/{workflow_id}"
        }

        return Document(page_content=page_content, metadata=metadata)

    def summarize_workflow(self, workflow: Dict[str, Any], path: str) -> Dict[str, Any]:
        """
        Summarize one Galaxy workflow into a compact JSON structure for RAG.

        If summary API credentials are available, this uses the expert prompt above.
        If the chat endpoint is unavailable or returns invalid JSON, it falls back to
        a deterministic summary extracted directly from the workflow definition.
        """
        workflow_context = self.workflow_summary_context(workflow, path)
        if self.summary_api_key and self.summary_base_url:
            try:
                return self.summarize_workflow_with_llm(workflow_context)
            except Exception as exc:
                workflow_name = workflow_context.get("workflow_name", "unknown")
                logger.warning("LLM summary failed for %s; using local summary: %s", workflow_name, exc)

        return self.summarize_workflow_locally(workflow_context)

    def summarize_workflow_with_llm(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured workflow summary with an OpenAI-compatible chat model."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LLM workflow summaries") from exc

        client = OpenAI(api_key=self.summary_api_key, base_url=self.summary_base_url)
        response = client.chat.completions.create(
            model=self.summary_model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(workflow_context, ensure_ascii=False),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        summary = json.loads(content)
        if not isinstance(summary, dict):
            raise RuntimeError("summary response was not a JSON object")
        return self.normalize_summary(summary, workflow_context)

    def summarize_workflow_locally(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deterministic summary using only fields present in the workflow JSON."""
        steps = []
        definition = workflow_context.get("definition", {})
        ordered_step_context = definition.get("steps", []) if isinstance(definition, dict) else []
        for step in ordered_step_context:
            steps.append({
                "step_id": step.get("step_id"),
                "tool_name": step.get("tool_name") or "unknown",
                "step_purpose": step.get("step_purpose") or "unknown",
                "input_datatypes": step.get("input_datatypes") or [],
                "output_datatypes": step.get("output_datatypes") or [],
            })

        summary = {
            "workflow_name": workflow_context.get("workflow_name") or "unknown",
            "workflow_purpose": workflow_context.get("annotation") or workflow_context.get("readme_excerpt") or "unknown",
            "scientific_domain": workflow_context.get("topic") or "unknown",
            "main_analysis_task": workflow_context.get("workflow_name") or "unknown",
            "ordered_steps": steps,
            "summary": {
                "categories": workflow_context.get("categories") or [],
                "collections": workflow_context.get("collections") or [],
                "source_path": workflow_context.get("path") or "",
            },
        }
        return self.normalize_summary(summary, workflow_context)

    def workflow_summary_context(self, workflow: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Build the bounded workflow JSON context passed to the summarizer."""
        definition = workflow.get("definition", {})
        workflow_name = definition.get("name", workflow.get("name", "Unnamed"))
        path_parts = path.strip("./").split("/")
        topic = path_parts[1] if len(path_parts) > 1 else "unknown"

        return {
            "workflow_id": workflow.get("iwcID", ""),
            "workflow_name": workflow_name,
            "annotation": definition.get("annotation", ""),
            "readme_excerpt": str(workflow.get("readme", ""))[:4000],
            "topic": topic,
            "path": path,
            "categories": workflow.get("categories", []),
            "collections": workflow.get("collections", []),
            "definition": {
                "name": definition.get("name"),
                "annotation": definition.get("annotation"),
                "license": definition.get("license"),
                "release": definition.get("release"),
                "steps": self.extract_ordered_step_context(definition.get("steps", {})),
            },
        }

    def extract_ordered_step_context(self, steps: Any) -> List[Dict[str, Any]]:
        """Extract ordered, compact step context from a Galaxy workflow definition."""
        if not isinstance(steps, dict):
            return []

        step_items = []
        for step_id, step in steps.items():
            if not isinstance(step, dict):
                continue
            step_items.append((self.step_sort_key(step_id, step), step_id, step))

        ordered_steps = []
        for _, step_id, step in sorted(step_items, key=lambda item: item[0]):
            tool_name = step.get("name") or step.get("tool_id") or step.get("tool_version")
            ordered_steps.append({
                "step_id": str(step.get("id", step_id)),
                "tool_name": tool_name,
                "step_purpose": step.get("annotation") or step.get("label") or "",
                "input_datatypes": self.extract_step_datatypes(step.get("input_connections")),
                "output_datatypes": self.extract_step_datatypes(step.get("outputs")),
            })
        return ordered_steps

    @staticmethod
    def step_sort_key(step_id: Any, step: Dict[str, Any]) -> tuple[int, str]:
        order = step.get("order_index", step.get("id", step_id))
        try:
            return (int(order), str(step_id))
        except (TypeError, ValueError):
            return (0, str(step_id))

    @staticmethod
    def extract_step_datatypes(value: Any) -> List[str]:
        """Pull datatype-like values from common Galaxy step fields."""
        datatypes = set()

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    if key in {"datatype", "format", "extension"} and child:
                        datatypes.add(str(child))
                    else:
                        visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
        return sorted(datatypes)

    def normalize_summary(self, summary: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure summaries have the expected RAG schema."""
        normalized = {
            "workflow_name": summary.get("workflow_name") or workflow_context.get("workflow_name") or "unknown",
            "workflow_purpose": summary.get("workflow_purpose") or workflow_context.get("annotation") or "unknown",
            "scientific_domain": summary.get("scientific_domain") or workflow_context.get("topic") or "unknown",
            "main_analysis_task": summary.get("main_analysis_task") or workflow_context.get("workflow_name") or "unknown",
            "ordered_steps": summary.get("ordered_steps") if isinstance(summary.get("ordered_steps"), list) else [],
            "summary": summary.get("summary") if isinstance(summary.get("summary"), dict) else {},
        }
        return normalized

    def summary_to_text(self, summary: Dict[str, Any]) -> str:
        """Convert structured summary JSON to text before Chroma embedding."""
        parts = [
            self.text_field("Workflow summary name", summary.get("workflow_name")),
            self.text_field("Workflow purpose", summary.get("workflow_purpose")),
            self.text_field("Scientific domain", summary.get("scientific_domain")),
            self.text_field("Main analysis task", summary.get("main_analysis_task")),
        ]

        ordered_steps = summary.get("ordered_steps")
        if isinstance(ordered_steps, list):
            step_lines = [
                self.format_summary_step(step)
                for step in ordered_steps
                if isinstance(step, dict)
            ]
            if step_lines:
                parts.extend(["Ordered steps:", *step_lines])

        nested_summary = summary.get("summary")
        if isinstance(nested_summary, dict) and nested_summary:
            parts.append("Summary details:")
            for key, value in nested_summary.items():
                formatted_value = self.format_key_value(key, value)
                if formatted_value:
                    parts.append(formatted_value)

        return "\n".join(part for part in parts if part)

    @staticmethod
    def text_field(label: str, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return f"{label}: {text}" if text else ""

    def format_summary_step(self, step: Dict[str, Any]) -> str:
        fields = [
            f"step_id={step.get('step_id')}" if step.get("step_id") is not None else "",
            f"tool_name={step.get('tool_name')}" if step.get("tool_name") else "",
            f"step_purpose={step.get('step_purpose')}" if step.get("step_purpose") else "",
            self.format_key_value("input_datatypes", step.get("input_datatypes")),
            self.format_key_value("output_datatypes", step.get("output_datatypes")),
        ]
        return "- " + " | ".join(field for field in fields if field)

    @staticmethod
    def format_key_value(key: str, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return f"{key}: {', '.join(str(item) for item in value)}" if value else ""
        if isinstance(value, dict):
            return f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}" if value else ""
        text = str(value).strip()
        return f"{key}: {text}" if text else ""

    def document_to_summary_record(self, doc: Document) -> Dict[str, Any] | None:
        """Convert a summarized workflow document into a persisted summary record."""
        raw_summary = doc.metadata.get("workflow_summary")
        if not raw_summary:
            return None

        try:
            summary = json.loads(raw_summary)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid workflow summary metadata for %s", doc.metadata.get("source"))
            return None

        categories = doc.metadata.get("categories", "")
        tags = [tag.strip() for tag in categories.split(",") if tag.strip()]
        text_summary = self.summary_to_text(summary)

        return {
            "workflow_id": doc.metadata.get("workflow_id", ""),
            "name": doc.metadata.get("workflow_name", ""),
            "source": doc.metadata.get("source", ""),
            "source_file": doc.metadata.get("path", ""),
            "url": doc.metadata.get("url", ""),
            "topic": doc.metadata.get("topic", ""),
            "tags": tags,
            "status": "summarised",
            "summary": summary,
            "text_summary": text_summary,
        }

    def write_summary_records(self, documents: List[Document]) -> None:
        """Write all generated workflow summaries to a JSON file."""
        if not self.summarize_workflows:
            logger.info("Workflow summaries disabled; not writing summaries file")
            return

        records = [
            record
            for doc in documents
            if (record := self.document_to_summary_record(doc)) is not None
        ]

        self.summaries_file.parent.mkdir(parents=True, exist_ok=True)
        self.summaries_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote %s workflow summaries to %s", len(records), self.summaries_file)

    def load_documents(self) -> List[Document]:
        """
        Load and parse all workflows from manifest.

        Returns:
            List of LangChain Documents
        """
        manifest = self.fetch_manifest()

        documents = []
        for j, entry in enumerate(tqdm(manifest, desc="IWC manifest entries")):
            path = entry.get('path', '')
            workflows = entry.get('workflows', [])

            for i, workflow in enumerate(
                tqdm(
                    workflows,
                    desc=f"IWC workflows {j + 1}/{len(manifest)}",
                    leave=False,
                )
            ):
                try:
                    doc = self.parse_workflow_to_document(workflow, path)
                    documents.append(doc)
                    logger.info(f"Parsed workflow {i / len(workflows) * 100:.1f}%: {doc.metadata['workflow_name']}' from {path}")
                except Exception as e:
                    logger.error(f"Error parsing workflow at {path}: {e}")
                    continue

        logger.info(f"Created {len(documents)} workflow documents")
        self.write_summary_records(documents)
        return documents

    def run(self, force_rebuild: bool = False):
        """
        Run the IWC pipeline.

        Args:
            force_rebuild: Force rebuild of all embeddings

        Returns:
            Chroma vectorstore
        """
        logger.info("=" * 60)
        logger.info("Starting IWC Pipeline")
        logger.info("=" * 60)

        logger.info("Loading workflow documents from manifest")
        # Load documents
        documents = self.load_documents()

        logger.info(f"Loaded {len(documents)} workflow documents")

        logger.info("Running embedding pipeline")
        # Run embedding pipeline
        vectorstore = self.pipeline.run(documents, force_rebuild=force_rebuild)

        logger.info("=" * 60)
        logger.info("IWC Pipeline Complete")
        logger.info("=" * 60)

        return vectorstore


def main():
    """Main entry point for IWC pipeline."""
    parser = argparse.ArgumentParser(
        description="IWC Workflow embedding pipeline"
    )
    parser.add_argument(
        "--manifest-url",
        default="https://iwc.galaxyproject.org/workflow_manifest.json",
        help="URL to workflow manifest JSON"
    )
    parser.add_argument(
        "--chroma-directory",
        default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings"),
        help="Path to Chroma database directory"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of all embeddings (ignore incremental updates)"
    )
    parser.add_argument(
        "--test-query",
        help="Test query after building embeddings"
    )
    parser.add_argument(
        "--no-summarize-workflows",
        action="store_true",
        help="Embed raw workflow metadata only; skip structured workflow summaries"
    )
    parser.add_argument(
        "--summary-model",
        help="Chat model used to summarize workflows before embedding"
    )
    parser.add_argument(
        "--summary-base-url",
        help="OpenAI-compatible chat API base URL used for workflow summaries"
    )
    parser.add_argument(
        "--summary-api-key",
        help="API key used for workflow summaries"
    )
    parser.add_argument(
        "--summaries-file",
        default=DEFAULT_SUMMARIES_FILE,
        help="JSON file where generated workflow text summaries are written"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run pipeline
    pipeline = IWCPipeline(
        manifest_url=args.manifest_url,
        chroma_directory=args.chroma_directory,
        summarize_workflows=not args.no_summarize_workflows,
        summary_model=args.summary_model,
        summary_base_url=args.summary_base_url,
        summary_api_key=args.summary_api_key,
        summaries_file=args.summaries_file,
    )

    vectorstore = pipeline.run(force_rebuild=args.force_rebuild)

    # Test query if provided
    if args.test_query:
        logger.info(f"Testing with query: {args.test_query}")
        results = vectorstore.similarity_search(args.test_query, k=3)
        logger.info(f"Found {len(results)} similar workflows")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result.metadata['workflow_name']}")
            logger.info(f"     Topic: {result.metadata['topic']}")
            logger.info(f"     Categories: {result.metadata['categories']}")


if __name__ == "__main__":
    main()
