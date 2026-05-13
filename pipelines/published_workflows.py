"""
Published Galaxy workflow embedding pipeline.

Downloads published workflows from a Galaxy instance, summarizes each workflow
with the same structured prompt and fallback logic used by the IWC pipeline, and
creates Chroma vectors for RAG.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from tqdm import tqdm

from pipelines.iwc import DEFAULT_OPENAI_BASE_URL, IWCPipeline

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_GALAXY_URL = "https://usegalaxy.eu"
DEFAULT_OUTPUT_DIR = "data/workflows_published"
DEFAULT_SUMMARIES_FILE = "data/published_workflow_summaries.json"
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_filename(value: str) -> str:
    safe = SAFE_NAME_PATTERN.sub("_", value).strip("._")
    return safe or "workflow"


class PublishedWorkflowsPipeline(IWCPipeline):
    """Pipeline for published workflows exposed by a Galaxy instance."""

    def __init__(
        self,
        galaxy_url: str = DEFAULT_GALAXY_URL,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        timeout: int = 60,
        delay: float = 0.3,
        overwrite: bool = False,
        download_workflows: bool = True,
        chroma_directory: str = "./data/embeddings_raw",
        collection_name: str = "published_workflows",
        incremental: bool = True,
        summarize_workflows: bool = True,
        summary_model: str | None = None,
        summary_base_url: str | None = None,
        summary_api_key: str | None = None,
        summaries_file: str | Path = DEFAULT_SUMMARIES_FILE,
    ):
        self.galaxy_url = galaxy_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.delay = delay
        self.overwrite = overwrite
        self.download_workflows = download_workflows

        super().__init__(
            manifest_url=f"{self.galaxy_url}/api/workflows?published=true",
            chroma_directory=chroma_directory,
            collection_name=collection_name,
            incremental=incremental,
            summarize_workflows=summarize_workflows,
            summary_model=summary_model,
            summary_base_url=summary_base_url or os.getenv("CHAT_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            summary_api_key=summary_api_key,
            summaries_file=summaries_file,
        )

        logger.info("Published workflows pipeline initialized")
        logger.info("Galaxy URL: %s", self.galaxy_url)
        logger.info("Workflow cache: %s", self.output_dir)

    def list_published_workflows(self, session: requests.Session) -> list[dict[str, Any]]:
        response = session.get(
            f"{self.galaxy_url}/api/workflows",
            params={"published": "true"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("Published workflows endpoint did not return a list.")
        return data

    def download_workflow(self, session: requests.Session, workflow_id: str) -> object:
        response = session.get(
            f"{self.galaxy_url}/api/workflows/{workflow_id}/download",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def workflow_output_path(self, workflow: dict[str, Any]) -> Path:
        workflow_id = str(workflow["id"])
        name = str(workflow.get("name") or workflow_id)
        return self.output_dir / f"{safe_filename(name)}__{safe_filename(workflow_id)}.ga.json"

    def write_manifest(self, manifest: list[dict[str, Any]]) -> None:
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Wrote manifest: %s", manifest_path)

    def collect_published_workflows(self) -> dict[str, int]:
        """Download all published workflows and write a local cache manifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "galaxy-tool-recommendation-ai-agent",
            }
        )

        workflows = self.list_published_workflows(session)
        logger.info("Found %s published workflows", len(workflows))

        manifest: list[dict[str, Any]] = []
        saved = 0
        skipped = 0
        failed = 0

        for index, workflow in enumerate(tqdm(workflows, desc="Downloading workflows"), start=1):
            workflow_id = str(workflow["id"])
            name = str(workflow.get("name") or workflow_id)
            outfile = self.workflow_output_path(workflow)

            record: dict[str, Any] = {
                "order": index,
                "workflow_id": workflow_id,
                "name": name,
                "local_path": str(outfile.relative_to(self.output_dir)),
                "download_url": f"{self.galaxy_url}/api/workflows/{workflow_id}/download",
            }

            if outfile.exists() and not self.overwrite:
                skipped += 1
                record["status"] = "skipped"
                manifest.append(record)
                continue

            try:
                data = self.download_workflow(session, workflow_id)
                outfile.write_text(json.dumps(data, indent=2), encoding="utf-8")
                saved += 1
                record["status"] = "saved"
                if self.delay > 0:
                    time.sleep(self.delay)
            except Exception as exc:
                failed += 1
                record["status"] = "failed"
                record["error"] = str(exc)
                logger.warning("Failed workflow %s %s: %s", workflow_id, name, exc)

            manifest.append(record)

        self.write_manifest(manifest)
        return {
            "found": len(workflows),
            "saved": saved,
            "skipped": skipped,
            "failed": failed,
        }

    def read_manifest(self) -> list[dict[str, Any]]:
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} does not exist. Run without --skip-download first."
            )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError(f"{manifest_path} did not contain a manifest list")
        return data

    def parse_workflow_file_to_document(self, record: dict[str, Any]) -> Document:
        local_path = self.output_dir / str(record["local_path"])
        workflow_definition = json.loads(local_path.read_text(encoding="utf-8"))
        if not isinstance(workflow_definition, dict):
            raise RuntimeError(f"{local_path} did not contain a Galaxy workflow object")

        workflow_id = str(record.get("workflow_id") or workflow_definition.get("uuid") or local_path.stem)
        workflow_name = str(
            workflow_definition.get("name")
            or record.get("name")
            or workflow_id
        )
        annotation = str(workflow_definition.get("annotation") or workflow_definition.get("description") or "")

        workflow = {
            "id": workflow_id,
            "name": workflow_name,
            "definition": workflow_definition,
            "categories": workflow_definition.get("tags", []),
            "collections": [],
            "readme": workflow_definition.get("readme", ""),
            "updated": workflow_definition.get("update_time", ""),
        }
        workflow_summary = self.summarize_workflow(workflow, str(local_path))

        content_parts = [f"# {workflow_name}"]
        if workflow_summary:
            content_parts.append("\n## Structured Workflow Summary\n")
            content_parts.append(self.summary_to_text(workflow_summary))
        if annotation:
            content_parts.append(f"\n{annotation}")

        step_lines = self.workflow_steps_to_text(workflow_definition.get("steps", {}))
        if step_lines:
            content_parts.append("\n## Workflow Steps\n")
            content_parts.append(step_lines)

        workflow_source = f"galaxy-published-workflow://{self.galaxy_host_label()}/{workflow_id}"
        metadata = {
            "source": workflow_source,
            "data_source": "published_galaxy_workflows",
            "content_type": "workflow",
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "topic": "published_workflows",
            "path": str(local_path),
            "categories": ", ".join(str(tag) for tag in workflow_definition.get("tags", [])),
            "collections": "",
            "authors": "",
            "orcids": "",
            "license": workflow_definition.get("license", ""),
            "release": workflow_definition.get("release", ""),
            "doi": workflow_definition.get("doi", ""),
            "updated": workflow_definition.get("update_time", ""),
            "summary_model": self.summary_model if workflow_summary else "",
            "workflow_summary": json.dumps(workflow_summary, ensure_ascii=False) if workflow_summary else "",
            "url": f"{self.galaxy_url}/api/workflows/{workflow_id}/download",
            "download_url": str(record.get("download_url", "")),
            "local_path": str(local_path),
        }
        return Document(page_content="\n".join(content_parts), metadata=metadata)

    def workflow_summary_context(self, workflow: dict[str, Any], path: str) -> dict[str, Any]:
        context = super().workflow_summary_context(workflow, path)
        context["workflow_id"] = workflow.get("id", context.get("workflow_id", ""))
        context["galaxy_url"] = self.galaxy_url
        context["topic"] = "published_workflows"
        return context

    def document_to_summary_record(self, doc: Document) -> dict[str, Any] | None:
        record = super().document_to_summary_record(doc)
        if record is not None:
            record["galaxy_url"] = self.galaxy_url
            record["download_url"] = doc.metadata.get("download_url", "")
            record["local_path"] = doc.metadata.get("local_path", "")
        return record

    def load_documents(self) -> list[Document]:
        if self.download_workflows:
            stats = self.collect_published_workflows()
            logger.info(
                "Published workflow collection complete: %s saved, %s skipped, %s failed",
                stats["saved"],
                stats["skipped"],
                stats["failed"],
            )

        manifest = self.read_manifest()
        documents = []
        for record in tqdm(manifest, desc="Parsing published workflows"):
            if record.get("status") == "failed":
                continue
            try:
                documents.append(self.parse_workflow_file_to_document(record))
            except Exception as exc:
                logger.warning("Could not parse %s: %s", record.get("local_path"), exc)

        logger.info("Created %s published workflow documents", len(documents))
        self.write_summary_records(documents)
        return documents

    def run(self, force_rebuild: bool = False):
        logger.info("=" * 60)
        logger.info("Starting Published Workflows Pipeline")
        logger.info("=" * 60)

        documents = self.load_documents()
        logger.info("Loaded %s published workflow documents", len(documents))

        vectorstore = self.pipeline.run(documents, force_rebuild=force_rebuild)

        logger.info("=" * 60)
        logger.info("Published Workflows Pipeline Complete")
        logger.info("=" * 60)
        return vectorstore

    def galaxy_host_label(self) -> str:
        parsed = urlparse(self.galaxy_url)
        return safe_filename(parsed.netloc or self.galaxy_url)

    def workflow_steps_to_text(self, steps: Any) -> str:
        step_lines = []
        for step in self.extract_ordered_step_context(steps):
            step_lines.append(self.format_summary_step(step))
        return "\n".join(step_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, summarize, and embed published Galaxy workflows."
    )
    parser.add_argument(
        "--galaxy-url",
        default=DEFAULT_GALAXY_URL,
        help="Galaxy instance URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where published workflow JSON files are written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between workflow downloads in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download workflows even when a local copy already exists.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing manifest/cache and only summarize/embed.",
    )
    parser.add_argument(
        "--chroma-directory",
        default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings_raw"),
        help="Path to Chroma database directory.",
    )
    parser.add_argument(
        "--collection-name",
        default="published_workflows",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of all embeddings.",
    )
    parser.add_argument(
        "--test-query",
        help="Test query after building embeddings.",
    )
    parser.add_argument(
        "--no-summarize-workflows",
        action="store_true",
        help="Embed raw workflow metadata only; skip structured workflow summaries.",
    )
    parser.add_argument(
        "--summary-model",
        help="Chat model used to summarize workflows before embedding.",
    )
    parser.add_argument(
        "--summary-base-url",
        help="OpenAI-compatible chat API base URL used for workflow summaries.",
    )
    parser.add_argument(
        "--summary-api-key",
        help="API key used for workflow summaries.",
    )
    parser.add_argument(
        "--summaries-file",
        default=DEFAULT_SUMMARIES_FILE,
        help="JSON file where generated workflow text summaries are written.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    pipeline = PublishedWorkflowsPipeline(
        galaxy_url=args.galaxy_url,
        output_dir=args.output_dir,
        timeout=args.timeout,
        delay=args.delay,
        overwrite=args.overwrite,
        download_workflows=not args.skip_download,
        chroma_directory=args.chroma_directory,
        collection_name=args.collection_name,
        summarize_workflows=not args.no_summarize_workflows,
        summary_model=args.summary_model,
        summary_base_url=args.summary_base_url,
        summary_api_key=args.summary_api_key,
        summaries_file=args.summaries_file,
    )

    vectorstore = pipeline.run(force_rebuild=args.force_rebuild)

    if args.test_query and vectorstore is not None:
        logger.info("Testing with query: %s", args.test_query)
        results = vectorstore.similarity_search(args.test_query, k=3)
        logger.info("Found %s similar workflows", len(results))
        for i, result in enumerate(results, 1):
            logger.info("  %s. %s", i, result.metadata["workflow_name"])
            logger.info("     Source: %s", result.metadata["source"])
            logger.info("     URL: %s", result.metadata["url"])


if __name__ == "__main__":
    main()
