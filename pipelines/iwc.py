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
from typing import List, Dict
from langchain.schema import Document
from dotenv import load_dotenv

from pipelines.common import EmbeddingPipeline

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class IWCPipeline:
    """Pipeline for processing IWC workflow documentation."""

    def __init__(
        self,
        manifest_url: str = "https://iwc.galaxyproject.org/workflow_manifest.json",
        chroma_directory: str = "./data/embeddings",
        collection_name: str = "iwc_workflows",
        incremental: bool = True
    ):
        """
        Initialize IWC pipeline.

        Args:
            manifest_url: URL to workflow manifest JSON
            chroma_directory: Path to Chroma database
            collection_name: Name of Chroma collection
            incremental: Enable incremental updates
        """
        self.manifest_url = manifest_url
        self.chroma_directory = Path(chroma_directory)
        self.collection_name = collection_name

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
        annotation = definition.get('annotation', '')
        readme = workflow.get('readme', '')

        # Build content from README and annotation
        content_parts = []

        # Title
        content_parts.append(f"# {workflow_name}")

        # Description from annotation
        if annotation:
            content_parts.append(f"\n{annotation}")

        # README content
        if readme:
            content_parts.append(f"\n{readme}")

        # Step annotations (for additional context)
        steps = definition.get('steps', {})
        if steps:
            step_annotations = [
                step.get('annotation', '').strip()
                for step in steps.values()
                if step.get('annotation', '').strip()
            ]
            if step_annotations:
                content_parts.append("\n## Workflow Steps\n")
                for i, ann in enumerate(step_annotations[:10], 1):  # Limit to 10
                    content_parts.append(f"{i}. {ann}")

        page_content = "\n".join(content_parts)

        # Build metadata
        authors = workflow.get('authors', [])
        author_names = [a.get('name', '') for a in authors if a.get('name')]
        author_orcids = [a.get('orcid', '') for a in authors if a.get('orcid')]

        # Extract topic from path (e.g., "computational-chemistry")
        path_parts = path.strip('./').split('/')
        topic = path_parts[1] if len(path_parts) > 1 else 'unknown'

        metadata = {
            "source": f"iwc://{path}",
            "data_source": "iwc",
            "content_type": "workflow",
            "workflow_name": workflow_name,
            "workflow_id": workflow.get('iwcID', ''),
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
            "url": f"https://iwc.galaxyproject.org/workflows/{workflow.get('iwcID', '')}"
        }

        return Document(page_content=page_content, metadata=metadata)

    def load_documents(self) -> List[Document]:
        """
        Load and parse all workflows from manifest.

        Returns:
            List of LangChain Documents
        """
        manifest = self.fetch_manifest()

        documents = []
        for entry in manifest:
            path = entry.get('path', '')
            workflows = entry.get('workflows', [])

            for workflow in workflows:
                try:
                    doc = self.parse_workflow_to_document(workflow, path)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error parsing workflow at {path}: {e}")
                    continue

        logger.info(f"Created {len(documents)} workflow documents")
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

        # Load documents
        documents = self.load_documents()

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

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run pipeline
    pipeline = IWCPipeline(
        manifest_url=args.manifest_url,
        chroma_directory=args.chroma_directory
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
