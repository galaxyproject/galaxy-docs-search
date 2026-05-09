#!/usr/bin/env python
"""
Galaxy Training Network (GTN) Pipeline

Processes GTN tutorial markdown files and creates embeddings.
Runs incrementally - only processes new/changed documents.

Usage:
    python -m pipelines.gtn --source-dir /path/to/training-material/topics
    python -m pipelines.gtn --source-dir /path/to/training-material/topics --force-rebuild
"""

import os
import logging
import argparse
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

from .common import EmbeddingPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GTNPipeline:
    """Pipeline for processing Galaxy Training Network tutorials."""

    def __init__(
        self,
        source_directory: str,
        chroma_directory: str = "./data/embeddings",
        collection_name: str = "gtn_tutorials",
        pattern: str = "**/tutorial.md",
        incremental: bool = True
    ):
        """
        Initialize GTN pipeline.

        Args:
            source_directory: Path to GTN training-material/topics directory
            chroma_directory: Path to persist Chroma database
            collection_name: Name of the Chroma collection
            pattern: Glob pattern for finding tutorial files
            incremental: Enable incremental updates
        """
        self.source_directory = Path(source_directory)
        self.pattern = pattern

        # Initialize embedding pipeline
        tracking_file = Path(chroma_directory) / ".gtn_tracking.json"
        self.pipeline = EmbeddingPipeline(
            chroma_directory=chroma_directory,
            collection_name=collection_name,
            tracking_file=str(tracking_file),
            incremental=incremental,
            splitter="markdown",
        )

        logger.info("GTN Pipeline initialized")
        logger.info(f"Source: {self.source_directory}")
        logger.info(f"Pattern: {self.pattern}")

    def load_documents(self) -> List[Document]:
        """
        Load GTN tutorial markdown files.

        Returns:
            List of Document objects
        """
        logger.info(f"Loading GTN tutorials from {self.source_directory}")

        try:
            # TextLoader (not UnstructuredMarkdownLoader) so the raw markdown
            # reaches the splitter -- Unstructured flattens `---` delimiters
            # and drops `#` characters, which would defeat the header-aware split.
            loader = DirectoryLoader(
                str(self.source_directory),
                glob=self.pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
                use_multithreading=True,
            )
            documents = loader.load()

            # Add metadata to identify source
            for doc in documents:
                doc.metadata["data_source"] = "gtn"
                doc.metadata["content_type"] = "tutorial"

            logger.info(f"Loaded {len(documents)} GTN tutorials")
            return documents
        except Exception as e:
            logger.error(f"Error loading GTN tutorials: {e}")
            raise

    def run(self, force_rebuild: bool = False, test_query: str = None):
        """
        Run the complete GTN pipeline.

        Args:
            force_rebuild: Force full rebuild ignoring incremental tracking
            test_query: Optional query to test the vectorstore
        """
        logger.info("=" * 60)
        logger.info("Starting GTN Pipeline")
        logger.info("=" * 60)

        # Load documents
        documents = self.load_documents()

        if not documents:
            logger.warning("No GTN documents found!")
            return None

        # Run embedding pipeline
        vectorstore = self.pipeline.run(documents, force_rebuild=force_rebuild)

        if vectorstore and test_query:
            logger.info(f"Testing with query: {test_query}")
            results = vectorstore.similarity_search(test_query, k=3)
            logger.info(f"Found {len(results)} similar documents")
            for i, result in enumerate(results, 1):
                source = result.metadata.get('source', 'Unknown')
                logger.info(f"  {i}. {source}")

        logger.info("=" * 60)
        logger.info("GTN Pipeline Complete")
        logger.info("=" * 60)

        return vectorstore


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Galaxy Training Network (GTN) Embedding Pipeline"
    )
    parser.add_argument(
        "--source-dir",
        default=os.getenv("GTN_SOURCE_DIRECTORY", "/path/to/training-material/topics"),
        help="Source directory containing GTN tutorial markdown files"
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings"),
        help="Directory to persist Chroma database"
    )
    parser.add_argument(
        "--collection",
        default="gtn_tutorials",
        help="Name of the Chroma collection"
    )
    parser.add_argument(
        "--pattern",
        default="**/tutorial.md",
        help="Glob pattern for finding tutorial files"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force full rebuild, ignoring incremental tracking"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental updates"
    )
    parser.add_argument(
        "--test-query",
        type=str,
        help="Run a test query after building embeddings"
    )

    args = parser.parse_args()

    # Validate source directory
    if not os.path.exists(args.source_dir):
        logger.error(f"Source directory does not exist: {args.source_dir}")
        logger.info("Please set GTN_SOURCE_DIRECTORY environment variable or use --source-dir")
        return

    try:
        pipeline = GTNPipeline(
            source_directory=args.source_dir,
            chroma_directory=args.chroma_dir,
            collection_name=args.collection,
            pattern=args.pattern,
            incremental=not args.no_incremental
        )

        pipeline.run(
            force_rebuild=args.force_rebuild,
            test_query=args.test_query
        )

    except Exception as e:
        logger.error(f"GTN Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
