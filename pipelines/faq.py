#!/usr/bin/env python
"""
Galaxy FAQ Pipeline

Processes Galaxy FAQ documents and creates embeddings.
Each FAQ (Q&A pair) becomes a single chunk for better retrieval.

Usage:
    python -m pipelines.faq
    python -m pipelines.faq --force-rebuild
"""

import os
import re
import logging
import argparse
from pathlib import Path
from typing import List
from dotenv import load_dotenv

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


class FAQPipeline:
    """Pipeline for processing Galaxy FAQ documents."""

    def __init__(
        self,
        faq_file: str,
        chroma_directory: str = "./data/embeddings",
        collection_name: str = "galaxy_faqs",
        incremental: bool = True
    ):
        """
        Initialize FAQ pipeline.

        Args:
            faq_file: Path to FAQ markdown file
            chroma_directory: Path to persist Chroma database
            collection_name: Name of the Chroma collection
            incremental: Enable incremental updates
        """
        self.faq_file = Path(faq_file)

        if not self.faq_file.exists():
            raise FileNotFoundError(f"FAQ file not found: {faq_file}")

        # Initialize embedding pipeline
        tracking_file = Path(chroma_directory) / ".faq_tracking.json"
        self.pipeline = EmbeddingPipeline(
            chroma_directory=chroma_directory,
            collection_name=collection_name,
            tracking_file=str(tracking_file),
            incremental=incremental,
            splitter="markdown",
        )

        logger.info("FAQ Pipeline initialized")
        logger.info(f"Source: {self.faq_file}")

    def parse_faq_file(self, content: str) -> List[dict]:
        """
        Parse FAQ markdown into individual Q&A pairs.

        Each ## heading is a question, followed by the answer.

        Args:
            content: Markdown content

        Returns:
            List of dicts with 'question', 'answer', and 'category'
        """
        faqs = []

        # Split by ## headings (questions)
        # First, extract the content after TOC
        toc_match = re.search(r'\{:toc\}', content)
        if toc_match:
            content = content[toc_match.end():]

        # Find all sections (# headers for categories)
        sections = re.split(r'\n# (.+?)\n', content)

        current_category = "General"

        for i, section in enumerate(sections):
            if i == 0:
                continue  # Skip content before first #

            # Odd indices are category names
            if i % 2 == 1:
                current_category = section.strip()
                continue

            # Even indices are section content
            # Split this section by ## (questions)
            questions = re.split(r'\n## (.+?)\n', section)

            for j in range(1, len(questions), 2):
                if j + 1 < len(questions):
                    question = questions[j].strip()
                    answer = questions[j + 1].strip()

                    # Clean up answer (remove multiple newlines, etc.)
                    answer = re.sub(r'\n\n+', '\n\n', answer)

                    faqs.append({
                        'question': question,
                        'answer': answer,
                        'category': current_category
                    })

        logger.info(f"Parsed {len(faqs)} FAQs from file")
        return faqs

    def create_faq_documents(self, faqs: List[dict]) -> List[Document]:
        """
        Convert FAQ dictionaries to LangChain Documents.

        Args:
            faqs: List of FAQ dictionaries

        Returns:
            List of Document objects
        """
        documents = []

        for i, faq in enumerate(faqs):
            # Create content with question and answer
            content = f"Q: {faq['question']}\n\nA: {faq['answer']}"

            # Metadata
            metadata = {
                "source": str(self.faq_file),
                "data_source": "gtn_faq",
                "content_type": "faq",
                "category": faq['category'],
                "question": faq['question'],
                "faq_index": i
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def load_documents(self) -> List[Document]:
        """
        Load and parse FAQ file into documents.

        Returns:
            List of Document objects (one per FAQ)
        """
        logger.info(f"Loading FAQ file: {self.faq_file}")

        try:
            with open(self.faq_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse into Q&A pairs
            faqs = self.parse_faq_file(content)

            # Convert to documents
            documents = self.create_faq_documents(faqs)

            logger.info(f"Created {len(documents)} FAQ documents")
            return documents

        except Exception as e:
            logger.error(f"Error loading FAQ file: {e}")
            raise

    def run(self, force_rebuild: bool = False, test_query: str = None):
        """
        Run the complete FAQ pipeline.

        Args:
            force_rebuild: Force full rebuild ignoring incremental tracking
            test_query: Optional query to test the vectorstore
        """
        logger.info("=" * 60)
        logger.info("Starting FAQ Pipeline")
        logger.info("=" * 60)

        # Load documents
        documents = self.load_documents()

        if not documents:
            logger.warning("No FAQ documents found!")
            return None

        # IMPORTANT: For FAQs, each Q&A pair should ideally remain as a single chunk.
        # Since FAQs are typically short (Q + A), they should fit within the
        # default chunk size (512 tokens). The pipeline will handle them as-is.

        # Run embedding pipeline
        vectorstore = self.pipeline.run(documents, force_rebuild=force_rebuild)

        if vectorstore and test_query:
            logger.info(f"Testing with query: {test_query}")
            results = vectorstore.similarity_search(test_query, k=3)
            logger.info(f"Found {len(results)} similar FAQs")
            for i, result in enumerate(results, 1):
                question = result.metadata.get('question', 'Unknown')
                category = result.metadata.get('category', 'Unknown')
                logger.info(f"  {i}. [{category}] {question}")

        logger.info("=" * 60)
        logger.info("FAQ Pipeline Complete")
        logger.info("=" * 60)

        return vectorstore


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Galaxy FAQ Embedding Pipeline"
    )
    parser.add_argument(
        "--faq-file",
        default=os.path.join(
            os.path.dirname(os.getenv("GTN_SOURCE_DIRECTORY", "/Users/dannon/work/training-material/topics").rstrip("/")),
            "faq.md"
        ),
        help="Path to FAQ markdown file"
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings"),
        help="Directory to persist Chroma database"
    )
    parser.add_argument(
        "--collection",
        default="galaxy_faqs",
        help="Name of the Chroma collection"
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

    # Validate FAQ file
    if not os.path.exists(args.faq_file):
        logger.error(f"FAQ file does not exist: {args.faq_file}")
        logger.info("Please set GTN_SOURCE_DIRECTORY or use --faq-file")
        return

    try:
        pipeline = FAQPipeline(
            faq_file=args.faq_file,
            chroma_directory=args.chroma_dir,
            collection_name=args.collection,
            incremental=not args.no_incremental
        )

        pipeline.run(
            force_rebuild=args.force_rebuild,
            test_query=args.test_query
        )

    except Exception as e:
        logger.error(f"FAQ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
