#!/usr/bin/env python
"""
Modern embedding pipeline for Galaxy Training Network tutorials.
Uses Chroma vector database and latest OpenAI SDK.
Supports incremental updates - only processes new/changed documents.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
from hashlib import sha256
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentTracker:
    """Tracks processed documents to enable incremental updates."""

    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self.data = self._load()

    def _load(self) -> Dict:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted tracking file, starting fresh")
                return {"documents": {}, "last_run": None}
        return {"documents": {}, "last_run": None}

    def _save(self):
        """Save tracking data to file."""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return sha256(content.encode('utf-8')).hexdigest()

    def is_changed(self, path: str, content: str) -> bool:
        """Check if document has changed since last processing."""
        current_hash = self.compute_hash(content)
        stored_data = self.data["documents"].get(path, {})
        return stored_data.get("hash") != current_hash

    def mark_processed(self, path: str, content: str, chunk_count: int):
        """Mark document as processed."""
        self.data["documents"][path] = {
            "hash": self.compute_hash(content),
            "chunk_count": chunk_count,
            "processed_at": datetime.now().isoformat()
        }

    def get_processed_paths(self) -> Set[str]:
        """Get set of all processed document paths."""
        return set(self.data["documents"].keys())

    def remove_document(self, path: str):
        """Remove document from tracking."""
        if path in self.data["documents"]:
            del self.data["documents"][path]

    def update_last_run(self):
        """Update the last run timestamp."""
        self.data["last_run"] = datetime.now().isoformat()
        self._save()

    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            "total_documents": len(self.data["documents"]),
            "last_run": self.data["last_run"],
            "total_chunks": sum(d.get("chunk_count", 0) for d in self.data["documents"].values())
        }


class EmbeddingPipeline:
    """Pipeline for processing markdown documents and creating embeddings."""

    def __init__(
        self,
        source_directory: str,
        chroma_directory: str = "./chroma_data",
        collection_name: str = "gtn_tutorials",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model: str = "text-embedding-3-small",
        incremental: bool = True
    ):
        """
        Initialize the embedding pipeline.

        Args:
            source_directory: Path to directory containing markdown files
            chroma_directory: Path to persist Chroma database
            collection_name: Name of the Chroma collection
            chunk_size: Size of text chunks in tokens (default: 512)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
            model: OpenAI embedding model to use
            incremental: Enable incremental updates (default: True)
        """
        self.source_directory = Path(source_directory)
        self.chroma_directory = Path(chroma_directory)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.incremental = incremental

        # Initialize document tracker
        tracking_file = self.chroma_directory / ".document_tracking.json"
        self.tracker = DocumentTracker(tracking_file)

        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=self.model)

        logger.info(f"Initialized pipeline with model: {self.model}")
        logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"Incremental updates: {'enabled' if incremental else 'disabled'}")

    def load_documents(self, pattern: str = "**/tutorial.md") -> List[Document]:
        """
        Load markdown documents from source directory.

        Args:
            pattern: Glob pattern for finding files

        Returns:
            List of Document objects
        """
        logger.info(f"Loading documents from {self.source_directory}")

        try:
            loader = DirectoryLoader(
                str(self.source_directory),
                glob=pattern,
                loader_cls=UnstructuredMarkdownLoader,
                show_progress=True,
                use_multithreading=True
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def filter_changed_documents(self, documents: List[Document]) -> tuple[List[Document], List[str]]:
        """
        Filter documents to only those that are new or changed.

        Args:
            documents: List of all documents

        Returns:
            Tuple of (changed_documents, removed_paths)
        """
        if not self.incremental:
            logger.info("Incremental mode disabled, processing all documents")
            return documents, []

        changed_docs = []
        current_paths = set()

        for doc in documents:
            path = str(doc.metadata.get("source", ""))
            current_paths.add(path)
            content = doc.page_content

            if self.tracker.is_changed(path, content):
                changed_docs.append(doc)
                logger.debug(f"Changed: {path}")

        # Find removed documents
        removed_paths = self.tracker.get_processed_paths() - current_paths

        logger.info(f"Incremental update: {len(changed_docs)} changed, {len(removed_paths)} removed")
        return changed_docs, list(removed_paths)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using recursive character splitting.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        logger.info("Splitting documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )

        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def remove_old_chunks(self, vectorstore: Chroma, removed_paths: List[str]):
        """
        Remove chunks for deleted/removed documents.

        Args:
            vectorstore: Chroma vectorstore instance
            removed_paths: List of document paths that were removed
        """
        if not removed_paths:
            return

        logger.info(f"Removing chunks for {len(removed_paths)} deleted documents")

        try:
            # Chroma doesn't have a direct "delete by metadata" so we need to:
            # 1. Query for documents with matching source paths
            # 2. Delete those IDs
            for path in removed_paths:
                # Get all chunks from this document
                results = vectorstore.get(where={"source": path})
                if results and results.get("ids"):
                    vectorstore.delete(ids=results["ids"])
                    logger.info(f"Removed {len(results['ids'])} chunks from {path}")

                # Remove from tracker
                self.tracker.remove_document(path)

        except Exception as e:
            logger.warning(f"Error removing old chunks: {e}")

    def create_or_update_vectorstore(
        self,
        documents: List[Document],
        removed_paths: List[str],
        batch_size: int = 100
    ) -> Chroma:
        """
        Create or update Chroma vectorstore with documents.

        Args:
            documents: List of Document objects to embed
            removed_paths: List of paths to remove
            batch_size: Number of documents to process at once

        Returns:
            Chroma vectorstore instance
        """
        logger.info(f"Updating vectorstore in {self.chroma_directory}")

        try:
            # Load or create vectorstore
            if self.chroma_directory.exists() and (self.chroma_directory / "chroma.sqlite3").exists():
                logger.info("Loading existing vectorstore")
                vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.chroma_directory)
                )
            else:
                logger.info("Creating new vectorstore")
                vectorstore = None

            # Remove old chunks for deleted documents
            if removed_paths and vectorstore:
                self.remove_old_chunks(vectorstore, removed_paths)

            # Add new/changed documents
            if documents:
                logger.info(f"Processing {len(documents)} changed documents")

                # Split into chunks
                chunks = self.split_documents(documents)

                # Process documents in batches with progress bar
                for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
                    batch = chunks[i:i + batch_size]

                    if vectorstore is None:
                        # Create new vectorstore with first batch
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            collection_name=self.collection_name,
                            persist_directory=str(self.chroma_directory)
                        )
                    else:
                        # Add subsequent batches
                        vectorstore.add_documents(batch)

                # Update tracker for processed documents
                for doc in documents:
                    path = str(doc.metadata.get("source", ""))
                    # Count chunks for this document
                    doc_chunks = [c for c in chunks if c.metadata.get("source") == path]
                    self.tracker.mark_processed(path, doc.page_content, len(doc_chunks))

                logger.info(f"Successfully updated vectorstore with {len(chunks)} chunks")
            elif vectorstore:
                logger.info("No new documents to process")
            else:
                logger.warning("No vectorstore exists and no documents to process")
                return None

            return vectorstore

        except Exception as e:
            logger.error(f"Error updating vectorstore: {e}")
            raise

    def run(self, pattern: str = "**/tutorial.md", force_rebuild: bool = False) -> Chroma:
        """
        Run the complete embedding pipeline.

        Args:
            pattern: Glob pattern for finding files
            force_rebuild: Force full rebuild ignoring incremental tracking

        Returns:
            Chroma vectorstore instance
        """
        logger.info("Starting embedding pipeline")

        if force_rebuild:
            logger.info("Force rebuild enabled - processing all documents")
            self.incremental = False

        # Load all documents
        all_documents = self.load_documents(pattern)

        if not all_documents:
            logger.warning("No documents found!")
            return None

        # Filter to changed documents (if incremental)
        changed_documents, removed_paths = self.filter_changed_documents(all_documents)

        if not changed_documents and not removed_paths:
            logger.info("No changes detected, vectorstore is up to date")
            # Load existing vectorstore for return
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.chroma_directory)
            )
            return vectorstore

        # Create or update vectorstore
        vectorstore = self.create_or_update_vectorstore(changed_documents, removed_paths)

        # Update tracking
        self.tracker.update_last_run()

        # Show stats
        stats = self.tracker.get_stats()
        logger.info(f"Pipeline complete! Stats: {stats}")

        return vectorstore


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Generate embeddings for markdown documents using Chroma"
    )
    parser.add_argument(
        "--source-dir",
        default=os.getenv("SOURCE_DIRECTORY", "/path/to/training-material/topics"),
        help="Source directory containing markdown files"
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIRECTORY", "./chroma_data"),
        help="Directory to persist Chroma database"
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("COLLECTION_NAME", "gtn_tutorials"),
        help="Name of the Chroma collection"
    )
    parser.add_argument(
        "--pattern",
        default="**/tutorial.md",
        help="Glob pattern for finding files (default: **/tutorial.md)"
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
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
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
        logger.info("Please set SOURCE_DIRECTORY environment variable or use --source-dir")
        return

    try:
        # Initialize and run pipeline
        pipeline = EmbeddingPipeline(
            source_directory=args.source_dir,
            chroma_directory=args.chroma_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model="text-embedding-3-small",
            incremental=not args.no_incremental
        )

        vectorstore = pipeline.run(
            pattern=args.pattern,
            force_rebuild=args.force_rebuild
        )

        if vectorstore and args.test_query:
            # Run test query
            logger.info(f"Testing with query: {args.test_query}")
            results = vectorstore.similarity_search(args.test_query, k=3)
            logger.info(f"Found {len(results)} similar documents")
            for i, result in enumerate(results, 1):
                source = result.metadata.get('source', 'Unknown')
                logger.info(f"{i}. {source}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
