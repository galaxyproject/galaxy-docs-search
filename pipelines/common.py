"""
Shared embedding logic for both GTN and Discourse pipelines.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Set
from hashlib import sha256
from datetime import datetime
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

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
    """
    Shared embedding pipeline logic for any document source.
    Handles incremental updates, chunking, and vector storage.
    """

    def __init__(
        self,
        chroma_directory: str,
        collection_name: str,
        tracking_file: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model: str = None,
        incremental: bool = True
    ):
        """
        Initialize the embedding pipeline.

        Args:
            chroma_directory: Path to persist Chroma database
            collection_name: Name of the Chroma collection
            tracking_file: Path to document tracking file
            chunk_size: Size of text chunks in tokens (default: 512)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
            model: Embedding model to use. If None, reads EMBEDDINGS_MODEL from
                env, falling back to the LM Studio nomic-v2-moe default.
            incremental: Enable incremental updates (default: True)
        """
        self.chroma_directory = Path(chroma_directory)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model or os.getenv(
            "EMBEDDINGS_MODEL", "text-embedding-nomic-embed-text-v2-moe"
        )
        self.incremental = incremental

        # Initialize document tracker
        self.tracker = DocumentTracker(Path(tracking_file))

        # Validate embeddings API credentials
        embeddings_api_key = os.getenv("EMBEDDINGS_API_KEY")
        embeddings_base_url = os.getenv("EMBEDDINGS_BASE_URL")

        if not embeddings_api_key:
            raise ValueError("EMBEDDINGS_API_KEY environment variable not set")
        if not embeddings_base_url:
            raise ValueError("EMBEDDINGS_BASE_URL environment variable not set")

        # Initialize embeddings using OpenAI-compatible API
        # Works with: OpenAI, SambaNova, LM Studio, or any OpenAI-compatible endpoint
        # Note: We disable tiktoken checking for custom models not in tiktoken's registry
        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=embeddings_api_key,
            openai_api_base=embeddings_base_url,
            tiktoken_enabled=False,  # Disable tiktoken to avoid DNS issues with custom models
            check_embedding_ctx_length=False  # Disable context length checking
        )

        logger.info(f"Initialized pipeline for collection: {collection_name}")
        logger.info(f"Model: {self.model}, Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        logger.info(f"Incremental updates: {'enabled' if incremental else 'disabled'}")

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
        logger.info(f"Splitting {len(documents)} documents into chunks")

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
            for path in removed_paths:
                results = vectorstore.get(where={"source": path})
                if results and results.get("ids"):
                    vectorstore.delete(ids=results["ids"])
                    logger.info(f"Removed {len(results['ids'])} chunks from {path}")

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
        logger.info(f"Updating collection '{self.collection_name}' in {self.chroma_directory}")

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

    def run(
        self,
        documents: List[Document],
        force_rebuild: bool = False
    ) -> Chroma:
        """
        Run the complete embedding pipeline.

        Args:
            documents: List of documents to process
            force_rebuild: Force full rebuild ignoring incremental tracking

        Returns:
            Chroma vectorstore instance
        """
        logger.info(f"Starting embedding pipeline for {len(documents)} documents")

        if force_rebuild:
            logger.info("Force rebuild enabled - processing all documents")
            self.incremental = False

        if not documents:
            logger.warning("No documents provided!")
            return None

        # Filter to changed documents (if incremental)
        changed_documents, removed_paths = self.filter_changed_documents(documents)

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
