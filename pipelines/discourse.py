#!/usr/bin/env python
"""
Discourse Forum Pipeline

Fetches Discourse forum posts and creates embeddings.
Runs incrementally - only processes new/changed posts.

Usage:
    python -m pipelines.discourse                    # Run all phases
    python -m pipelines.discourse --fetch-only       # Just fetch from API
    python -m pipelines.discourse --convert-only     # Just convert to documents
    python -m pipelines.discourse --embed-only       # Just create embeddings
    python -m pipelines.discourse --force-rebuild
"""

import os
import json
import logging
import argparse
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from time import sleep
from datetime import datetime
from hashlib import sha256
from dotenv import load_dotenv
import requests
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from langchain_core.documents import Document

from .common import EmbeddingPipeline
from .database import DiscourseDB

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _convert_topic_worker(topic: Dict, api_base: str, api_key: str, username: str,
                          cache_folder: str, sleep_time: float) -> Optional[Tuple[Dict, Dict]]:
    """
    Worker function for parallel topic conversion.
    Creates its own fetcher to avoid pickling issues.

    Args:
        topic: Topic metadata
        api_base: Discourse API base URL
        api_key: API key
        username: API username
        cache_folder: Cache directory path
        sleep_time: Rate limiting delay

    Returns:
        Tuple of (topic, post_data) or None if fetch failed
    """
    try:
        # Create fetcher instance for this worker
        fetcher = DiscourseFetcher(
            api_base=api_base,
            api_key=api_key,
            username=username,
            cache_folder=cache_folder,
            sleep_time=sleep_time
        )

        # Fetch post data
        post_data = fetcher.fetch_topic_posts(topic)
        if post_data:
            sleep(sleep_time)  # Rate limiting
            return (topic, post_data)
        return None
    except Exception as e:
        logger.error(f"Error converting topic {topic.get('id')}: {e}")
        return None


class DiscourseFetcher:
    """Fetches and caches posts from Discourse forum."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        username: str,
        cache_folder: str = "./data/discourse/.cache",
        sleep_time: float = 1.5
    ):
        """
        Initialize Discourse fetcher.

        Args:
            api_base: Base URL of Discourse instance
            api_key: Discourse API key
            username: Discourse API username
            cache_folder: Directory for caching API responses
            sleep_time: Delay between API calls (rate limiting)
        """
        self.api_base = api_base
        self.headers = {"Api-Key": api_key, "Api-Username": username}
        self.cache_folder = Path(cache_folder)
        self.sleep_time = sleep_time

        # Create cache folder
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Discourse Fetcher initialized")
        logger.info(f"API Base: {api_base}")

    def _read_json(self, path: Path) -> Optional[Dict]:
        """Read JSON file, handling errors."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write_json(self, data: Dict, path: Path):
        """Write JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path for URL."""
        cache_key = sha256(url.encode('utf-8')).hexdigest()
        return self.cache_folder / f"{cache_key}.json"

    def fetch_with_cache(self, url: str, use_cache: bool = True) -> Optional[Dict]:
        """Fetch data from URL with caching."""
        cache_path = self._get_cache_path(url)

        if use_cache and cache_path.exists():
            logger.debug(f"Cache hit: {url}")
            return self._read_json(cache_path)

        logger.info(f"Fetching: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            self._write_json(data, cache_path)
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def fetch_categories(self) -> List[Dict]:
        """Fetch all available categories from Discourse."""
        url = f"{self.api_base}/categories.json"
        data = self.fetch_with_cache(url)

        if not data:
            logger.warning("Could not fetch categories")
            return []

        categories = data.get("category_list", {}).get("categories", [])
        logger.info(f"Found {len(categories)} categories")

        return categories

    def fetch_topics_for_category(self, category_slug: str) -> List[Dict]:
        """Fetch all topics from a specific category."""
        url = f"{self.api_base}/c/{category_slug}.json"
        topics = []

        while url:
            data = self.fetch_with_cache(url)
            if not data:
                break

            results = data.get("topic_list", {})
            category_topics = results.get("topics", [])

            # Add category slug to each topic for tracking
            for topic in category_topics:
                topic['_category_slug'] = category_slug

            topics.extend(category_topics)

            # Check for more pages
            more_url = results.get("more_topics_url")
            if more_url:
                url = f"{self.api_base}{more_url}"
                if not url.endswith(".json"):
                    url_part, param_part = url.split("?", 1)
                    url = f"{url_part}.json?{param_part}"
                sleep(self.sleep_time)
            else:
                url = None

        return topics

    def fetch_topics(self, show_progress: bool = True) -> List[Dict]:
        """Fetch all topics from all categories."""
        # Get all categories
        categories = self.fetch_categories()

        if not categories:
            logger.warning("No categories found")
            return []

        # Fetch topics from each category
        all_topics = []
        category_iter = tqdm(categories, desc="Fetching categories", disable=not show_progress) if show_progress else categories

        for category in category_iter:
            category_slug = category.get('slug')
            category_name = category.get('name', category_slug)

            if not category_slug:
                continue

            if show_progress:
                category_iter.set_description(f"Fetching {category_name}")
            logger.info(f"Fetching topics from category: {category_name} ({category_slug})")
            topics = self.fetch_topics_for_category(category_slug)
            logger.info(f"  Found {len(topics)} topics in {category_name}")
            all_topics.extend(topics)

            # Rate limiting between categories
            sleep(self.sleep_time)

        logger.info(f"Total: Fetched {len(all_topics)} topics from {len(categories)} categories")
        return all_topics

    def fetch_topic_posts(self, topic: Dict) -> Optional[Dict]:
        """Fetch full post content for a topic."""
        url = f"{self.api_base}/t/{topic['slug']}/{topic['id']}.json"
        return self.fetch_with_cache(url)


class DiscoursePipeline:
    """Pipeline for processing Discourse forum posts."""

    def __init__(
        self,
        api_base: str = None,
        api_key: str = None,
        username: str = None,
        chroma_directory: str = "./data/embeddings",
        discourse_data_dir: str = "./data/discourse",
        collection_name: str = "discourse_posts",
        incremental: bool = True,
        workers: int = None
    ):
        """
        Initialize Discourse pipeline.

        Args:
            api_base: Discourse API base URL
            api_key: Discourse API key
            username: Discourse API username
            chroma_directory: Path to persist Chroma database
            discourse_data_dir: Path to store Discourse data
            collection_name: Name of the Chroma collection
            incremental: Enable incremental updates
            workers: Number of parallel workers for conversion (None = auto-detect)
        """
        # Get config from environment if not provided
        self.api_base = api_base or os.getenv("DISCOURSE_BASE", "https://help.galaxyproject.org")
        self.api_key = api_key or os.getenv("DISCOURSE_API_KEY")
        self.username = username or os.getenv("DISCOURSE_API_USER")

        if not self.api_key or not self.username:
            raise ValueError("DISCOURSE_API_KEY and DISCOURSE_API_USER must be set")

        self.discourse_data_dir = Path(discourse_data_dir)

        # Initialize fetcher
        self.fetcher = DiscourseFetcher(
            api_base=self.api_base,
            api_key=self.api_key,
            username=self.username,
            cache_folder=str(self.discourse_data_dir / ".cache")
        )

        # Initialize embedding pipeline
        tracking_file = Path(chroma_directory) / ".discourse_tracking.json"
        self.pipeline = EmbeddingPipeline(
            chroma_directory=chroma_directory,
            collection_name=collection_name,
            tracking_file=str(tracking_file),
            incremental=incremental
        )

        # Initialize database
        db_path = self.discourse_data_dir / "discourse.db"
        self.db = DiscourseDB(db_path)

        # Parallel processing
        self.workers = workers if workers is not None else max(1, cpu_count() - 1)

        # Keep old paths for backwards compatibility (deprecated)
        self.documents_file = self.discourse_data_dir / "documents.pkl"
        self.topics_file = self.discourse_data_dir / "topics.json"

        logger.info(f"Discourse Pipeline initialized (workers={self.workers})")

    def save_topics(self, topics: List[Dict]):
        """Save topics to database."""
        logger.info(f"Saving {len(topics)} topics to database")
        count = self.db.bulk_insert_topics(topics)
        logger.info(f"✓ Saved {count} topics")

    def load_topics(self) -> List[Dict]:
        """Load topics from database."""
        logger.info("Loading topics from database")
        topics = self.db.get_all_topics()

        if not topics:
            logger.error("No topics found in database")
            logger.error("Run with --fetch-only first to fetch topics")
            return []

        logger.info(f"Loaded {len(topics)} topics")
        # Convert back to original format (extract raw_data)
        return [topic['raw_data'] for topic in topics]

    def save_documents(self, documents: List[Document]):
        """Save documents to database."""
        logger.info(f"Saving {len(documents)} documents to database")

        # Convert LangChain Documents to database format
        db_documents = []
        for doc in documents:
            # Extract topic ID from metadata source
            source = doc.metadata.get('source', '')
            topic_id = int(source.split('/')[-1]) if source else None

            if topic_id:
                db_documents.append({
                    'topic_id': topic_id,
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

        count = self.db.bulk_insert_documents(db_documents)
        logger.info(f"✓ Saved {count} documents")

    def load_documents_from_file(self) -> List[Document]:
        """Load documents from database."""
        logger.info("Loading documents from database")
        db_documents = self.db.get_all_documents()

        if not db_documents:
            logger.error("No documents found in database")
            logger.error("Run with --convert-only first to convert topics to documents")
            return []

        # Convert database documents back to LangChain Documents
        documents = []
        for db_doc in db_documents:
            # Clean metadata: convert lists to strings (Chroma doesn't accept lists)
            metadata = db_doc['metadata'].copy()
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ", ".join(str(v) for v in value) if value else ""

            doc = Document(
                page_content=db_doc['content'],
                metadata=metadata
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def topic_to_document(self, topic: Dict, post_data: Dict) -> Document:
        """
        Convert a Discourse topic/post to a LangChain Document.

        Args:
            topic: Topic metadata (includes _category_slug)
            post_data: Full post data with replies

        Returns:
            Document object
        """
        # Combine all post content (original + replies)
        posts = post_data.get("post_stream", {}).get("posts", [])

        # Build content
        content_parts = []

        # Title and first post
        if posts:
            first_post = posts[0]
            content_parts.append(f"# {topic['title']}\n\n{first_post.get('cooked', '')}")

        # Replies
        for post in posts[1:]:
            content_parts.append(f"\n\n## Reply by {post.get('username', 'Unknown')}\n\n{post.get('cooked', '')}")

        content = "\n".join(content_parts)

        # Get category from topic metadata (added during fetching)
        category_slug = topic.get('_category_slug', 'unknown')

        # Create unique source identifier
        source = f"discourse://{category_slug}/{topic['id']}"

        # Metadata (Chroma only accepts scalar values, not lists)
        tags = topic.get('tags', [])
        metadata = {
            "source": source,
            "data_source": "discourse",
            "content_type": "forum_post",
            "topic_id": topic['id'],
            "title": topic['title'],
            "slug": topic['slug'],
            "category": category_slug,
            "url": f"{self.api_base}/t/{topic['slug']}/{topic['id']}",
            "reply_count": topic.get('reply_count', 0),
            "views": topic.get('views', 0),
            "created_at": topic.get('created_at', ''),
            "tags": ", ".join(tags) if tags else ""  # Convert list to comma-separated string
        }

        return Document(page_content=content, metadata=metadata)

    def fetch_all_topics(self, show_progress: bool = True) -> List[Dict]:
        """
        Fetch all topics from Discourse API.

        Returns:
            List of topic dictionaries
        """
        logger.info("Fetching Discourse topics from API")
        topics = self.fetcher.fetch_topics(show_progress=show_progress)

        if not topics:
            logger.warning("No topics found!")
            return []

        return topics

    def convert_topics_to_documents(self, topics: List[Dict], show_progress: bool = True) -> List[Document]:
        """
        Convert topics to documents by fetching full post content.
        Uses parallel processing if workers > 1.

        Args:
            topics: List of topic metadata
            show_progress: Show progress bar

        Returns:
            List of Document objects
        """
        logger.info(f"Converting {len(topics)} topics to documents (workers={self.workers})")

        if self.workers <= 1:
            # Serial processing (original behavior)
            return self._convert_topics_serial(topics, show_progress)
        else:
            # Parallel processing
            return self._convert_topics_parallel(topics, show_progress)

    def _convert_topics_serial(self, topics: List[Dict], show_progress: bool = True) -> List[Document]:
        """Serial topic conversion (original implementation)."""
        documents = []
        topic_iter = tqdm(topics, desc="Converting topics", unit="topic", disable=not show_progress) if show_progress else topics

        for topic in topic_iter:
            post_data = self.fetcher.fetch_topic_posts(topic)
            if post_data:
                doc = self.topic_to_document(topic, post_data)
                documents.append(doc)
            sleep(self.fetcher.sleep_time)

        logger.info(f"Converted {len(documents)} topics to documents")
        return documents

    def _convert_topics_parallel(self, topics: List[Dict], show_progress: bool = True) -> List[Document]:
        """Parallel topic conversion using multiprocessing."""
        documents = []

        # Create partial function with fixed parameters
        worker_func = partial(
            _convert_topic_worker,
            api_base=self.api_base,
            api_key=self.api_key,
            username=self.username,
            cache_folder=str(self.discourse_data_dir / ".cache"),
            sleep_time=self.fetcher.sleep_time
        )

        # Process in parallel
        with Pool(processes=self.workers) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(worker_func, topics),
                total=len(topics),
                desc=f"Converting topics ({self.workers} workers)",
                unit="topic",
                disable=not show_progress
            ))

        # Convert results to documents
        for result in results:
            if result:
                topic, post_data = result
                doc = self.topic_to_document(topic, post_data)
                documents.append(doc)

        logger.info(f"Converted {len(documents)} topics to documents")
        return documents

    def run_fetch(self, show_progress: bool = True):
        """Phase 1: Fetch topics from Discourse API."""
        logger.info("=" * 60)
        logger.info("Phase 1: Fetching Topics")
        logger.info("=" * 60)

        topics = self.fetch_all_topics(show_progress=show_progress)
        self.save_topics(topics)

        logger.info(f"✓ Fetched and saved {len(topics)} topics")
        logger.info("=" * 60)
        return topics

    def run_convert(self, show_progress: bool = True):
        """Phase 2: Convert topics to documents."""
        logger.info("=" * 60)
        logger.info("Phase 2: Converting Topics to Documents")
        logger.info("=" * 60)

        topics = self.load_topics()
        if not topics:
            return []

        documents = self.convert_topics_to_documents(topics, show_progress=show_progress)
        self.save_documents(documents)

        logger.info(f"✓ Converted and saved {len(documents)} documents")
        logger.info("=" * 60)
        return documents

    def run_embed(self, force_rebuild: bool = False, test_query: str = None):
        """Phase 3: Create embeddings."""
        logger.info("=" * 60)
        logger.info("Phase 3: Creating Embeddings")
        logger.info("=" * 60)

        documents = self.load_documents_from_file()
        if not documents:
            return None

        # Run embedding pipeline
        vectorstore = self.pipeline.run(documents, force_rebuild=force_rebuild)

        if vectorstore and test_query:
            logger.info(f"Testing with query: {test_query}")
            results = vectorstore.similarity_search(test_query, k=3)
            logger.info(f"Found {len(results)} similar documents")
            for i, result in enumerate(results, 1):
                title = result.metadata.get('title', 'Unknown')
                url = result.metadata.get('url', '')
                logger.info(f"  {i}. {title}")
                logger.info(f"     {url}")

        logger.info("=" * 60)
        return vectorstore

    def run(self, force_rebuild: bool = False, test_query: str = None, show_progress: bool = True):
        """
        Run the complete Discourse pipeline (all phases).

        Args:
            force_rebuild: Force full rebuild ignoring incremental tracking
            test_query: Optional query to test the vectorstore
            show_progress: Show progress bars
        """
        logger.info("=" * 60)
        logger.info("Starting Complete Discourse Pipeline")
        logger.info("=" * 60)

        # Phase 1: Fetch
        self.run_fetch(show_progress=show_progress)

        # Phase 2: Convert
        self.run_convert(show_progress=show_progress)

        # Phase 3: Embed
        vectorstore = self.run_embed(force_rebuild=force_rebuild, test_query=test_query)

        logger.info("=" * 60)
        logger.info("Discourse Pipeline Complete")
        logger.info("=" * 60)

        return vectorstore


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Discourse Forum Embedding Pipeline - Fetches from ALL categories"
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("DISCOURSE_BASE", "https://help.galaxyproject.org"),
        help="Discourse API base URL"
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings"),
        help="Directory to persist Chroma database"
    )
    parser.add_argument(
        "--discourse-data-dir",
        default="./data/discourse",
        help="Directory to store Discourse data and cache"
    )
    parser.add_argument(
        "--collection",
        default="discourse_posts",
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

    # Phase control
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch topics from Discourse API"
    )
    phase_group.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert cached topics to documents"
    )
    phase_group.add_argument(
        "--embed-only",
        action="store_true",
        help="Only create embeddings from cached documents"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for conversion (default: auto-detect)"
    )

    args = parser.parse_args()

    # Validate API credentials
    if not os.getenv("DISCOURSE_API_KEY") or not os.getenv("DISCOURSE_API_USER"):
        logger.error("DISCOURSE_API_KEY and DISCOURSE_API_USER environment variables must be set")
        sys.exit(1)

    try:
        pipeline = DiscoursePipeline(
            api_base=args.api_base,
            chroma_directory=args.chroma_dir,
            discourse_data_dir=args.discourse_data_dir,
            collection_name=args.collection,
            incremental=not args.no_incremental,
            workers=args.workers
        )

        show_progress = not args.no_progress

        # Run specific phase or all phases
        if args.fetch_only:
            pipeline.run_fetch(show_progress=show_progress)
        elif args.convert_only:
            pipeline.run_convert(show_progress=show_progress)
        elif args.embed_only:
            pipeline.run_embed(
                force_rebuild=args.force_rebuild,
                test_query=args.test_query
            )
        else:
            # Run all phases
            pipeline.run(
                force_rebuild=args.force_rebuild,
                test_query=args.test_query,
                show_progress=show_progress
            )

    except Exception as e:
        logger.error(f"Discourse Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
