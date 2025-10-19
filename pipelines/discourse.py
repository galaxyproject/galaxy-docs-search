#!/usr/bin/env python
"""
Discourse Forum Pipeline

Fetches Discourse forum posts and creates embeddings.
Runs incrementally - only processes new/changed posts.

Usage:
    python -m pipelines.discourse
    python -m pipelines.discourse --force-rebuild
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from time import sleep
from datetime import datetime
from hashlib import sha256
from dotenv import load_dotenv
import requests

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

    def fetch_topics(self) -> List[Dict]:
        """Fetch all topics from all categories."""
        # Get all categories
        categories = self.fetch_categories()

        if not categories:
            logger.warning("No categories found")
            return []

        # Fetch topics from each category
        all_topics = []
        for category in categories:
            category_slug = category.get('slug')
            category_name = category.get('name', category_slug)

            if not category_slug:
                continue

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
        incremental: bool = True
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

        logger.info("Discourse Pipeline initialized")

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

        # Metadata
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
            "tags": topic.get('tags', [])
        }

        return Document(page_content=content, metadata=metadata)

    def load_documents(self) -> List[Document]:
        """
        Fetch Discourse posts and convert to documents.

        Returns:
            List of Document objects
        """
        logger.info("Fetching Discourse topics and posts")

        # Fetch all topics
        topics = self.fetcher.fetch_topics()

        if not topics:
            logger.warning("No topics found!")
            return []

        # Convert each topic to a document
        documents = []
        for topic in topics:
            post_data = self.fetcher.fetch_topic_posts(topic)
            if post_data:
                doc = self.topic_to_document(topic, post_data)
                documents.append(doc)
            sleep(self.fetcher.sleep_time)

        logger.info(f"Converted {len(documents)} topics to documents")
        return documents

    def run(self, force_rebuild: bool = False, test_query: str = None):
        """
        Run the complete Discourse pipeline.

        Args:
            force_rebuild: Force full rebuild ignoring incremental tracking
            test_query: Optional query to test the vectorstore
        """
        logger.info("=" * 60)
        logger.info("Starting Discourse Pipeline")
        logger.info("=" * 60)

        # Load documents (fetch and convert)
        documents = self.load_documents()

        if not documents:
            logger.warning("No Discourse documents found!")
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
            incremental=not args.no_incremental
        )

        pipeline.run(
            force_rebuild=args.force_rebuild,
            test_query=args.test_query
        )

    except Exception as e:
        logger.error(f"Discourse Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
