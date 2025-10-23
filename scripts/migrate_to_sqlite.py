"""
Migration script to load existing topics.json and documents.pkl into SQLite database.

Run this once to migrate from file-based storage to SQLite.
"""

import json
import pickle
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.database import DiscourseDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_topics(db: DiscourseDB, topics_file: Path):
    """
    Migrate topics from JSON file to database.

    Args:
        db: Database instance
        topics_file: Path to topics.json
    """
    logger.info(f"Loading topics from {topics_file}")

    with open(topics_file) as f:
        topics = json.load(f)

    logger.info(f"Loaded {len(topics)} topics")

    # Bulk insert
    count = db.bulk_insert_topics(topics)

    logger.info(f"✓ Migrated {count} topics to database")
    return count


def migrate_documents(db: DiscourseDB, documents_file: Path):
    """
    Migrate documents from pickle file to database.

    Args:
        db: Database instance
        documents_file: Path to documents.pkl
    """
    logger.info(f"Loading documents from {documents_file}")

    with open(documents_file, 'rb') as f:
        langchain_documents = pickle.load(f)

    logger.info(f"Loaded {len(langchain_documents)} LangChain documents")

    # Convert LangChain Documents to database format
    documents = []
    for doc in langchain_documents:
        # Extract topic ID from metadata
        source = doc.metadata.get('source', '')
        # Source format: "discourse://category-name/topic_id"
        topic_id = int(source.split('/')[-1]) if source else None

        if topic_id:
            documents.append({
                'topic_id': topic_id,
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        else:
            logger.warning(f"Document missing topic_id in metadata: {doc.metadata}")

    logger.info(f"Prepared {len(documents)} documents for migration")

    # Bulk insert
    count = db.bulk_insert_documents(documents)

    logger.info(f"✓ Migrated {count} documents to database")
    return count


def main():
    """Run the migration."""
    logger.info("=" * 60)
    logger.info("Discourse Data Migration to SQLite")
    logger.info("=" * 60)

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "discourse"
    topics_file = data_dir / "topics.json"
    documents_file = data_dir / "documents.pkl"
    db_file = data_dir / "discourse.db"

    # Verify files exist
    if not topics_file.exists():
        logger.error(f"Topics file not found: {topics_file}")
        sys.exit(1)

    if not documents_file.exists():
        logger.error(f"Documents file not found: {documents_file}")
        sys.exit(1)

    # Check if database already exists
    if db_file.exists():
        response = input(f"Database {db_file} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            logger.info("Migration cancelled")
            sys.exit(0)
        db_file.unlink()
        logger.info(f"Deleted existing database: {db_file}")

    # Initialize database
    logger.info(f"Creating database: {db_file}")
    db = DiscourseDB(db_file)

    # Migrate topics
    logger.info("")
    logger.info("Step 1: Migrating Topics")
    logger.info("-" * 60)
    topic_count = migrate_topics(db, topics_file)

    # Migrate documents
    logger.info("")
    logger.info("Step 2: Migrating Documents")
    logger.info("-" * 60)
    doc_count = migrate_documents(db, documents_file)

    # Show stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Complete")
    logger.info("=" * 60)
    stats = db.get_stats()
    logger.info(f"Topics:     {stats['topics']}")
    logger.info(f"Documents:  {stats['documents']}")
    logger.info(f"Embeddings: {stats['embeddings']}")
    logger.info("")
    logger.info(f"Database file: {db_file}")
    logger.info(f"Database size: {db_file.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Update pipelines/discourse.py to use DiscourseDB")
    logger.info("  2. Test with: make convert")
    logger.info("  3. Run embeddings with: make embed")


if __name__ == "__main__":
    main()
