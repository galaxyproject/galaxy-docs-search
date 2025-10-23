"""
Database module for storing and managing Discourse topics and documents.

Replaces pickle/JSON file storage with SQLite for better performance,
querying, and incremental updates.
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DiscourseDB:
    """SQLite database for Discourse topics and documents."""

    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Create database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Topics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    category_id INTEGER,
                    category_name TEXT,
                    tags TEXT,
                    created_at TIMESTAMP,
                    last_posted_at TIMESTAMP,
                    posts_count INTEGER,
                    views INTEGER,
                    like_count INTEGER,
                    raw_data TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT,
                    UNIQUE(id)
                )
            """)

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT,
                    metadata TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (topic_id) REFERENCES topics(id),
                    UNIQUE(topic_id)
                )
            """)

            # Embeddings tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    document_id INTEGER PRIMARY KEY,
                    chroma_id TEXT,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_topics_last_posted
                ON topics(last_posted_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_topics_hash
                ON topics(content_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_hash
                ON documents(content_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_processed
                ON documents(processed_at)
            """)

            logger.info(f"Database schema initialized at {self.db_path}")

    def insert_topic(self, topic: Dict[str, Any]) -> int:
        """
        Insert or update a single topic.

        Args:
            topic: Topic dictionary from Discourse API

        Returns:
            Topic ID
        """
        # Calculate content hash for change detection
        hash_content = json.dumps({
            'id': topic.get('id'),
            'title': topic.get('title'),
            'last_posted_at': topic.get('last_posted_at'),
            'posts_count': topic.get('posts_count')
        }, sort_keys=True)
        content_hash = hashlib.sha256(hash_content.encode()).hexdigest()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO topics (
                    id, title, category_id, category_name, tags,
                    created_at, last_posted_at, posts_count, views, like_count,
                    raw_data, content_hash, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                topic.get('id'),
                topic.get('title'),
                topic.get('category_id'),
                topic.get('category_name'),
                json.dumps(topic.get('tags', [])),
                topic.get('created_at'),
                topic.get('last_posted_at'),
                topic.get('posts_count'),
                topic.get('views'),
                topic.get('like_count'),
                json.dumps(topic),  # Store full raw data
                content_hash,
                datetime.now().isoformat()
            ))

        return topic['id']

    def bulk_insert_topics(self, topics: List[Dict[str, Any]]) -> int:
        """
        Bulk insert topics efficiently.

        Args:
            topics: List of topic dictionaries

        Returns:
            Number of topics inserted
        """
        logger.info(f"Bulk inserting {len(topics)} topics...")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            data = []
            for topic in topics:
                hash_content = json.dumps({
                    'id': topic.get('id'),
                    'title': topic.get('title'),
                    'last_posted_at': topic.get('last_posted_at'),
                    'posts_count': topic.get('posts_count')
                }, sort_keys=True)
                content_hash = hashlib.sha256(hash_content.encode()).hexdigest()

                data.append((
                    topic.get('id'),
                    topic.get('title'),
                    topic.get('category_id'),
                    topic.get('category_name'),
                    json.dumps(topic.get('tags', [])),
                    topic.get('created_at'),
                    topic.get('last_posted_at'),
                    topic.get('posts_count'),
                    topic.get('views'),
                    topic.get('like_count'),
                    json.dumps(topic),
                    content_hash,
                    datetime.now().isoformat()
                ))

            cursor.executemany("""
                INSERT OR REPLACE INTO topics (
                    id, title, category_id, category_name, tags,
                    created_at, last_posted_at, posts_count, views, like_count,
                    raw_data, content_hash, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)

        logger.info(f"✓ Inserted {len(topics)} topics")
        return len(topics)

    def insert_document(self, topic_id: int, content: str, metadata: Dict[str, Any]) -> int:
        """
        Insert or update a document.

        Args:
            topic_id: Associated topic ID
            content: Document text content
            metadata: Document metadata dictionary

        Returns:
            Document ID
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    topic_id, content, content_hash, metadata, processed_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                topic_id,
                content,
                content_hash,
                json.dumps(metadata),
                datetime.now().isoformat()
            ))

            return cursor.lastrowid

    def bulk_insert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Bulk insert documents efficiently.

        Args:
            documents: List of dicts with 'topic_id', 'content', 'metadata'

        Returns:
            Number of documents inserted
        """
        logger.info(f"Bulk inserting {len(documents)} documents...")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            data = []
            for doc in documents:
                content = doc['content']
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                data.append((
                    doc['topic_id'],
                    content,
                    content_hash,
                    json.dumps(doc['metadata']),
                    datetime.now().isoformat()
                ))

            cursor.executemany("""
                INSERT OR REPLACE INTO documents (
                    topic_id, content, content_hash, metadata, processed_at
                ) VALUES (?, ?, ?, ?, ?)
            """, data)

        logger.info(f"✓ Inserted {len(documents)} documents")
        return len(documents)

    def get_topic(self, topic_id: int) -> Optional[Dict[str, Any]]:
        """Get a topic by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM topics WHERE id = ?", (topic_id,))
            row = cursor.fetchone()

            if row:
                topic = dict(row)
                topic['tags'] = json.loads(topic['tags']) if topic['tags'] else []
                topic['raw_data'] = json.loads(topic['raw_data']) if topic['raw_data'] else {}
                return topic
            return None

    def get_all_topics(self) -> List[Dict[str, Any]]:
        """Get all topics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM topics ORDER BY id")
            rows = cursor.fetchall()

            topics = []
            for row in rows:
                topic = dict(row)
                topic['tags'] = json.loads(topic['tags']) if topic['tags'] else []
                topic['raw_data'] = json.loads(topic['raw_data']) if topic['raw_data'] else {}
                topics.append(topic)

            return topics

    def get_document(self, topic_id: int) -> Optional[Dict[str, Any]]:
        """Get a document by topic ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE topic_id = ?", (topic_id,))
            row = cursor.fetchone()

            if row:
                doc = dict(row)
                doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
                return doc
            return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY id")
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                doc = dict(row)
                doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
                documents.append(doc)

            return documents

    def get_topics_needing_conversion(self) -> List[Dict[str, Any]]:
        """
        Get topics that haven't been converted to documents yet,
        or where the topic has changed since last conversion.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.* FROM topics t
                LEFT JOIN documents d ON t.id = d.topic_id
                WHERE d.id IS NULL
                   OR t.content_hash != (
                       SELECT content_hash FROM topics WHERE id = t.id
                   )
                ORDER BY t.id
            """)
            rows = cursor.fetchall()

            topics = []
            for row in rows:
                topic = dict(row)
                topic['tags'] = json.loads(topic['tags']) if topic['tags'] else []
                topic['raw_data'] = json.loads(topic['raw_data']) if topic['raw_data'] else {}
                topics.append(topic)

            return topics

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM topics")
            topic_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM embeddings")
            embedding_count = cursor.fetchone()[0]

            return {
                'topics': topic_count,
                'documents': doc_count,
                'embeddings': embedding_count
            }

    def mark_embedded(self, document_id: int, chroma_id: str, model: str):
        """Mark a document as embedded in Chroma."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (
                    document_id, chroma_id, model, embedded_at
                ) VALUES (?, ?, ?, ?)
            """, (document_id, chroma_id, model, datetime.now().isoformat()))
