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
import re
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
    List
)
from dataclasses import dataclass, field
from hashlib import md5
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

from .common import EmbeddingPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Tutorial:
    """Represents a GTN tutorial with all its metadata."""

    topic: str
    tutorial: str
    title: str
    description: str = ""
    url: str = ""
    difficulty: str = "intermediate"
    hands_on: bool = True
    time_estimation: str = ""
    content: str = ""
    questions: str = ""
    objectives: str = ""
    key_points: str = ""
    tools: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    content_hash: str = ""
    last_modified: str = ""
    gtn_commit: str = ""
    zenodo_link: str = ""
    workflows: list[str] = field(default_factory=list)


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
            chunk_size=1000,
            chunk_overlap=50
        )

        logger.info(f"GTN Pipeline initialized")
        logger.info(f"Source: {self.source_directory}")
        logger.info(f"Pattern: {self.pattern}")

    def parse_tutorial(self, tutorial_file: Path, content: str, topic: str, tutorial_name: str) -> Optional[Tutorial]:
        """Parse a tutorial markdown file, returning None on failure."""
        try:
            #with open(tutorial_file, encoding="utf-8") as f:
            #    content = f.read()

            # Extract frontmatter (YAML between --- markers)
            frontmatter = {}
            if content.startswith("---"):
                try:
                    end_index = content.index("---", 3)
                    yaml_content = content[3:end_index]
                    frontmatter = self.parse_yaml_simple(yaml_content)
                    # Remove frontmatter from content
                    content = content[end_index + 3 :].strip()
                except ValueError:
                    pass

            # Extract key sections from content
            questions = self.extract_section(content, "questions")
            objectives = self.extract_section(content, "objectives")
            key_points = self.extract_section(content, "keypoints") or self.extract_section(content, "key_points")

            # Build tutorial URL
            base_url = "https://training.galaxyproject.org/training-material"
            url = f"{base_url}/topics/{topic}/tutorials/{tutorial_name}/tutorial.html"

            # Get file stats
            stat = tutorial_file.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Calculate content hash
            content_bytes = content.encode("utf-8")
            content_hash = md5(content_bytes).hexdigest()

            # Create tutorial object
            tutorial = Tutorial(
                topic=topic,
                tutorial=tutorial_name,
                title=frontmatter.get("title", tutorial_name.replace("-", " ").title()),
                description=frontmatter.get("description", ""),
                url=url,
                difficulty=str(frontmatter.get("level", "intermediate")).lower(),
                hands_on=frontmatter.get("hands_on", True) not in (False, "false", "False"),
                time_estimation=frontmatter.get("time_estimation", ""),
                content=content[:50000],  # Limit content size
                questions=questions,
                objectives=objectives,
                key_points=key_points,
                tools=self.extract_list(frontmatter.get("tools", [])),
                requirements=self.extract_list(frontmatter.get("requirements", [])),
                tags=self.extract_list(frontmatter.get("tags", [])),
                content_hash=content_hash,
                last_modified=last_modified,
                zenodo_link=str(frontmatter.get("zenodo_link", "") or ""),
            )

            return tutorial

        except (OSError, ValueError, KeyError) as e:
            logging.log.warning(f"Error parsing tutorial {tutorial_file}: {e}")
            return None
        
    def parse_yaml_simple(self, yaml_content: str) -> dict[str, Any]:
        """Simple YAML frontmatter parser (no external dependencies)."""
        result: dict[str, Any] = {}
        current_list: Optional[list[str]] = None

        for line in yaml_content.split("\n"):
            line = line.rstrip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle list items
            if line.startswith("  - ") or line.startswith("- "):
                if current_list is not None:
                    item = line.strip("- ").strip()
                    if item:
                        current_list.append(item)
                continue

            # Handle key-value pairs (only top-level keys, not indented)
            if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                parts = line.split(":", 1)
                key = parts[0].strip()
                str_value = parts[1].strip() if len(parts) > 1 else ""

                # Remove quotes if present
                was_quoted = False
                if str_value.startswith('"') and str_value.endswith('"'):
                    str_value = str_value[1:-1]
                    was_quoted = True
                elif str_value.startswith("'") and str_value.endswith("'"):
                    str_value = str_value[1:-1]
                    was_quoted = True

                # Check for boolean values
                parsed_value: Union[str, bool, list[str]]
                if str_value.lower() == "true":
                    parsed_value = True
                elif str_value.lower() == "false":
                    parsed_value = False
                elif str_value == "" and not was_quoted:
                    # Unquoted empty value — this might be a list header
                    current_list = []
                    result[key] = current_list
                    continue
                else:
                    parsed_value = str_value

                result[key] = parsed_value
                current_list = None

        return result
    
    def extract_list(self, value: Any) -> list[str]:
        """Extract a list from various input formats."""
        if isinstance(value, list):
            return [str(item) for item in value]
        elif isinstance(value, str):
            return [value]
        else:
            return []
        
    def extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from markdown content."""
        escaped = re.escape(section_name)
        patterns = [
            rf"{{:\s*\.{escaped}.*?}}(.*?)(?:{{:|^#|\Z)",
            rf">{{\%\s*icon\s+{escaped}.*?\%}}.*?\n(.*?)(?:^#|\Z)",
            rf"<{escaped}.*?>(.*?)</{escaped}>",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Clean up the text
                text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
                text = re.sub(r"{%.*?%}", "", text)
                text = re.sub(r"{:.*?}", "", text)
                return text.strip() #[:1000]  # Limit section size

        return ""
    
    def clean_markdown_for_embeddings(self, text):
        # Remove fenced code blocks
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

        # Remove inline code
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Markdown links
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Images
        text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

        # Headers
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

        # Bold/italic
        text = re.sub(r"[*_~]", "", text)

        # HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    
    def create_document(self, tutorial: Tutorial) -> Document:
        """Create LlamaIndex documents from tutorials and FAQs."""
        # Combine relevant text for embedding
        text_parts = [
            tutorial.title,
            tutorial.description,
            tutorial.content,
            tutorial.questions,
            tutorial.objectives,
            tutorial.key_points,
        ]
        text = "\n".join(part for part in text_parts if part)

        #text = self.clean_markdown_for_embeddings(text)

        metadata = {
            "type": "tutorial",
            "topic": tutorial.topic,
            "tutorial": tutorial.tutorial,
            "title": tutorial.title,
            "url": tutorial.url,
            "difficulty": tutorial.difficulty,
            "hands_on": tutorial.hands_on,
            "time_estimation": tutorial.time_estimation,
            "tools": json.dumps(tutorial.tools),
            "requirements": json.dumps(tutorial.requirements),
            "tags": json.dumps(tutorial.tags),
            "content_hash": tutorial.content_hash,
            "last_modified": tutorial.last_modified,
            "zenodo_link": tutorial.zenodo_link,
        }

        doc = Document(
            page_content=text,
            metadata=metadata,
            id_=f"tutorial_{tutorial.topic}_{tutorial.tutorial}"
        )
        return doc

    def load_documents(self) -> List[Document]:
        """
        Load GTN tutorial markdown files.

        Returns:
            List of Document objects
        """
        logger.info(f"Loading GTN tutorials from {self.source_directory}")

        try:
            loader = DirectoryLoader(
                str(self.source_directory),
                glob=self.pattern,
                loader_cls=UnstructuredMarkdownLoader,
                show_progress=True,
                use_multithreading=True
            )
            documents = loader.load()

            parsed_documents = []

            # Add metadata to identify source
            for doc in documents:
                src = doc.metadata.get("source", "")
                topic, tutorial = self.extract_names(src)
                tutorial = self.parse_tutorial(Path(src), doc.page_content, topic, tutorial)
                updated_doc = self.create_document(tutorial)
                updated_doc.metadata["source"] = src
                updated_doc.metadata["data_source"] = "gtn"
                updated_doc.metadata["content_type"] = "tutorial"
                parsed_documents.append(updated_doc)
            logger.info(f"Loaded {len(parsed_documents)} GTN tutorials")
            return parsed_documents
        except Exception as e:
            logger.error(f"Error loading GTN tutorials: {e}")
            raise

    def extract_names(self, tutorial_path: str) -> tuple[str, str]:
        p = Path(tutorial_path)

        # Extract parts
        parts = p.parts

        # Find topic name
        topic_index = parts.index("topics") + 1
        topic_name = parts[topic_index]

        # Find tutorial name
        tutorial_index = parts.index("tutorials") + 1
        tutorial_name = parts[tutorial_index]

        print("Topic name:", topic_name)
        print("Tutorial name:", tutorial_name)

        return topic_name, tutorial_name

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
            #results = vectorstore.similarity_search(test_query, k=1)
            results_with_scores = vectorstore.similarity_search_with_score(test_query, k=5)
            logger.info(f"Found {len(results_with_scores)} similar documents")
            for i, (result, score) in enumerate(results_with_scores, 1):
                source = result.metadata.get('url', 'Unknown')
                logger.info(f"  {i}. {source}")
                logger.info(f"Score  {score}")
                logger.info(f"Content  {result.page_content}...")
                logger.info("-" * 40)

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
