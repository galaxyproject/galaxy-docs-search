#!/usr/bin/env python
"""
Galaxy FAQ Pipeline

Walks the GTN training-material repo for FAQ files and embeds them.

We discover FAQs the same way the GTN Jekyll site does -- any `.md` file
whose YAML frontmatter has `layout: faq` is an FAQ. That catches all four
tiers without hard-coding paths:

    faqs/galaxy/*.md
    faqs/gtn/*.md
    topics/<topic>/faqs/*.md
    topics/<topic>/tutorials/<tut>/faqs/*.md

Usage:
    python -m pipelines.faq
    python -m pipelines.faq --source-dir /path/to/training-material
    python -m pipelines.faq --force-rebuild
"""

import os
import re
import logging
import argparse
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

import yaml
from langchain.schema import Document

from .common import EmbeddingPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)\Z", re.DOTALL)
LIQUID_TAG_RE = re.compile(r"\{%.*?%\}", re.DOTALL)
LIQUID_VAR_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)

# Locale-mirror FAQ directories that duplicate `faqs/galaxy/` in other
# languages. Skipped by default to keep retrieval results in English.
TRANSLATION_DIRS = {"galaxy-de", "galaxy-es", "galaxy-fr", "galaxy-it"}


def split_frontmatter(text: str) -> tuple[Optional[dict], str]:
    """Return (frontmatter_dict, body). frontmatter_dict is None if absent."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None, text
    if not isinstance(fm, dict):
        return None, text
    return fm, m.group(2)


def strip_liquid(text: str) -> str:
    """Drop Liquid tags ({% %}) and variables ({{ }}) so embeddings see clean prose."""
    text = LIQUID_TAG_RE.sub("", text)
    text = LIQUID_VAR_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def classify_tier(rel_path: Path) -> tuple[str, Optional[str], Optional[str]]:
    """
    Determine which FAQ tier a file belongs to.

    Returns (tier, topic, tutorial) where tier is one of:
        galaxy | gtn | topic | tutorial | other
    """
    parts = rel_path.parts
    # faqs/{galaxy,gtn,galaxy-de,...}/*.md
    if len(parts) >= 3 and parts[0] == "faqs":
        bucket = parts[1]
        if bucket == "gtn":
            return "gtn", None, None
        if bucket == "galaxy" or bucket in TRANSLATION_DIRS:
            return "galaxy", None, None
        return "other", None, None
    # topics/<topic>/faqs/*.md
    # topics/<topic>/tutorials/<tut>/faqs/*.md
    if len(parts) >= 4 and parts[0] == "topics" and "faqs" in parts:
        topic = parts[1]
        if "tutorials" in parts:
            try:
                tut_idx = parts.index("tutorials")
                tutorial = parts[tut_idx + 1]
                return "tutorial", topic, tutorial
            except (ValueError, IndexError):
                pass
        return "topic", topic, None
    return "other", None, None


def is_translation(rel_path: Path) -> bool:
    parts = rel_path.parts
    return len(parts) >= 2 and parts[0] == "faqs" and parts[1] in TRANSLATION_DIRS


class FAQPipeline:
    """Walks the GTN repo for FAQ files and embeds them."""

    def __init__(
        self,
        source_directory: str,
        chroma_directory: str = "./data/embeddings",
        collection_name: str = "galaxy_faqs",
        incremental: bool = True,
        include_translations: bool = False,
    ):
        self.source_directory = Path(source_directory)
        if not self.source_directory.is_dir():
            raise FileNotFoundError(f"FAQ source directory not found: {source_directory}")
        self.include_translations = include_translations

        tracking_file = Path(chroma_directory) / ".faq_tracking.json"
        self.pipeline = EmbeddingPipeline(
            chroma_directory=chroma_directory,
            collection_name=collection_name,
            tracking_file=str(tracking_file),
            incremental=incremental,
            splitter="markdown",
        )

        logger.info("FAQ Pipeline initialized")
        logger.info(f"Source: {self.source_directory}")
        logger.info(f"Include translations: {include_translations}")

    def iter_faq_files(self):
        """Yield (path, frontmatter, body) for every FAQ file under source dir."""
        for md_path in self.source_directory.rglob("*.md"):
            try:
                text = md_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Skipping unreadable file {md_path}: {e}")
                continue

            fm, body = split_frontmatter(text)
            if not fm or fm.get("layout") != "faq":
                continue

            # The legacy root `faq.md` has `layout: faq` but no `title:` --
            # it predates the per-file format and isn't a single FAQ.
            if not fm.get("title"):
                continue

            rel = md_path.relative_to(self.source_directory)
            if not self.include_translations and is_translation(rel):
                continue

            yield md_path, rel, fm, body

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []
        skipped_no_title = 0
        for md_path, rel, fm, body in self.iter_faq_files():
            question = str(fm.get("title", "")).strip()
            if not question:
                skipped_no_title += 1
                continue

            answer = strip_liquid(body)
            content = f"Q: {question}\n\nA: {answer}"

            tier, topic, tutorial = classify_tier(rel)
            metadata = {
                "source": str(rel),
                "data_source": "gtn_faq",
                "content_type": "faq",
                "question": question,
                "area": str(fm.get("area") or ""),
                "tier": tier,
                "topic": topic or "",
                "tutorial": tutorial or "",
            }
            if isinstance(fm.get("contributors"), list):
                metadata["contributors"] = ",".join(str(c) for c in fm["contributors"])

            documents.append(Document(page_content=content, metadata=metadata))

        logger.info(f"Loaded {len(documents)} FAQ documents")
        if skipped_no_title:
            logger.info(f"Skipped {skipped_no_title} files with empty title")
        return documents

    def run(self, force_rebuild: bool = False, test_query: str = None):
        logger.info("=" * 60)
        logger.info("Starting FAQ Pipeline")
        logger.info("=" * 60)

        documents = self.load_documents()
        if not documents:
            logger.warning("No FAQ documents found!")
            return None

        vectorstore = self.pipeline.run(documents, force_rebuild=force_rebuild)

        if vectorstore and test_query:
            logger.info(f"Testing with query: {test_query}")
            results = vectorstore.similarity_search(test_query, k=3)
            logger.info(f"Found {len(results)} similar FAQs")
            for i, result in enumerate(results, 1):
                question = result.metadata.get('question', 'Unknown')
                tier = result.metadata.get('tier', 'unknown')
                logger.info(f"  {i}. [{tier}] {question}")

        logger.info("=" * 60)
        logger.info("FAQ Pipeline Complete")
        logger.info("=" * 60)
        return vectorstore


def _default_source_dir() -> str:
    """Resolve the FAQ source directory.

    Priority: FAQ_SOURCE_DIRECTORY -> parent of GTN_SOURCE_DIRECTORY (which
    points at `.../training-material/topics`) -> a placeholder that fails the
    existence check below with a hint about which env var to set.
    """
    explicit = os.getenv("FAQ_SOURCE_DIRECTORY")
    if explicit:
        return explicit
    gtn = os.getenv("GTN_SOURCE_DIRECTORY")
    if gtn:
        # GTN_SOURCE_DIRECTORY points at `.../training-material/topics`; we
        # want the repo root one level up.
        return str(Path(gtn.rstrip("/")).parent)
    return "/path/to/training-material"


def main():
    parser = argparse.ArgumentParser(description="Galaxy FAQ Embedding Pipeline")
    parser.add_argument(
        "--source-dir",
        default=_default_source_dir(),
        help="Path to the training-material repo (walked recursively for layout:faq files)",
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings"),
    )
    parser.add_argument("--collection", default="galaxy_faqs")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-incremental", action="store_true")
    parser.add_argument(
        "--include-translations",
        action="store_true",
        help="Include localized galaxy-{de,es,fr,it} FAQ mirrors (default: English only)",
    )
    parser.add_argument("--test-query", type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        logger.error(f"FAQ source directory does not exist: {args.source_dir}")
        logger.info("Set FAQ_SOURCE_DIRECTORY or GTN_SOURCE_DIRECTORY, or pass --source-dir")
        return

    try:
        pipeline = FAQPipeline(
            source_directory=args.source_dir,
            chroma_directory=args.chroma_dir,
            collection_name=args.collection,
            incremental=not args.no_incremental,
            include_translations=args.include_translations,
        )
        pipeline.run(force_rebuild=args.force_rebuild, test_query=args.test_query)
    except Exception as e:
        logger.error(f"FAQ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
