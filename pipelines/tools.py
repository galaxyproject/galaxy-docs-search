#!/usr/bin/env python
"""
Galaxy tools help-text embedding pipeline.

Fetches the Galaxy tool panel from /api/tools, extracts tool metadata and each
tool wrapper's <help> text from /api/tools/{tool_id}/raw_tool_source, writes
JSON/TSV inspection artifacts, and creates embeddings for the tool help text.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from markdown import markdown
from tqdm import tqdm

from .common import EmbeddingPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_URL = "https://usegalaxy.org/api/tools"
DEFAULT_BASE_URL = "https://usegalaxy.org"
DEFAULT_OUT_JSON = "data/tools_only.json"
DEFAULT_OUT_TABLE = "data/tools_metadata.tsv"
DEFAULT_OUT_HELP_TABLE = "data/tools_helptext.tsv"

HELP_BLOCK_RE = re.compile(
    r"<help\b[^>]*>\s*(?P<body>.*?)\s*</help>",
    flags=re.DOTALL | re.IGNORECASE,
)


class HTMLTextExtractor(HTMLParser):
    """Small HTML-to-text helper to avoid a BeautifulSoup dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


@dataclass
class GalaxyTool:
    """Normalized Galaxy tool metadata plus cleaned help text."""

    tool_id: str
    name: str
    description: str
    panel_section_id: str
    panel_section_name: str
    help_text: str = ""


def fetch_json(url: str, timeout: int = 120) -> Any:
    response = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_tool_objects(
    node: Any,
    inherited_section_id: str | None = None,
    inherited_section_name: str | None = None,
) -> Iterable[tuple[dict[str, Any], str, str]]:
    """
    Walk a Galaxy /api/tools payload, yielding Tool objects and panel section
    context. The panel commonly nests ToolSection -> elems -> Tool.
    """
    if isinstance(node, dict):
        model_class = node.get("model_class")

        if model_class == "ToolSection":
            section_id = str(node.get("id") or inherited_section_id or "")
            section_name = str(node.get("name") or inherited_section_name or "")
            elems = node.get("elems")
            if isinstance(elems, list):
                for child in elems:
                    yield from iter_tool_objects(child, section_id, section_name)
            return

        if model_class == "Tool":
            section_id = str(node.get("panel_section_id") or inherited_section_id or "")
            section_name = str(node.get("panel_section_name") or inherited_section_name or "")
            yield node, section_id, section_name
            return

        for key in ("elems", "items", "children", "tools"):
            value = node.get(key)
            if isinstance(value, list):
                for child in value:
                    yield from iter_tool_objects(child, inherited_section_id, inherited_section_name)
                return

        for value in node.values():
            yield from iter_tool_objects(value, inherited_section_id, inherited_section_name)

    elif isinstance(node, list):
        for item in node:
            yield from iter_tool_objects(item, inherited_section_id, inherited_section_name)


def extract_tools(payload: Any, limit: int | None = None) -> tuple[list[GalaxyTool], list[dict[str, Any]]]:
    """Extract and deduplicate tool metadata from a Galaxy tool panel payload."""
    tools: list[GalaxyTool] = []
    raw_tools: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for tool_obj, section_id, section_name in iter_tool_objects(payload):
        tool_id = str(tool_obj.get("id", "")).strip()
        if not tool_id or tool_id in seen_ids:
            continue

        seen_ids.add(tool_id)
        raw_tools.append(tool_obj)
        tools.append(
            GalaxyTool(
                tool_id=tool_id,
                name=str(tool_obj.get("name", "") or ""),
                description=str(tool_obj.get("description", "") or ""),
                panel_section_id=section_id,
                panel_section_name=section_name,
            )
        )

        if limit is not None and len(tools) >= limit:
            break

    tools.sort(key=lambda tool: (tool.panel_section_name, tool.name, tool.tool_id))
    return tools, raw_tools


def extract_help_block(xml_text: str) -> str:
    match = HELP_BLOCK_RE.search(xml_text)
    return match.group("body").strip() if match else ""


def markdown_to_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    parser = HTMLTextExtractor()
    parser.feed(markdown(text))
    return parser.get_text()


def fetch_tool_help_text(
    session: requests.Session,
    base_url: str,
    tool_id: str,
    timeout: int,
) -> str:
    quoted_tool_id = quote(tool_id, safe="/")
    url = f"{base_url.rstrip('/')}/api/tools/{quoted_tool_id}/raw_tool_source"
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return markdown_to_text(extract_help_block(response.text))


def fetch_all_help_texts(
    tools: list[GalaxyTool],
    base_url: str,
    timeout: int,
) -> None:
    """Mutate each tool with its cleaned help text."""
    with requests.Session() as session:
        session.headers.update({"User-Agent": "chatgxy-tool-help-scraper/1.0"})
        for tool in tqdm(tools, desc="Galaxy tool help"):
            try:
                tool.help_text = fetch_tool_help_text(session, base_url, tool.tool_id, timeout)
                if not tool.help_text:
                    logger.info("No <help> text found for tool id: %s", tool.tool_id)
            except requests.RequestException as exc:
                logger.warning("Error fetching help text for tool id %s: %s", tool.tool_id, exc)
                tool.help_text = ""


def write_tools_json(raw_tools: list[dict[str, Any]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(raw_tools, ensure_ascii=False, indent=2), encoding="utf-8")


def write_table(tools: list[GalaxyTool], out_path: Path, include_help_text: bool, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "," if fmt == "csv" else "\t"
    columns = ["tool_id", "name", "description", "panel_section_id", "panel_section_name"]
    if include_help_text:
        columns.append("help_text")

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        for tool in tools:
            row = {
                "tool_id": tool.tool_id,
                "name": tool.name,
                "description": tool.description,
                "panel_section_id": tool.panel_section_id,
                "panel_section_name": tool.panel_section_name,
            }
            if include_help_text:
                row["help_text"] = tool.help_text
            writer.writerow(row)


class GalaxyToolsPipeline:
    """Pipeline for processing Galaxy tool help documentation."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        base_url: str = DEFAULT_BASE_URL,
        chroma_directory: str = "./data/embeddings",
        collection_name: str = "galaxy_tools",
        incremental: bool = True,
        timeout: int = 120,
        help_timeout: int = 30,
        input_json: str | None = None,
        tool_limit: int | None = None,
        out_json: str = DEFAULT_OUT_JSON,
        out_table: str = DEFAULT_OUT_TABLE,
        out_help_table: str = DEFAULT_OUT_HELP_TABLE,
        table_format: str = "tsv",
    ) -> None:
        self.url = url
        self.base_url = base_url
        self.timeout = timeout
        self.help_timeout = help_timeout
        self.input_json = Path(input_json) if input_json else None
        self.tool_limit = tool_limit
        self.out_json = Path(out_json)
        self.out_table = Path(out_table)
        self.out_help_table = Path(out_help_table)
        self.table_format = table_format
        self.chroma_directory = Path(os.getenv("CHROMA_DIRECTORY", chroma_directory))
        self.collection_name = collection_name

        tracking_file = self.chroma_directory / f".{collection_name}_tracking.json"
        self.pipeline = EmbeddingPipeline(
            chroma_directory=str(self.chroma_directory),
            collection_name=collection_name,
            tracking_file=str(tracking_file),
            incremental=incremental,
            chunk_size=1000,
            chunk_overlap=100,
        )

    def load_payload(self) -> Any:
        if self.input_json:
            logger.info("Loading Galaxy tool panel payload from %s", self.input_json)
            return load_json_file(self.input_json)

        logger.info("Fetching Galaxy tool panel from %s", self.url)
        return fetch_json(self.url, timeout=self.timeout)

    def tool_to_document(self, tool: GalaxyTool) -> Document:
        content_parts = [
            f"# {tool.name or tool.tool_id}",
            self.text_field("Tool ID", tool.tool_id),
            self.text_field("Description", tool.description),
            self.text_field("Panel section", tool.panel_section_name),
        ]
        if tool.help_text:
            content_parts.extend(["\n## Help\n", tool.help_text])

        source = f"galaxy-tool://{tool.tool_id}"
        metadata = {
            "source": source,
            "data_source": "usegalaxy_tools",
            "content_type": "tool_help",
            "tool_id": tool.tool_id,
            "name": tool.name,
            "description": tool.description,
            "panel_section_id": tool.panel_section_id,
            "panel_section_name": tool.panel_section_name,
            "url": f"{self.base_url.rstrip('/')}/root?tool_id={tool.tool_id}",
            "raw_tool_source_url": f"{self.base_url.rstrip('/')}/api/tools/{quote(tool.tool_id, safe='/')}/raw_tool_source",
        }
        return Document(page_content="\n".join(part for part in content_parts if part), metadata=metadata)

    @staticmethod
    def text_field(label: str, value: str) -> str:
        value = value.strip()
        return f"{label}: {value}" if value else ""

    def load_documents(self) -> list[Document]:
        payload = self.load_payload()
        tools, raw_tools = extract_tools(payload, limit=self.tool_limit)
        logger.info("Extracted metadata for %s Galaxy tools", len(tools))

        fetch_all_help_texts(tools, base_url=self.base_url, timeout=self.help_timeout)
        logger.info("Fetched help text for %s Galaxy tools", len(tools))

        write_tools_json(raw_tools, self.out_json)
        write_table(tools, self.out_table, include_help_text=False, fmt=self.table_format)
        write_table(tools, self.out_help_table, include_help_text=True, fmt=self.table_format)
        logger.info("Wrote tool artifacts to %s, %s, and %s", self.out_json, self.out_table, self.out_help_table)

        documents = [self.tool_to_document(tool) for tool in tools]
        logger.info("Created %s Galaxy tool documents", len(documents))
        return documents

    def run(self, force_rebuild: bool = False, test_query: str | None = None):
        logger.info("=" * 60)
        logger.info("Starting Galaxy Tools Pipeline")
        logger.info("=" * 60)

        documents = self.load_documents()
        if not documents:
            logger.warning("No Galaxy tool documents found")
            return None

        vectorstore = self.pipeline.run(documents, force_rebuild=force_rebuild)

        logger.info("=" * 60)
        logger.info("Galaxy Tools Pipeline Complete")
        logger.info("=" * 60)
        return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Galaxy tools help-text embedding pipeline")
    parser.add_argument("--url", default=DEFAULT_URL, help="Galaxy /api/tools URL")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Galaxy instance base URL")
    parser.add_argument("--input-json", help="Path to a previously downloaded /api/tools JSON")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Output JSON for raw Tool objects")
    parser.add_argument("--out-table", default=DEFAULT_OUT_TABLE, help="Output TSV/CSV for tool metadata")
    parser.add_argument("--out-table-helptext", default=DEFAULT_OUT_HELP_TABLE, help="Output TSV/CSV for tool help text")
    parser.add_argument("--table-format", choices=["tsv", "csv"], default="tsv", help="Output table format")
    parser.add_argument("--chroma-directory", default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings"), help="Path to Chroma database directory")
    parser.add_argument("--collection", default="galaxy_tools", help="Name of the Chroma collection")
    parser.add_argument("--timeout", type=int, default=120, help="Tool panel HTTP timeout in seconds")
    parser.add_argument("--help-timeout", type=int, default=30, help="Tool help HTTP timeout in seconds")
    parser.add_argument("--tool-limit", type=int, help="Optional maximum number of tools to process")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of all embeddings")
    parser.add_argument("--no-incremental", action="store_true", help="Disable incremental updates")
    parser.add_argument("--test-query", help="Test query after building embeddings")
    args = parser.parse_args()

    pipeline = GalaxyToolsPipeline(
        url=args.url,
        base_url=args.base_url,
        chroma_directory=args.chroma_directory,
        collection_name=args.collection,
        incremental=not args.no_incremental,
        timeout=args.timeout,
        help_timeout=args.help_timeout,
        input_json=args.input_json,
        tool_limit=args.tool_limit,
        out_json=args.out_json,
        out_table=args.out_table,
        out_help_table=args.out_table_helptext,
        table_format=args.table_format,
    )
    pipeline.run(force_rebuild=args.force_rebuild, test_query=args.test_query)


if __name__ == "__main__":
    main()
