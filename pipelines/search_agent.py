#!/usr/bin/env python
"""
Configurable RAG agent for ChatGXY materials.

The agent retrieves context from configured Chroma collections and frames the
final LLM answer using instructions loaded from a SKILL.md file. It intentionally
uses dependencies already present in this project: Chroma, OpenAI-compatible
embeddings, and the OpenAI client.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from openai import OpenAI

if load_dotenv:
    load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_COLLECTIONS: dict[str, str] = {
    "faq": "galaxy_faqs",
    "tutorials": "gtn_tutorials",
    "workflows": "iwc_workflows",
    "community": "discourse_posts",
}

DEFAULT_TITLE_FIELDS = (
    "title",
    "question",
    "workflow_name",
    "history_name",
    "name",
    "slug",
)

DEFAULT_SOURCE_FIELDS = (
    "url",
    "source",
    "path",
    "history_url",
    "workflow_url",
    "doi",
)

DEFAULT_SKILL_TEXT = """# Galaxy Training Network (GTN) RAG Answering Skill

You are a Galaxy training specialist. Answer user questions using real retrieved
Galaxy context, not invented steps. Context may include GTN tutorials, FAQs,
workflow records, history records, community posts, or future indexed
collections.

Before framing an answer, call the available RAG tools to retrieve context. Do
not answer from memory. Use vector_search(query, top_k, collection) to search a
configured Chroma collection alias, and use read_context_file(path) only when a
local context file is explicitly supplied.

Default collection aliases are faq, tutorials, workflows, histories when
configured, and community. For definitional or quick how-to questions, search
faq first. For broad analysis questions, search tutorials first, then workflows
if the user asks for runnable workflows or the tutorial hit is weak. For
workflow-specific questions, search workflows first. For history-specific
questions, search histories first if configured. For troubleshooting, prefer
official FAQ/tutorial/workflow results before community posts.

Before answering, infer which retrieved sources are most relevant:

- Definitional or quick how-to questions should prefer FAQ results.
- Broader analysis questions should prefer GTN tutorial and workflow results.
- Tool-driven questions should prefer results whose metadata or content lists
  the requested tools.
- Histories are evidence for example completed analyses or reproducible Galaxy
  states, not a substitute for tutorial steps unless the context supports it.

Evaluate match quality:

- Vector search scores are distance-based, so lower is better.
- Treat a top vector score above about 0.5 as weak unless title, topic, metadata,
  and excerpt are clearly aligned with the user question.
- If titles or topics clearly do not match, treat the result as a miss.
- Prefer official GTN tutorials, FAQs, workflows, and histories over community
  posts when sources conflict.

On a weak match, say that you could not find a specific matching GTN tutorial or
source in the retrieved context. Point to the most relevant GTN topic landing
page if a topic slug is available, otherwise use
https://training.galaxyproject.org/training-material/. Do not invent tutorial
steps from loosely related context.

For strong matches, answer from the retrieved content only, give practical steps
when supported, and cite 1-3 relevant sources with URLs, paths, workflow names,
history names, or clear source titles.
"""


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration for retrieval and answer framing."""

    chroma_directory: str = "./data/embeddings"
    collections: dict[str, str] | None = None
    skill_file: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    context_chars: int = 2500
    local_context_chars: int = 12000
    title_fields: tuple[str, ...] = DEFAULT_TITLE_FIELDS
    source_fields: tuple[str, ...] = DEFAULT_SOURCE_FIELDS


@dataclass(frozen=True)
class ToolCall:
    """A registered tool callable."""

    name: str
    description: str
    func: Callable[..., Any]


class ToolAgent:
    """Minimal tool registry with an `agent.tool` decorator interface."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolCall] = {}

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or inner.__name__
            self.tools[tool_name] = ToolCall(
                name=tool_name,
                description=description or (inner.__doc__ or "").strip(),
                func=inner,
            )
            return inner

        if func is None:
            return decorator
        return decorator(func)

    def run_tool(self, name: str, **kwargs: Any) -> Any:
        if name not in self.tools:
            available = ", ".join(sorted(self.tools))
            raise KeyError(f"Unknown tool '{name}'. Available tools: {available}")
        return self.tools[name].func(**kwargs)


class MarkdownRAGAgent:
    """Agent that retrieves configured RAG context and writes skill-guided answers."""

    def __init__(
        self,
        chroma_directory: str = "./data/embeddings",
        collections: dict[str, str] | None = None,
        skill_file: str | None = None,
        embedding_model: str | None = None,
        chat_model: str | None = None,
        context_chars: int = 2500,
        local_context_chars: int = 12000,
        title_fields: tuple[str, ...] = DEFAULT_TITLE_FIELDS,
        source_fields: tuple[str, ...] = DEFAULT_SOURCE_FIELDS,
    ) -> None:
        self.config = AgentConfig(
            chroma_directory=chroma_directory,
            collections=collections,
            skill_file=skill_file,
            embedding_model=embedding_model,
            chat_model=chat_model,
            context_chars=context_chars,
            local_context_chars=local_context_chars,
            title_fields=title_fields,
            source_fields=source_fields,
        )
        self.chroma_directory = Path(chroma_directory)
        self.collections = collections or DEFAULT_COLLECTIONS
        self.skill_file = skill_file or os.getenv("AGENT_SKILL_FILE") or "SKILL.md"
        self.skill_text = self._load_skill_text(self.skill_file)
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDINGS_MODEL",
            "text-embedding-nomic-embed-text-v2-moe",
        )
        self.chat_model = chat_model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.context_chars = context_chars
        self.local_context_chars = local_context_chars
        self.title_fields = title_fields
        self.source_fields = source_fields
        self.agent = ToolAgent()

        self.embeddings = self._create_embeddings()
        self.vectorstores = self._load_vectorstores()
        self.chat_client = self._create_chat_client()
        self._register_tools()

    def _create_embeddings(self) -> "OpenAIEmbeddings":
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("EMBEDDINGS_API_KEY")
        base_url = os.getenv("EMBEDDINGS_BASE_URL")
        if not api_key:
            raise ValueError("EMBEDDINGS_API_KEY environment variable not set")
        if not base_url:
            raise ValueError("EMBEDDINGS_BASE_URL environment variable not set")

        return OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )

    def _create_chat_client(self) -> "OpenAI":
        from openai import OpenAI

        api_key = (
            os.getenv("CHAT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("EMBEDDINGS_API_KEY")
        )
        base_url = (
            os.getenv("CHAT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("EMBEDDINGS_BASE_URL")
        )
        if not api_key:
            raise ValueError("Set CHAT_API_KEY, OPENAI_API_KEY, or EMBEDDINGS_API_KEY")
        return OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _load_skill_text(skill_file: str | None) -> str:
        if not skill_file:
            return DEFAULT_SKILL_TEXT

        path = Path(skill_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            logger.warning("Skill file not found: %s. Falling back to default skill.", path)
            return DEFAULT_SKILL_TEXT

        return path.read_text(encoding="utf-8", errors="replace").strip() or DEFAULT_SKILL_TEXT

    def _load_vectorstores(self) -> dict[str, "Chroma"]:
        from langchain_community.vectorstores import Chroma

        if not self.chroma_directory.exists():
            raise FileNotFoundError(f"Chroma directory not found: {self.chroma_directory}")

        stores: dict[str, Chroma] = {}
        for alias, collection_name in self.collections.items():
            stores[alias] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.chroma_directory),
            )
        return stores

    def _register_tools(self) -> None:
        @self.agent.tool(
            name="vector_search",
            description="Search configured Chroma collections for relevant RAG context.",
        )
        def vector_search(query: str, top_k: int = 6, collection: str | None = None) -> list[dict[str, Any]]:
            return self.search(query=query, top_k=top_k, collection=collection)

        @self.agent.tool(
            name="read_context_file",
            description="Read one local context file and return bounded content.",
        )
        def read_context_file(path: str, max_chars: int | None = None) -> dict[str, str]:
            return self.read_context_file(path=path, max_chars=max_chars or self.local_context_chars)

    def search(
        self,
        query: str,
        top_k: int = 6,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search one or all vector collections and return serializable results."""
        aliases = [collection] if collection else list(self.vectorstores)
        unknown = [alias for alias in aliases if alias not in self.vectorstores]
        if unknown:
            available = ", ".join(sorted(self.vectorstores))
            raise ValueError(f"Unknown collection(s): {unknown}. Available: {available}")

        per_collection_k = top_k if collection else max(1, min(top_k, 3))
        results: list[dict[str, Any]] = []
        for alias in aliases:
            store = self.vectorstores[alias]
            for doc, score in store.similarity_search_with_score(query, k=per_collection_k):
                print(f"score for {alias}: {score}")
                print(f"metadata for {alias}: {doc.metadata}")
                print(f"content for {alias}: {doc.page_content}")
                print("-" * 40)
                results.append(self._format_result(alias, doc, score))

        results.sort(key=lambda item: item["score"])
        return results[:top_k]

    def read_context_file(self, path: str, max_chars: int | None = None) -> dict[str, str]:
        """Read a local context file for additional prompt context."""
        context_path = Path(path).expanduser()
        if not context_path.is_absolute():
            context_path = Path.cwd() / context_path
        context_path = context_path.resolve()

        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found: {context_path}")
        if not context_path.is_file():
            raise ValueError(f"Context path is not a file: {context_path}")

        limit = max_chars or self.local_context_chars
        content = context_path.read_text(encoding="utf-8", errors="replace")
        return {
            "path": str(context_path),
            "content": content[:limit],
            "truncated": str(len(content) > limit).lower(),
        }

    def read_markdown_file(self, path: str, max_chars: int = 12000) -> dict[str, str]:
        """Backward-compatible alias for reading a context file."""
        return self.read_context_file(path=path, max_chars=max_chars)

    def ask(
        self,
        question: str,
        context_file: str | None = None,
        markdown_file: str | None = None,
        top_k: int = 8,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Run tools, send their context to the chat model, and return the answer."""
        search_results = self.agent.run_tool(
            "vector_search",
            query=question,
            top_k=top_k,
            collection=collection,
        )
        local_context = None
        context_path = context_file or markdown_file
        if context_path:
            local_context = self.agent.run_tool("read_context_file", path=context_path)

        messages = self._build_messages(question, search_results, local_context)
        response = self.chat_client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=messages,
        )
        answer = response.choices[0].message.content or ""
        return {
            "answer": answer.strip(),
            "sources": self._sources(search_results, local_context),
            "tool_results": {
                "vector_search": search_results,
                "read_context_file": local_context,
            },
            "config": {
                "collections": self.collections,
                "skill_file": self.skill_file,
                "chat_model": self.chat_model,
                "embedding_model": self.embedding_model,
            },
        }

    def _build_messages(
        self,
        question: str,
        search_results: list[dict[str, Any]],
        local_context: dict[str, str] | None,
    ) -> list[dict[str, str]]:
        context_blocks = []
        if local_context:
            context_blocks.append(
                "Local context file:\n"
                f"Path: {local_context['path']}\n"
                f"Content:\n{local_context['content']}"
            )

        for index, result in enumerate(search_results, start=1):
            metadata = json.dumps(result["metadata"], ensure_ascii=False, sort_keys=True)
            context_blocks.append(
                f"Vector result [{index}]\n"
                f"Collection: {result['collection']}\n"
                f"Score: {result['score']:.4f}\n"
                f"Title: {result['title']}\n"
                f"Source: {result['source']}\n"
                f"Metadata: {metadata}\n"
                f"Content:\n{result['content']}"
            )

        context = "\n\n---\n\n".join(context_blocks)
        system = (
            "You are ChatGXY's configurable RAG agent. Follow the SKILL.md "
            "instructions below when framing the answer.\n\n"
            f"{self.skill_text}"
        )
        user = f"Question: {question}\n\nContext:\n{context}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _format_result(self, alias: str, doc: Any, score: float) -> dict[str, Any]:
        metadata = dict(doc.metadata)
        title = self._first_metadata_value(metadata, self.title_fields) or "Untitled"
        source = self._first_metadata_value(metadata, self.source_fields) or ""
        return {
            "collection": alias,
            "score": float(score),
            "title": str(title),
            "source": str(source),
            "metadata": metadata,
            "content": doc.page_content[: self.context_chars],
        }

    @staticmethod
    def _first_metadata_value(metadata: dict[str, Any], fields: tuple[str, ...]) -> Any:
        for field in fields:
            value = metadata.get(field)
            if value:
                return value
        return None

    @staticmethod
    def _sources(
        search_results: list[dict[str, Any]],
        local_context: dict[str, str] | None,
    ) -> list[str]:
        sources: list[str] = []
        if local_context:
            sources.append(local_context["path"])
        for result in search_results:
            label = result["source"] or result["title"]
            if label and label not in sources:
                sources.append(label)
        return sources


def load_agent_config(config_file: str | None) -> dict[str, Any]:
    """Load optional JSON agent config from disk."""
    if not config_file:
        return {}

    path = Path(config_file).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Agent config must be a JSON object: {path}")
    return config


def parse_collections(collections_json: str | None) -> dict[str, str] | None:
    """Parse a JSON collection mapping, e.g. '{"histories": "galaxy_histories"}'."""
    if not collections_json:
        return None
    parsed = json.loads(collections_json)
    if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
        raise ValueError("--collections-json must be a JSON object mapping aliases to collection names")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions with the configurable ChatGXY RAG agent")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--context-file", help="Optional local context file to include")
    parser.add_argument(
        "--markdown-file",
        help="Deprecated alias for --context-file",
    )
    parser.add_argument(
        "--skill-file",
        default=os.getenv("AGENT_SKILL_FILE", "SKILL.md"),
        help="SKILL.md file used to frame the final answer",
    )
    parser.add_argument(
        "--config-file",
        help="Optional JSON config file for collections, models, skill file, and context limits",
    )
    parser.add_argument(
        "--collections-json",
        help='JSON mapping of aliases to Chroma collections, e.g. \'{"histories":"galaxy_histories"}\'',
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIRECTORY", "./data/embeddings"),
        help="Directory containing persisted Chroma collections",
    )
    parser.add_argument(
        "--collection",
        help="Restrict vector search to one configured collection alias",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of vector results")
    parser.add_argument("--context-chars", type=int, help="Maximum characters per vector result")
    parser.add_argument("--local-context-chars", type=int, help="Maximum characters from --context-file")
    parser.add_argument("--chat-model", help="Override CHAT_MODEL")
    parser.add_argument("--embedding-model", help="Override EMBEDDINGS_MODEL")
    parser.add_argument("--json", action="store_true", help="Print JSON response")
    args = parser.parse_args()

    file_config = load_agent_config(args.config_file)
    collections = (
        parse_collections(args.collections_json)
        or file_config.get("collections")
        or DEFAULT_COLLECTIONS
    )
    if not isinstance(collections, dict):
        raise ValueError("Configured collections must be a mapping of aliases to collection names")

    context_file = args.context_file or args.markdown_file

    rag_agent = MarkdownRAGAgent(
        chroma_directory=file_config.get("chroma_directory", args.chroma_dir),
        collections=collections,
        skill_file=file_config.get("skill_file", args.skill_file),
        embedding_model=args.embedding_model or file_config.get("embedding_model"),
        chat_model=args.chat_model or file_config.get("chat_model"),
        context_chars=args.context_chars or file_config.get("context_chars", 2500),
        local_context_chars=args.local_context_chars or file_config.get("local_context_chars", 12000),
        title_fields=tuple(file_config.get("title_fields", DEFAULT_TITLE_FIELDS)),
        source_fields=tuple(file_config.get("source_fields", DEFAULT_SOURCE_FIELDS)),
    )
    result = rag_agent.ask(
        question=args.question,
        context_file=context_file,
        top_k=args.top_k,
        collection=args.collection,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source}")


if __name__ == "__main__":
    main()
