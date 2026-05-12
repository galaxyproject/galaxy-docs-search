from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.md_agent import ToolAgent
from pipelines.md_agent import MarkdownRAGAgent
from pipelines.md_agent import parse_collections


class MarkdownFileOnlyAgent(MarkdownRAGAgent):
    def __init__(self) -> None:
        self.local_context_chars = 12000


def test_tool_agent_registers_and_runs_tool():
    agent = ToolAgent()

    @agent.tool(name="echo")
    def echo(value: str) -> str:
        return value

    assert "echo" in agent.tools
    assert agent.run_tool("echo", value="hello") == "hello"


def test_read_context_file_rejects_missing_file(tmp_path: Path):
    agent = MarkdownFileOnlyAgent()

    with pytest.raises(FileNotFoundError):
        agent.read_context_file(str(tmp_path / "missing.ga"))


def test_read_context_file_accepts_non_markdown(tmp_path: Path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not markdown context", encoding="utf-8")
    agent = MarkdownFileOnlyAgent()

    result = agent.read_context_file(str(text_file))

    assert result["content"] == "not markdown context"


def test_read_context_file_returns_bounded_content(tmp_path: Path):
    context_file = tmp_path / "workflow.ga"
    context_file.write_text("# Heading\n\n" + ("content " * 20), encoding="utf-8")
    agent = MarkdownFileOnlyAgent()

    result = agent.read_context_file(str(context_file), max_chars=20)

    assert result["path"] == str(context_file.resolve())
    assert result["content"] == "# Heading\n\ncontent c"
    assert result["truncated"] == "true"


def test_parse_collections_accepts_json_mapping():
    assert parse_collections('{"histories": "galaxy_histories"}') == {
        "histories": "galaxy_histories"
    }


def test_load_skill_text_reads_skill_file(tmp_path: Path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("# Skill\n\nUse workflows and histories.", encoding="utf-8")

    assert "workflows and histories" in MarkdownRAGAgent._load_skill_text(str(skill_file))
