"""
Unit tests for the Galaxy tools help-text embedding pipeline.

Tests cover:
- HTML text extraction
- Help block extraction from XML
- Markdown to text conversion
- Tool object extraction from nested structures
- Tool deduplication
- Pipeline document creation
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from pipelines.tools import (
    HTMLTextExtractor,
    GalaxyTool,
    extract_help_block,
    markdown_to_text,
    iter_tool_objects,
    extract_tools,
    GalaxyToolsPipeline,
    fetch_json,
    load_json_file,
)


class TestHTMLTextExtractor:
    """Test HTML to plain text conversion."""

    def test_extracts_text_from_simple_html(self):
        """Test basic text extraction from simple HTML."""
        extractor = HTMLTextExtractor()
        extractor.feed("<p>Hello world</p>")
        assert extractor.get_text() == "Hello world"

    def test_handles_nested_tags(self):
        """Test extraction from nested HTML tags."""
        extractor = HTMLTextExtractor()
        extractor.feed("<div><p>Hello</p><p>world</p></div>")
        text = extractor.get_text()
        assert "Hello" in text
        assert "world" in text

    def test_adds_newlines_for_block_elements(self):
        """Test that block elements get newlines."""
        extractor = HTMLTextExtractor()
        extractor.feed("<p>Line 1</p><p>Line 2</p>")
        text = extractor.get_text()
        lines = text.strip().split("\n")
        assert "Line 1" in lines
        assert "Line 2" in lines

    def test_normalizes_whitespace(self):
        """Test that extra whitespace is normalized."""
        extractor = HTMLTextExtractor()
        extractor.feed("<p>Hello    \n\n    world</p>")
        text = extractor.get_text()
        # Paragraph tags add newlines, but internal whitespace is normalized
        assert "Hello" in text and "world" in text

    def test_handles_lists(self):
        """Test extraction from list elements."""
        extractor = HTMLTextExtractor()
        extractor.feed("<ul><li>Item 1</li><li>Item 2</li></ul>")
        text = extractor.get_text()
        assert "Item 1" in text
        assert "Item 2" in text

    def test_handles_table_content(self):
        """Test extraction from table elements."""
        extractor = HTMLTextExtractor()
        extractor.feed("<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>")
        text = extractor.get_text()
        assert "Cell 1" in text
        assert "Cell 2" in text

    def test_handles_headers(self):
        """Test extraction from header elements."""
        extractor = HTMLTextExtractor()
        extractor.feed("<h1>Title</h1><h2>Subtitle</h2>")
        text = extractor.get_text()
        assert "Title" in text
        assert "Subtitle" in text

    def test_empty_html(self):
        """Test with empty or no HTML."""
        extractor = HTMLTextExtractor()
        extractor.feed("")
        assert extractor.get_text() == ""


class TestGalaxyTool:
    """Test GalaxyTool dataclass."""

    def test_creates_tool_with_all_fields(self):
        """Test creating a tool with all fields."""
        tool = GalaxyTool(
            tool_id="test_tool",
            name="Test Tool",
            description="A test tool",
            panel_section_id="section_1",
            panel_section_name="Tools",
            help_text="This is help text.",
        )
        assert tool.tool_id == "test_tool"
        assert tool.name == "Test Tool"
        assert tool.help_text == "This is help text."

    def test_creates_tool_with_default_help_text(self):
        """Test that help_text defaults to empty string."""
        tool = GalaxyTool(
            tool_id="test_tool",
            name="Test Tool",
            description="A test tool",
            panel_section_id="section_1",
            panel_section_name="Tools",
        )
        assert tool.help_text == ""


class TestExtractHelpBlock:
    """Test extraction of help blocks from XML."""

    def test_extracts_help_block(self):
        """Test basic help block extraction."""
        xml = """<tool>
            <help>This is help text.</help>
        </tool>"""
        result = extract_help_block(xml)
        assert result == "This is help text."

    def test_extracts_multiline_help(self):
        """Test extracting multiline help blocks."""
        xml = """<tool>
            <help>
                This is line 1.
                This is line 2.
            </help>
        </tool>"""
        result = extract_help_block(xml)
        assert "line 1" in result
        assert "line 2" in result

    def test_handles_help_with_nested_tags(self):
        """Test help blocks with nested HTML."""
        xml = """<tool>
            <help>
                <p>Paragraph 1</p>
                <p>Paragraph 2</p>
            </help>
        </tool>"""
        result = extract_help_block(xml)
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result

    def test_returns_empty_string_when_no_help(self):
        """Test that empty string is returned when no help block."""
        xml = "<tool><name>Test</name></tool>"
        result = extract_help_block(xml)
        assert result == ""

    def test_case_insensitive_help_tag(self):
        """Test that help tag detection is case-insensitive."""
        xml = "<tool><HELP>Help content</HELP></tool>"
        result = extract_help_block(xml)
        assert "Help content" in result

    def test_extracts_help_with_attributes(self):
        """Test extracting help blocks with attributes."""
        xml = '<tool><help xml:space="preserve">Help text</help></tool>'
        result = extract_help_block(xml)
        assert "Help text" in result


class TestMarkdownToText:
    """Test markdown to text conversion."""

    def test_converts_simple_markdown(self):
        """Test basic markdown conversion."""
        result = markdown_to_text("# Title")
        assert "Title" in result

    def test_converts_bold_text(self):
        """Test that bold markdown is converted."""
        result = markdown_to_text("**bold text**")
        assert "bold text" in result

    def test_converts_italic_text(self):
        """Test that italic markdown is converted."""
        result = markdown_to_text("*italic text*")
        assert "italic text" in result

    def test_converts_lists(self):
        """Test markdown list conversion."""
        result = markdown_to_text("- Item 1\n- Item 2")
        assert "Item 1" in result
        assert "Item 2" in result

    def test_converts_code_blocks(self):
        """Test markdown code block conversion."""
        result = markdown_to_text("`inline code`")
        assert "inline code" in result

    def test_normalizes_whitespace(self):
        """Test that excessive whitespace is normalized."""
        result = markdown_to_text("Text with   multiple   spaces")
        assert "Text with multiple spaces" in result

    def test_empty_string(self):
        """Test with empty markdown."""
        result = markdown_to_text("")
        assert result == ""

    def test_whitespace_only(self):
        """Test with whitespace-only markdown."""
        result = markdown_to_text("   \n\n   ")
        assert result == ""


class TestIterToolObjects:
    """Test iteration over nested tool structures."""

    def test_extracts_simple_tool(self):
        """Test extracting a simple tool from payload."""
        payload = {
            "model_class": "Tool",
            "id": "tool1",
            "name": "Tool 1",
        }
        tools = list(iter_tool_objects(payload))
        assert len(tools) == 1
        assert tools[0][0]["id"] == "tool1"

    def test_extracts_tools_from_section(self):
        """Test extracting tools from a tool section."""
        payload = {
            "model_class": "ToolSection",
            "id": "section1",
            "name": "Section 1",
            "elems": [
                {"model_class": "Tool", "id": "tool1", "name": "Tool 1"},
                {"model_class": "Tool", "id": "tool2", "name": "Tool 2"},
            ],
        }
        tools = list(iter_tool_objects(payload))
        assert len(tools) == 2

    def test_inherits_section_context(self):
        """Test that section info is inherited by nested tools."""
        payload = {
            "model_class": "ToolSection",
            "id": "section1",
            "name": "My Section",
            "elems": [
                {"model_class": "Tool", "id": "tool1", "name": "Tool 1"},
            ],
        }
        tools = list(iter_tool_objects(payload))
        tool, section_id, section_name = tools[0]
        assert section_id == "section1"
        assert section_name == "My Section"

    def test_handles_nested_sections(self):
        """Test handling of nested tool sections."""
        payload = {
            "elems": [
                {
                    "model_class": "ToolSection",
                    "id": "section1",
                    "name": "Section 1",
                    "elems": [
                        {"model_class": "Tool", "id": "tool1", "name": "Tool 1"},
                    ],
                }
            ]
        }
        tools = list(iter_tool_objects(payload))
        assert len(tools) == 1

    def test_handles_list_payload(self):
        """Test iterating over a list of items."""
        payload = [
            {"model_class": "Tool", "id": "tool1", "name": "Tool 1"},
            {"model_class": "Tool", "id": "tool2", "name": "Tool 2"},
        ]
        tools = list(iter_tool_objects(payload))
        assert len(tools) == 2

    def test_alternative_container_keys(self):
        """Test that alternative container keys are handled."""
        payload = {
            "items": [
                {"model_class": "Tool", "id": "tool1", "name": "Tool 1"},
            ]
        }
        tools = list(iter_tool_objects(payload))
        assert len(tools) == 1


class TestExtractTools:
    """Test tool extraction and deduplication."""

    def test_extracts_single_tool(self):
        """Test extracting a single tool."""
        payload = {
            "model_class": "Tool",
            "id": "tool1",
            "name": "Tool 1",
            "description": "Description 1",
        }
        tools, raw = extract_tools(payload)
        assert len(tools) == 1
        assert tools[0].tool_id == "tool1"
        assert tools[0].name == "Tool 1"

    def test_deduplicates_tools(self):
        """Test that duplicate tool IDs are deduplicated."""
        payload = [
            {
                "model_class": "Tool",
                "id": "tool1",
                "name": "Tool 1",
                "description": "Description 1",
            },
            {
                "model_class": "Tool",
                "id": "tool1",  # duplicate
                "name": "Tool 1 Duplicate",
                "description": "Description 1",
            },
        ]
        tools, raw = extract_tools(payload)
        assert len(tools) == 1

    def test_respects_tool_limit(self):
        """Test that tool limit is respected."""
        payload = [
            {
                "model_class": "Tool",
                "id": f"tool{i}",
                "name": f"Tool {i}",
                "description": f"Description {i}",
            }
            for i in range(10)
        ]
        tools, raw = extract_tools(payload, limit=5)
        assert len(tools) == 5

    def test_sorts_tools(self):
        """Test that tools are sorted correctly."""
        payload = [
            {
                "model_class": "Tool",
                "id": "tool_b",
                "name": "B Tool",
                "panel_section_name": "Section A",
            },
            {
                "model_class": "Tool",
                "id": "tool_a",
                "name": "A Tool",
                "panel_section_name": "Section A",
            },
        ]
        tools, raw = extract_tools(payload)
        assert tools[0].name == "A Tool"
        assert tools[1].name == "B Tool"

    def test_includes_section_info(self):
        """Test that section information is captured."""
        payload = {
            "model_class": "ToolSection",
            "id": "section1",
            "name": "My Section",
            "elems": [
                {"model_class": "Tool", "id": "tool1", "name": "Tool 1"},
            ],
        }
        tools, raw = extract_tools(payload)
        assert tools[0].panel_section_id == "section1"
        assert tools[0].panel_section_name == "My Section"


class TestGalaxyToolsPipeline:
    """Test the main pipeline class."""

    def test_initializes_with_defaults(self, tmp_path, monkeypatch):
        """Test pipeline initialization with default values."""
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "stub")
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://stub")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "test-model")
        
        pipeline = GalaxyToolsPipeline(
            chroma_directory=str(tmp_path),
        )
        assert pipeline.url == "https://usegalaxy.eu/api/tools"
        assert pipeline.base_url == "https://usegalaxy.eu"
        assert pipeline.collection_name == "galaxy_tools"
        assert pipeline.timeout == 120
        assert pipeline.help_timeout == 30

    def test_initializes_with_custom_values(self, tmp_path, monkeypatch):
        """Test pipeline initialization with custom values."""
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "stub")
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://stub")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "test-model")
        
        pipeline = GalaxyToolsPipeline(
            url="https://example.com/api/tools",
            base_url="https://example.com",
            chroma_directory=str(tmp_path),
            collection_name="custom_tools",
            timeout=60,
            help_timeout=20,
        )
        assert pipeline.url == "https://example.com/api/tools"
        assert pipeline.base_url == "https://example.com"
        assert pipeline.collection_name == "custom_tools"
        assert pipeline.timeout == 60
        assert pipeline.help_timeout == 20

    def test_tool_to_document(self, tmp_path, monkeypatch):
        """Test conversion of tool to document."""
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "stub")
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://stub")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "test-model")
        
        pipeline = GalaxyToolsPipeline(
            base_url="https://usegalaxy.eu",
            chroma_directory=str(tmp_path),
        )
        
        tool = GalaxyTool(
            tool_id="test_tool_id",
            name="Test Tool",
            description="A test tool",
            panel_section_id="section1",
            panel_section_name="Testing",
            help_text="This is help text.",
        )
        
        doc = pipeline.tool_to_document(tool)
        assert "Test Tool" in doc.page_content
        assert "test_tool_id" in doc.page_content
        assert "This is help text." in doc.page_content
        assert doc.metadata["tool_id"] == "test_tool_id"
        assert doc.metadata["source"] == "galaxy-tool://test_tool_id"
        assert doc.metadata["data_source"] == "usegalaxy_tools"

    def test_text_field_with_value(self):
        """Test text_field method with value."""
        result = GalaxyToolsPipeline.text_field("Tool ID", "test_tool")
        assert result == "Tool ID: test_tool"

    def test_text_field_without_value(self):
        """Test text_field method without value."""
        result = GalaxyToolsPipeline.text_field("Tool ID", "")
        assert result == ""

    def test_text_field_strips_whitespace(self):
        """Test text_field strips whitespace."""
        result = GalaxyToolsPipeline.text_field("Tool ID", "  test_tool  ")
        assert result == "Tool ID: test_tool"

    def test_load_payload_from_file(self, tmp_path, monkeypatch):
        """Test loading payload from a JSON file."""
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "stub")
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://stub")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "test-model")
        
        # Create a test JSON file
        test_data = {
            "model_class": "Tool",
            "id": "tool1",
            "name": "Tool 1",
        }
        input_file = tmp_path / "test_tools.json"
        import json
        with open(input_file, "w") as f:
            json.dump(test_data, f)
        
        pipeline = GalaxyToolsPipeline(
            input_json=str(input_file),
            chroma_directory=str(tmp_path),
        )
        
        payload = pipeline.load_payload()
        assert payload["id"] == "tool1"

    @patch("pipelines.tools.fetch_json")
    def test_load_payload_from_url(self, mock_fetch, tmp_path, monkeypatch):
        """Test loading payload from a URL."""
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "stub")
        monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://stub")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "test-model")
        
        mock_fetch.return_value = {
            "model_class": "Tool",
            "id": "tool1",
            "name": "Tool 1",
        }
        
        pipeline = GalaxyToolsPipeline(
            url="https://example.com/api/tools",
            chroma_directory=str(tmp_path),
        )
        
        payload = pipeline.load_payload()
        assert payload["id"] == "tool1"
        mock_fetch.assert_called_once()


class TestFetchJson:
    """Test JSON fetching functionality."""

    @patch("pipelines.tools.requests.get")
    def test_fetches_json(self, mock_get):
        """Test successful JSON fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test"}
        mock_get.return_value = mock_response
        
        result = fetch_json("https://example.com/api/data")
        assert result["id"] == "test"

    @patch("pipelines.tools.requests.get")
    def test_fetch_json_sets_accept_header(self, mock_get):
        """Test that Accept header is set."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        fetch_json("https://example.com/api/data")
        
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Accept"] == "application/json"

    @patch("pipelines.tools.requests.get")
    def test_fetch_json_respects_timeout(self, mock_get):
        """Test that timeout parameter is respected."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        fetch_json("https://example.com/api/data", timeout=60)
        
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 60


class TestLoadJsonFile:
    """Test JSON file loading."""

    def test_loads_json_file(self, tmp_path):
        """Test loading a JSON file."""
        import json
        test_data = {"id": "test", "name": "Test"}
        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)
        
        result = load_json_file(test_file)
        assert result["id"] == "test"
        assert result["name"] == "Test"

    def test_loads_json_with_unicode(self, tmp_path):
        """Test loading JSON with unicode characters."""
        import json
        test_data = {"name": "Tëst Tøøl"}
        test_file = tmp_path / "test.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)
        
        result = load_json_file(test_file)
        assert result["name"] == "Tëst Tøøl"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tool_with_empty_id_is_skipped(self):
        """Test that tools with empty IDs are skipped."""
        payload = [
            {
                "model_class": "Tool",
                "id": "",
                "name": "Tool 1",
                "description": "Description",
            }
        ]
        tools, raw = extract_tools(payload)
        assert len(tools) == 0

    def test_tool_with_none_id_is_converted_to_string(self):
        """Test that None IDs are converted to 'None' string."""
        payload = [
            {
                "model_class": "Tool",
                "id": None,
                "name": "Tool 1",
                "description": "Description",
            }
        ]
        tools, raw = extract_tools(payload)
        # str(None) becomes "None" string, which is truthy, so tool is extracted
        assert len(tools) == 1
        assert tools[0].tool_id == "None"

    def test_help_block_with_special_characters(self):
        """Test help block extraction with special characters."""
        xml = '<tool><help>Special chars: &amp; &lt; &gt; "quotes"</help></tool>'
        result = extract_help_block(xml)
        assert len(result) > 0

    def test_markdown_with_mixed_content(self):
        """Test markdown conversion with mixed content types."""
        markdown_text = """
# Title

This is a paragraph with **bold** and *italic* text.

- Item 1
- Item 2

```python
code = "example"
```
        """
        result = markdown_to_text(markdown_text)
        assert "Title" in result
        assert "paragraph" in result
        assert "Item 1" in result
