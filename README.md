# ChatGXY Materials Pipeline

A complete RAG (Retrieval-Augmented Generation) system for the Galaxy Project that fetches, processes, and embeds documentation from multiple sources. This enables semantic search and question-answering capabilities for the ChatGXY chatbot.

## Overview

The system consists of five independent pipelines that populate separate Chroma vector store collections:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CHATGXY MATERIALS PIPELINE                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DISCOURSE     GTN          FAQ          IWC WORKFLOWS                       │
│  ┌─────────┐  ┌────────┐   ┌────────┐   ┌─────────────┐                      │
│  │ 5,986   │  │  168   │   │   25   │   │     105     │                      │
│  │ topics  │  │tutorial│   │  FAQs  │   │  workflows  │                      │
│  │ 70,516  │  │ 9,295  │   │   62   │   │  772 chunks │                      │
│  │ chunks  │  │ chunks │   │ chunks │   └─────────────┘                      │
│  └─────────┘  └────────┘   └────────┘          ↓                             │
│       ↓            ↓            ↓         iwc_workflows                      │
│  discourse_    gtn_        galaxy_faqs    collection                         │
│  posts         tutorials   collection                                        │
│  collection    collection                                                    │
│                                                                              │
│                       CHROMA VECTOR STORE                                    │
│                     (./data/embeddings/)                                     │
│                                                                              │
│                    Total: ~81,600 embedded chunks                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Environment Variables** (`.env` file):
```bash
# Embeddings API (LM Studio, SambaNova, or OpenAI-compatible)
EMBEDDINGS_API_KEY=your-api-key
EMBEDDINGS_BASE_URL=http://127.0.0.1:1234/v1  # or your endpoint
EMBEDDINGS_MODEL=text-embedding-nomic-embed-text-v2-moe

# Discourse API
DISCOURSE_BASE=https://help.galaxyproject.org
DISCOURSE_API_KEY=your_api_key_here
DISCOURSE_API_USER=your_username_here

# GTN Source (local clone of training-material repo)
GTN_SOURCE_DIRECTORY=/path/to/training-material/topics

# Storage
CHROMA_DIRECTORY=./data/embeddings
```

2. **LM Studio** running with embedding model (recommended):
   - Model: `text-embedding-nomic-embed-text-v2-moe`
   - Server: `http://127.0.0.1:1234`

3. **Dependencies**:
```bash
uv sync  # Install all dependencies
```

### Running the Pipelines

```bash
# Show all available commands
make help

# Run individual pipelines
make discourse-all   # Discourse forum posts
make gtn            # GTN tutorials
make faq            # Galaxy FAQs
make iwc            # IWC workflows
make tools          # Galaxy tool help text

# Run all pipelines
make update-all

# Ask the configurable RAG agent after embeddings exist
make md-agent QUESTION="How do I create a workflow in Galaxy?"

# Check status
make status
```

## The Five Pipelines

### 1. Discourse Pipeline

**Purpose**: Fetches and embeds Galaxy Project forum posts from Discourse.

**Source**: https://help.galaxyproject.org

**Output**:
- 5,986 topics
- 70,516 embedding chunks
- Collection: `discourse_posts`

**Features**:
- SQLite database for persistent storage
- Incremental updates (content hashing)
- Parallel processing (4+ workers)
- API response caching
- Rate limiting (1.5s between requests)

**Usage**:
```bash
# Full pipeline
make discourse-all

# Individual phases
make fetch      # Fetch topics from API
make convert    # Convert to documents (parallel)
make embed      # Create embeddings

# Force full rebuild
uv run python -m pipelines.discourse --force-rebuild
```

**Implementation**: See `.claude/plans/DISCOURSE_PIPELINE.md` for detailed architecture.

### 2. GTN Pipeline

**Purpose**: Processes Galaxy Training Network tutorial markdown files.

**Source**: Local clone of https://github.com/galaxyproject/training-material

**Output**:
- 168 tutorials
- 9,295 embedding chunks
- Collection: `gtn_tutorials`

**Features**:
- Recursive directory scanning for `tutorial.md` files
- Markdown front matter extraction
- Topic/subtopic organization
- Incremental updates

**Usage**:
```bash
# Run GTN pipeline
make gtn

# With options
uv run python -m pipelines.gtn --force-rebuild
uv run python -m pipelines.gtn --test-query "your question"
```

**Document Structure**:
```python
Document(
    page_content="Tutorial content...",
    metadata={
        "source": "path/to/tutorial.md",
        "data_source": "gtn",
        "content_type": "tutorial",
        "title": "Tutorial Title",
        "topic": "assembly",
        "subtopic": "introduction",
        "tags": "assembly, genome, nanopore",
        "contributors": "author1, author2"
    }
)
```

### 3. FAQ Pipeline

**Purpose**: Processes Galaxy FAQ document with individual Q&A pairs.

**Source**: `training-material/faq.md`

**Output**:
- 25 FAQs
- 62 embedding chunks
- Collection: `galaxy_faqs`

**Features**:
- Markdown parsing (## headers = questions)
- Each FAQ becomes a single document
- Category extraction (# headers)
- Semantic search optimized for Q&A format

**Usage**:
```bash
# Run FAQ pipeline
make faq

# With test query
uv run python -m pipelines.faq --test-query "your question"

# Force rebuild
uv run python -m pipelines.faq --force-rebuild
```

**Document Format**:
```
Q: How do I use Galaxy workflows?

A: Galaxy workflows allow you to...
```

**Metadata**:
```python
{
    "source": "faq.md",
    "data_source": "gtn_faq",
    "content_type": "faq",
    "category": "For Individual Learners",
    "question": "How do I use Galaxy workflows?",
    "faq_index": 5
}
```

### 4. IWC Workflows Pipeline

**Purpose**: Processes IWC (Intergalactic Workflow Commission) workflows documentation.

**Source**: https://iwc.galaxyproject.org/workflow_manifest.json

**Output**:
- 105 workflows
- 772 embedding chunks
- Collection: `iwc_workflows`

**Features**:
- Fetches workflow manifest from IWC website
- Extracts README, annotations, and metadata
- Workflow step documentation
- Category and collection organization
- Author and DOI information

**Usage**:
```bash
# Run IWC pipeline
make iwc

# With test query
uv run python -m pipelines.iwc --test-query "your question"

# Force rebuild
uv run python -m pipelines.iwc --force-rebuild
```

**Document Structure**:
```python
Document(
    page_content="# Workflow Name\n\nDescription...\n\nREADME content...",
    metadata={
        "source": "iwc://path/to/workflow",
        "data_source": "iwc",
        "content_type": "workflow",
        "workflow_name": "Protein Structure Analysis",
        "workflow_id": "protein-structure-analysis-main",
        "topic": "computational-chemistry",
        "categories": "Structural Biology",
        "collections": "Computational Chemistry",
        "authors": "Author Name",
        "license": "MIT",
        "doi": "10.5281/zenodo.12345",
        "url": "https://iwc.galaxyproject.org/workflows/..."
    }
)
```

### 5. Galaxy Tools Pipeline

**Purpose**: Fetches Galaxy tool metadata and embeds each tool wrapper's help text.

**Source**: https://usegalaxy.org/api/tools and each tool's `raw_tool_source` API endpoint.

**Output**:
- One document per Galaxy tool
- Collection: `galaxy_tools`
- Inspection artifacts: `data/tools_only.json`, `data/tools_metadata.tsv`, `data/tools_helptext.tsv`

**Features**:
- Recursively walks the Galaxy tool panel and keeps section metadata
- Extracts `<help>...</help>` from raw Galaxy tool wrappers
- Converts Galaxy wrapper help Markdown/reStructuredText-like text to plain text for embeddings
- Incremental updates through the shared embedding tracker

**Usage**:
```bash
# Run Galaxy tools pipeline
make tools

# Process only the first N tools while testing
uv run python -m pipelines.tools --tool-limit 25

# With test query
uv run python -m pipelines.tools --test-query "How do I run fastqc?"

# Force rebuild
uv run python -m pipelines.tools --force-rebuild
```

**Document Structure**:
```python
Document(
    page_content="# FastQC\nTool ID: ...\nDescription: ...\n\n## Help\n...",
    metadata={
        "source": "galaxy-tool://toolshed.g2.bx.psu.edu/repos/...",
        "data_source": "usegalaxy_tools",
        "content_type": "tool_help",
        "tool_id": "toolshed.g2.bx.psu.edu/repos/...",
        "name": "FastQC",
        "panel_section_name": "FASTA/FASTQ",
        "url": "https://usegalaxy.org/root?tool_id=..."
    }
)
```

## System Architecture

### Data Flow

```
Source Data → Load/Parse → Documents → Chunks → Embeddings → Vector Store
```

1. **Load**: Fetch/read source data (API, files, database)
2. **Parse**: Convert to LangChain Documents with metadata
3. **Chunk**: Split long documents (512 tokens, 50 overlap)
4. **Embed**: Generate embeddings via LM Studio
5. **Store**: Persist in Chroma vector database

### Common Components

All pipelines share the `EmbeddingPipeline` class (`pipelines/common.py`):

```python
from pipelines.common import EmbeddingPipeline

pipeline = EmbeddingPipeline(
    chroma_directory="./data/embeddings",
    collection_name="my_collection",
    tracking_file="./data/embeddings/.tracking.json",
    incremental=True  # Enable incremental updates
)

# Process documents
vectorstore = pipeline.run(documents, force_rebuild=False)

# Query
results = vectorstore.similarity_search("your query", k=5)
```

**Features**:
- Text chunking with RecursiveCharacterTextSplitter
- Batch embedding (100 chunks per batch)
- Incremental updates (only process changed documents)
- Content hashing for change detection
- Progress tracking with tqdm

### Storage Structure

```
chatgxy-materials/
├── data/
│   ├── discourse/
│   │   ├── discourse.db          # SQLite database
│   │   └── .cache/                # API response cache
│   │
│   └── embeddings/
│       ├── chroma.sqlite3         # Chroma vector DB
│       ├── .discourse_tracking.json
│       ├── .gtn_tracking.json
│       └── .faq_tracking.json
```

## Incremental Updates

All pipelines support incremental updates to only process new or changed content:

**How it works**:
1. Content hashing (SHA256) detects changes in documents
2. Tracking files record processed documents
3. Only changed documents are re-chunked and re-embedded
4. Removed documents are deleted from vector store

**Example**:
```bash
# Initial run: processes everything
make discourse-all

# Later runs: only process changes
make fetch      # Fetch new topics
make convert    # Convert only new/changed
make embed      # Embed only new/changed

# Or force full rebuild
uv run python -m pipelines.discourse --force-rebuild
```

**Cost savings**:
- First run: 6,179 documents → ~70 minutes
- Daily updates: ~10 changed → <2 minutes
- 98%+ savings after first run (time and API costs)

## Performance

### Current Stats

| Pipeline | Documents | Chunks | Processing Time | Storage |
|----------|-----------|--------|-----------------|---------|
| Discourse | 5,986 | 70,516 | ~67 min | ~2 GB |
| GTN | 168 | 9,295 | ~2 min | ~200 MB |
| FAQ | 25 | 62 | <1 min | ~10 MB |
| IWC | 105 | 772 | ~15 sec | ~20 MB |
| Tools | varies | varies | varies | varies |
| **Total** | **6,284** | **81,645** | **~70 min** | **~2.2 GB** |

### Optimization Tips

1. **Parallel Processing**: Use `--workers N` for discourse conversion
2. **Incremental Updates**: Run regularly to avoid full rebuilds
3. **Batch Size**: Adjust in `common.py` if memory constrained
4. **Caching**: API responses cached to avoid re-fetching

## Integration with ChatGXY

The pipelines produce Chroma vector stores that can be queried by the ChatGXY chatbot:

### Configurable RAG Agent

`pipelines.md_agent` provides a small tool-driven RAG agent. It searches
configured Chroma collections, then sends the retrieved context to an
OpenAI-compatible LLM with answer-framing instructions loaded from `SKILL.md`.
The context can come from GTN Markdown tutorials, workflows, Galaxy tool help,
histories, FAQs, community posts, or any future collection you add.

It registers tools through `agent.tool`:

- `vector_search`: searches configured Chroma collections.
- `read_context_file`: adds one local context file as extra context.

Example:

```bash
uv run python -m pipelines.md_agent \
  "How do I run RNA-seq analysis in Galaxy?" \
  --skill-file SKILL.md \
  --context-file README.md \
  --top-k 8
```

Use `agent_config.example.json` as a starting point for adding more collections,
for example `histories`.

```bash
uv run python -m pipelines.md_agent \
  "Which workflow or history should I use for variant calling?" \
  --config-file agent_config.example.json \
  --collection histories
```

You can also pass a collection mapping directly:

```bash
uv run python -m pipelines.md_agent \
  "Find a relevant Galaxy history for RNA-seq" \
  --collections-json '{"histories":"galaxy_histories","workflows":"iwc_workflows","tools":"galaxy_tools"}'
```

The agent uses `EMBEDDINGS_API_KEY`, `EMBEDDINGS_BASE_URL`, and
`EMBEDDINGS_MODEL` for vector search. For response generation, set
`CHAT_API_KEY`, `CHAT_BASE_URL`, and `CHAT_MODEL`; if omitted, it falls back to
the embedding API key/base URL and `gpt-4o-mini`.

To trace retrieval, answer generation, and LLM-as-judge evaluation with Opik,
run the agent with `--evaluate`. The OpenAI client is wrapped with Opik's
OpenAI integration, and the agent uses `@opik.track` spans around vector search,
answer generation, and evaluation. Set `EVALUATION_MODEL` or pass
`--evaluation-model` to choose the judge model. For OpenAI-compatible judge
endpoints, set `EVALUATION_BASE_URL` and `EVALUATION_API_KEY`; if omitted, the
evaluator falls back to `CHAT_BASE_URL`/`CHAT_API_KEY`, then the OpenAI or
embedding API settings.

```bash
uv run python -m pipelines.search_agent \
  "How do I use FastQC in Galaxy?" \
  --collection tools \
  --evaluate \
  --json
```

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LMStudioEmbeddings

# Initialize embeddings
embeddings = LMStudioEmbeddings(
    base_url="http://localhost:1234/v1",
    model="text-embedding-nomic-embed-text-v2-moe"
)

# Load all collections
discourse = Chroma(
    persist_directory="./data/embeddings",
    collection_name="discourse_posts",
    embedding_function=embeddings
)

gtn = Chroma(
    persist_directory="./data/embeddings",
    collection_name="gtn_tutorials",
    embedding_function=embeddings
)

faq = Chroma(
    persist_directory="./data/embeddings",
    collection_name="galaxy_faqs",
    embedding_function=embeddings
)

# Query all sources
def search_all(query: str, k: int = 5):
    results = {
        'discourse': discourse.similarity_search(query, k=k),
        'gtn': gtn.similarity_search(query, k=k),
        'faq': faq.similarity_search(query, k=k)
    }
    return results

# Example
results = search_all("How do I install Galaxy?")
for source, docs in results.items():
    print(f"\n{source.upper()}:")
    for doc in docs:
        print(f"  - {doc.metadata.get('title', doc.metadata.get('question'))}")
```

### RAG Integration Strategies

Separate collections enable flexible retrieval:

1. **Search only curated content** (tutorials + FAQs)
2. **Search only community Q&A** (discourse)
3. **Search all with different weights** (prioritize FAQs, then tutorials)
4. **Show sources separately** in results
5. **Filter by metadata** (tags, dates, reply counts, etc.)

**Example RAG prompt**:
```
Quick Answer (if available):
{faq_results}

Official Tutorials:
{gtn_results}

Community Discussions:
{discourse_results}

User question: {query}
```

## File Structure

```
chatgxy-materials/
├── README.md                    # Main documentation
├── INTEGRATION_GUIDE.md         # RAG integration guide
├── Makefile                     # Pipeline commands
├── pyproject.toml               # Dependencies
├── .env                         # Configuration (not in repo)
│
├── pipelines/
│   ├── common.py                # Shared embedding pipeline
│   ├── database.py              # SQLite database layer
│   ├── discourse.py             # Discourse forum pipeline
│   ├── gtn.py                   # GTN tutorial pipeline
│   └── faq.py                   # FAQ pipeline
│
├── scripts/
│   └── migrate_to_sqlite.py     # Migration utilities
│
└── data/                        # Generated data (not in repo)
    ├── discourse/
    │   ├── discourse.db          # SQLite database
    │   └── .cache/               # API response cache
    │
    └── embeddings/
        ├── chroma.sqlite3        # Chroma vector database
        ├── .discourse_tracking.json
        ├── .gtn_tracking.json
        └── .faq_tracking.json
```

## Makefile Commands

```bash
# Help
make help                 # Show all commands

# Discourse Pipeline
make fetch                # Fetch topics from Discourse API
make convert              # Convert topics to documents
make embed                # Create embeddings
make discourse-all        # Run complete Discourse pipeline

# GTN Pipeline
make gtn                  # Run GTN pipeline

# FAQ Pipeline
make faq                  # Run FAQ pipeline

# IWC Pipeline
make iwc                  # Run IWC workflows pipeline

# Galaxy Tools Pipeline
make tools                # Run Galaxy tools help-text pipeline

# All Pipelines
make update-all           # Run all pipelines

# Utilities
make status               # Show database stats
make clean                # Remove deprecated files
```

## Environment Variables

### Required

```bash
# Embeddings API (choose one)
EMBEDDINGS_API_KEY=your-api-key
EMBEDDINGS_BASE_URL=http://127.0.0.1:1234/v1  # LM Studio
EMBEDDINGS_MODEL=text-embedding-nomic-embed-text-v2-moe

# Discourse
DISCOURSE_API_KEY=your-discourse-key
DISCOURSE_API_USER=your-username
DISCOURSE_BASE=https://help.galaxyproject.org

# GTN
GTN_SOURCE_DIRECTORY=/path/to/training-material/topics

# Storage
CHROMA_DIRECTORY=./data/embeddings
```

### Embedding Models

**Local LM Studio** (recommended for development):
- Model: `text-embedding-nomic-embed-text-v2-moe`
- Dimensions: 768
- Runs locally, no API costs, unlimited usage

**SambaNova via TACC**:
- Model: `e5-mistral-7B-instruct`
- Dimensions: 4096
- Free but may have connectivity issues

**OpenAI** (production):
- Model: `text-embedding-3-small`
- Dimensions: 1536
- ~$0.02/1M tokens, reliable uptime

## Troubleshooting

### Common Issues

**1. Missing Environment Variables**
```
KeyError: 'DISCOURSE_BASE'
```
**Solution**: Create `.env` file with required variables.

**2. LM Studio Not Running**
```
ConnectionError: Failed to connect to http://127.0.0.1:1234
```
**Solution**: Start LM Studio with embedding model loaded.

**3. GTN Source Not Found**
```
FileNotFoundError: /path/to/training-material/topics
```
**Solution**: Set `GTN_SOURCE_DIRECTORY` in `.env` or use `--source-dir` flag.

**4. FAQ File Not Found**
```
FileNotFoundError: /path/to/training-material/faq.md
```
**Solution**: Set `GTN_SOURCE_DIRECTORY` to parent directory of `faq.md`.

**5. Out of Memory**
```
MemoryError during embedding
```
**Solution**: Reduce batch size in `pipelines/common.py` (default: 100).

### Debugging

**Check database**:
```bash
make status

# Or manually
sqlite3 data/discourse/discourse.db "SELECT COUNT(*) FROM topics"
```

**Test embeddings**:
```bash
uv run python -m pipelines.discourse --test-query "your question"
uv run python -m pipelines.gtn --test-query "your question"
uv run python -m pipelines.faq --test-query "your question"
```

**View Chroma collections**:
```python
from chromadb import PersistentClient
client = PersistentClient(path="./data/embeddings")

collections = client.list_collections()
for coll in collections:
    print(f"{coll.name}: {coll.count()} chunks")
```

**Check tracking files**:
```bash
cat data/embeddings/.discourse_tracking.json | jq '.documents | length'
cat data/embeddings/.gtn_tracking.json | jq '.documents | length'
cat data/embeddings/.faq_tracking.json | jq '.documents | length'
```

## Maintenance

### Regular Updates

**Weekly**: Fetch new content
```bash
make fetch      # Get new Discourse topics
make convert    # Convert new topics
make embed      # Embed changes
make gtn        # Update GTN tutorials
make faq        # Update FAQs
```

**Or run all at once**:
```bash
make update-all
```

### Monthly: Full Rebuild

To catch edits and ensure data quality:
```bash
uv run python -m pipelines.discourse --force-rebuild
uv run python -m pipelines.gtn --force-rebuild
uv run python -m pipelines.faq --force-rebuild
```

### Scheduling with Cron

```bash
# Daily at 2am
0 2 * * * cd /path/to/chatgxy-materials && make update-all

# Or individually
0 2 * * * cd /path/to/chatgxy-materials && make discourse-all
0 3 * * * cd /path/to/chatgxy-materials && make gtn
0 3 * * * cd /path/to/chatgxy-materials && make faq
```

### Monitoring

```bash
# Overall stats
make status

# Recent Discourse topics
sqlite3 data/discourse/discourse.db \
  "SELECT title, created_at FROM topics ORDER BY created_at DESC LIMIT 10"

# Embedding coverage
sqlite3 data/discourse/discourse.db \
  "SELECT COUNT(*) as total_docs,
          (SELECT COUNT(*) FROM embeddings) as embedded_docs"
```

## Future Enhancements

### Planned Features

1. **Multi-Source FAQ Pipeline (Option 3)**
   - Combine FAQs from multiple sources
   - Discourse posts tagged/categorized as FAQs
   - Galaxy documentation FAQ pages
   - Unified FAQ collection with source tracking

2. **Scheduled Updates**
   - Cron jobs for daily/weekly updates
   - Webhook support for real-time updates
   - Differential sync

3. **Advanced Search**
   - Hybrid search (keyword + semantic)
   - User/post filtering
   - Time-based weighting
   - Cross-collection search

4. **Performance**
   - Parallel embedding (GPU acceleration)
   - Caching layer for frequent queries
   - Query optimization

5. **Monitoring**
   - Dashboard for pipeline health
   - Error tracking and alerts
   - Performance metrics
   - Data quality checks

## Development

### Running Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_db_integration.py

# With verbose output
pytest -v tests/
```

### Adding a New Pipeline

1. Create new file: `pipelines/your_pipeline.py`
2. Inherit from `EmbeddingPipeline` or create custom class
3. Implement document loading and metadata extraction
4. Add Makefile target
5. Update this README
6. Add tests

Example template:
```python
from pipelines.common import EmbeddingPipeline
from langchain.schema import Document
from typing import List

class YourPipeline:
    def __init__(self, source_dir: str, chroma_directory: str):
        self.pipeline = EmbeddingPipeline(
            chroma_directory=chroma_directory,
            collection_name="your_collection",
            incremental=True
        )

    def load_documents(self) -> List[Document]:
        # Load and parse your source data
        documents = []
        # ... your loading logic ...
        return documents

    def run(self, force_rebuild: bool = False):
        documents = self.load_documents()
        return self.pipeline.run(documents, force_rebuild=force_rebuild)
```

### Code Formatting

```bash
# Install dev dependencies
uv sync --extra dev

# Format code
uv run black .

# Lint
uv run ruff check .
```

## Additional Documentation

- `INTEGRATION_GUIDE.md` - Complete RAG integration guide with code examples
- `.claude/plans/` - Planning docs, detailed architecture notes, migration guides
