# Galaxy Documentation Embedding Pipelines

RAG-ready embedding pipelines for Galaxy project documentation sources:
- **GTN Tutorials** - Curated educational training materials
- **Discourse Forum** - Community Q&A and support threads

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Setup environment
# Edit .env with:
#   - Your embeddings API configuration (LM Studio, SambaNova, or OpenAI)
#   - Your Discourse API key and username
#   - Path to GTN training materials

# 3. Run both pipelines
./scripts/update_all.sh
```

## Two Data Sources

### 1. Galaxy Training Network (GTN)
**Source**: Curated markdown tutorial files
**Collection**: `gtn_tutorials`
**Update frequency**: Occasional (when training materials change)
**Characteristics**: Authoritative, structured, educational

### 2. Discourse Forum
**Source**: help.galaxyproject.org API (all categories)
**Collection**: `discourse_posts`
**Update frequency**: Frequent (daily or more)
**Characteristics**: Real-world problems, community solutions, Q&A format
**Note**: Automatically fetches from ALL categories in the forum

## Architecture

```
chatgxy-materials/
├── pipelines/
│   ├── common.py        # Shared embedding logic
│   ├── gtn.py           # GTN pipeline
│   └── discourse.py     # Discourse pipeline
├── data/
│   ├── embeddings/      # Chroma database (both collections)
│   ├── discourse/       # Discourse cache
│   └── gtn/             # GTN data (if needed)
├── scripts/
│   └── update_all.sh    # Master update script
└── old/                 # Deprecated files
```

## Usage

### Run Both Pipelines (Recommended)

```bash
# Incremental updates (only processes changes)
./scripts/update_all.sh

# Force full rebuild
./scripts/update_all.sh --force-rebuild
```

### Run Individual Pipelines

**GTN Pipeline:**
```bash
uv run python -m pipelines.gtn --source-dir /path/to/training-material/topics
uv run python -m pipelines.gtn --source-dir /path/to/training-material/topics --force-rebuild
uv run python -m pipelines.gtn --source-dir /path/to/training-material/topics --test-query "RNA-seq"
```

**Discourse Pipeline:**
```bash
uv run python -m pipelines.discourse
uv run python -m pipelines.discourse --force-rebuild
uv run python -m pipelines.discourse --test-query "workflow error"
```

## Incremental Updates

Both pipelines are **idempotent** - safe to run repeatedly:

- ✅ SHA256 content hashing detects changes
- ✅ Only processes new/modified documents
- ✅ Removes embeddings for deleted documents
- ✅ Tracks processing history
- ✅ Saves API costs (98%+ savings after first run)

**Example cost savings:**
- First run: 1000 documents → **$0.00 (FREE via TACC!)**
- Daily updates: 10 changed → **$0.00 (FREE!)**
- No changes: Skip processing → $0.00

**Annual cost estimate** (daily updates, 1% change rate):
- **$0.00/year** - Completely free via TACC infrastructure!

## Scheduling Periodic Updates

**Cron (recommended):**
```bash
# Daily at 2am
0 2 * * * cd /path/to/chatgxy-materials && ./scripts/update_all.sh

# Or separately
0 2 * * * cd /path/to/chatgxy-materials && uv run python -m pipelines.gtn
0 3 * * * cd /path/to/chatgxy-materials && uv run python -m pipelines.discourse
```

**systemd timer:**
```ini
# /etc/systemd/system/galaxy-embeddings.service
[Unit]
Description=Update Galaxy documentation embeddings

[Service]
Type=oneshot
WorkingDirectory=/path/to/chatgxy-materials
ExecStart=/path/to/chatgxy-materials/scripts/update_all.sh
```

## Querying the Vector Database

Both collections are stored in the same Chroma database but are **separate collections** for flexible retrieval.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

# Using configured embeddings endpoint (local LM Studio, SambaNova, OpenAI, etc.)
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-nomic-embed-text-v2-moe"),
    openai_api_key=os.getenv("EMBEDDINGS_API_KEY"),
    openai_api_base=os.getenv("EMBEDDINGS_BASE_URL")
)

# Query GTN tutorials
gtn_store = Chroma(
    collection_name="gtn_tutorials",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

gtn_results = gtn_store.similarity_search("RNA-seq tutorial", k=5)

# Query Discourse posts
discourse_store = Chroma(
    collection_name="discourse_posts",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

discourse_results = discourse_store.similarity_search("workflow error", k=5)

# Search both (combine results)
all_results = gtn_results + discourse_results
```

## RAG Integration

Separate collections enable flexible retrieval strategies:

1. **Search only curated content** (tutorials)
2. **Search only community Q&A** (discourse)
3. **Search both with different weights** (prioritize tutorials)
4. **Show sources separately** in results
5. **Filter by metadata** (tags, dates, reply counts, etc.)

**Example RAG prompt:**
```
Context from official tutorials:
{gtn_results}

Context from community discussions:
{discourse_results}

User question: {query}
```

## Environment Variables

Configure your `.env` file with the following options:

```bash
# Required - Embeddings API configuration
# Works with: OpenAI, SambaNova (via TACC), LM Studio, or any OpenAI-compatible endpoint
EMBEDDINGS_API_KEY=your-api-key
EMBEDDINGS_BASE_URL=https://api.openai.com/v1  # or http://127.0.0.1:1234/v1 for LM Studio
EMBEDDINGS_MODEL=text-embedding-3-small  # or text-embedding-nomic-embed-text-v2-moe for local

# GTN
GTN_SOURCE_DIRECTORY=/path/to/training-material/topics

# Discourse (fetches from ALL categories)
DISCOURSE_API_KEY=your-discourse-key
DISCOURSE_API_USER=your-username
DISCOURSE_BASE=https://help.galaxyproject.org

# Shared
CHROMA_DIRECTORY=./data/embeddings
```

### Embedding Models

This project supports multiple embedding providers via OpenAI-compatible APIs:

**Local LM Studio (recommended for development):**
- ✅ **FREE** - Runs on your own hardware
- ✅ **Private** - Data never leaves your machine
- ✅ **Fast** - text-embedding-nomic-embed-text-v2-moe (768 dimensions)
- ✅ **No API limits** - Process as much as you want

**SambaNova via TACC (alternative):**
- ✅ **FREE** - No external API costs (when working)
- ✅ **High quality** - e5-mistral-7B-instruct (4096 dimensions)
- ⚠️ **Note** - May have connectivity issues

**OpenAI (production):**
- 💰 **Low cost** - text-embedding-3-small (~$0.02/1M tokens)
- ✅ **Reliable** - High uptime and performance
- ✅ **High quality** - 1536 dimensions

## Data Organization

```
data/
├── embeddings/
│   ├── chroma.sqlite3                  # Vector database
│   ├── .gtn_tracking.json              # GTN processing history
│   └── .discourse_tracking.json        # Discourse processing history
└── discourse/
    ├── .cache/                          # API response cache
    └── [topic data]
```

## Metadata

Each document includes rich metadata for filtering:

**GTN:**
- `data_source`: "gtn"
- `content_type`: "tutorial"
- `source`: file path

**Discourse:**
- `data_source`: "discourse"
- `content_type`: "forum_post"
- `source`: `discourse://category/topic_id`
- `category`: category slug (e.g., "usegalaxy-org-support", "galaxy-development")
- `title`, `url`, `reply_count`, `views`, `tags`, `created_at`

## Troubleshooting

**"SAMBANOVA_API_KEY not set"**
```bash
cp .env.example .env
# API key and base URL are pre-configured in .env.example
```

**"No GTN documents found"**
```bash
# Check path in .env
echo $GTN_SOURCE_DIRECTORY
ls -la $GTN_SOURCE_DIRECTORY/**/*.md
```

**"Discourse API error"**
```bash
# Verify credentials
curl -H "Api-Key: $DISCOURSE_API_KEY" -H "Api-Username: $DISCOURSE_API_USER" \
  $DISCOURSE_BASE/c/$DISCOURSE_CATEGORY.json
```

**Force rebuild from scratch**
```bash
rm -rf data/embeddings/
./scripts/update_all.sh
```

## Migration from Old Structure

Old files (`main.py`, `export_discourse.py`) are deprecated. The new structure provides:

- Clear separation of data sources
- Separate collections for flexible querying
- Better organization
- Same incremental update logic
- Same cost savings

See `MIGRATION_NOTES.md` for details.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Format code
uv run black .

# Lint
uv run ruff check .
```

## Documentation

- `CLAUDE.md` - Technical details and architecture
- `MIGRATION_NOTES.md` - Changes from old structure
- `QUICKSTART.md` - Quick reference guide

## License

See `LICENSE` file.
