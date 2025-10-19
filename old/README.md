# Deprecated Files

These files are from the old structure and are no longer used.

## Old Files

- `main.py` - Original embedding pipeline (GTN only)
- `export_discourse.py` - Original Discourse fetcher (no embedding)

## New Structure

The project has been reorganized for clarity and to support both data sources:

### Use These Instead:

**GTN Pipeline:**
```bash
python -m pipelines.gtn --source-dir /path/to/training-material/topics
```

**Discourse Pipeline:**
```bash
python -m pipelines.discourse
```

**Both Pipelines:**
```bash
./scripts/update_all.sh
```

See the main README.md for details.

## Why Changed?

The new structure:
- Separates GTN and Discourse into distinct pipelines
- Creates separate collections for flexible RAG queries
- Provides better organization
- Maintains all incremental update features
- Uses shared embedding logic (DRY principle)

## Migration

Old data locations:
- `chroma_data/` → `data/embeddings/`
- `posts/` → `data/discourse/`

No data migration needed - just use the new pipelines going forward.
