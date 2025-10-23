.PHONY: help fetch convert embed all clean status

help:
	@echo "Galaxy Documentation Embedding Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  make fetch      - Fetch topics from Discourse API"
	@echo "  make convert    - Convert cached topics to documents"
	@echo "  make embed      - Create embeddings from cached documents"
	@echo "  make all        - Run complete pipeline (fetch → convert → embed)"
	@echo "  make clean      - Remove intermediate files"
	@echo "  make status     - Show pipeline status"
	@echo ""
	@echo "Discourse Pipeline (incremental):"
	@echo "  make discourse-fetch   - Fetch Discourse topics"
	@echo "  make discourse-convert - Convert Discourse topics"
	@echo "  make discourse-embed   - Embed Discourse documents"
	@echo "  make discourse-all     - Run complete Discourse pipeline"
	@echo ""
	@echo "GTN Pipeline:"
	@echo "  make gtn        - Run GTN pipeline"
	@echo ""
	@echo "FAQ Pipeline:"
	@echo "  make faq        - Run FAQ pipeline"
	@echo ""
	@echo "All Pipelines:"
	@echo "  make update-all - Run Discourse, GTN, and FAQ pipelines"

# Discourse pipeline targets
fetch: discourse-fetch
convert: discourse-convert
embed: discourse-embed

discourse-fetch:
	@echo "📥 Fetching topics from Discourse API..."
	uv run python -m pipelines.discourse --fetch-only

discourse-convert:
	@echo "🔄 Converting topics to documents..."
	uv run python -m pipelines.discourse --convert-only --workers 4

discourse-embed:
	@echo "🧠 Creating embeddings..."
	uv run python -m pipelines.discourse --embed-only

discourse-all:
	@echo "🚀 Running complete Discourse pipeline..."
	uv run python -m pipelines.discourse

# Convenience targets
all: discourse-all

# GTN pipeline
gtn:
	@echo "📚 Running GTN pipeline..."
	uv run python -m pipelines.gtn

# FAQ pipeline
faq:
	@echo "❓ Running FAQ pipeline..."
	uv run python -m pipelines.faq

# Run all pipelines
update-all:
	@echo "🌐 Running all pipelines..."
	@echo "Running Discourse pipeline..."
	$(MAKE) discourse-all
	@echo ""
	@echo "Running GTN pipeline..."
	$(MAKE) gtn
	@echo ""
	@echo "Running FAQ pipeline..."
	$(MAKE) faq
	@echo ""
	@echo "✅ All pipelines complete!"

# Cleanup
clean:
	@echo "🧹 Cleaning intermediate files..."
	rm -f data/discourse/topics.json
	rm -f data/discourse/documents.pkl
	@echo "✓ Intermediate files removed"

# Status
status:
	@echo "📊 Pipeline Status"
	@echo "=================="
	@if [ -f "data/discourse/discourse.db" ]; then \
		echo "✓ Database: $$(sqlite3 data/discourse/discourse.db 'SELECT COUNT(*) FROM topics') topics, $$(sqlite3 data/discourse/discourse.db 'SELECT COUNT(*) FROM documents') documents"; \
	else \
		echo "✗ Database: Not initialized yet"; \
	fi
	@if [ -f "data/embeddings/.discourse_tracking.json" ]; then \
		echo "✓ Embeddings: $$(jq '.documents | length' data/embeddings/.discourse_tracking.json) documents tracked"; \
	else \
		echo "✗ Embeddings: Not created yet"; \
	fi
