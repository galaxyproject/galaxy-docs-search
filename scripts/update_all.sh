#!/bin/bash
#
# Master Update Script
#
# Runs both GTN and Discourse pipelines to update embeddings.
# Safe to run periodically via cron - only processes changes.
#
# Usage:
#   ./scripts/update_all.sh                  # Incremental updates
#   ./scripts/update_all.sh --force-rebuild  # Full rebuild
#
# Cron example (daily at 2am):
#   0 2 * * * cd /path/to/galaxy-docs-search && ./scripts/update_all.sh

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env exists
if [ ! -f ".env" ]; then
    log_error ".env file not found. Copy .env.example and configure it."
    exit 1
fi

# Source environment variables
set -a
source .env
set +a

# Parse arguments
FORCE_REBUILD=""
if [[ "${1:-}" == "--force-rebuild" ]]; then
    FORCE_REBUILD="--force-rebuild"
    log_warn "Force rebuild enabled"
fi

# Timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
log_info "Starting update at $TIMESTAMP"
echo "=========================================="

# Run GTN Pipeline
log_info "Running GTN Pipeline..."
if uv run python -m pipelines.gtn $FORCE_REBUILD; then
    log_success "GTN Pipeline completed"
else
    log_error "GTN Pipeline failed"
    exit 1
fi

echo ""
echo "=========================================="
echo ""

# Run Discourse Pipeline
log_info "Running Discourse Pipeline..."
if uv run python -m pipelines.discourse $FORCE_REBUILD; then
    log_success "Discourse Pipeline completed"
else
    log_error "Discourse Pipeline failed"
    exit 1
fi

echo ""
echo "=========================================="
log_success "All pipelines completed successfully"
TIMESTAMP_END=$(date '+%Y-%m-%d %H:%M:%S')
log_info "Finished at $TIMESTAMP_END"
