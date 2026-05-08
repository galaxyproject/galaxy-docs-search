"""
Read data/embeddings/manifest.json and validate it matches the configured
embedder. The manifest is written by scripts/write_manifest.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


@dataclass
class Manifest:
    model: str
    embeddings_base_url: str
    built_at: str
    collections: dict[str, dict[str, int]]


class ManifestMismatch(RuntimeError):
    """Raised when the on-disk manifest disagrees with the configured embedder."""


def load(embeddings_dir: Path) -> Manifest | None:
    """Load manifest.json, or None if it's missing."""
    path = embeddings_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Manifest(
        model=data.get("model", ""),
        embeddings_base_url=data.get("embeddings_base_url", ""),
        built_at=data.get("built_at", ""),
        collections=data.get("collections", {}),
    )


def validate(manifest: Manifest | None, configured_model: str, *, strict: bool) -> None:
    """
    Confirm the manifest matches what we're about to use as the query embedder.

    - No manifest: warn (older snapshots predate this); if strict, raise.
    - Manifest model mismatch: always raise -- mixing a 4096-dim Chroma store
      with a different-dim query embedder produces silent garbage.
    """
    if manifest is None:
        msg = (
            "No manifest.json next to the embeddings dir. Snapshot may predate "
            "manifest support; run `make manifest` to write one."
        )
        if strict:
            raise ManifestMismatch(msg)
        logger.warning(msg)
        return

    if manifest.model and configured_model and manifest.model != configured_model:
        raise ManifestMismatch(
            f"Embedder mismatch: manifest says {manifest.model!r} produced this "
            f"snapshot, but EMBEDDINGS_MODEL is {configured_model!r}. Either point "
            f"EMBEDDINGS_MODEL at the original embedder or rebuild the snapshot."
        )
