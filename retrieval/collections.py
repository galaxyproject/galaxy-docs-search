"""
Collection metadata: short keys (used in the MCP tool's `collection` enum),
the actual Chroma collection names, and the human-readable descriptions
that the LLM reads to pick between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CollectionKey = Literal["faqs", "tutorials", "discourse", "workflows"]


@dataclass(frozen=True)
class CollectionInfo:
    key: CollectionKey
    chroma_name: str
    title: str
    description: str


COLLECTIONS: dict[CollectionKey, CollectionInfo] = {
    "faqs": CollectionInfo(
        key="faqs",
        chroma_name="galaxy_faqs",
        title="Galaxy FAQs",
        description=(
            "Short canonical Q&A pairs from the Galaxy FAQ. Best for quick "
            "'how do I X' questions where you want the official one-paragraph "
            "answer. Small corpus."
        ),
    ),
    "tutorials": CollectionInfo(
        key="tutorials",
        chroma_name="gtn_tutorials",
        title="GTN Tutorials",
        description=(
            "Galaxy Training Network tutorial chunks. Best when the user "
            "wants a walkthrough or step-by-step explanation of a specific "
            "analysis or technique."
        ),
    ),
    "discourse": CollectionInfo(
        key="discourse",
        chroma_name="discourse_posts",
        title="Galaxy Discourse",
        description=(
            "Galaxy community forum threads (help.galaxyproject.org). Best "
            "for troubleshooting specific error messages, edge cases, or "
            "'I tried X and got Y' questions that benefit from real-world "
            "resolutions. Largest corpus."
        ),
    ),
    "workflows": CollectionInfo(
        key="workflows",
        chroma_name="iwc_workflows",
        title="IWC Workflows",
        description=(
            "Curated workflow descriptions from the Intergalactic Workflow "
            "Commission (IWC). Best for 'is there a workflow for X' or "
            "recommending a published analysis pipeline."
        ),
    ),
}


def collection_by_chroma_name(name: str) -> CollectionInfo | None:
    """Reverse-lookup so the manifest's chroma_name keys can be mapped back."""
    for info in COLLECTIONS.values():
        if info.chroma_name == name:
            return info
    return None
