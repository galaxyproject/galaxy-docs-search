"""
In-process retrieval over the galaxy-docs-search Chroma collections.

This module is the single source of truth for "given a query, return the
top-k most-similar chunks from one of the four collections." It's used by:

- GalaxyAI (Python, in-process inside a running Galaxy server) -- imports
  GalaxyDocsRetriever directly.
- galaxy-docs-mcp -- the FastMCP server in mcp_server/ wraps this module
  and exposes a search tool over MCP for non-Python consumers (Loom, etc).
"""

from retrieval.client import GalaxyDocsRetriever, Hit
from retrieval.collections import COLLECTIONS, CollectionKey

__all__ = ["GalaxyDocsRetriever", "Hit", "COLLECTIONS", "CollectionKey"]
