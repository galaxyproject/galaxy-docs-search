"""
galaxy-docs-mcp -- FastMCP server exposing semantic search over the
galaxy-docs-search Chroma collections. Thin wrapper around retrieval/.
"""

from mcp_server.server import build_server

__all__ = ["build_server"]
