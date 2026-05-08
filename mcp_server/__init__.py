"""
galaxy-docs-mcp -- FastMCP server exposing semantic search over the
chatgxy-materials Chroma collections. Thin wrapper around retrieval/.
"""

from mcp_server.server import build_server

__all__ = ["build_server"]
