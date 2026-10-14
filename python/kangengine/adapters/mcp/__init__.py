"""Optional local MCP access to an App's automation service."""

from ._dependency import McpUnavailableError
from .server import McpServer

__all__ = ["McpServer", "McpUnavailableError"]
