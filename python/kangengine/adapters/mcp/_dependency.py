"""Load the optional SDK only when MCP is enabled."""

from importlib import import_module
from importlib.metadata import version


class McpUnavailableError(ImportError):
    """The optional MCP SDK is missing or outside the supported version range."""


def load_mcp():
    try:
        sdk = import_module("mcp")
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        raise McpUnavailableError(
            "KangEngine MCP requires the optional SDK; install kangengine[mcp]."
        ) from error
    parts = version("mcp").split(".")
    if int(parts[0]) != 2 or int(parts[1]) < 3:
        raise McpUnavailableError(
            "KangEngine MCP requires mcp>=2.3,<3; install kangengine[mcp]."
        )
    return sdk
