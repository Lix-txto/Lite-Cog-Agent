"""
Tool registry for Lite-Cog-Agent.

The registry manages tool registration, lookup,
unregistration, and LLM tool definition generation.
"""

from __future__ import annotations

from lite_cog_agent.llm.schemas import ToolDefinition
from lite_cog_agent.tools.base import BaseTool


class ToolRegistryError(Exception):
    """Base exception for tool registry operations."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a tool with the same name is already registered."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool does not exist."""


class ToolRegistry:
    """
    Central registry for managing executable tools.

    Tools are stored by their unique names. Registration
    order is preserved by the underlying dictionary.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool.

        Raises:
            TypeError: If the object is not a BaseTool.
            ValueError: If the tool name is empty or invalid.
            DuplicateToolError: If the name is already registered.
        """

        if not isinstance(tool, BaseTool):
            raise TypeError("Registered tool must be an instance of BaseTool.")

        name = tool.name

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")

        if name in self._tools:
            raise DuplicateToolError(f"Tool '{name}' is already registered.")

        self._tools[name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Returns:
            True if a tool was removed, otherwise False.
        """

        if not isinstance(name, str):
            raise TypeError("Tool name must be a string.")

        if name not in self._tools:
            return False

        del self._tools[name]
        return True

    def get(self, name: str) -> BaseTool:
        """
        Retrieve a registered tool by name.

        Raises:
            TypeError: If the name is not a string.
            ToolNotFoundError: If the tool is not registered.
        """

        if not isinstance(name, str):
            raise TypeError("Tool name must be a string.")

        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.") from exc

    def list_tools(self) -> list[BaseTool]:
        """Return a new list containing all registered tools."""

        return list(self._tools.values())

    def get_definitions(self) -> list[ToolDefinition]:
        """
        Return LLM-compatible definitions for all registered tools.

        Each definition is generated through BaseTool.to_definition().
        """

        return [tool.to_definition() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Return the number of registered tools."""

        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        """Check whether a tool name is registered."""

        if not isinstance(name, str):
            return False

        return name in self._tools
