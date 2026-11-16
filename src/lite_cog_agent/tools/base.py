"""
Abstract base class for all Lite-Cog-Agent tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lite_cog_agent.llm.schemas import ToolDefinition
from lite_cog_agent.tools.schemas import ToolResult


class BaseTool(ABC):
    """
    Abstract interface for executable tools.

    Every tool must provide:
    - A unique name.
    - A human-readable description.
    - A JSON Schema describing its arguments.
    - An asynchronous execute method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """
        Return the JSON Schema describing tool arguments.

        The schema should use an object as its root type.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Execute the tool using validated arguments.

        Implementations should return ToolResult for both
        successful and expected failed executions.
        """
        raise NotImplementedError

    def to_definition(self) -> ToolDefinition:
        """
        Convert the tool metadata into the unified LLM schema.

        This definition can be passed to LLMGateway so that
        the model knows which tools are available.
        """

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
