"""Unit tests for Agent execution budget."""

import pytest

from lite_cog_agent.core.agent_budget import (
    AgentBudget,
    AgentBudgetExceededError,
)


class TestAgentBudget:
    """Tests for AgentBudget configuration."""

    def test_default_budget_values(self):
        budget = AgentBudget()

        assert budget.max_iterations == 20
        assert budget.max_tool_calls == 30

    def test_custom_budget_values(self):
        budget = AgentBudget(
            max_iterations=5,
            max_tool_calls=12,
        )

        assert budget.max_iterations == 5
        assert budget.max_tool_calls == 12

    @pytest.mark.parametrize(
        "field",
        ["max_iterations", "max_tool_calls"],
    )
    @pytest.mark.parametrize(
        "value",
        [True, False, 1.5, "10", None],
    )
    def test_budget_rejects_non_integer_values(self, field, value):
        with pytest.raises(TypeError):
            AgentBudget(**{field: value})

    @pytest.mark.parametrize(
        "field",
        ["max_iterations", "max_tool_calls"],
    )
    @pytest.mark.parametrize(
        "value",
        [0, -1, -10],
    )
    def test_budget_rejects_non_positive_values(self, field, value):
        with pytest.raises(ValueError):
            AgentBudget(**{field: value})

    def test_budget_is_immutable(self):
        budget = AgentBudget()

        with pytest.raises(AttributeError):
            budget.max_iterations = 100

        with pytest.raises(AttributeError):
            budget.max_tool_calls = 100


class TestAgentBudgetExceededError:
    """Tests for AgentBudgetExceededError."""

    def test_iteration_budget_error_attributes(self):
        error = AgentBudgetExceededError(
            resource="iterations",
            limit=20,
            used=20,
        )

        assert error.resource == "iterations"
        assert error.limit == 20
        assert error.used == 20
        assert error.requested == 1

    def test_tool_call_budget_error_attributes(self):
        error = AgentBudgetExceededError(
            resource="tool_calls",
            limit=30,
            used=25,
            requested=8,
        )

        assert error.resource == "tool_calls"
        assert error.limit == 30
        assert error.used == 25
        assert error.requested == 8

    def test_error_message(self):
        error = AgentBudgetExceededError(
            resource="iterations",
            limit=20,
            used=20,
        )

        assert str(error) == (
            "Agent iterations budget exceeded: "
            "limit=20, used=20, requested=1."
        )

    def test_error_inherits_runtime_error(self):
        error = AgentBudgetExceededError(
            resource="tool_calls",
            limit=30,
            used=30,
            requested=2,
        )

        assert isinstance(error, RuntimeError)
