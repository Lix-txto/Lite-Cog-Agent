"""End-to-end integration tests for the composed analysis Agent."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lite_cog_agent.core.agent_budget import AgentBudget, AgentBudgetExceededError
from lite_cog_agent.core.agent_factory import create_agent
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    ToolCall,
)
from lite_cog_agent.core.tool_calling_config import ToolCallingConfig
from lite_cog_agent.llm.exceptions import LLMGatewayError


ResponseFactory = Callable[[LLMRequest], LLMResponse]


def make_response(
    *,
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> LLMResponse:
    """Build a deterministic fake LLM response."""
    calls = tool_calls or []

    return LLMResponse(
        content=content,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        model="fake-model",
    )


class FakeProvider:
    """Return predefined or request-dependent LLM responses."""

    def __init__(
        self,
        responses: list[LLMResponse | ResponseFactory],
    ) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []
        self.closed = False

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.requests.append(copy.deepcopy(request))

        if not self._responses:
            raise AssertionError(
                "FakeProvider received an unexpected LLM request."
            )

        response = self._responses.pop(0)

        if callable(response):
            return response(request)

        return response

    async def close(self) -> None:
        self.closed = True


def make_tool_call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
    )


def find_nested_value(
    value: Any,
    key: str,
) -> Any:
    """Find the first occurrence of a key in nested JSON data."""
    if isinstance(value, dict):
        if key in value:
            return value[key]

        for child in value.values():
            try:
                return find_nested_value(child, key)
            except KeyError:
                continue

    elif isinstance(value, list):
        for child in value:
            try:
                return find_nested_value(child, key)
            except KeyError:
                continue

    raise KeyError(f"Key not found: {key}")


def get_latest_tool_payload(request: LLMRequest) -> dict[str, Any]:
    """Decode the most recent tool message from an LLM request."""
    tool_messages = [
        message
        for message in request.messages
        if message.role == "tool"
    ]

    if not tool_messages:
        raise AssertionError("Request does not contain a tool message.")

    return json.loads(tool_messages[-1].content)


def get_nested_value(payload: dict[str, Any], key: str) -> Any:
    return find_nested_value(payload, key)


@pytest.mark.asyncio
async def test_composed_agent_runs_complete_data_analysis_workflow(
    tmp_path: Path,
) -> None:
    """Test dataset loading, cleaning, statistics and visualization."""
    dataset_root = tmp_path / "datasets"
    artifact_root = tmp_path / "artifacts"

    dataset_root.mkdir(parents=True)

    source_file = dataset_root / "users.csv"
    source_file.write_text(
        "name,age,score\n"
        "Alice,20,80\n"
        "Bob,30,90\n"
        ",40,100\n"
        "David,50,110\n",
        encoding="utf-8",
    )

    # Dataset IDs are generated at runtime. Each subsequent fake
    # response reads the actual output of the preceding tool call.
    def respond_with_cleaning(request: LLMRequest) -> LLMResponse:
        load_payload = get_latest_tool_payload(request)
        original_dataset_id = get_nested_value(
            load_payload,
            "dataset_id",
        )

        return make_response(
            tool_calls=[
                make_tool_call(
                    "call_clean",
                    "dataset.clean",
                    {
                        "dataset_id": original_dataset_id,
                        "steps": [
                            {
                                "operation": "missing.drop",
                                "parameters": {"how": "any"},
                            }
                        ],
                    },
                )
            ]
        )

    def respond_with_descriptive_analysis(
        request: LLMRequest,
    ) -> LLMResponse:
        cleaning_payload = get_latest_tool_payload(request)
        cleaned_dataset_id = get_nested_value(
            cleaning_payload,
            "output_dataset_id",
        )

        return make_response(
            tool_calls=[
                make_tool_call(
                    "call_descriptive",
                    "analysis.descriptive",
                    {
                        "dataset_id": cleaned_dataset_id,
                        "columns": ["age", "score"],
                    },
                )
            ]
        )

    def respond_with_histogram(
        request: LLMRequest,
    ) -> LLMResponse:
        """Build a histogram request using the analyzed dataset."""
        descriptive_payload = get_latest_tool_payload(request)

        cleaned_dataset_id = get_nested_value(
            descriptive_payload,
            "dataset_id",
        )

        return make_response(
            tool_calls=[
                make_tool_call(
                    "call_histogram",
                    "visualization.histogram",
                    {
                        "dataset_id": cleaned_dataset_id,
                        "column": "age",
                        "bins": 5,
                    },
                )
            ]
        )

    provider = FakeProvider(
        [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "call_load",
                        "dataset.load",
                        {"file_path": "users.csv"},
                    )
                ]
            ),
            respond_with_cleaning,
            respond_with_descriptive_analysis,
            respond_with_histogram,
            make_response(
                content=(
                    "数据分析完成：已加载数据、删除缺失值、"
                    "完成描述性统计，并生成 age 直方图。"
                )
            ),
        ]
    )

    gateway = LLMGateway(provider=provider)
    context = ExecutionContext()

    try:
        agent = create_agent(
            gateway,
            context=context,
            dataset_root=dataset_root,
            artifact_root=artifact_root,
            session_id="integration-analysis-session",
            budget=AgentBudget(
                max_iterations=10,
                max_tool_calls=10,
            ),
        )

        response = await agent.run(
            "加载 users.csv，删除缺失值，分析 age 和 score，"
            "并生成 age 直方图。"
        )

        # ---------------------------------------------------------
        # 1. Agent composition and final response
        # ---------------------------------------------------------
        assert response.content is not None
        assert "数据分析完成" in response.content
        assert len(provider.requests) == 5

        # The Agent facade exposes its loop internally.
        state = agent._loop._state

        assert state.session_id == "integration-analysis-session"
        assert state.is_running is False
        assert state.iteration_count == 5
        assert state.tool_call_count == 4

        # ---------------------------------------------------------
        # 2. Dataset loading and cleaning
        # ---------------------------------------------------------
        load_payload = get_latest_tool_payload(provider.requests[1])
        original_dataset_id = get_nested_value(
            load_payload,
            "dataset_id",
        )

        cleaning_payload = get_latest_tool_payload(provider.requests[2])
        cleaned_dataset_id = get_nested_value(
            cleaning_payload,
            "output_dataset_id",
        )

        assert original_dataset_id != cleaned_dataset_id
        assert set(context.dataset_ids) == {
            original_dataset_id,
            cleaned_dataset_id,
        }

        original_dataframe = context.get_dataset(original_dataset_id)
        cleaned_dataframe = context.get_dataset(cleaned_dataset_id)

        assert len(original_dataframe) == 4
        assert len(cleaned_dataframe) == 3
        assert cleaned_dataframe["name"].isna().sum() == 0

        # The original dataset must remain unchanged.
        assert original_dataframe["name"].isna().sum() == 1

        # ---------------------------------------------------------
        # 3. Descriptive analysis uses the cleaned dataset
        # ---------------------------------------------------------
        descriptive_payload = get_latest_tool_payload(
            provider.requests[3]
        )

        descriptive_dataset_id = get_nested_value(
            descriptive_payload,
            "dataset_id",
        )

        assert descriptive_dataset_id == cleaned_dataset_id

        statistics = get_nested_value(
            descriptive_payload,
            "statistics",
        )

        statistics_by_column = {
            item["column"]: item
            for item in statistics
        }

        assert statistics_by_column["age"]["count"] == 3
        assert statistics_by_column["age"]["mean"] == pytest.approx(
            100 / 3
        )

        assert statistics_by_column["score"]["count"] == 3
        assert statistics_by_column["score"]["mean"] == pytest.approx(
            280 / 3
        )

        # ---------------------------------------------------------
        # 4. Histogram uses the cleaned dataset and saves an artifact
        # ---------------------------------------------------------
        histogram_payload = get_latest_tool_payload(
            provider.requests[4]
        )

        histogram_dataset_id = get_nested_value(
            histogram_payload,
            "dataset_id",
        )

        assert histogram_dataset_id == cleaned_dataset_id

        artifact_filename = get_nested_value(
            histogram_payload,
            "filename",
        )

        artifact_size = get_nested_value(
            histogram_payload,
            "size_bytes",
        )

        artifact_path = artifact_root / artifact_filename

        assert artifact_path.is_file()
        assert artifact_path.stat().st_size > 0
        assert artifact_path.stat().st_size == artifact_size
        assert artifact_path.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )

        # ---------------------------------------------------------
        # 5. Agent message history
        # ---------------------------------------------------------
        messages = state.message_manager.get_messages()

        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[-1].role == "assistant"
        assert messages[-1].content == response.content

        tool_call_names = [
            tool_call.name
            for message in messages
            if message.role == "assistant"
            for tool_call in (message.tool_calls or [])
        ]

        assert tool_call_names == [
            "dataset.load",
            "dataset.clean",
            "analysis.descriptive",
            "visualization.histogram",
        ]

        tool_messages = [
            message
            for message in messages
            if message.role == "tool"
        ]

        assert len(tool_messages) == 4

    finally:
        await gateway.close()
        context.close()



@pytest.mark.asyncio
async def test_composed_agent_recovers_after_provider_failure(
    tmp_path: Path,
) -> None:
    """A failed provider request must not corrupt Agent history."""

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    def fail_provider(request: LLMRequest) -> LLMResponse:
        raise RuntimeError("Simulated provider failure.")

    provider = FakeProvider(
        [
            fail_provider,
            make_response(content="Retry succeeded."),
        ]
    )

    gateway = LLMGateway(provider=provider)
    context = ExecutionContext()

    try:
        agent = create_agent(
            gateway,
            context=context,
            dataset_root=dataset_root,
            session_id="provider-failure-session",
        )

        state = agent._loop._state
        manager = state.message_manager

        with pytest.raises(LLMGatewayError):
            await agent.run("第一次请求。")

        # The failed request must not commit partial history.
        assert manager.get_messages() == []
        assert state.is_running is False
        assert state.iteration_count == 0
        assert state.tool_call_count == 0

        # The same Agent instance must remain usable.
        response = await agent.run("重试请求。")

        assert response.content == "Retry succeeded."
        assert len(provider.requests) == 2

        messages = manager.get_messages()

        assert [message.role for message in messages] == [
            "system",
            "user",
            "assistant",
        ]
        assert messages[1].content == "重试请求。"
        assert messages[-1].content == response.content

        assert state.is_running is False
        assert state.iteration_count == 1
        assert state.tool_call_count == 0

    finally:
        await gateway.close()
        context.close()


@pytest.mark.asyncio
async def test_composed_agent_iteration_budget_blocks_next_request(
    tmp_path: Path,
) -> None:
    """Iteration budget must block the next LLM request."""

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    source_file = dataset_root / "users.csv"
    source_file.write_text(
        "name,age\nAlice,20\nBob,30\n",
        encoding="utf-8",
    )

    provider = FakeProvider(
        [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "call_load_iteration",
                        "dataset.load",
                        {"file_path": "users.csv"},
                    )
                ]
            )
        ]
    )

    gateway = LLMGateway(provider=provider)
    context = ExecutionContext()

    try:
        agent = create_agent(
            gateway,
            context=context,
            dataset_root=dataset_root,
            session_id="iteration-budget-session",
            budget=AgentBudget(
                max_iterations=1,
                max_tool_calls=10,
            ),
        )

        state = agent._loop._state
        original_history = state.message_manager.get_messages()

        with pytest.raises(AgentBudgetExceededError) as exc_info:
            await agent.run("加载 users.csv。")

        assert exc_info.value.resource == "iterations"
        assert exc_info.value.limit == 1
        assert exc_info.value.used == 1

        # One LLM request was completed. The next was blocked
        # before reaching the provider.
        assert len(provider.requests) == 1

        # The first tool call was allowed and executed.
        assert len(context.dataset_ids) == 1
        assert state.tool_call_count == 1

        # The entire Agent request failed transactionally.
        assert state.message_manager.get_messages() == original_history
        assert state.iteration_count == 1
        assert state.is_running is False

    finally:
        await gateway.close()
        context.close()


@pytest.mark.asyncio
async def test_composed_agent_tool_call_budget_rejects_entire_batch(
    tmp_path: Path,
) -> None:
    """An oversized tool-call batch must never reach the dispatcher."""

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    source_file = dataset_root / "users.csv"
    source_file.write_text(
        "name,age\nAlice,20\nBob,30\n",
        encoding="utf-8",
    )

    provider = FakeProvider(
        [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "call_load_1",
                        "dataset.load",
                        {"file_path": "users.csv"},
                    ),
                    make_tool_call(
                        "call_load_2",
                        "dataset.load",
                        {"file_path": "users.csv"},
                    ),
                ]
            )
        ]
    )

    gateway = LLMGateway(provider=provider)
    context = ExecutionContext()

    try:
        agent = create_agent(
            gateway,
            context=context,
            dataset_root=dataset_root,
            session_id="tool-budget-session",
            budget=AgentBudget(
                max_iterations=10,
                max_tool_calls=1,
            ),
            calling_config=ToolCallingConfig(
                max_tool_calls_per_round=2,
            ),
        )

        state = agent._loop._state
        original_history = state.message_manager.get_messages()

        with pytest.raises(AgentBudgetExceededError) as exc_info:
            await agent.run("加载 users.csv。")

        assert exc_info.value.resource == "tool_calls"
        assert exc_info.value.limit == 1
        assert exc_info.value.used == 0
        assert exc_info.value.requested == 2

        # The LLM response was received, but the entire batch
        # was rejected before dispatch.
        assert len(provider.requests) == 1
        assert context.dataset_ids == ()

        assert state.iteration_count == 1
        assert state.tool_call_count == 0
        assert state.is_running is False

        # No partial tool-call history is committed.
        assert state.message_manager.get_messages() == original_history

    finally:
        await gateway.close()
        context.close()


@pytest.mark.asyncio
async def test_composed_agent_recovers_from_tool_failure(
    tmp_path: Path,
) -> None:
    """A failed tool result should be available for final synthesis."""

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    def synthesize_after_tool_failure(
        request: LLMRequest,
    ) -> LLMResponse:
        payload = get_latest_tool_payload(request)

        assert payload["success"] is False
        assert payload["error"] is not None

        return make_response(
            content="数据文件加载失败，未执行后续分析。"
        )

    provider = FakeProvider(
        [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "call_missing_dataset",
                        "dataset.load",
                        {"file_path": "missing.csv"},
                    )
                ]
            ),
            synthesize_after_tool_failure,
        ]
    )

    gateway = LLMGateway(provider=provider)
    context = ExecutionContext()

    try:
        agent = create_agent(
            gateway,
            context=context,
            dataset_root=dataset_root,
            session_id="tool-failure-session",
        )

        response = await agent.run(
            "加载 missing.csv 并分析数据。"
        )

        assert response.content == (
            "数据文件加载失败，未执行后续分析。"
        )

        assert len(provider.requests) == 2
        assert context.dataset_ids == ()

        state = agent._loop._state

        assert state.is_running is False
        assert state.iteration_count == 2
        assert state.tool_call_count == 1

        messages = state.message_manager.get_messages()

        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[-2].role == "tool"
        assert messages[-1].role == "assistant"

        tool_payload = json.loads(messages[-2].content)

        assert tool_payload["success"] is False
        assert messages[-1].content == response.content

    finally:
        await gateway.close()
        context.close()
