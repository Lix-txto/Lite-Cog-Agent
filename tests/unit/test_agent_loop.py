"""Unit tests for AgentLoop."""

from __future__ import annotations

import pytest

from lite_cog_agent.core.agent_budget import AgentBudget, AgentBudgetExceededError
from lite_cog_agent.core.agent_loop import (
    AgentLoop,
    AgentLoopError,
)
from lite_cog_agent.core.agent_state import (
    AgentState,
    AgentStateError,
)
from lite_cog_agent.core.messages import MessageManager
from lite_cog_agent.core.prompts import AGENT_SYSTEM_PROMPT
from lite_cog_agent.core.tool_calling import ToolCallingExecution
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMResponse,
    ToolCall,
)


class FakeToolCallingService:
    """Fake ToolCallingService for AgentLoop tests."""

    def __init__(
        self,
        *,
        response: LLMResponse | None = None,
        error: Exception | None = None,
        transform=None,
        llm_request_count: int = 1,
        tool_call_batches: list[int] | None = None,
    ) -> None:
        self.response = response or LLMResponse(
            content="分析完成。"
        )
        self.error = error
        self.transform = transform
        self.llm_request_count = llm_request_count
        self.tool_call_batches = (
            tool_call_batches
            if tool_call_batches is not None
            else []
        )

        self.requests = []
        self.dispatched_tool_calls = 0

    async def run_with_history(
        self,
        request,
        *,
        on_iteration=None,
        before_llm_request=None,
        before_tool_dispatch=None,
    ):
        self.requests.append(request)

        for _ in range(self.llm_request_count):
            if before_llm_request is not None:
                before_llm_request()

            if self.error is not None:
                raise self.error

            if on_iteration is not None:
                on_iteration()

        for batch_size in self.tool_call_batches:
            if before_tool_dispatch is not None:
                before_tool_dispatch(batch_size)

            self.dispatched_tool_calls += batch_size

        if self.transform is not None:
            return self.transform(request)

        return ToolCallingExecution(
            response=self.response,
            messages=[
                *request.messages,
                LLMMessage(
                    role="assistant",
                    content=self.response.content,
                ),
            ],
        )



def create_loop(
    *,
    messages: list[LLMMessage] | None = None,
    service: FakeToolCallingService | None = None,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
    budget: AgentBudget | None = None,
):
    manager = MessageManager(messages)
    state = AgentState(message_manager=manager)

    if service is None:
        service = FakeToolCallingService()

    loop = AgentLoop(
        tool_calling_service=service,
        state=state,
        system_prompt=system_prompt,
        budget=budget,
    )

    return loop, state, manager, service


@pytest.mark.asyncio
async def test_first_request_commits_system_user_and_assistant_messages():
    loop, state, manager, service = create_loop()

    response = await loop.run("分析销售数据。")

    messages = manager.get_messages()

    assert response.content == "分析完成。"
    assert len(messages) == 3

    assert messages[0].role == "system"
    assert messages[0].content == AGENT_SYSTEM_PROMPT

    assert messages[1].role == "user"
    assert messages[1].content == "分析销售数据。"

    assert messages[2].role == "assistant"
    assert messages[2].content == "分析完成。"

    assert len(service.requests) == 1
    assert state.is_running is False


@pytest.mark.asyncio
async def test_subsequent_request_preserves_history_without_duplicate_prompt():
    existing_messages = [
        LLMMessage(
            role="system",
            content=AGENT_SYSTEM_PROMPT,
        ),
        LLMMessage(
            role="user",
            content="第一次请求。",
        ),
        LLMMessage(
            role="assistant",
            content="第一次回答。",
        ),
    ]

    loop, _, manager, _ = create_loop(
        messages=existing_messages
    )

    await loop.run("第二次请求。")

    messages = manager.get_messages()

    assert len(messages) == 5
    assert sum(
        message.role == "system"
        for message in messages
    ) == 1

    assert messages[-2].role == "user"
    assert messages[-2].content == "第二次请求。"

    assert messages[-1].role == "assistant"
    assert messages[-1].content == "分析完成。"


@pytest.mark.asyncio
async def test_failed_execution_does_not_commit_new_messages():
    loop, state, manager, _ = create_loop(
        service=FakeToolCallingService(
            error=RuntimeError("LLM provider unavailable.")
        )
    )

    with pytest.raises(
        RuntimeError,
        match="LLM provider unavailable",
    ):
        await loop.run("分析数据。")

    assert len(manager) == 0
    assert state.is_running is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_input",
    [
        "",
        " ",
        "\n\t",
    ],
)
async def test_empty_user_input_is_rejected_before_start(user_input):
    loop, state, manager, service = create_loop()

    with pytest.raises(ValueError, match="cannot be empty"):
        await loop.run(user_input)

    assert state.is_running is False
    assert state.iteration_count == 0
    assert len(manager) == 0
    assert service.requests == []


@pytest.mark.asyncio
async def test_non_string_user_input_is_rejected_before_start():
    loop, state, manager, service = create_loop()

    with pytest.raises(TypeError, match="must be a string"):
        await loop.run(123)

    assert state.is_running is False
    assert len(manager) == 0
    assert service.requests == []


@pytest.mark.asyncio
async def test_existing_history_without_system_prompt_is_rejected():
    existing_messages = [
        LLMMessage(
            role="user",
            content="历史请求。",
        ),
        LLMMessage(
            role="assistant",
            content="历史回答。",
        ),
    ]

    loop, state, manager, service = create_loop(
        messages=existing_messages
    )

    with pytest.raises(
        AgentLoopError,
        match="system prompt",
    ):
        await loop.run("新请求。")

    assert len(manager) == 2
    assert state.is_running is False
    assert service.requests == []


@pytest.mark.asyncio
async def test_existing_history_with_different_system_prompt_is_rejected():
    existing_messages = [
        LLMMessage(
            role="system",
            content="Different system prompt.",
        ),
        LLMMessage(
            role="user",
            content="历史请求。",
        ),
        LLMMessage(
            role="assistant",
            content="历史回答。",
        ),
    ]

    loop, state, manager, service = create_loop(
        messages=existing_messages
    )

    with pytest.raises(
        AgentLoopError,
        match="system prompt",
    ):
        await loop.run("新请求。")

    assert len(manager) == 3
    assert state.is_running is False
    assert service.requests == []


@pytest.mark.asyncio
async def test_modified_execution_prefix_is_rejected():
    def transform(request):
        request.messages[0].content = "Tampered prompt."

        return ToolCallingExecution(
            response=LLMResponse(content="分析完成。"),
            messages=[
                *request.messages,
                LLMMessage(
                    role="assistant",
                    content="分析完成。",
                ),
            ],
        )

    loop, state, manager, _ = create_loop(
        service=FakeToolCallingService(
            transform=transform
        )
    )

    with pytest.raises(
        AgentLoopError,
        match="modified the original",
    ):
        await loop.run("分析数据。")

    assert len(manager) == 0
    assert state.is_running is False


@pytest.mark.asyncio
async def test_invalid_execution_history_is_rejected():
    def transform(request):
        return ToolCallingExecution(
            response=LLMResponse(content="分析完成。"),
            messages=[
                *request.messages,
                LLMMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="dataset.profile",
                            arguments={},
                        )
                    ],
                ),
                LLMMessage(
                    role="assistant",
                    content="分析完成。",
                ),
            ],
        )

    loop, state, manager, _ = create_loop(
        service=FakeToolCallingService(
            transform=transform
        )
    )

    with pytest.raises(
        AgentLoopError,
        match="invalid message history",
    ):
        await loop.run("分析数据。")

    assert len(manager) == 0
    assert state.is_running is False


@pytest.mark.asyncio
async def test_final_response_must_match_final_assistant_message():
    def transform(request):
        return ToolCallingExecution(
            response=LLMResponse(
                content="Expected response."
            ),
            messages=[
                *request.messages,
                LLMMessage(
                    role="assistant",
                    content="Different response.",
                ),
            ],
        )

    loop, state, manager, _ = create_loop(
        service=FakeToolCallingService(
            transform=transform
        )
    )

    with pytest.raises(
        AgentLoopError,
        match="valid final assistant response",
    ):
        await loop.run("分析数据。")

    assert len(manager) == 0
    assert state.is_running is False


@pytest.mark.asyncio
async def test_run_rejects_already_running_state():
    loop, state, manager, service = create_loop()

    state.start()

    with pytest.raises(AgentStateError):
        await loop.run("分析数据。")

    assert state.is_running is True
    assert len(manager) == 0
    assert service.requests == []

    state.finish()


def test_reset_clears_history_and_preserves_session_id():
    loop, state, manager, _ = create_loop(
        messages=[
            LLMMessage(
                role="system",
                content=AGENT_SYSTEM_PROMPT,
            ),
            LLMMessage(
                role="user",
                content="历史请求。",
            ),
            LLMMessage(
                role="assistant",
                content="历史回答。",
            ),
        ]
    )

    session_id = state.session_id

    loop.reset()

    assert len(manager) == 0
    assert state.session_id == session_id
    assert state.iteration_count == 0
    assert state.is_running is False


def test_custom_system_prompt_is_supported():
    custom_prompt = "You are a custom data analysis agent."

    loop, _, manager, _ = create_loop(
        system_prompt=custom_prompt
    )

    assert loop.session_id
    assert len(manager) == 0


@pytest.mark.asyncio
async def test_iteration_count_is_one_for_direct_response():
    loop, state, _, _ = create_loop()

    await loop.run("直接分析数据。")

    assert state.iteration_count == 1
    assert state.is_running is False


@pytest.mark.asyncio
async def test_iteration_count_resets_for_each_request():
    loop, state, _, _ = create_loop()

    await loop.run("第一次请求。")

    assert state.iteration_count == 1

    await loop.run("第二次请求。")

    assert state.iteration_count == 1
    assert state.is_running is False


@pytest.mark.asyncio
async def test_iteration_count_remains_zero_when_llm_fails():
    loop, state, manager, _ = create_loop(
        service=FakeToolCallingService(
            error=RuntimeError("LLM provider unavailable.")
        )
    )

    with pytest.raises(
        RuntimeError,
        match="LLM provider unavailable",
    ):
        await loop.run("分析数据。")

    assert state.iteration_count == 0
    assert state.is_running is False
    assert len(manager) == 0



@pytest.mark.asyncio
async def test_iteration_budget_blocks_next_llm_request():
    service = FakeToolCallingService(
        llm_request_count=2,
    )

    loop, state, manager, service = create_loop(
        service=service,
        budget=AgentBudget(
            max_iterations=1,
            max_tool_calls=10,
        ),
    )

    original_history = manager.get_messages()

    with pytest.raises(AgentBudgetExceededError) as exc_info:
        await loop.run("分析销售数据")

    assert exc_info.value.resource == "iterations"
    assert exc_info.value.limit == 1
    assert exc_info.value.used == 1

    assert state.iteration_count == 1
    assert state.is_running is False
    assert len(service.requests) == 1
    assert manager.get_messages() == original_history


@pytest.mark.asyncio
async def test_tool_call_budget_rejects_entire_batch():
    service = FakeToolCallingService(
        tool_call_batches=[3],
    )

    loop, state, manager, service = create_loop(
        service=service,
        budget=AgentBudget(
            max_iterations=10,
            max_tool_calls=2,
        ),
    )

    original_history = manager.get_messages()

    with pytest.raises(AgentBudgetExceededError) as exc_info:
        await loop.run("分析销售数据")

    assert exc_info.value.resource == "tool_calls"
    assert exc_info.value.limit == 2
    assert exc_info.value.used == 0
    assert exc_info.value.requested == 3

    assert state.tool_call_count == 0
    assert service.dispatched_tool_calls == 0
    assert state.is_running is False
    assert manager.get_messages() == original_history


@pytest.mark.asyncio
async def test_tool_call_budget_allows_exact_limit():
    service = FakeToolCallingService(
        tool_call_batches=[2],
    )

    loop, state, manager, service = create_loop(
        service=service,
        budget=AgentBudget(
            max_iterations=10,
            max_tool_calls=2,
        ),
    )

    response = await loop.run("分析销售数据")

    assert response.content == "分析完成。"
    assert state.tool_call_count == 2
    assert service.dispatched_tool_calls == 2
    assert state.is_running is False
    assert len(manager.get_messages()) == 3


@pytest.mark.asyncio
async def test_agent_budget_counters_reset_for_next_request():
    service = FakeToolCallingService(
        llm_request_count=1,
        tool_call_batches=[1],
    )

    loop, state, _manager, service = create_loop(
        service=service,
        budget=AgentBudget(
            max_iterations=1,
            max_tool_calls=1,
        ),
    )

    await loop.run("第一次分析")
    assert state.iteration_count == 1
    assert state.tool_call_count == 1

    await loop.run("第二次分析")

    assert state.iteration_count == 1
    assert state.tool_call_count == 1
    assert state.is_running is False
    assert len(service.requests) == 2


def test_get_messages_returns_empty_history_initially():
    loop, _, _, _ = create_loop()

    assert loop.get_messages() == []


@pytest.mark.asyncio
async def test_get_messages_returns_isolated_history_copy():
    loop, _, manager, _ = create_loop()

    await loop.run("分析销售数据。")

    messages = loop.get_messages()

    assert messages == manager.get_messages()
    assert len(messages) == 3

    # 修改返回值不能污染 AgentLoop 内部历史。
    messages[1].content = "被外部修改的内容"
    messages.append(
        LLMMessage(
            role="user",
            content="额外消息",
        )
    )

    internal_messages = loop.get_messages()

    assert len(internal_messages) == 3
    assert internal_messages[1].content == "分析销售数据。"