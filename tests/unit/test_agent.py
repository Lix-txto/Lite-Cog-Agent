"""Unit tests for the public Agent interface."""

from __future__ import annotations

import pytest

from lite_cog_agent.core.agent import Agent
from lite_cog_agent.llm.schemas import LLMMessage, LLMResponse


class FakeAgentLoop:
    def __init__(self):
        self.session_id = "test-session"
        self.received_inputs = []
        self.reset_called = False
        self.messages = []
        self.response = LLMResponse(
            content="Agent response."
        )

    async def run(self, user_input: str) -> LLMResponse:
        self.received_inputs.append(user_input)
        return self.response

    def get_messages(self) -> list[LLMMessage]:
        return self.messages

    def reset(self) -> None:
        self.reset_called = True


@pytest.mark.asyncio
async def test_agent_delegates_run_to_agent_loop():
    loop = FakeAgentLoop()
    agent = Agent(loop)

    response = await agent.run("分析数据。")

    assert response.content == "Agent response."
    assert loop.received_inputs == ["分析数据。"]


def test_agent_exposes_session_id():
    agent = Agent(FakeAgentLoop())

    assert agent.session_id == "test-session"

def test_agent_delegates_get_messages_to_agent_loop():
    loop = FakeAgentLoop()
    expected_messages = [
        LLMMessage(
            role="user",
            content="分析销售数据。",
        ),
    ]
    loop.messages = expected_messages

    agent = Agent(loop)

    messages = agent.get_messages()

    assert messages == expected_messages
    assert messages is loop.messages

def test_agent_delegates_reset_to_agent_loop():
    loop = FakeAgentLoop()
    agent = Agent(loop)

    agent.reset()

    assert loop.reset_called is True


def test_agent_rejects_invalid_loop():
    with pytest.raises(TypeError, match="provide run"):
        Agent(object())


def test_agent_rejects_loop_without_reset():
    class InvalidLoop:
        session_id = "test-session"

        async def run(self, user_input):
            return LLMResponse(content="response")

    with pytest.raises(TypeError, match="provide reset"):
        Agent(InvalidLoop())


def test_agent_rejects_loop_without_string_session_id():
    class InvalidLoop:
        session_id = None

        async def run(self, user_input):
            return LLMResponse(content="response")

        def reset(self):
            pass

    with pytest.raises(TypeError, match="session_id"):
        Agent(InvalidLoop())

def test_agent_rejects_loop_without_get_messages():
    class InvalidLoop:
        session_id = "test-session"

        async def run(self, user_input):
            return LLMResponse(content="response")

        def reset(self):
            pass

    with pytest.raises(
        TypeError,
        match="provide get_messages",
    ):
        Agent(InvalidLoop())
