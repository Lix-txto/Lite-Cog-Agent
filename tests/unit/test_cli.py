import asyncio
from types import SimpleNamespace

from lite_cog_agent import cli
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMResponse,
    LLMRequest,
    TokenUsage,
)


def test_print_help(capsys):
    cli.print_help()

    output = capsys.readouterr().out

    assert "help" in output
    assert "status" in output
    assert "usage" in output
    assert "reset" in output
    assert "exit" in output


def test_print_status(capsys):
    gateway = SimpleNamespace(
        usage_summary=SimpleNamespace(request_count=2)
    )

    cli.print_status(gateway, connected=True)

    output = capsys.readouterr().out

    assert "Lite-Cog-Agent" in output
    assert "Connected (request succeeded)" in output
    assert "Requests     : 2" in output


def test_print_usage(capsys):
    gateway = SimpleNamespace(
        usage_summary=SimpleNamespace(
            request_count=3,
            prompt_tokens=300,
            completion_tokens=150,
            total_tokens=450,
        )
    )

    cli.print_usage(gateway)

    output = capsys.readouterr().out

    assert "Requests           : 3" in output
    assert "Prompt Tokens      : 300" in output
    assert "Completion Tokens  : 150" in output
    assert "Total Tokens       : 450" in output


def test_send_message_builds_llm_request(monkeypatch):
    captured = {}

    class FakeGateway:
        async def chat(self, request):
            captured["request"] = request

            return LLMResponse(
                content="测试回答",
                model="deepseek-flash",
                finish_reason="stop",
                usage=TokenUsage(10, 20, 30),
            )

    response = asyncio.run(
        cli.send_message(
            FakeGateway(),
            "请介绍一下 pandas。",
        )
    )

    request = captured["request"]

    assert isinstance(request, LLMRequest)
    assert len(request.messages) == 2

    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.messages[1].content == "请介绍一下 pandas。"

    assert request.temperature == 0.7
    assert request.max_tokens == 1024

    assert response.content == "测试回答"
