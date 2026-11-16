import pytest

from lite_cog_agent.llm.exceptions import LLMResponseError
from lite_cog_agent.llm.response_validator import (
    LLMResponseValidator,
)
from lite_cog_agent.llm.schemas import (
    LLMResponse,
    TokenUsage,
    ToolCall,
)


def test_validate_normal_text_response():
    response = LLMResponse(
        content="Analysis completed.",
        finish_reason="stop",
        model="deepseek-flash",
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )

    result = LLMResponseValidator.validate(response)

    assert result is response


def test_validate_tool_call_response():
    response = LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_001",
                name="analyze_dataset",
                arguments={"file_path": "data.csv"},
            )
        ],
        finish_reason="tool_calls",
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130,
        ),
    )

    result = LLMResponseValidator.validate(response)

    assert len(result.tool_calls) == 1


def test_reject_empty_response():
    response = LLMResponse(
        content=None,
        tool_calls=[],
    )

    with pytest.raises(LLMResponseError):
        LLMResponseValidator.validate(response)


def test_reject_inconsistent_token_usage():
    response = LLMResponse(
        content="Completed.",
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=200,
        ),
    )

    with pytest.raises(LLMResponseError):
        LLMResponseValidator.validate(response)


@pytest.mark.parametrize(
    "usage",
    [
        TokenUsage(-1, 10, 9),
        TokenUsage(10, -1, 9),
        TokenUsage(10, 10, -1),
        TokenUsage(1.5, 2, 3),
        TokenUsage(True, 2, 3),
    ],
)
def test_reject_invalid_token_values(usage):
    with pytest.raises(LLMResponseError):
        LLMResponseValidator.validate_usage(usage)


def test_reject_wrong_response_type():
    with pytest.raises(LLMResponseError):
        LLMResponseValidator.validate("not a response")
