from types import SimpleNamespace

import pytest

from lite_cog_agent.llm.exceptions import LLMConfigurationError
from lite_cog_agent.llm.providers.deepseek import DeepSeekProvider
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    ToolCall,
    ToolDefinition,
)


def test_provider_rejects_missing_api_key():
    with pytest.raises(LLMConfigurationError):
        DeepSeekProvider(
            api_key="",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
        )


def test_provider_rejects_empty_base_url():
    with pytest.raises(LLMConfigurationError):
        DeepSeekProvider(
            api_key="test-key",
            base_url="",
            model="deepseek-flash",
        )


def test_provider_rejects_empty_model():
    with pytest.raises(LLMConfigurationError):
        DeepSeekProvider(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="",
        )


def test_provider_accepts_valid_configuration():
    provider = DeepSeekProvider(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )

    assert provider.base_url == "https://api.deepseek.com"
    assert provider.model == "deepseek-flash"

    import asyncio

    asyncio.run(provider.close())


@pytest.mark.parametrize(
    ("internal_name", "expected_external_name"),
    [
        ("dataset.load", "lc_dataset_dload"),
        ("analysis.grouped", "lc_analysis_dgrouped"),
        ("analysis.derived_metric", "lc_analysis_dderived_umetric"),
        (
            "visualization.grouped_bar",
            "lc_visualization_dgrouped_ubar",
        ),
        ("inspect_file", "lc_inspect_ufile"),
        ("foo.bar", "lc_foo_dbar"),
        ("foo_bar", "lc_foo_ubar"),
    ],
)
def test_encode_tool_name(internal_name, expected_external_name):
    assert (
        DeepSeekProvider._encode_tool_name(internal_name)
        == expected_external_name
    )


@pytest.mark.parametrize(
    "internal_name",
    [
        "dataset.load",
        "analysis.grouped",
        "analysis.derived_metric",
        "visualization.grouped_bar",
        "inspect_file",
        "foo.bar",
        "foo_bar",
        "name-with-hyphen",
        "中文工具",
    ],
)
def test_tool_name_encoding_is_reversible(internal_name):
    encoded = DeepSeekProvider._encode_tool_name(internal_name)

    decoded = DeepSeekProvider._decode_tool_name(encoded)

    assert decoded == internal_name


def test_tool_name_encoding_avoids_collisions():
    first = DeepSeekProvider._encode_tool_name("foo.bar")
    second = DeepSeekProvider._encode_tool_name("foo_bar")

    assert first != second


def test_tool_name_encoding_always_uses_valid_api_characters():
    import re

    names = [
        "dataset.load",
        "analysis.grouped",
        "analysis.derived_metric",
        "visualization.grouped_bar",
        "中文工具",
        "foo.bar",
        "foo_bar",
    ]

    for name in names:
        encoded = DeepSeekProvider._encode_tool_name(name)

        assert re.fullmatch(r"[a-zA-Z0-9_-]+", encoded)


def test_convert_tools_encodes_tool_names():
    tools = [
        ToolDefinition(
            name="dataset.load",
            description="Load a dataset.",
            parameters={
                "type": "object",
                "properties": {},
            },
        ),
        ToolDefinition(
            name="analysis.grouped",
            description="Perform grouped analysis.",
            parameters={
                "type": "object",
                "properties": {},
            },
        ),
        ToolDefinition(
            name="analysis.derived_metric",
            description="Calculate a controlled derived metric.",
            parameters={
                "type": "object",
                "properties": {
                    "left_column": {"type": "string"},
                    "right_column": {"type": "string"},
                },
                "required": ["left_column", "right_column"],
            },
        ),
    ]

    converted = DeepSeekProvider._convert_tools(tools)

    assert converted[0]["function"]["name"] == "lc_dataset_dload"
    assert converted[1]["function"]["name"] == "lc_analysis_dgrouped"
    assert (
        converted[2]["function"]["name"]
        == "lc_analysis_dderived_umetric"
    )

    # Schema and descriptions must remain unchanged.
    assert converted[0]["function"]["description"] == "Load a dataset."
    assert converted[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }

    assert converted[2]["function"]["description"] == (
        "Calculate a controlled derived metric."
    )
    assert converted[2]["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "left_column": {"type": "string"},
            "right_column": {"type": "string"},
        },
        "required": ["left_column", "right_column"],
    }


def test_convert_messages_encodes_historical_tool_calls():
    messages = [
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="dataset.load",
                    arguments={"file_path": "data/raw/example.xlsx"},
                )
            ],
        )
    ]

    converted = DeepSeekProvider._convert_messages(messages)

    assert converted[0]["role"] == "assistant"
    assert converted[0]["tool_calls"][0]["id"] == "call_001"
    assert (
        converted[0]["tool_calls"][0]["function"]["name"]
        == "lc_dataset_dload"
    )
    assert converted[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"file_path": "data/raw/example.xlsx"}'
    )


def test_convert_messages_encodes_history_without_tool_definitions():
    """
    Historical tool calls must be encoded independently of request.tools.

    This is important because the final synthesis request can have
    tools=[], while its messages still contain historical tool calls.
    """

    messages = [
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_002",
                    name="analysis.grouped",
                    arguments={"column": "Country"},
                )
            ],
        ),
        LLMMessage(
            role="tool",
            tool_call_id="call_002",
            name="analysis.grouped",
            content='{"status": "success"}',
        ),
    ]

    # Simulate a request with no tool definitions.
    tools = []

    assert tools == []

    converted = DeepSeekProvider._convert_messages(messages)

    assert (
        converted[0]["tool_calls"][0]["function"]["name"]
        == "lc_analysis_dgrouped"
    )

    # Tool result names must match the encoded historical tool call name.
    assert converted[1]["name"] == "lc_analysis_dgrouped"


def test_convert_response_decodes_tool_call_name():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_003",
                            function=SimpleNamespace(
                                name="lc_dataset_dload",
                                arguments='{"file_path": "data/raw/example.xlsx"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
        model="deepseek-flash",
    )

    converted = DeepSeekProvider._convert_response(response)

    assert len(converted.tool_calls) == 1
    assert converted.tool_calls[0].id == "call_003"
    assert converted.tool_calls[0].name == "dataset.load"
    assert converted.tool_calls[0].arguments == {
        "file_path": "data/raw/example.xlsx"
    }


def test_convert_response_preserves_normal_text_response():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="The dataset contains 100 rows.",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="deepseek-flash",
    )

    converted = DeepSeekProvider._convert_response(response)

    assert converted.content == "The dataset contains 100 rows."
    assert converted.tool_calls == []
    assert converted.finish_reason == "stop"
    assert converted.model == "deepseek-flash"