import pytest

from lite_cog_agent.core.messages import (
    MessageHistoryError,
    MessageManager,
    validate_message_history,
)
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    ToolCall,
)


def make_tool_call(
    call_id: str = "call_001",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="analysis.profile",
        arguments={"dataset_id": "dataset_001"},
    )


def test_message_manager_starts_with_empty_history():
    manager = MessageManager()

    assert len(manager) == 0
    assert manager.get_messages() == []


def test_message_manager_adds_system_user_and_assistant_messages():
    manager = MessageManager()

    manager.add_system_message("You are a data analysis agent.")
    manager.add_user_message("Analyze my dataset.")
    manager.add_assistant_message("I will inspect the dataset.")

    messages = manager.get_messages()

    assert len(messages) == 3
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[0].content == "You are a data analysis agent."
    assert messages[1].content == "Analyze my dataset."
    assert messages[2].content == "I will inspect the dataset."


def test_message_manager_adds_tool_call_and_tool_result():
    manager = MessageManager()
    tool_call = make_tool_call()

    manager.add_assistant_message(
        content=None,
        tool_calls=[tool_call],
    )

    manager.add_tool_message(
        content='{"row_count": 100}',
        tool_call_id="call_001",
        name="analysis.profile",
    )

    messages = manager.get_messages()

    assert len(messages) == 2
    assert messages[0].role == "assistant"
    assert messages[0].tool_calls[0].id == "call_001"

    assert messages[1].role == "tool"
    assert messages[1].tool_call_id == "call_001"
    assert messages[1].name == "analysis.profile"


def test_message_manager_allows_pending_tool_calls():
    manager = MessageManager()

    manager.add_assistant_message(
        content=None,
        tool_calls=[
            make_tool_call("call_001"),
            make_tool_call("call_002"),
        ],
    )

    assert len(manager) == 1

    # Pending Tool Calls are permitted during message construction.
    validate_message_history(
        manager.get_messages(),
        allow_pending=True,
    )


def test_message_manager_rejects_unresolved_tool_calls_on_validate():
    manager = MessageManager()

    manager.add_assistant_message(
        content=None,
        tool_calls=[make_tool_call()],
    )

    with pytest.raises(
        MessageHistoryError,
        match="unresolved Tool Calls",
    ):
        manager.validate()


def test_message_manager_rejects_orphan_tool_message():
    manager = MessageManager()

    with pytest.raises(
        MessageHistoryError,
        match="unknown or already resolved",
    ):
        manager.add_tool_message(
            content="result",
            tool_call_id="unknown_call",
        )

    assert len(manager) == 0


def test_message_manager_rejects_duplicate_tool_call_ids():
    manager = MessageManager()

    manager.add_assistant_message(
        content=None,
        tool_calls=[make_tool_call("call_001")],
    )

    manager.add_tool_message(
        content="result",
        tool_call_id="call_001",
    )

    with pytest.raises(
        MessageHistoryError,
        match="Duplicate tool call ID",
    ):
        manager.add_assistant_message(
            content=None,
            tool_calls=[make_tool_call("call_001")],
        )

    assert len(manager) == 2


@pytest.mark.parametrize(
    "role",
    [
        "developer",
        "",
        None,
        [],
    ],
)
def test_validate_message_history_rejects_invalid_roles(role):
    message = LLMMessage(
        role=role,
        content="test",
    )

    with pytest.raises(
        MessageHistoryError,
        match="Unsupported message role",
    ):
        validate_message_history([message])


def test_validate_message_history_rejects_non_assistant_tool_calls():
    message = LLMMessage(
        role="user",
        content="test",
        tool_calls=[make_tool_call()],
    )

    with pytest.raises(
        MessageHistoryError,
        match="Only assistant messages",
    ):
        validate_message_history([message])


def test_validate_message_history_rejects_non_tool_tool_call_id():
    message = LLMMessage(
        role="assistant",
        content="test",
        tool_call_id="call_001",
    )

    with pytest.raises(
        MessageHistoryError,
        match="Only tool messages",
    ):
        validate_message_history([message])


def test_message_manager_returns_deep_copied_messages():
    manager = MessageManager()

    manager.add_assistant_message(
        content=None,
        tool_calls=[make_tool_call()],
    )

    copied_messages = manager.get_messages()

    copied_messages[0].content = "modified"
    copied_messages[0].tool_calls[0].name = "modified.tool"
    copied_messages[0].tool_calls[0].arguments["dataset_id"] = "changed"

    original_messages = manager.get_messages()

    assert original_messages[0].content is None
    assert original_messages[0].tool_calls[0].name == "analysis.profile"
    assert (
        original_messages[0].tool_calls[0].arguments["dataset_id"]
        == "dataset_001"
    )


def test_message_manager_add_messages_is_atomic_on_validation_failure():
    manager = MessageManager()
    manager.add_user_message("Existing message.")

    original_messages = manager.get_messages()

    with pytest.raises(
        MessageHistoryError,
        match="unknown or already resolved",
    ):
        manager.add_messages(
            [
                LLMMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[make_tool_call()],
                ),
                LLMMessage(
                    role="tool",
                    content="orphan result",
                    tool_call_id="unknown_call",
                ),
            ]
        )

    assert manager.get_messages() == original_messages
    assert len(manager) == 1


def test_message_manager_clear_removes_all_messages():
    manager = MessageManager()

    manager.add_system_message("System.")
    manager.add_user_message("User.")

    manager.clear()

    assert len(manager) == 0
    assert manager.get_messages() == []
    manager.validate()


def test_message_manager_rejects_empty_text_messages():
    manager = MessageManager()

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.add_system_message("   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.add_user_message("")

    assert len(manager) == 0
