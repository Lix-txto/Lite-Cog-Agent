import pytest

from lite_cog_agent.core.agent_state import (
    AgentState,
    AgentStateError,
)
from lite_cog_agent.core.messages import MessageManager


def test_agent_state_initializes_with_default_values():
    state = AgentState()

    assert state.session_id
    assert state.iteration_count == 0
    assert state.is_running is False
    assert isinstance(state.message_manager, MessageManager)
    assert len(state.message_manager) == 0


def test_agent_state_generates_unique_session_ids():
    first = AgentState()
    second = AgentState()

    assert first.session_id != second.session_id


def test_agent_state_accepts_custom_session_id():
    state = AgentState(session_id="session_test_001")

    assert state.session_id == "session_test_001"


def test_agent_state_rejects_invalid_session_id():
    with pytest.raises(ValueError, match="session_id"):
        AgentState(session_id="")

    with pytest.raises(ValueError, match="session_id"):
        AgentState(session_id="   ")


def test_agent_state_accepts_injected_message_manager():
    manager = MessageManager()
    manager.add_user_message("Existing conversation.")

    state = AgentState(
        session_id="session_test_001",
        message_manager=manager,
    )

    assert state.message_manager is manager
    assert len(state.message_manager) == 1


def test_agent_state_rejects_invalid_message_manager():
    with pytest.raises(TypeError, match="MessageManager"):
        AgentState(message_manager=object())


def test_agent_state_starts_successfully():
    state = AgentState()

    state.start()

    assert state.is_running is True
    assert state.iteration_count == 0


def test_agent_state_rejects_duplicate_start():
    state = AgentState()
    state.start()

    with pytest.raises(
        AgentStateError,
        match="already running",
    ):
        state.start()


def test_agent_state_increments_iteration():
    state = AgentState()
    state.start()

    assert state.increment_iteration() == 1
    assert state.increment_iteration() == 2
    assert state.increment_iteration() == 3

    assert state.iteration_count == 3


def test_agent_state_rejects_increment_when_not_running():
    state = AgentState()

    with pytest.raises(
        AgentStateError,
        match="not running",
    ):
        state.increment_iteration()

    assert state.iteration_count == 0


def test_agent_state_finishes_successfully():
    state = AgentState()
    state.start()
    state.increment_iteration()

    state.finish()

    assert state.is_running is False
    assert state.iteration_count == 1


def test_agent_state_finish_is_idempotent():
    state = AgentState()

    state.finish()
    state.finish()

    assert state.is_running is False
    assert state.iteration_count == 0


def test_agent_state_resets_iteration_count_on_new_request():
    state = AgentState()

    state.start()
    state.increment_iteration()
    state.increment_iteration()
    state.finish()

    assert state.iteration_count == 2

    state.start()

    assert state.is_running is True
    assert state.iteration_count == 0


def test_agent_state_reset_clears_messages_and_iteration_count():
    state = AgentState()
    session_id = state.session_id

    state.message_manager.add_system_message("System.")
    state.message_manager.add_user_message("User.")

    state.start()
    state.increment_iteration()
    state.finish()

    state.reset()

    assert state.session_id == session_id
    assert state.iteration_count == 0
    assert state.is_running is False
    assert len(state.message_manager) == 0


def test_agent_state_rejects_reset_while_running():
    state = AgentState()
    state.message_manager.add_user_message("Keep this message.")

    state.start()
    state.increment_iteration()

    with pytest.raises(
        AgentStateError,
        match="while it is running",
    ):
        state.reset()

    assert state.is_running is True
    assert state.iteration_count == 1
    assert len(state.message_manager) == 1


def test_different_agent_states_have_isolated_message_histories():
    first = AgentState()
    second = AgentState()

    first.message_manager.add_user_message(
        "First session message."
    )

    second.message_manager.add_user_message(
        "Second session message."
    )

    assert first.session_id != second.session_id

    assert len(first.message_manager) == 1
    assert len(second.message_manager) == 1

    assert (
        first.message_manager.get_messages()[0].content
        == "First session message."
    )

    assert (
        second.message_manager.get_messages()[0].content
        == "Second session message."
    )


def test_agent_state_exposes_read_only_properties():
    state = AgentState()

    with pytest.raises(AttributeError):
        state.session_id = "modified"

    with pytest.raises(AttributeError):
        state.iteration_count = 10

    with pytest.raises(AttributeError):
        state.is_running = True
