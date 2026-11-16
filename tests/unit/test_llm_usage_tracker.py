import pytest

from lite_cog_agent.llm.exceptions import LLMResponseError
from lite_cog_agent.llm.schemas import TokenUsage
from lite_cog_agent.llm.usage_tracker import TokenUsageTracker


def test_record_single_usage():
    tracker = TokenUsageTracker()

    tracker.record(
        TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
    )

    summary = tracker.snapshot()

    assert summary.request_count == 1
    assert summary.prompt_tokens == 100
    assert summary.completion_tokens == 50
    assert summary.total_tokens == 150


def test_accumulate_multiple_requests():
    tracker = TokenUsageTracker()

    tracker.record(TokenUsage(100, 50, 150))
    tracker.record(TokenUsage(200, 80, 280))
    tracker.record(TokenUsage(150, 70, 220))

    summary = tracker.snapshot()

    assert summary.request_count == 3
    assert summary.prompt_tokens == 450
    assert summary.completion_tokens == 200
    assert summary.total_tokens == 650


def test_reject_invalid_usage():
    tracker = TokenUsageTracker()

    with pytest.raises(LLMResponseError):
        tracker.record(
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=999,
            )
        )

    assert tracker.snapshot().request_count == 0


def test_snapshot_does_not_modify_internal_state():
    tracker = TokenUsageTracker()

    tracker.record(TokenUsage(100, 50, 150))

    summary = tracker.snapshot()
    summary.total_tokens = 999999

    assert tracker.snapshot().total_tokens == 150


def test_reset_usage():
    tracker = TokenUsageTracker()

    tracker.record(TokenUsage(100, 50, 150))
    tracker.reset()

    summary = tracker.snapshot()

    assert summary.request_count == 0
    assert summary.prompt_tokens == 0
    assert summary.completion_tokens == 0
    assert summary.total_tokens == 0
