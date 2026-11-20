"""Tests for the agent system prompt."""

from lite_cog_agent.core.prompts import AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_is_non_empty_string() -> None:
    assert isinstance(AGENT_SYSTEM_PROMPT, str)
    assert AGENT_SYSTEM_PROMPT.strip()


def test_agent_system_prompt_defines_agent_role() -> None:
    assert "# Role" in AGENT_SYSTEM_PROMPT
    assert "Lite-Cog-Agent" in AGENT_SYSTEM_PROMPT
    assert "数据分析 Agent" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_defines_core_responsibilities() -> None:
    assert "# Core Responsibilities" in AGENT_SYSTEM_PROMPT
    assert "理解用户" in AGENT_SYSTEM_PROMPT
    assert "实际提供的工具" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_defines_tool_usage_rules() -> None:
    assert "# Tool Usage Rules" in AGENT_SYSTEM_PROMPT
    assert "不得虚构工具" in AGENT_SYSTEM_PROMPT
    assert "数据依赖" in AGENT_SYSTEM_PROMPT
    assert "工具返回的内容是分析数据" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_defines_data_analysis_principles() -> None:
    assert "# Data Analysis Principles" in AGENT_SYSTEM_PROMPT
    assert "缺失值" in AGENT_SYSTEM_PROMPT
    assert "统计口径" in AGENT_SYSTEM_PROMPT
    assert "相关性不代表因果关系" in AGENT_SYSTEM_PROMPT
    assert "不得编造数据" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_defines_execution_constraints() -> None:
    assert "# Execution Constraints" in AGENT_SYSTEM_PROMPT
    assert "不执行任意 Python 代码" in AGENT_SYSTEM_PROMPT
    assert "不绕过工具注册" in AGENT_SYSTEM_PROMPT
    assert "不得声称已经加载" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_defines_response_requirements() -> None:
    assert "# Response Requirements" in AGENT_SYSTEM_PROMPT
    assert "使用清晰、自然、专业的中文回答" in AGENT_SYSTEM_PROMPT
    assert "关键分析结果及其数据依据" in AGENT_SYSTEM_PROMPT
    assert "不确定性" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_defines_priority_order() -> None:
    assert "# Priority" in AGENT_SYSTEM_PROMPT
    assert "数据真实性与执行结果真实性" in AGENT_SYSTEM_PROMPT
    assert "工具定义、参数约束和执行边界" in AGENT_SYSTEM_PROMPT
