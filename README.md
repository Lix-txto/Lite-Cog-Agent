# Lite-Cog-Agent

**A Lightweight Data Analysis Agent Powered by ReAct and Tool Calling.**

Lite-Cog-Agent 是一个基于 Python 构建的轻量级数据分析智能体项目，旨在探索如何利用大语言模型的推理能力与工具调用机制，将自然语言分析需求转化为可执行的数据分析任务。

项目采用 **ReAct + Tool Calling** 作为核心 Agent 设计模式，通过 LLM 与 Python 分析工具之间的交互，逐步构建从用户需求理解、工具选择、数据处理到分析结果生成的自动化工作流。

项目以轻量化、模块化和可扩展性为设计原则，不依赖复杂的 Agent 编排框架，重点实现 Agent 核心运行机制及其基础工程能力。

> **Project Status:** Under Development

---

## 1. Project Overview

### 1.1 Background

传统数据分析通常要求用户具备一定的数据处理和编程能力。面对 CSV、Excel 等数据文件，用户需要手动完成数据读取、清洗、统计分析和可视化等操作。

Lite-Cog-Agent 希望通过自然语言交互降低数据分析的使用门槛。

用户只需描述分析目标，Agent 即可根据任务需求选择合适的工具，并通过多轮推理与工具调用逐步完成分析任务。

### 1.2 Project Goals

项目的核心目标包括：

- **Natural Language Interaction**：通过自然语言描述数据分析需求。
- **Autonomous Tool Selection**：根据用户需求自主选择合适的分析工具。
- **Python-based Execution**：利用 Python 执行数据处理、统计分析和可视化任务。
- **Iterative Reasoning**：通过 ReAct 模式实现推理、行动与观察的循环。
- **Lightweight Architecture**：采用模块化单体架构，避免不必要的基础设施复杂度。
- **Extensibility**：为后续扩展数据分析工具、LLM Provider 和 Agent 能力预留接口。

---

## 2. Core Architecture

### 2.1 Agent Execution Model

项目采用 ReAct（Reasoning and Acting）模式，并结合 LLM Tool Calling 实现 Agent 的基本运行机制。

核心执行流程：

1. 接收用户的数据分析需求。
2. 将用户请求与上下文发送给 LLM。
3. LLM 根据任务需求决定是否调用工具。
4. Agent 解析 Tool Calling 请求并分发工具。
5. 工具执行数据处理或 Python 分析任务。
6. 将工具执行结果作为 Observation 返回给 LLM。
7. LLM 根据 Observation 决定下一步行动。
8. 完成任务后向用户返回分析结果。

核心流程：

```text
User Request
     |
     v
Agent Controller
     |
     v
LLM Gateway
     |
     v
DeepSeek API
     |
     v
Tool Calling Decision
     |
     v
Tool Registry
     |
     v
Tool Execution
     |
     v
Observation
     |
     +------> LLM Reasoning
     |              |
     |              v
     |        Continue / Finish
     |
     v
Final Analysis Result
```

### 2.2 Architecture Layers

项目采用模块化单体架构，按照职责划分为以下模块：

| Module | Responsibility |
|---|---|
| Core | Agent 状态管理、消息结构、提示词及推理循环 |
| LLM | LLM Gateway、Provider Adapter 和统一响应结构 |
| Tools | 工具定义、工具注册及内置数据分析工具 |
| Executor | Python 代码执行、子进程管理及执行限制 |
| Storage | Session 管理和本地文件存储 |
| Services | 组织完整的数据分析业务流程 |
| Config | 环境变量与应用配置 |
| Utils | 日志及通用辅助功能 |

各模块通过明确的接口协作，降低 Agent 推理、模型通信、工具执行和存储之间的耦合。

---

## 3. Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.12 | Core development language |
| Package Manager | uv | Dependency and virtual environment management |
| LLM | DeepSeek API | Language model inference |
| LLM SDK | OpenAI Python SDK | DeepSeek-compatible API integration |
| Configuration | python-dotenv | Environment variable management |
| Agent Pattern | ReAct + Tool Calling | Agent reasoning and tool orchestration |
| Data Analysis | pandas / NumPy | Data processing and numerical computation |
| Visualization | Matplotlib | Data visualization |
| Execution | Subprocess | Isolated process-based Python execution |
| Concurrency | asyncio | Asynchronous task coordination |
| Version Control | Git / GitHub | Source code management |

**Note:** pandas、NumPy 和 Matplotlib 等数据分析依赖将在对应功能开发阶段按需引入；当前基础依赖以 `pyproject.toml` 为准。

---

## 4. Project Structure

```text
Lite-Cog-Agent/
│
├── src/
│   └── lite_cog_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── agent_loop.py
│       │   ├── agent_state.py
│       │   ├── messages.py
│       │   └── prompts.py
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── gateway.py
│       │   ├── schemas.py
│       │   └── providers/
│       │       ├── __init__.py
│       │       └── deepseek.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── schemas.py
│       │   └── builtin/
│       │       ├── __init__.py
│       │       ├── dataset_tools.py
│       │       ├── analysis_tools.py
│       │       └── visualization_tools.py
│       │
│       ├── executor/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── code_executor.py
│       │   └── subprocess_manager.py
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── session_manager.py
│       │   └── file_manager.py
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   └── analysis_service.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── logging.py
│
├── data/
│   ├── sessions/
│   └── temp/
│
├── logs/
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
├── scripts/
│
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
└── CLAUDE.md
```

---

## 5. Getting Started

### 5.1 Prerequisites

确保本地开发环境满足以下要求：

- Python 3.12+
- uv
- Git
- DeepSeek API Key

建议在 Linux 或 WSL 2 环境下进行开发。

### 5.2 Clone Repository

```bash
git clone https://github.com/Lix-txto/Lite-Cog-Agent.git

cd Lite-Cog-Agent
```

将 `<YOUR_USERNAME>` 替换为实际 GitHub 用户名。

### 5.3 Install Dependencies

使用 uv 同步项目依赖：

```bash
uv sync
```

uv 将根据 `pyproject.toml` 和 `uv.lock` 创建虚拟环境并安装依赖。

### 5.4 Configure Environment Variables

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`，填写自己的 DeepSeek API Key。

```dotenv
# LLM Provider Configuration
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Agent Configuration
AGENT_MAX_ITERATIONS=10
AGENT_MAX_TOOL_CALLS=15

# Python Execution Configuration
PYTHON_EXECUTION_TIMEOUT=30
PYTHON_MAX_OUTPUT_SIZE=10000

# Storage Configuration
DATA_DIR=./data
SESSION_DIR=./data/sessions
TEMP_DIR=./data/temp

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=./logs
```

**Security Notice:**

- 不要将 `.env` 文件提交到 GitHub。
- 不要在源代码中硬编码 API Key。
- 不要公开分享包含真实 API Key 的终端截图。
- `.env.example` 仅用于提供配置模板，不应包含真实密钥。

### 5.5 Run Application

通过 Python 模块入口启动：

```bash
uv run python -m lite_cog_agent
```

或者使用项目注册的命令行入口：

```bash
uv run lite-cog-agent
```

两种方式都会启动 Lite-Cog-Agent CLI。

---

## 6. CLI Commands

当前 CLI 提供基础交互命令：

| Command | Description |
|---|---|
| `help` | 显示可用命令 |
| `status` | 查看应用运行状态 |
| `exit` | 退出程序 |
| `quit` | 退出程序 |

启动后示例：

```text
============================================================
  Lite-Cog-Agent
  Lightweight Data Analysis Agent
============================================================
  Type 'help' to see available commands.
  Type 'exit' or 'quit' to close the application.

lite-cog-agent> help

Available commands:
  help       Show available commands
  status     Show current application status
  exit       Exit the application
  quit       Exit the application

lite-cog-agent> status

Application Status
----------------------------------------
  Application : Lite-Cog-Agent
  Version     : 0.1.0
  Status      : Running
  Agent       : Not initialized
  LLM         : Not connected

lite-cog-agent> exit
```

当前 CLI 主要用于验证程序启动入口和基础交互能力，尚未接入完整 Agent 推理流程。

---

## 7. Configuration

主要配置项如下：

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM Provider | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | Empty |
| `DEEPSEEK_BASE_URL` | API Base URL | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | Model Name | `deepseek-chat` |
| `AGENT_MAX_ITERATIONS` | Maximum Agent iterations | `10` |
| `AGENT_MAX_TOOL_CALLS` | Maximum tool calls | `15` |
| `PYTHON_EXECUTION_TIMEOUT` | Python execution timeout (seconds) | `30` |
| `PYTHON_MAX_OUTPUT_SIZE` | Maximum execution output size | `10000` |
| `DATA_DIR` | Data directory | `./data` |
| `SESSION_DIR` | Session storage directory | `./data/sessions` |
| `TEMP_DIR` | Temporary file directory | `./data/temp` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_DIR` | Log directory | `./logs` |

---

## 8. Development Roadmap

项目采用分阶段开发方式，逐步从基础 Python 工程演进为具备自主工具调用能力的数据分析 Agent。

| Milestone | Scope | Status |
|---|---|---|
| M1 | Project Bootstrap | In Progress |
| M2 | LLM Gateway & DeepSeek Integration | Planned |
| M3 | Tool System & Registry | Planned |
| M4 | Python Execution Engine | Planned |
| M5 | ReAct Agent Loop | Planned |
| M6 | Data Analysis Tools | Planned |
| M7 | Session & File Management | Planned |
| M8 | Analysis Workflow Integration | Planned |
| M9 | Logging & Observability | Planned |
| M10 | Testing & Project Refinement | Planned |

后续开发将围绕 Agent 核心能力展开，包括模型通信、工具调用、Python 执行、上下文管理和分析结果生成。

---

## 9. Design Principles

### Lightweight by Design

优先使用 Python 原生能力和必要依赖，避免引入与当前项目规模不匹配的基础设施。

### Modular Architecture

通过模块化设计明确不同组件的职责，为后续功能扩展提供基础。

### Tool-oriented Agent

将数据处理、分析和可视化能力封装为可调用工具，由 Agent 根据任务需求选择和组合。

### Controlled Code Execution

通过独立子进程执行 Python 分析代码，并设计超时与输出大小限制。

**Security limitation:** Subprocess execution with basic restrictions is not a complete security sandbox. It should not be exposed directly to untrusted users or arbitrary public code execution workloads.

### Incremental Development

按照工程依赖关系逐步实现功能，优先保证核心运行链路可验证，再扩展分析能力。

---

## 10. License

License information will be added when the project license is selected.

---

## 11. Acknowledgements

This project is developed as an exploration of lightweight LLM Agent architecture, Python-based data analysis automation, and tool-calling workflows.