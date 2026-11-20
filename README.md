# Lite-Cog-Agent(轻量级数据分析Agent)

**A Lightweight Data Analysis Agent Powered by ReAct and Tool Calling.**

Lite-Cog-Agent 是一个基于 Python 构建的轻量级数据分析 Agent 项目，旨在探索如何利用大语言模型的推理能力与 Tool Calling 机制，将自然语言分析需求转化为受控、可执行的数据分析任务。

项目采用 **ReAct + Tool Calling** 作为核心 Agent 设计模式，通过 LLM 与注册工具之间的多轮交互，实现从用户需求理解、数据加载、数据清洗、统计分析到可视化结果生成的自动化工作流。

项目采用轻量级模块化单体架构，不依赖 LangGraph 等复杂 Agent 编排框架。重点实现 Agent Runtime、Tool Calling、数据分析执行基础设施及 CLI 集成。

> **Project Status:** M1–M5 Completed — Core Agent Runtime, Data Analysis Pipeline and CLI Integration. The project remains under active development.

---

## 1. Project Overview

### 1.1 Background

传统数据分析通常要求用户具备一定的数据处理和编程能力。面对 CSV、Excel 等数据文件，用户需要手动完成数据读取、清洗、统计分析和可视化等操作。

Lite-Cog-Agent 希望通过自然语言交互降低数据分析的使用门槛。

用户只需描述分析目标，Agent 即可结合当前对话上下文，选择合适的注册工具，并通过多轮推理与 Tool Calling 逐步完成分析任务。

### 1.2 Project Goals

项目的核心目标包括：

- **Natural Language Interaction**：通过自然语言描述数据分析需求。
- **Autonomous Tool Selection**：根据用户需求选择并组合合适的分析工具。
- **Controlled Analysis Execution**：通过注册的分析操作执行数据处理，避免执行 LLM 生成的任意 Python 代码。
- **Iterative Reasoning**：通过 ReAct 模式实现推理、行动与观察的循环。
- **Data Analysis Pipeline**：支持数据加载、清洗、概览、统计分析和可视化。
- **Lightweight Architecture**：采用模块化单体架构，避免不必要的基础设施复杂度。
- **Extensibility**：为后续扩展分析工具、LLM Provider 和 Agent 能力预留接口。

---

## 2. Core Architecture

### 2.1 Agent Execution Model

项目采用 ReAct（Reasoning and Acting）模式，并结合 LLM Tool Calling 实现 Agent 的运行机制。

一次 Agent 请求的核心执行流程：

1. 接收用户的数据分析需求。
2. Agent 将系统 Prompt、对话历史和当前请求组织为 LLM messages。
3. LLM 根据任务需求决定返回最终回答或发起 Tool Calling。
4. Tool Calling Service 验证工具调用请求。
5. Tool Dispatcher 将调用分发到 Tool Registry 中已注册的工具。
6. Builtin Tools 调用受控的分析服务及 Execution Registry 中的操作。
7. 工具执行结果以 Tool Message 的形式加入对话历史。
8. Agent 将更新后的上下文再次发送给 LLM。
9. Agent 根据工具观察结果继续调用工具，或生成最终回答。
10. Agent 完成请求并提交本次对话历史。

核心流程：

```text
                    User Request
                          |
                          v
                    Agent Facade
                          |
                          v
                      AgentLoop
                          |
                 Agent State / Messages
                          |
                          v
                 ToolCallingService
                          |
                          v
                     LLM Gateway
                          |
                          v
                     DeepSeek API
                          |
                  Tool Call / Response
                          |
                          v
                   Tool Dispatcher
                          |
                          v
                     Tool Registry
                          |
                          v
                     Builtin Tools
                          |
                          v
                  Analysis Services
                          |
                          v
                  Execution Registry
                          |
                          v
                 Registered Operations
                          |
                          v
                    Tool Results
                          |
                          +------> AgentLoop
                                      |
                                      v
                               LLM Reasoning
                                      |
                               Continue / Finish
                                      |
                                      v
                              Final Response
```

Agent 负责请求生命周期和对话状态管理；Tool Calling Service 负责模型交互与工具调用循环；Tool Registry 和 Dispatcher 负责工具注册与分发；Execution Registry 管理受控的数据分析操作。

### 2.2 Architecture Layers

项目采用模块化单体架构，按照职责划分为以下模块：

| Module | Responsibility |
|---|---|
| Core | Agent Facade、Agent State、Message Management、Prompt、Reasoning Loop、Execution Budget |
| LLM | LLM Gateway、DeepSeek Provider、Request/Response Validation、Token Usage Tracking |
| Tools | Tool Definition、Registry、Dispatcher、Parameter Validation、Builtin Tools |
| Loading | CSV、Excel、JSON Table 数据集加载 |
| Profiling | 数据集结构、列信息、缺失值和数据质量概览 |
| Cleaning | 注册式数据清洗操作及清洗结果管理 |
| Statistics | 描述性统计、频率、相关性、分组和分布分析 |
| Visualization | 图表生成及可视化结果输出 |
| Execution | Execution Context、Dataset Store、Execution Registry、Analysis Executor |
| Storage | 文件管理及可视化 Artifact 存储 |
| Executor | 独立代码执行相关模块，与当前 Agent 注册式分析工具链区分 |
| Services | 分析服务封装 |
| Config | 环境变量和应用配置 |
| Utils | 日志等通用工具 |

各模块通过明确的接口协作，降低 Agent 推理、模型通信、工具执行和存储之间的耦合。

### 2.3 Tool and Execution Architecture

项目将 Agent-facing Tool 与底层 Analysis Operation 分开管理。

**Tool Registry**

管理 Agent 可以调用的工具，包括工具名称、描述和参数 Schema。

**Tool Dispatcher**

负责工具调用的参数验证、分发和结果组织。

**Execution Registry**

管理已经注册的 Python 分析操作。分析操作由应用代码实现和注册，不由 LLM 动态生成。

**Execution Context**

为一次运行提供数据集及相关执行资源的上下文。

当前 Agent 的数据分析主链路通过上述机制执行受控操作，不会直接执行 LLM 生成的任意 Python 代码。

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
| Data Analysis | pandas | Data loading and analysis |
| Visualization | Matplotlib | Data visualization |
| Concurrency | asyncio | Asynchronous Agent and LLM coordination |
| Blocking I/O Integration | asyncio.to_thread | Offload synchronous analysis operations |
| Validation | jsonschema | Tool parameter and schema validation |
| Testing | pytest / pytest-asyncio | Unit and integration testing |
| Version Control | Git / GitHub | Source code management |

**Implementation Note:**

- Agent 使用异步 LLM Gateway 与 DeepSeek API 通信。
- 同步数据分析操作通过 `asyncio.to_thread` 接入异步工具调用流程。
- Agent 通过注册的 Builtin Tools 调用受控分析服务。
- 当前 Agent 分析主链路不依赖任意代码执行或复杂 Agent 编排框架。

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
│       │   ├── agent_budget.py
│       │   ├── agent_factory.py
│       │   ├── agent_loop.py
│       │   ├── agent_state.py
│       │   ├── messages.py
│       │   ├── prompts.py
│       │   ├── tool_calling.py
│       │   └── tool_calling_config.py
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── exceptions.py
│       │   ├── gateway.py
│       │   ├── response_validator.py
│       │   ├── schemas.py
│       │   ├── usage_tracker.py
│       │   └── providers/
│       │       ├── __init__.py
│       │       └── deepseek.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── dispatcher.py
│       │   ├── exceptions.py
│       │   ├── execution_config.py
│       │   ├── factory.py
│       │   ├── message_converter.py
│       │   ├── parameter_validator.py
│       │   ├── registry.py
│       │   ├── schemas.py
│       │   └── builtin/
│       │       ├── __init__.py
│       │       ├── analysis_tools.py
│       │       ├── cleaning_tools.py
│       │       ├── common.py
│       │       ├── dataset_tools.py
│       │       ├── factory.py
│       │       └── visualization_tools.py
│       │
│       ├── loading/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── contracts.py
│       │   ├── detector.py
│       │   ├── exceptions.py
│       │   ├── loaders.py
│       │   └── service.py
│       │
│       ├── profiling/
│       │   ├── __init__.py
│       │   ├── column_profiler.py
│       │   ├── contracts.py
│       │   ├── dataset_profiler.py
│       │   ├── exceptions.py
│       │   ├── quality.py
│       │   └── service.py
│       │
│       ├── cleaning/
│       │   ├── __init__.py
│       │   ├── columns.py
│       │   ├── contracts.py
│       │   ├── duplicates.py
│       │   ├── exceptions.py
│       │   ├── missing.py
│       │   ├── operations.py
│       │   ├── service.py
│       │   └── type_converter.py
│       │
│       ├── statistics/
│       │   ├── __init__.py
│       │   ├── contracts.py
│       │   ├── correlation.py
│       │   ├── descriptive.py
│       │   ├── distribution.py
│       │   ├── exceptions.py
│       │   ├── frequency.py
│       │   ├── grouped.py
│       │   └── service.py
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── contracts.py
│       │   ├── exceptions.py
│       │   ├── matplotlib_renderer.py
│       │   ├── outputs.py
│       │   ├── renderer.py
│       │   └── service.py
│       │
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── context.py
│       │   ├── contracts.py
│       │   ├── dataset_store.py
│       │   ├── exceptions.py
│       │   ├── executor.py
│       │   └── registry.py
│       │
│       ├── executor/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── code_executor.py
│       │   └── subprocess_manager.py
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── file_manager.py
│       │   └── session_manager.py
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   └── analysis_service.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── logging.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── data/
├── logs/
├── docs/
├── scripts/
│   └── test_deepseek_connection.py
│
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
└── CLAUDE.md
```

**Structure Note:**

- `core/`：Agent 生命周期、状态、消息历史、Prompt、预算和运行循环。
- `llm/`：模型 Provider、Gateway、消息 Schema 和 Token Usage。
- `tools/`：Agent-facing 工具协议、注册、分发及 Builtin Tools。
- `loading/`、`profiling/`、`cleaning/`、`statistics/`、`visualization/`：数据分析领域能力。
- `execution/`：受控分析操作的执行上下文、数据集存储和注册机制。
- `executor/`：独立代码执行模块，不应与当前 Agent 注册式分析主链路混为一谈。
- `storage/`、`services/`、`utils/`：其他基础设施和服务模块。

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

### 5.3 Install Dependencies

使用 uv 同步项目依赖：

```bash
uv sync
```

uv 将根据 `pyproject.toml` 和 `uv.lock` 创建虚拟环境并安装项目依赖。

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
DEEPSEEK_MODEL=deepseek-flash

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
- 当前 `LLMGateway` 在初始化 DeepSeek Provider 时就会检查 API Key 配置。缺失密钥会导致初始化失败，而不是等到第一次对话时才检查。

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

### 5.6 Dataset Input

将待分析的数据文件放入配置的 `DATA_DIR` 目录。

当前数据加载能力包括：

- CSV 文件
- Excel `.xlsx` 文件
- JSON Table Object Arrays

启动 CLI 后，可以通过自然语言请求 Agent 加载数据集并开展分析。具体数据文件名和分析目标以实际数据为准。

---

## 6. CLI Commands

当前 CLI 已经集成 Agent Runtime。除内置命令外，普通文本输入都会作为自然语言请求交给 Agent 执行。

| Command | Description |
|---|---|
| `help` | 显示可用命令和使用说明 |
| `status` | 查看 Agent Session、数据集数量和 LLM 请求状态 |
| `usage` | 查看当前 Gateway 的 Token Usage |
| `reset` | 重置 Gateway Token Usage 统计 |
| `clear` | 清空 Agent 对话历史及当前执行计数 |
| `exit` | 退出程序 |
| `quit` | 退出程序 |

### 6.1 Command Semantics

**`status`**

展示当前 Session ID、数据集数量和 LLM 请求状态。连接状态依据当前 CLI 会话中是否曾经成功完成 Agent 请求判断，不代表实时网络探测。

**`usage`**

展示 Gateway 记录的 Token Usage。

**`reset`**

只重置 Gateway 的 Token Usage 统计，不清除 Agent 对话历史，也不更换 Session ID。

**`clear`**

调用 Agent Reset，清空当前对话历史和 Agent 计数器，但保留：

- 当前 Session ID
- 已加载的数据集
- 已生成的可视化 Artifacts
- Gateway Token Usage 统计

**`exit` / `quit`**

退出 CLI，释放当前 Execution Context 和 LLM Gateway 资源。

### 6.2 Natural Language Interaction

普通文本会被发送给 Agent，例如：

```text
You: Load the sales.csv dataset and summarize its structure.

You: Check missing values and data quality.

You: Remove rows with missing values in required columns.

You: Calculate descriptive statistics for the sales column.

You: Generate a histogram for the sales distribution.
```

以上是交互示例，具体工具选择由 Agent 根据数据和任务需求决定。

---

## 7. Configuration

主要配置项如下：

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM Provider | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | Empty |
| `DEEPSEEK_BASE_URL` | API Base URL | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | Model Name | `deepseek-flash` |
| `AGENT_MAX_ITERATIONS` | Maximum Agent iterations per request | `10` |
| `AGENT_MAX_TOOL_CALLS` | Maximum tool calls per request | `15` |
| `PYTHON_EXECUTION_TIMEOUT` | Python execution timeout configuration | `30` |
| `PYTHON_MAX_OUTPUT_SIZE` | Maximum execution output size configuration | `10000` |
| `DATA_DIR` | Dataset and artifact root directory | `./data` |
| `SESSION_DIR` | Session storage directory configuration | `./data/sessions` |
| `TEMP_DIR` | Temporary file directory configuration | `./data/temp` |
| `LOG_LEVEL` | Logging level configuration | `INFO` |
| `LOG_DIR` | Log directory configuration | `./logs` |

### 7.1 Configuration Integration Status

**Used by the current Agent CLI runtime:**

- `LLM_PROVIDER`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `AGENT_MAX_ITERATIONS`
- `AGENT_MAX_TOOL_CALLS`
- `DATA_DIR`

**Configured but not integrated into the current Agent CLI runtime:**

- `PYTHON_EXECUTION_TIMEOUT`
- `PYTHON_MAX_OUTPUT_SIZE`
- `SESSION_DIR`
- `TEMP_DIR`
- `LOG_LEVEL`
- `LOG_DIR`

这些配置项目前存在于 Settings 中，但不能据此认为对应的超时控制、Session 持久化或日志管理功能已经完整接入当前 Agent CLI 流程。

---

## 8. Development Roadmap

项目按照工程依赖关系分阶段开发，当前已经完成基础工程、LLM 集成、工具系统、数据分析执行基础设施和 Agent Runtime 集成。

| Milestone | Scope | Status |
|---|---|---|
| M1 | Project Bootstrap & Configuration | Completed |
| M2 | LLM Gateway & DeepSeek Integration | Completed |
| M3 | Tool Calling Infrastructure | Completed |
| M4 | Data Analysis Execution Infrastructure | Completed |
| M5 | Agent Analysis Workflow & CLI Integration | Completed |

### Future Development Directions

后续可以围绕以下方向继续迭代，具体范围将在后续阶段设计和确认：

- **Session Persistence**：对话历史和 Agent Session 的持久化。
- **Observability**：Agent 执行轨迹、工具调用耗时和错误信息。
- **Execution Reliability**：任务恢复、资源限制和执行失败处理。
- **Analysis Capabilities**：扩展数据分析和可视化工具。
- **CLI Experience**：提升交互体验和结果展示能力。
- **Documentation & Packaging**：完善开发文档、使用文档和分发流程。

这些属于后续开发方向，不代表当前版本已经实现。

---

## 9. Design Principles

### Lightweight by Design

优先使用 Python 原生能力和必要依赖，避免引入与当前项目规模不匹配的基础设施。

### Modular Architecture

通过模块化设计明确不同组件的职责，为后续功能扩展提供基础。

### Tool-oriented Agent

将数据加载、清洗、分析和可视化能力封装为可调用工具，由 Agent 根据任务需求选择和组合。

### Controlled Analysis Execution

Agent 只能调用应用注册并经过参数校验的工具。

工具进一步通过 Execution Registry 调用预先实现的 Python 分析操作。

LLM 负责理解任务、选择工具和解释结果，而不是生成并执行任意 Python 代码。

### Separation of Responsibilities

明确区分以下职责：

- Agent：管理请求生命周期、对话状态和推理流程。
- LLM Gateway：处理模型请求与响应。
- Tool Registry：管理 Agent 可调用的工具。
- Tool Dispatcher：验证和分发工具调用。
- Execution Registry：管理注册的分析操作。
- Analysis Services：实现数据处理、统计和可视化逻辑。

### Bounded Execution

通过 Agent iteration budget 和 tool-call budget 限制单次请求的执行规模，避免无限制的模型交互和工具调用。

### Incremental Development

按照工程依赖关系逐步实现功能，优先保证核心运行链路可验证，再扩展分析能力。

---

## 10. License

License information will be added when the project license is selected.

---

## 11. Acknowledgements

This project is developed as an exploration of lightweight LLM Agent architecture, Python-based data analysis automation, and tool-calling workflows.
