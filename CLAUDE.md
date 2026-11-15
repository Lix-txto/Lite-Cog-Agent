
# Lite-Cog-Agent Development Guide

## 1. Project Overview

Lite-Cog-Agent is a lightweight data analysis agent
powered by ReAct and LLM Tool Calling.

The agent can understand natural language data analysis
requests, select tools, execute Python analysis code,
observe execution results, and generate final answers.

## 2. Technology Stack

- Python 3.12
- uv
- DeepSeek API
- OpenAI Python SDK
- pandas
- numpy
- matplotlib
- openpyxl
- asyncio
- subprocess
- Local filesystem

## 3. Architecture Principles

- Use a modular monolith architecture.
- Follow the src layout.
- Use ReAct + Tool Calling as the agent pattern.
- Keep LLM Gateway independent from Agent Core.
- Keep tool definitions separate from tool execution.
- Use subprocess for Python code execution.
- Use local filesystem for session and artifact storage.
- Avoid unnecessary infrastructure and abstractions.

## 4. Restrictions

Do not introduce the following unless explicitly approved:

- FastAPI
- Docker
- PostgreSQL
- Redis
- LangGraph
- MCP
- Multi-Agent frameworks
- Frontend frameworks
- DDD architecture

## 5. Coding Standards

- Use Python type hints.
- Prefer clear and explicit function names.
- Keep modules focused on a single responsibility.
- Use async/await for asynchronous I/O.
- Keep API keys in environment variables.
- Never commit secrets.
- Handle exceptions explicitly.
- Add timeout controls to subprocess execution.

## 6. Development Workflow

- Implement one module at a time.
- Follow the development roadmap.
- Verify each module before proceeding.
- Do not prematurely implement future milestones.
- Keep README and documentation synchronized.
