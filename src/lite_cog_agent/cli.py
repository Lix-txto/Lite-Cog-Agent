"""
Command-line interface for Lite-Cog-Agent.

Provides an interactive interface for communicating
with the configured LLM through LLMGateway.
"""

from __future__ import annotations

import asyncio

from lite_cog_agent.config import settings
from lite_cog_agent.llm.exceptions import LLMError
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import LLMMessage, LLMRequest


SYSTEM_PROMPT = """
You are Lite-Cog-Agent, a lightweight data analysis assistant.

Your responsibilities:
1. Help users understand and analyze structured data.
2. Explain data analysis concepts clearly.
3. Suggest practical Python and pandas solutions.
4. Answer in the same language as the user.
5. Do not claim to have analyzed files unless data was actually provided
   and processed by an available tool.

At this stage, you are operating in text-only mode.
You cannot directly read files or execute Python code.
""".strip()


def print_banner() -> None:
    """Display the application startup banner."""

    print()
    print("=" * 60)
    print("  Lite-Cog-Agent")
    print("  Lightweight Data Analysis Agent")
    print("=" * 60)
    print(f"  Provider : {settings.LLM_PROVIDER}")
    print(f"  Model    : {settings.DEEPSEEK_MODEL}")
    print("  Type 'help' to see available commands.")
    print("  Type 'exit' or 'quit' to close the application.")
    print()


def print_help() -> None:
    """Display available CLI commands."""

    print()
    print("Available commands:")
    print("  help       Show available commands")
    print("  status     Show application status")
    print("  usage      Show accumulated token usage")
    print("  reset      Reset token usage statistics")
    print("  exit       Exit the application")
    print("  quit       Exit the application")
    print()
    print("Any other input will be sent to the LLM.")
    print()


def print_status(gateway: LLMGateway, connected: bool) -> None:
    """Display application and LLM status."""

    summary = gateway.usage_summary

    print()
    print("Application Status")
    print("-" * 40)
    print("  Application  : Lite-Cog-Agent")
    print("  Version      : 0.1.0")
    print(f"  LLM Provider : {settings.LLM_PROVIDER}")
    print(f"  LLM Model    : {settings.DEEPSEEK_MODEL}")

    if connected:
        llm_status = "Connected (request succeeded)"
    else:
        llm_status = "Configured (not tested yet)"

    print(f"  LLM Status   : {llm_status}")
    print(f"  Requests     : {summary.request_count}")
    print()


def print_usage(gateway: LLMGateway) -> None:
    """Display accumulated token usage."""

    summary = gateway.usage_summary

    print()
    print("Token Usage")
    print("-" * 40)
    print(f"  Requests           : {summary.request_count}")
    print(f"  Prompt Tokens      : {summary.prompt_tokens}")
    print(f"  Completion Tokens  : {summary.completion_tokens}")
    print(f"  Total Tokens       : {summary.total_tokens}")
    print()


def print_response(response) -> None:
    """Display an LLM response and its token usage."""

    print()
    print("Assistant:")
    print(response.content or "(No text content returned.)")

    if response.tool_calls:
        print("\nTool Calls:")
        for tool_call in response.tool_calls:
            print(f"  Name: {tool_call.name}")
            print(f"  Arguments: {tool_call.arguments}")

    print()
    print("-" * 40)
    print(f"  Model              : {response.model}")
    print(f"  Finish Reason      : {response.finish_reason}")
    print(f"  Prompt Tokens      : {response.usage.prompt_tokens}")
    print(f"  Completion Tokens  : {response.usage.completion_tokens}")
    print(f"  Total Tokens       : {response.usage.total_tokens}")
    print("-" * 40)
    print()


async def send_message(
    gateway: LLMGateway,
    user_input: str,
):
    """Send a single user message through the LLM Gateway."""

    request = LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=SYSTEM_PROMPT,
            ),
            LLMMessage(
                role="user",
                content=user_input,
            ),
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    return await gateway.chat(request)


async def run_interactive() -> None:
    """Run the asynchronous interactive CLI."""

    print_banner()

    connected = False

    async with LLMGateway() as gateway:
        while True:
            try:
                user_input = input("lite-cog-agent> ").strip()

                if not user_input:
                    continue

                command = user_input.lower()

                if command == "help":
                    print_help()
                    continue

                if command == "status":
                    print_status(gateway, connected)
                    continue

                if command == "usage":
                    print_usage(gateway)
                    continue

                if command == "reset":
                    gateway.reset_usage()
                    print("\nToken usage statistics have been reset.\n")
                    continue

                if command in {"exit", "quit"}:
                    print("\nShutting down Lite-Cog-Agent...")
                    print("Goodbye!")
                    break

                print("\nThinking...")

                try:
                    response = await send_message(
                        gateway,
                        user_input,
                    )

                    connected = True
                    print_response(response)

                except LLMError as exc:
                    print("\nLLM request failed.")
                    print(f"Error type: {type(exc).__name__}")
                    print(f"Error: {exc}")
                    print()

            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal. Exiting...")
                break

            except EOFError:
                print("\n\nInput stream closed. Exiting...")
                break


def main() -> int:
    """Application entry point."""

    try:
        asyncio.run(run_interactive())
        return 0

    except KeyboardInterrupt:
        print("\nApplication interrupted.")
        return 130

    except LLMError as exc:
        print("\nApplication initialization failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        return 1

    except Exception as exc:
        print("\nUnexpected application error.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
