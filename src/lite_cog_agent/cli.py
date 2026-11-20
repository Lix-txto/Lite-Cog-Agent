"""
Command-line interface for Lite-Cog-Agent.

Provides an interactive interface for communicating
with the configured LLM through LLMGateway.
"""

from __future__ import annotations

import asyncio

from lite_cog_agent.config import settings
from lite_cog_agent.core.agent_budget import (
    AgentBudgetExceededError,
)
from lite_cog_agent.core.agent_factory import create_agent
from lite_cog_agent.core.agent_loop import AgentLoopError
from lite_cog_agent.core.agent_state import AgentStateError
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.llm.exceptions import LLMError
from lite_cog_agent.llm.gateway import LLMGateway


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
    print("  clear      Clear conversation history")
    print("  exit       Exit the application")
    print("  quit       Exit the application")
    print()
    print("Any other input will be sent to the Agent.")
    print()
    print()
    print("Data analysis examples:")
    print("  Analyze sales.csv")
    print("  Analyze the distribution of sales in sales.csv")
    print("  Remove rows with missing values in users.csv")
    print("  Calculate average age and score in users.csv")
    print("  Generate a histogram of the age column")
    print()
    print("Dataset files must be placed under the data/ directory.")



def print_status(
    gateway: LLMGateway,
    connected: bool,
    *,
    agent=None,
    context=None,
) -> None:
    """Print current CLI and Agent runtime status."""

    print("\n========== Agent Status ==========")

    print("Application: Lite-Cog-Agent")
    print(f"Provider: {settings.LLM_PROVIDER}")
    print(f"Model: {settings.DEEPSEEK_MODEL}")

    if connected:
        print("LLM Status: Connected (request succeeded)")
    else:
        print("LLM Status: Not connected yet")

    if agent is not None:
        print(f"Session ID: {agent.session_id}")

    if context is not None:
        print(
            f"Loaded Datasets: {len(context.dataset_ids)}"
        )

    print(
        f"LLM Requests: {gateway.usage_summary.request_count}"
    )

    print("==================================")



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

async def run_interactive() -> None:
    """Run the interactive Agent CLI session."""

    print_banner()

    connected = False

    # Ensure runtime directories exist.
    dataset_root = settings.DATA_DIR
    artifact_root = (
        settings.DATA_DIR
        / "artifacts"
        / "visualizations"
    )

    dataset_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # One Gateway and one ExecutionContext per CLI session.
    async with LLMGateway() as gateway:
        context = ExecutionContext()

        try:
            # Compose the complete Agent runtime.
            agent = create_agent(
                gateway,
                context=context,
                dataset_root=dataset_root,
                artifact_root=artifact_root,
            )

            print(
                "\nAgent initialized successfully."
            )
            print(
                f"Session ID: {agent.session_id}"
            )
            print(
                f"Dataset directory: {dataset_root}"
            )

            while True:
                try:
                    user_input = input(
                        "\nYou: "
                    ).strip()

                except EOFError:
                    print("\nInput stream closed.")
                    break

                if not user_input:
                    continue

                command = user_input.lower()

                if command in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                if command == "help":
                    print_help()
                    continue

                if command == "status":
                    print_status(
                        gateway,
                        connected,
                        agent=agent,
                        context=context,
                    )
                    continue

                if command == "usage":
                    print_usage(gateway)
                    continue

                if command == "reset":
                    gateway.reset_usage()
                    print(
                        "LLM usage statistics reset."
                    )
                    continue

                if command == "clear":
                    try:
                        agent.reset()
                        print(
                            "Conversation history cleared."
                        )
                        print(
                            "Loaded datasets and artifacts "
                            "have been preserved."
                        )
                    except AgentStateError as exc:
                        print(f"Unable to clear session: {exc}")

                    continue

                # All ordinary input goes through Agent.
                try:
                    response = await agent.run(
                        user_input
                    )

                    connected = True
                    print_response(response)

                except AgentBudgetExceededError as exc:
                    print(
                        f"\nAgent budget exceeded: {exc}"
                    )

                except LLMError as exc:
                    print(
                        f"\nLLM request failed: {exc}"
                    )

                except (
                    AgentLoopError,
                    AgentStateError,
                ) as exc:
                    print(
                        f"\nAgent execution failed: {exc}"
                    )

                except Exception as exc:
                    print(
                        "\nUnexpected error during "
                        f"Agent execution: {exc}"
                    )

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        finally:
            # ExecutionContext owns the in-memory datasets.
            # Close it when the interactive session ends.
            context.close()


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
