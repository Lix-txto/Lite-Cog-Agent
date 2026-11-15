"""
Command-line interface for Lite-Cog-Agent.
"""

from __future__ import annotations


def print_banner() -> None:
    """Display the application startup banner."""
    print()
    print("=" * 60)
    print("  Lite-Cog-Agent")
    print("  Lightweight Data Analysis Agent")
    print("=" * 60)
    print("  Type 'help' to see available commands.")
    print("  Type 'exit' or 'quit' to close the application.")
    print()


def print_help() -> None:
    """Display available CLI commands."""
    print()
    print("Available commands:")
    print("  help       Show available commands")
    print("  status     Show current application status")
    print("  exit       Exit the application")
    print("  quit       Exit the application")
    print()


def print_status() -> None:
    """Display the current application status."""
    print()
    print("Application Status")
    print("-" * 40)
    print("  Application : Lite-Cog-Agent")
    print("  Version     : 0.1.0")
    print("  Status      : Running")
    print("  Agent       : Not initialized")
    print("  LLM         : Not connected")
    print()
    print("Note: Agent and LLM integration will be implemented")
    print("      in subsequent development milestones.")
    print()


def run_interactive() -> None:
    """Start the interactive command-line loop."""
    print_banner()

    while True:
        try:
            command = input("lite-cog-agent> ").strip().lower()

            if not command:
                continue

            if command == "help":
                print_help()

            elif command == "status":
                print_status()

            elif command in {"exit", "quit"}:
                print("\nShutting down Lite-Cog-Agent...")
                print("Goodbye!")
                break

            else:
                print(f"Unknown command: {command}")
                print("Type 'help' to see available commands.")

        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal. Exiting...")
            break

        except EOFError:
            print("\n\nInput stream closed. Exiting...")
            break


def main() -> int:
    """Main entry point of the application."""
    try:
        run_interactive()
        return 0

    except Exception as exc:
        print(f"\nApplication error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
