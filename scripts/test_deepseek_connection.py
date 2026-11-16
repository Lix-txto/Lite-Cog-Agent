"""
Integration test for the real DeepSeek API.

This script sends one real request through LLMGateway
and verifies the returned LLMResponse.
"""

from __future__ import annotations

import asyncio

from lite_cog_agent.llm.exceptions import LLMError
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import LLMMessage, LLMRequest


async def main() -> int:
    print("=" * 60)
    print("Lite-Cog-Agent | DeepSeek API Integration Test")
    print("=" * 60)

    try:
        async with LLMGateway() as gateway:
            request = LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a concise assistant. "
                            "Answer in Chinese."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            "请简要介绍你能如何帮助分析 CSV 数据，"
                            "并列举三个具体例子。"
                        ),
                    ),
                ],
                max_tokens=128,
            )

            print("\n[1/3] Sending request to DeepSeek...")

            response = await gateway.chat(request)

            print("[2/3] Response received successfully.")

            if not response.content:
                raise RuntimeError(
                    "DeepSeek returned an empty text response."
                )

            print("\n[3/3] Validating response...")

            print("\n" + "-" * 60)
            print("Model:", response.model)
            print("Finish reason:", response.finish_reason)
            print("Prompt tokens:", response.usage.prompt_tokens)
            print("Completion tokens:", response.usage.completion_tokens)
            print("Total tokens:", response.usage.total_tokens)
            print("-" * 60)

            print("\nDeepSeek Response:")
            print(response.content)

            assert response.model is not None
            assert response.usage.total_tokens > 0

            print("\n" + "=" * 60)
            print("DeepSeek API integration test PASSED.")
            print("=" * 60)

            return 0

    except LLMError as exc:
        print("\nDeepSeek LLM operation failed.")
        print("Error type:", type(exc).__name__)
        print("Error:", str(exc))
        return 1

    except Exception as exc:
        print("\nUnexpected integration test failure.")
        print("Error type:", type(exc).__name__)
        print("Error:", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
