from __future__ import annotations

import asyncio

from openai.types.chat import ChatCompletionMessage

import agent as agent_module
from gateway import gateway


def test_deep_agent_preserves_provider_tool_metadata(monkeypatch) -> None:
    assistant_with_tool_call = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_document",
                        "arguments": '{"query":"offer letter candidate name"}',
                    },
                    "extra_content": {
                        "google": {"thought_signature": "signature-value"},
                    },
                }
            ],
        }
    )
    second_request_messages: list[dict] = []
    request_count = 0

    async def complete_with_tools(messages, tools, purpose):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return assistant_with_tool_call
        second_request_messages.extend(messages)
        return ChatCompletionMessage(role="assistant", content="Jane Doe")

    class SearchTool:
        @staticmethod
        def invoke(args: dict) -> str:
            return "Candidate Name: Jane Doe"

    monkeypatch.setattr(gateway, "complete_with_tools", complete_with_tools)
    monkeypatch.setitem(agent_module._TOOL_MAP, "search_document", SearchTool())

    answer, _tool = asyncio.run(
        agent_module.DeepAgent().run_async("Who is the candidate?", intent="rag")
    )

    assistant_messages = [
        message for message in second_request_messages if message.get("role") == "assistant"
    ]
    assert answer == "Jane Doe"
    assert (
        assistant_messages[0]["tool_calls"][0]["extra_content"]["google"][
            "thought_signature"
        ]
        == "signature-value"
    )
