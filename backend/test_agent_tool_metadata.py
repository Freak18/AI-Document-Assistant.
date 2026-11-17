from __future__ import annotations

import asyncio
import json

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


def test_deep_agent_handles_multi_query_and_finishes_after_tool_limit(monkeypatch) -> None:
    requests: list[tuple[list[dict], list[dict]]] = []
    searched_queries: list[str] = []

    def tool_call_message(call_id: str, query: str | list[str]) -> ChatCompletionMessage:
        return ChatCompletionMessage.model_validate(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "search_document",
                            "arguments": json.dumps({"query": query}),
                        },
                    }
                ],
            }
        )

    async def complete_with_tools(messages, tools, purpose):
        requests.append((messages, tools))
        if len(requests) == 1:
            return tool_call_message(
                "call_1", ["employee name", "Dear", "Name", "offer letter"]
            )
        if len(requests) <= 4:
            return tool_call_message(f"call_{len(requests)}", "employee name")
        return ChatCompletionMessage(role="assistant", content="Jane Doe")

    class SearchTool:
        @staticmethod
        def invoke(args: dict) -> str:
            searched_queries.append(args["query"])
            return "Candidate name appears in the offer letter."

    monkeypatch.setattr(gateway, "complete_with_tools", complete_with_tools)
    monkeypatch.setitem(agent_module._TOOL_MAP, "search_document", SearchTool())

    answer, tool_label = asyncio.run(
        agent_module.DeepAgent().run_async("Who is the employee?", intent="rag")
    )

    assert answer == "Jane Doe"
    assert searched_queries[:4] == ["employee name", "Dear", "Name", "offer letter"]
    assert len(requests) == 5
    assert requests[-1][1] == []
    assert "search_document(no_match)" not in tool_label
