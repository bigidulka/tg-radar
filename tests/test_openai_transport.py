from __future__ import annotations

import respx
from httpx import Response

from tg_radar.agent_runtime import ChatCompletionsTransport, OpenAITransportConfig, ResponsesTransport


@respx.mock
async def test_chat_completions_transport_returns_message_content():
    route = respx.post("http://localhost:8317/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={"choices": [{"message": {"content": "{\"type\":\"final\",\"reason\":\"done\"}"}}]},
        )
    )
    transport = ChatCompletionsTransport(
        OpenAITransportConfig(
            base_url="http://localhost:8317/v1/",
            api_key="sk-local-dev-key",
            model="gpt-5.5",
        )
    )

    content = await transport.complete_json([{"role": "user", "content": "x"}])

    assert content == "{\"type\":\"final\",\"reason\":\"done\"}"
    assert route.calls.last.request.content
    assert b"response_format" in route.calls.last.request.content


@respx.mock
async def test_chat_completions_transport_can_skip_json_response_format():
    route = respx.post("http://localhost:8317/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={"choices": [{"message": {"content": "{\"type\":\"final\",\"reason\":\"done\"}"}}]},
        )
    )
    transport = ChatCompletionsTransport(
        OpenAITransportConfig(
            base_url="http://localhost:8317/v1/",
            api_key="sk-local-dev-key",
            model="gpt-5.5",
            json_response_format=False,
        )
    )

    content = await transport.complete_json([{"role": "user", "content": "x"}])

    assert content == "{\"type\":\"final\",\"reason\":\"done\"}"
    assert b"response_format" not in route.calls.last.request.content


@respx.mock
async def test_responses_transport_returns_output_text():
    respx.post("http://localhost:8317/v1/responses").mock(
        return_value=Response(200, json={"output_text": "{\"type\":\"final\",\"reason\":\"done\"}"})
    )
    transport = ResponsesTransport(
        OpenAITransportConfig(
            base_url="http://localhost:8317/v1/",
            api_key="sk-local-dev-key",
            model="gpt-5.5",
        )
    )

    content = await transport.complete_json([{"role": "user", "content": "x"}])

    assert content == "{\"type\":\"final\",\"reason\":\"done\"}"


@respx.mock
async def test_responses_transport_returns_nested_output_text():
    respx.post("http://localhost:8317/v1/responses").mock(
        return_value=Response(
            200,
            json={"output": [{"content": [{"type": "output_text", "text": "{\"type\":\"final\",\"reason\":\"done\"}"}]}]},
        )
    )
    transport = ResponsesTransport(
        OpenAITransportConfig(
            base_url="http://localhost:8317/v1/",
            api_key="sk-local-dev-key",
            model="gpt-5.5",
        )
    )

    content = await transport.complete_json([{"role": "user", "content": "x"}])

    assert content == "{\"type\":\"final\",\"reason\":\"done\"}"
