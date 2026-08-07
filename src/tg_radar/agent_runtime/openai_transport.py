from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field


class OpenAITransportConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = Field(default=90.0, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=1)
    retries: int = Field(default=3, ge=1, le=10)
    json_response_format: bool = True

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")


class OpenAITransport(Protocol):
    async def complete_json(self, messages: list[dict[str, str]]) -> str:
        ...


async def _post_json(config: OpenAITransportConfig, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    url = f"{config.normalized_base_url}{path}"
    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        for attempt in range(config.retries):
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    json=payload,
                )
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    if attempt + 1 < config.retries:
                        await asyncio.sleep(0.7 * (attempt + 1))
                        continue
                if response.status_code >= 400:
                    raise RuntimeError(_http_error(response))
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("LLM returned non-object response")
                return data
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt + 1 < config.retries:
                    await asyncio.sleep(0.7 * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM transport error: {type(exc).__name__}: {exc}") from exc
    raise RuntimeError(f"LLM transport error: {type(last_error).__name__}: {last_error}") from last_error


def _http_error(response: httpx.Response) -> str:
    body = response.text.strip()
    if len(body) > 500:
        body = body[:497].rstrip() + "..."
    if body:
        return f"LLM HTTP {response.status_code}: {body}"
    return f"LLM HTTP {response.status_code}"


class ChatCompletionsTransport:
    def __init__(self, config: OpenAITransportConfig) -> None:
        self.config = config

    async def complete_json(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.json_response_format:
            payload["response_format"] = {"type": "json_object"}
        data = await _post_json(self.config, "/chat/completions", payload)
        return str(data["choices"][0]["message"]["content"])


class ResponsesTransport:
    def __init__(self, config: OpenAITransportConfig) -> None:
        self.config = config

    async def complete_json(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": messages,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        return self._content(await _post_json(self.config, "/responses", payload))

    def _content(self, data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        if isinstance(data.get("text"), str):
            return data["text"]
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"]
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            item = choices[0]
            message = item.get("message") if isinstance(item, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        raise ValueError("response content not found")
