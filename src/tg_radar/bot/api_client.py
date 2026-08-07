from __future__ import annotations

import asyncio
from typing import Any

import httpx

from tg_radar.text import clean_username


class AirbotApiError(RuntimeError):
    pass


class TgRadarApiClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        timeout_seconds: float = 60.0,
        retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.retries = max(retries, 1)
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_messages(self, channel: str, limit: int) -> list[dict[str, Any]]:
        clean = clean_username(channel)
        if not clean:
            raise AirbotApiError("invalid channel username")
        data = await self._post(f"/core/channels/{clean}/fetch-messages", {"limit": limit})
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise AirbotApiError("invalid fetch response")
        return [message for message in messages if isinstance(message, dict)]

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = await self._client.post(f"{self.base_url}{path}", headers=headers, json=payload)
                if response.status_code >= 500 and attempt + 1 < self.retries:
                    await asyncio.sleep(0.4 * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    raise AirbotApiError(self._error_detail(response))
                data = response.json()
                if not isinstance(data, dict):
                    raise AirbotApiError("invalid api response")
                return data
            except AirbotApiError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    await asyncio.sleep(0.4 * (attempt + 1))
                    continue
        raise AirbotApiError(f"{type(last_error).__name__}: {last_error}") from last_error

    def _error_detail(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except Exception:
            return response.text[:500] or f"HTTP {response.status_code}"
        detail = data.get("detail") if isinstance(data, dict) else None
        return str(detail or f"HTTP {response.status_code}")
