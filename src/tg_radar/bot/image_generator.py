from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx

from tg_radar.bot.avatar import ChannelAvatarLoader
from tg_radar.config import Settings


class AirbotImageGenerationError(RuntimeError):
    pass


class AirbotImageGenerator:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        avatar_loader: ChannelAvatarLoader | None = None,
    ) -> None:
        self.settings = settings
        self.root = Path(settings.bot_generated_images_path)
        self._client = client
        self.avatar_loader = avatar_loader
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_or_create(self, result: dict[str, Any]) -> Path:
        channel = str(result.get("channel") or "")
        clean = self.clean_channel(channel)
        if not clean:
            raise AirbotImageGenerationError("channel is missing")
        existing = self.image_path(clean)
        if existing:
            return existing
        lock = self._locks.setdefault(clean, asyncio.Lock())
        async with lock:
            existing = self.image_path(clean)
            if existing:
                return existing
            self.root.mkdir(parents=True, exist_ok=True)
            image_bytes = await self._generate(result)
            path = self.root / f"llmref-{clean}.png"
            path.write_bytes(image_bytes)
            self._write_prompt(result, clean)
            return path

    def image_path(self, channel: str) -> Path | None:
        clean = self.clean_channel(channel)
        if not clean or not self.root.exists():
            return None
        reference = self.root / f"llmref-{clean}.png"
        if reference.exists():
            return reference
        if self.avatar_loader:
            return None
        matches = sorted(self.root.glob(f"*-{clean}.png"))
        if matches:
            return matches[0]
        direct = self.root / f"{clean}.png"
        return direct if direct.exists() else None

    def clean_channel(self, channel: str) -> str:
        return "".join(char for char in channel.lower().lstrip("@") if char.isascii() and (char.isalnum() or char in {"_", "-"}))

    async def _generate(self, result: dict[str, Any]) -> bytes:
        if not self.settings.llm_base_url or not self.settings.llm_api_key:
            raise AirbotImageGenerationError("image model is not configured")
        avatar = await self._avatar_bytes(str(result.get("channel") or ""))
        if avatar:
            data = await self._post_edit(avatar, self._reference_prompt(result))
            return self._image_bytes(data)
        payload = {
            "model": self.settings.bot_image_model,
            "prompt": self._prompt(result),
            "size": self.settings.bot_image_size,
        }
        data = await self._post_json("/images/generations", payload)
        return self._image_bytes(data)

    def _image_bytes(self, data: dict[str, Any]) -> bytes:
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise AirbotImageGenerationError("image response is empty")
        item = items[0] if isinstance(items[0], dict) else {}
        if isinstance(item.get("b64_json"), str):
            return base64.b64decode(item["b64_json"])
        if isinstance(item.get("url"), str):
            raise AirbotImageGenerationError("image URL responses are not supported here")
        raise AirbotImageGenerationError("image payload is missing")

    async def _avatar_bytes(self, channel: str) -> tuple[str, bytes, str] | None:
        clean = self.clean_channel(channel)
        if not clean or not self.avatar_loader:
            return None
        local = self._local_avatar(clean)
        if local:
            return local
        try:
            avatar = await self.avatar_loader.fetch_avatar(clean)
        except Exception:
            return None
        if not avatar.data:
            return None
        content_type = avatar.content_type or "image/jpeg"
        suffix = ".png" if "png" in content_type else ".jpg"
        return f"{clean}-avatar{suffix}", avatar.data, content_type

    def _local_avatar(self, clean: str) -> tuple[str, bytes, str] | None:
        for suffix, content_type in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".webp", "image/webp")):
            path = self.root / f"{clean}-avatar{suffix}"
            if path.exists():
                return path.name, path.read_bytes(), content_type
        return None

    async def _post_edit(self, avatar: tuple[str, bytes, str], prompt: str) -> dict[str, Any]:
        client = self._client
        if client is not None:
            return await self._post_edit_with_client(client, avatar, prompt)
        async with httpx.AsyncClient(timeout=self.settings.bot_image_timeout_seconds) as owned:
            return await self._post_edit_with_client(owned, avatar, prompt)

    async def _post_edit_with_client(self, client: httpx.AsyncClient, avatar: tuple[str, bytes, str], prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f"{self.settings.llm_base_url.rstrip('/')}/images/edits"
        filename, data, content_type = avatar
        for attempt in range(3):
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                    data={"model": self.settings.bot_image_model, "prompt": prompt, "size": self.settings.bot_image_size},
                    files={"image": (filename, data, content_type)},
                )
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    if attempt + 1 < 3:
                        await asyncio.sleep(0.7 * (attempt + 1))
                        continue
                if response.status_code >= 400:
                    raise AirbotImageGenerationError(self._http_error(response))
                payload = response.json()
                if not isinstance(payload, dict):
                    raise AirbotImageGenerationError("image edit response is not an object")
                return payload
            except AirbotImageGenerationError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt + 1 < 3:
                    await asyncio.sleep(0.7 * (attempt + 1))
                    continue
                raise AirbotImageGenerationError(f"{type(exc).__name__}: {exc}") from exc
        raise AirbotImageGenerationError(f"{type(last_error).__name__}: {last_error}") from last_error

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        if client is not None:
            return await self._post_with_client(client, path, payload)
        async with httpx.AsyncClient(timeout=self.settings.bot_image_timeout_seconds) as owned:
            return await self._post_with_client(owned, path, payload)

    async def _post_with_client(self, client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f"{self.settings.llm_base_url.rstrip('/')}{path}"
        for attempt in range(3):
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                    json=payload,
                )
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    if attempt + 1 < 3:
                        await asyncio.sleep(0.7 * (attempt + 1))
                        continue
                if response.status_code >= 400:
                    raise AirbotImageGenerationError(self._http_error(response))
                data = response.json()
                if not isinstance(data, dict):
                    raise AirbotImageGenerationError("image response is not an object")
                return data
            except AirbotImageGenerationError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt + 1 < 3:
                    await asyncio.sleep(0.7 * (attempt + 1))
                    continue
                raise AirbotImageGenerationError(f"{type(exc).__name__}: {exc}") from exc
        raise AirbotImageGenerationError(f"{type(last_error).__name__}: {last_error}") from last_error

    def _reference_prompt(self, result: dict[str, Any]) -> str:
        payload = self._prompt_payload(result)
        return (
            "Use the uploaded image only as the exact graphic reference for the shirt badge. "
            "Create a realistic funny square Telegram avatar mascot: one human-like character in a black tuxedo, "
            "with a weather-device head selected by ELO from the analysis JSON: desk fan, air conditioner, tornado vortex, hurricane eye, or cosmic vortex. "
            "On the center of the white shirt, place a large circular printed badge that preserves the uploaded avatar image as recognizably as possible: "
            "same main shapes, same contrast, same circular framing, not redrawn as a different symbol. "
            "The uploaded avatar must appear on the shirt badge only, not as the character face or head. "
            "Background: wind, hurricanes, tornado trails, storm pressure, clean cinematic lighting. "
            "Composition must be centered, simple, readable at Telegram avatar size, high detail, not cluttered. "
            "No readable text, no logos, no watermark, no real person likeness. Analysis JSON: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _prompt(self, result: dict[str, Any]) -> str:
        payload = self._prompt_payload(result)
        return (
            "Create a realistic funny Telegram avatar mascot from this public channel analysis JSON. "
            "Do not depict a real person, do not copy a real face, do not use logos, do not add readable text. "
            "Make one human-like character in a tuxedo, with a weather-device head selected by ELO: "
            "low ELO desk fan head, medium ELO air conditioner or industrial fan head, high ELO tornado, hurricane eye, or cosmic vortex head. "
            "The tuxedo shirt must have a clearly visible circular badge on the center of the chest. "
            "Background: wind, hurricanes, tornado trails, storm pressure, clean cinematic lighting. "
            "Composition must be simple, centered, readable at Telegram avatar size, square 1:1, high detail, not cluttered. "
            "Use the metrics and tactics as visual hints only. Analysis JSON: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _prompt_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
        rank = result.get("air_rank") if isinstance(result.get("air_rank"), dict) else {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        top_metrics = sorted(
            (
                {"name": str(name), "score": self._int(item.get("score"))}
                for name, item in metrics.items()
                if isinstance(item, dict)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[:8]
        return {
            "channel": result.get("channel"),
            "elo": result.get("air_elo"),
            "rank": rank.get("tag") or profile.get("rank_name") or result.get("humorous_label"),
            "measure": result.get("air_rank_measure") or profile.get("rank_measure"),
            "niche": self._strings(result.get("niche_tags"), 8),
            "traits": self._strings(result.get("core_traits"), 8),
            "top_metrics": top_metrics,
            "tactics": [
                {
                    "name": item.get("name") or item.get("code"),
                    "intensity": item.get("intensity"),
                    "hint": item.get("hint"),
                }
                for item in (result.get("tactics") if isinstance(result.get("tactics"), list) else [])[:6]
                if isinstance(item, dict)
            ],
            "one_liner": profile.get("one_liner") or result.get("summary"),
            "fun_metrics": result.get("fun_metrics") if isinstance(result.get("fun_metrics"), list) else [],
        }

    def _write_prompt(self, result: dict[str, Any], clean: str) -> None:
        payload = {"prompt": self._prompt(result), "result": self._prompt_payload(result)}
        path = self.root / f"generated-{clean}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _http_error(self, response: httpx.Response) -> str:
        body = response.text.strip()
        if len(body) > 500:
            body = body[:497].rstrip() + "..."
        return f"image HTTP {response.status_code}: {body}" if body else f"image HTTP {response.status_code}"

    def _strings(self, value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:80] for item in value if str(item).strip()][:limit]

    def _int(self, value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0
