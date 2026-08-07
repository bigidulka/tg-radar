from __future__ import annotations

import asyncio
import base64
from html import escape
import json
from pathlib import Path
from typing import Any

import httpx

from tg_radar.bot.image_generator import AirbotImageGenerationError, AirbotImageGenerator
from tg_radar.bot.store import BotFightRun, BotRun
from tg_radar.bot.texts import text
from tg_radar.config import Settings


class AirbotFightImageGenerator:
    def __init__(
        self,
        settings: Settings,
        profile_generator: AirbotImageGenerator,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.profile_generator = profile_generator
        self.root = Path(settings.bot_generated_images_path)
        self._client = client
        self._locks: dict[str, asyncio.Lock] = {}

    async def generate_stage(self, fight: BotFightRun, left: BotRun, right: BotRun, result: dict[str, Any], stage: str) -> Path:
        path = self.stage_path(fight.fight_id, stage)
        if path.exists():
            return path
        lock = self._locks.setdefault(f"{fight.fight_id}:{stage}", asyncio.Lock())
        async with lock:
            if path.exists():
                return path
            left_profile = await self.profile_generator.get_or_create(dict(left.result or {}, channel=left.channel))
            right_profile = await self.profile_generator.get_or_create(dict(right.result or {}, channel=right.channel))
            prompt = self.stage_prompt(result, stage)
            payload = await self._post_edit([left_profile, right_profile], prompt)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._image_bytes(payload))
            self.stage_meta_path(fight.fight_id, stage).write_text(
                json.dumps(
                    {
                        "fight_id": fight.fight_id,
                        "stage": stage,
                        "inputs": [left_profile.name, right_profile.name],
                        "prompt": prompt,
                        "output": path.name,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return path

    def stage_path(self, fight_id: str, stage: str) -> Path:
        return self.root / f"fight-{self._clean(fight_id)}-{self._clean(stage)}.png"

    def stage_meta_path(self, fight_id: str, stage: str) -> Path:
        return self.root / f"fight-{self._clean(fight_id)}-{self._clean(stage)}.json"

    def image_paths(self, result: dict[str, Any]) -> list[Path]:
        images = result.get("images") if isinstance(result.get("images"), dict) else {}
        paths = []
        for stage in self.stages(result):
            value = images.get(stage)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = self.root / path.name
            if path.exists():
                paths.append(path)
        return paths

    def stages(self, result: dict[str, Any]) -> list[str]:
        rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
        stages = [f"round{index}" for index in range(1, min(len(rounds), 3) + 1)]
        stages.append("final")
        return stages

    def attach_image(self, result: dict[str, Any], stage: str, path: Path) -> dict[str, Any]:
        updated = dict(result)
        images = updated.get("images") if isinstance(updated.get("images"), dict) else {}
        updated["images"] = {**images, stage: str(path)}
        return updated

    def stage_caption(self, result: dict[str, Any], stage: str) -> str:
        if stage == "final":
            final = result.get("final") if isinstance(result.get("final"), dict) else {}
            return text(
                "fight.image_final_caption",
                winner=self._channel_link(str(final.get("channel") or "")),
                method=escape(str(final.get("method") or "")),
            )[:1024]
        number = self._stage_number(stage)
        rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
        item = rounds[number - 1] if 0 < number <= len(rounds) and isinstance(rounds[number - 1], dict) else {}
        return text(
            "fight.image_round_caption",
            number=number,
            title=escape(str(item.get("title") or "")),
            winner=self._channel_link(str(item.get("winner_channel") or "")),
            body=escape(str(item.get("snapshot") or "")),
        )[:1024]

    def stage_prompt(self, result: dict[str, Any], stage: str) -> str:
        data = self._stage_data(result, stage)
        left = str(result.get("left_channel") or "")
        right = str(result.get("right_channel") or "")
        return (
            "Use image 1 as the LEFT airbot profile reference and image 2 as the RIGHT airbot profile reference. "
            "Create one square cinematic meme battle frame for an airbot fight. "
            "Keep both fighters distinct and recognizable from references: tuxedos, weather-device heads, and shirt badges. "
            "No readable text, no logos, no watermark, no real person likeness, no gore, no weapons. "
            "Use a clean 1:1 composition, dramatic storm arena, flying wind, pressure waves, funny epic boss-battle energy. "
            f"Left channel: @{left}. Right channel: @{right}. Stage: {stage}. "
            f"Fight data JSON: {json.dumps(data, ensure_ascii=False)}"
        )

    def _stage_data(self, result: dict[str, Any], stage: str) -> dict[str, Any]:
        if stage == "final":
            return {
                "topic": result.get("topic"),
                "stage": "final",
                "final": result.get("final"),
                "share_line": result.get("share_line"),
                "visual_instruction": (
                    "Show the final winner in a dominant foreground pose. Show the defeated side pushed back or dissolved into harmless wind. "
                    "Make the winner obvious without any text."
                ),
            }
        number = self._stage_number(stage)
        rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
        item = rounds[number - 1] if 0 < number <= len(rounds) and isinstance(rounds[number - 1], dict) else {}
        return {
            "topic": result.get("topic"),
            "stage": stage,
            "round": item,
            "visual_instruction": (
                "Show this round as an active clash. The round winner must visibly overpower the other side, "
                "but both fighters stay intact and recognizable."
            ),
        }

    async def _post_edit(self, image_paths: list[Path], prompt: str) -> dict[str, Any]:
        if not self.settings.llm_base_url or not self.settings.llm_api_key:
            raise AirbotImageGenerationError("image model is not configured")
        client = self._client
        if client is not None:
            return await self._post_edit_with_client(client, image_paths, prompt)
        async with httpx.AsyncClient(timeout=self.settings.bot_image_timeout_seconds) as owned:
            return await self._post_edit_with_client(owned, image_paths, prompt)

    async def _post_edit_with_client(self, client: httpx.AsyncClient, image_paths: list[Path], prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f"{self.settings.llm_base_url.rstrip('/')}/images/edits"
        for attempt in range(3):
            opened = []
            try:
                files = []
                for path in image_paths:
                    handle = path.open("rb")
                    opened.append(handle)
                    files.append(("image", (path.name, handle, "image/png")))
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                    data={"model": self.settings.bot_image_model, "prompt": prompt, "size": self.settings.bot_image_size},
                    files=files,
                )
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    if attempt + 1 < 3:
                        await asyncio.sleep(0.7 * (attempt + 1))
                        continue
                if response.status_code >= 400:
                    raise AirbotImageGenerationError(self._http_error(response))
                payload = response.json()
                if not isinstance(payload, dict):
                    raise AirbotImageGenerationError("fight image response is not an object")
                return payload
            except AirbotImageGenerationError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt + 1 < 3:
                    await asyncio.sleep(0.7 * (attempt + 1))
                    continue
                raise AirbotImageGenerationError(f"{type(exc).__name__}: {exc}") from exc
            finally:
                for handle in opened:
                    handle.close()
        raise AirbotImageGenerationError(f"{type(last_error).__name__}: {last_error}") from last_error

    def _image_bytes(self, data: dict[str, Any]) -> bytes:
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise AirbotImageGenerationError("fight image response is empty")
        item = items[0] if isinstance(items[0], dict) else {}
        if isinstance(item.get("b64_json"), str):
            return base64.b64decode(item["b64_json"])
        raise AirbotImageGenerationError("fight image payload is missing")

    def _http_error(self, response: httpx.Response) -> str:
        body = response.text.strip()
        if len(body) > 500:
            body = body[:497].rstrip() + "..."
        return f"fight image HTTP {response.status_code}: {body}" if body else f"fight image HTTP {response.status_code}"

    def _stage_number(self, stage: str) -> int:
        try:
            return int(stage.removeprefix("round"))
        except Exception:
            return 0

    def _clean(self, value: str) -> str:
        return "".join(char for char in value.lower() if char.isascii() and (char.isalnum() or char in {"_", "-"}))[:96] or "fight"

    def _channel_link(self, channel: str) -> str:
        clean = channel.lower().strip().lstrip("@")
        if not clean:
            return escape(text("fight.draw"))
        value = escape(clean)
        url = escape(f"https://t.me/{clean}", quote=True)
        return f"<a href=\"{url}\">@{value}</a>"
