from __future__ import annotations

from pathlib import Path
from typing import Any

import aiohttp
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup, Message

from tg_radar.bot.avatar import ChannelAvatarLoader
from tg_radar.bot.cards import AirbotCardRenderer, result_card_caption, result_profile_html, result_profile_plain
from tg_radar.config import get_settings
from tg_radar.bot.runner import render_result
from tg_radar.bot.texts import bot_texts, text


class AirbotTelegramRenderer:
    def __init__(
        self,
        card_renderer: AirbotCardRenderer | None = None,
        avatar_loader: ChannelAvatarLoader | None = None,
    ) -> None:
        self.card_renderer = card_renderer or AirbotCardRenderer()
        self.avatar_loader = avatar_loader
        self._avatar_cache: dict[str, str | None] = {}
        settings = get_settings()
        self.generated_images_path = Path(settings.bot_generated_images_path)
        self.generated_images_base_url = (settings.bot_generated_images_base_url or "").rstrip("/")

    async def send_text(
        self,
        message: Message,
        value: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        if len(value) <= 3900:
            await message.answer(value, reply_markup=reply_markup)
            return
        chunk = []
        size = 0
        for line in value.splitlines():
            if size + len(line) + 1 > 3900 and chunk:
                await message.answer("\n".join(chunk))
                chunk = []
                size = 0
            chunk.append(line)
            size += len(line) + 1
        if chunk:
            await message.answer("\n".join(chunk), reply_markup=reply_markup)

    async def send_result(
        self,
        message: Message,
        result: dict,
        reply_markup: InlineKeyboardMarkup | None = None,
        referrer_user_id: int | None = None,
    ) -> None:
        bot_username = await self.bot_username(message.bot)
        try:
            await message.bot.send_chat_action(chat_id=message.chat.id, action="upload_photo")
            png = self.card_renderer.render(result)
            await message.answer_photo(
                BufferedInputFile(png, filename=self._filename(result)),
                caption=result_card_caption(result, bot_username=bot_username, referrer_user_id=referrer_user_id),
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        except Exception:
            await self.send_text(message, render_result(result), reply_markup=reply_markup)
            return
        await self.send_profile(message, result, referrer_user_id=referrer_user_id)

    async def send_profile(
        self,
        message: Message,
        result: dict,
        reply_markup: InlineKeyboardMarkup | None = None,
        referrer_user_id: int | None = None,
    ) -> None:
        bot_username = await self.bot_username(message.bot)
        await self.send_rich_html_variants(
            message,
            await self.result_html_variants(result, bot_username=bot_username, referrer_user_id=referrer_user_id),
            fallback=result_profile_plain(result, bot_username=bot_username, referrer_user_id=referrer_user_id),
            reply_markup=reply_markup,
        )

    def generated_image_path(self, channel: str) -> Path | None:
        clean = self._clean_channel(channel)
        if not clean or not self.generated_images_path.exists():
            return None
        matches = sorted(self.generated_images_path.glob(f"*-{clean}.png"))
        if not matches:
            direct = self.generated_images_path / f"{clean}.png"
            return direct if direct.exists() else None
        return matches[0]

    def generated_image_url(self, channel: str) -> str | None:
        path = self.generated_image_path(channel)
        if not path or not self.generated_images_base_url:
            return None
        return f"{self.generated_images_base_url}/{path.name}"

    def generated_image_urls(self, channels: list[str]) -> dict[str, str]:
        out = {}
        for channel in channels:
            url = self.generated_image_url(channel)
            if url:
                out[channel.lower()] = url
        return out

    async def result_html(
        self,
        result: dict,
        bot_username: str | None = None,
        referrer_user_id: int | None = None,
        include_avatar: bool = True,
    ) -> str:
        return result_profile_html(
            result,
            bot_username=bot_username,
            referrer_user_id=referrer_user_id,
            avatar_url=await self._avatar_url(result) if include_avatar else None,
            generated_image_url=self.generated_image_url(str(result.get("channel") or "")),
        )

    async def result_html_variants(
        self,
        result: dict,
        bot_username: str | None = None,
        referrer_user_id: int | None = None,
    ) -> list[str]:
        with_avatar = await self.result_html(
            result,
            bot_username=bot_username,
            referrer_user_id=referrer_user_id,
            include_avatar=True,
        )
        without_avatar = await self.result_html(
            result,
            bot_username=bot_username,
            referrer_user_id=referrer_user_id,
            include_avatar=False,
        )
        return [with_avatar] if with_avatar == without_avatar else [with_avatar, without_avatar]

    async def send_json(
        self,
        message: Message,
        result: dict,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        await self.send_text(message, text("card.json_intro"), reply_markup=reply_markup)
        await self.send_text(message, render_result(result))

    async def send_rich_screen(
        self,
        message: Message,
        html: str | list[str],
        fallback: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> int:
        token = getattr(message.bot, "token", None)
        if token:
            for value in self._html_values(html):
                for decorate_buttons in (True, False):
                    payload = self._rich_payload(message.chat.id, value, reply_markup, decorate_buttons=decorate_buttons)
                    try:
                        data = await self._tg_call(str(token), "sendRichMessage", payload)
                        if isinstance(data, dict) and data.get("message_id"):
                            return int(data["message_id"])
                    except Exception:
                        continue
        sent = await message.answer(fallback, reply_markup=reply_markup)
        return sent.message_id

    async def edit_rich_screen(
        self,
        bot,
        chat_id: int,
        message_id: int,
        html: str | list[str],
        fallback: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> bool:
        token = getattr(bot, "token", None)
        if token:
            for value in self._html_values(html):
                for decorate_buttons in (True, False):
                    payload = self._rich_payload(chat_id, value, reply_markup, decorate_buttons=decorate_buttons)
                    payload["message_id"] = message_id
                    try:
                        await self._tg_call(str(token), "editMessageText", payload)
                        return True
                    except Exception as exc:
                        if "message is not modified" in str(exc).lower():
                            return True
                        continue
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=fallback[:3900],
                reply_markup=reply_markup,
            )
            return True
        except Exception:
            return False

    async def send_rich_html(
        self,
        message: Message,
        html: str,
        fallback: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        token = getattr(message.bot, "token", None)
        if token:
            for decorate_buttons in (True, False):
                payload = self._rich_payload(message.chat.id, html, reply_markup, decorate_buttons=decorate_buttons)
                try:
                    await self._tg_call(str(token), "sendRichMessage", payload)
                    return
                except Exception:
                    continue
        await self.send_text(message, fallback, reply_markup=reply_markup)

    async def send_rich_html_variants(
        self,
        message: Message,
        html: str | list[str],
        fallback: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        token = getattr(message.bot, "token", None)
        if token:
            for value in self._html_values(html):
                for decorate_buttons in (True, False):
                    payload = self._rich_payload(message.chat.id, value, reply_markup, decorate_buttons=decorate_buttons)
                    try:
                        await self._tg_call(str(token), "sendRichMessage", payload)
                        return
                    except Exception:
                        continue
        await self.send_text(message, fallback, reply_markup=reply_markup)

    async def bot_username(self, bot) -> str | None:
        cached = getattr(bot, "_airbot_username", None)
        if cached:
            return str(cached)
        try:
            me = await bot.get_me()
        except Exception:
            return None
        username = getattr(me, "username", None)
        if username:
            setattr(bot, "_airbot_username", username)
        return str(username) if username else None

    def _filename(self, result: dict) -> str:
        channel = str(result.get("channel") or "channel")
        cleaned = self._clean_channel(channel)
        if not cleaned:
            cleaned = "channel"
        return f"airbot-{cleaned[:64]}.png"

    def _clean_channel(self, channel: str) -> str:
        return "".join(char for char in channel.lower() if char.isascii() and (char.isalnum() or char in {"_", "-"}))

    def _rich_payload(
        self,
        chat_id: int,
        html: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        decorate_buttons: bool = True,
    ) -> dict:
        payload = {
            "chat_id": chat_id,
            "rich_message": {
                "html": html,
                "skip_entity_detection": True,
            },
        }
        if reply_markup is not None:
            payload["reply_markup"] = self._reply_markup_payload(reply_markup, decorate_buttons=decorate_buttons)
        return payload

    def _html_values(self, html: str | list[str]) -> list[str]:
        if isinstance(html, str):
            return [html]
        values = []
        for value in html:
            if value and value not in values:
                values.append(value)
        return values

    def _reply_markup_payload(self, reply_markup: InlineKeyboardMarkup, decorate_buttons: bool = True) -> dict[str, Any]:
        payload = reply_markup.model_dump(exclude_none=True)
        if not decorate_buttons:
            return payload
        rows = payload.get("inline_keyboard")
        if not isinstance(rows, list):
            return payload
        for row in rows:
            if not isinstance(row, list):
                continue
            for button in row:
                if not isinstance(button, dict):
                    continue
                style = self._button_style(button)
                value = style.get("style")
                if value:
                    button["style"] = value
                emoji_id = self._button_emoji_id(str(style.get("emoji") or ""))
                if emoji_id:
                    button["icon_custom_emoji_id"] = emoji_id
        return payload

    def _button_style(self, button: dict[str, Any]) -> dict[str, Any]:
        styles = bot_texts().get("button_styles")
        config = styles if isinstance(styles, dict) else {}
        by_text = config.get("text") if isinstance(config.get("text"), dict) else {}
        text_value = str(button.get("text") or "")
        item = by_text.get(text_value)
        if isinstance(item, dict):
            return item
        callback_data = str(button.get("callback_data") or "")
        exact = config.get("exact") if isinstance(config.get("exact"), dict) else {}
        item = exact.get(callback_data)
        if isinstance(item, dict):
            return item
        prefixes = config.get("prefix") if isinstance(config.get("prefix"), dict) else {}
        for prefix, value in prefixes.items():
            if callback_data.startswith(str(prefix)) and isinstance(value, dict):
                return value
        return {}

    def _button_emoji_id(self, name: str) -> str:
        registry = bot_texts().get("rich_emoji")
        item = registry.get(name) if isinstance(registry, dict) else None
        data = item if isinstance(item, dict) else {}
        return str(data.get("id") or "")

    async def _tg_call(self, token: str, method: str, payload: dict) -> dict | bool:
        url = f"https://api.telegram.org/bot{token}/{method}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json(content_type=None)
        if not data.get("ok"):
            raise RuntimeError(data)
        return data["result"]

    async def _avatar_url(self, result: dict) -> str | None:
        if not self.avatar_loader:
            return None
        channel = str(result.get("channel") or "")
        if not channel:
            return None
        if channel in self._avatar_cache:
            return self._avatar_cache[channel]
        try:
            avatar = await self.avatar_loader.fetch_avatar_url(channel)
            value = avatar.url
        except Exception:
            value = None
        self._avatar_cache[channel] = value
        return value
