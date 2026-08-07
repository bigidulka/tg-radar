from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from tg_radar.bot.store import BotRunStore


class ReferralMiddleware(BaseMiddleware):
    def __init__(self, store: BotRunStore) -> None:
        self.store = store

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message) and event.from_user:
            payload = _start_payload(event.text or "")
            self.store.record_referral(event.from_user.id, _referrer_user_id(payload), payload)
        return await handler(event, data)


def _start_payload(value: str) -> str | None:
    parts = value.strip().split(maxsplit=1)
    if not parts or parts[0] != "/start":
        return None
    if len(parts) < 2:
        return None
    return parts[1][:128]


def _referrer_user_id(payload: str | None) -> int | None:
    if not payload or not payload.startswith("ref_"):
        return None
    try:
        return int(payload.removeprefix("ref_"))
    except ValueError:
        return None
