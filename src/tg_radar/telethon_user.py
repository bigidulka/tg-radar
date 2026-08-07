from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar

from tg_radar.config import Settings
from tg_radar.schemas import ChannelCandidate
from tg_radar.text import clean_username, extract_mentions


def walk_strings(value: Any, depth: int = 4) -> list[str]:
    if depth <= 0 or value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in list(value)[:100]:
            out.extend(walk_strings(item, depth - 1))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for item in list(value.values())[:100]:
            out.extend(walk_strings(item, depth - 1))
        return out
    if hasattr(value, "to_dict"):
        try:
            return walk_strings(value.to_dict(), depth - 1)
        except Exception:
            return []
    return []


def obj_to_dict(obj: Any, depth: int = 2) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [obj_to_dict(x, depth - 1) for x in obj[:100]]
    if isinstance(obj, dict):
        return {str(k): obj_to_dict(v, depth - 1) for k, v in list(obj.items())[:100]}
    if depth <= 0:
        return str(obj)
    if hasattr(obj, "to_dict"):
        try:
            data = obj.to_dict()
            for key in ("phone", "access_hash"):
                data.pop(key, None)
            return obj_to_dict(data, depth - 1)
        except Exception:
            pass
    out: dict[str, Any] = {}
    for key in (
        "id",
        "username",
        "first_name",
        "last_name",
        "title",
        "bot",
        "verified",
        "scam",
        "fake",
        "restricted",
        "broadcast",
        "megagroup",
        "gigagroup",
        "participants_count",
        "date",
    ):
        if hasattr(obj, key):
            try:
                out[key] = obj_to_dict(getattr(obj, key), depth - 1)
            except Exception:
                pass
    return out or str(obj)


def hidden_links_from_message(message: Any) -> list[str]:
    links: set[str] = set()
    for entity in getattr(message, "entities", []) or []:
        url = getattr(entity, "url", None)
        if url:
            links.add(str(url))
    webpage = getattr(getattr(message, "media", None), "webpage", None)
    for key in ("url", "display_url"):
        value = getattr(webpage, key, None)
        if value:
            links.add(str(value))
    for row in getattr(getattr(message, "reply_markup", None), "rows", []) or []:
        for button in getattr(row, "buttons", []) or []:
            url = getattr(button, "url", None)
            if url:
                links.add(str(url))
    return sorted(links)


def message_to_dict(message: Any) -> dict[str, Any]:
    chat = getattr(message, "chat", None)
    sender = getattr(message, "sender", None)
    reply_to = getattr(message, "reply_to", None)
    links = hidden_links_from_message(message)
    text = getattr(message, "raw_text", None) or ""
    thread = {
        "reply_to_msg_id": getattr(message, "reply_to_msg_id", None),
        "reply_to_top_id": getattr(reply_to, "reply_to_top_id", None) if reply_to else None,
        "forum_topic": bool(getattr(reply_to, "forum_topic", False)) if reply_to else False,
        "reply_to_peer_id": obj_to_dict(getattr(reply_to, "reply_to_peer_id", None), 1) if reply_to else None,
    }
    return {
        "id": getattr(message, "id", None),
        "date": message.date.isoformat() if getattr(message, "date", None) else None,
        "text": text,
        "links": links,
        "mentions": extract_mentions(" ".join([text, *links])),
        "sender_id": getattr(message, "sender_id", None),
        "chat_id": getattr(message, "chat_id", None),
        "reply_to_msg_id": getattr(message, "reply_to_msg_id", None),
        "thread": thread,
        "reply_to": obj_to_dict(reply_to, 2) if reply_to else None,
        "action": type(getattr(message, "action", None)).__name__ if getattr(message, "action", None) else None,
        "media": type(getattr(message, "media", None)).__name__ if getattr(message, "media", None) else None,
        "chat": obj_to_dict(chat, 1) if chat else None,
        "sender": obj_to_dict(sender, 1) if sender else None,
        "forward": obj_to_dict(getattr(message, "fwd_from", None), 2) if getattr(message, "fwd_from", None) else None,
    }


def dialog_to_dict(dialog: Any) -> dict[str, Any]:
    return {
        "id": dialog.id,
        "title": dialog.title,
        "name": dialog.name,
        "is_user": dialog.is_user,
        "is_group": dialog.is_group,
        "is_channel": dialog.is_channel,
        "unread_count": dialog.unread_count,
    }


def is_chat_entity(data: dict[str, Any]) -> bool:
    return bool(
        data.get("username")
        and (
            data.get("broadcast")
            or data.get("megagroup")
            or data.get("gigagroup")
            or (data.get("title") and not data.get("first_name") and not data.get("last_name"))
        )
    )


def _creds_path(session_path: Path) -> Path:
    return session_path.with_suffix(session_path.suffix + ".creds.json")


class TelethonUserClient:
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _entity_cache: ClassVar[dict[str, dict[str, Any]]] = {}
    _cache_loaded: ClassVar[bool] = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.home = Path(settings.telethon_home).expanduser().resolve()
        self.profile = settings.telethon_profile
        self.session_path = self.home / "sessions" / self.profile
        self._lock = self._locks.setdefault(str(self.session_path), asyncio.Lock())
        self.cache_path = Path.cwd() / "tmp" / "telethon_entity_cache.json"

    @property
    def enabled(self) -> bool:
        return self.settings.telethon_enabled

    def _load_creds(self) -> tuple[int, str]:
        path = _creds_path(self.session_path)
        if not path.exists():
            raise RuntimeError(f"Telethon creds not found for profile {self.profile}")
        data = json.loads(path.read_text())
        return int(data["api_id"]), str(data["api_hash"])

    def _load_entity_cache(self) -> None:
        if self.__class__._cache_loaded:
            return
        self.__class__._cache_loaded = True
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text())
        except Exception:
            return
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and isinstance(value, dict):
                    self.__class__._entity_cache[key.lower()] = value

    def _save_entity_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.__class__._entity_cache, ensure_ascii=False, sort_keys=True))
        except Exception:
            pass

    @asynccontextmanager
    async def client(self) -> AsyncIterator[Any]:
        if not self.enabled:
            raise RuntimeError("Telethon is disabled")
        from telethon import TelegramClient

        api_id, api_hash = self._load_creds()
        async with self._lock:
            client = TelegramClient(str(self.session_path), api_id, api_hash)
            await client.connect()
            try:
                if not await client.is_user_authorized():
                    raise RuntimeError(f"Telethon profile {self.profile} is not authorized")
                yield client
            finally:
                await client.disconnect()

    async def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "enabled": self.enabled,
            "profile": self.profile,
            "home": str(self.home),
            "session": str(self.session_path) + ".session",
            "creds_present": _creds_path(self.session_path).exists(),
            "authorized": False,
        }
        if not self.enabled or not out["creds_present"]:
            return out
        async with self.client() as client:
            me = await client.get_me()
            out["authorized"] = True
            out["me"] = obj_to_dict(me, 2)
        return out

    async def search_posts(self, query: str, limit: int) -> list[dict[str, Any]]:
        from telethon import functions, types

        async with self.client() as client:
            result = await client(
                functions.channels.SearchPostsRequest(
                    query=query,
                    offset_rate=0,
                    offset_peer=types.InputPeerEmpty(),
                    offset_id=0,
                    limit=min(limit, self.settings.telethon_search_limit),
                )
            )
            return self._messages_result(result)

    async def search_global(self, query: str, limit: int, broadcasts_only: bool = False, groups_only: bool = False, users_only: bool = False) -> list[dict[str, Any]]:
        from telethon import functions, types

        async with self.client() as client:
            result = await client(
                functions.messages.SearchGlobalRequest(
                    q=query,
                    filter=types.InputMessagesFilterEmpty(),
                    min_date=None,
                    max_date=None,
                    offset_rate=0,
                    offset_peer=types.InputPeerEmpty(),
                    offset_id=0,
                    limit=min(limit, self.settings.telethon_search_limit),
                    broadcasts_only=broadcasts_only or None,
                    groups_only=groups_only or None,
                    users_only=users_only or None,
                )
            )
            return self._messages_result(result)

    async def search_contacts(self, query: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        from telethon import functions

        async with self.client() as client:
            result = await client(functions.contacts.SearchRequest(q=query, limit=min(limit, self.settings.telethon_search_limit)))
            return {
                "chats": [obj_to_dict(chat, 2) for chat in getattr(result, "chats", [])],
                "users": [obj_to_dict(user, 2) for user in getattr(result, "users", [])],
            }

    async def dialogs(self, search: str | None, limit: int) -> list[dict[str, Any]]:
        async with self.client() as client:
            rows = []
            async for dialog in client.iter_dialogs(limit=limit):
                item = dialog_to_dict(dialog)
                if search and search.lower() not in (item.get("title") or "").lower():
                    continue
                rows.append(item)
            return rows

    async def entity(self, entity: str) -> dict[str, Any]:
        async with self.client() as client:
            resolved = await client.get_entity(entity)
            return obj_to_dict(resolved, 3)

    async def full_info(self, entity: str) -> dict[str, Any]:
        from telethon import functions, types

        async with self.client() as client:
            resolved = await client.get_entity(entity)
            if isinstance(resolved, types.User):
                full = await client(functions.users.GetFullUserRequest(resolved))
            elif isinstance(resolved, types.Chat):
                full = await client(functions.messages.GetFullChatRequest(resolved.id))
            else:
                full = await client(functions.channels.GetFullChannelRequest(resolved))
            data = obj_to_dict(full, 4)
            data["entity"] = obj_to_dict(resolved, 3)
            data["pinned_message"] = await self._pinned_message(client, resolved, full)
            return data

    async def _pinned_message(self, client: Any, entity: Any, full: Any) -> dict[str, Any] | None:
        full_chat = getattr(full, "full_chat", None) or getattr(full, "full_user", None)
        pinned_id = getattr(full_chat, "pinned_msg_id", None)
        if not pinned_id:
            return None
        try:
            message = await client.get_messages(entity, ids=pinned_id)
        except Exception:
            return {"id": pinned_id, "error": "pinned message is not accessible"}
        return message_to_dict(message) if message else {"id": pinned_id, "error": "pinned message not found"}

    async def messages(self, entity: str, query: str | None, limit: int, with_sender: bool = True) -> list[dict[str, Any]]:
        async with self.client() as client:
            rows = []
            async for message in client.iter_messages(entity, search=query, limit=min(limit, self.settings.telethon_search_limit)):
                if with_sender:
                    try:
                        await message.get_sender()
                    except Exception:
                        pass
                rows.append(message_to_dict(message))
            return rows

    async def forum_topics(self, entity: str, query: str | None, limit: int) -> dict[str, Any]:
        from telethon import functions

        async with self.client() as client:
            peer = await client.get_input_entity(entity)
            result = await client(
                functions.messages.GetForumTopicsRequest(
                    peer=peer,
                    offset_date=None,
                    offset_id=0,
                    offset_topic=0,
                    limit=min(limit, self.settings.telethon_search_limit),
                    q=query,
                )
            )
            return obj_to_dict(result, 4)

    async def thread_messages(self, entity: str, thread_id: int, limit: int) -> list[dict[str, Any]]:
        async with self.client() as client:
            rows = []
            try:
                async for message in client.iter_messages(entity, reply_to=thread_id, limit=min(limit, self.settings.telethon_search_limit)):
                    rows.append(message_to_dict(message))
            except Exception:
                rows = []
                async for message in client.iter_messages(entity, limit=min(max(limit * 5, 20), 100)):
                    reply_to = getattr(message, "reply_to", None)
                    top_id = getattr(reply_to, "reply_to_top_id", None) if reply_to else None
                    reply_id = getattr(message, "reply_to_msg_id", None)
                    if thread_id in {top_id, reply_id, getattr(message, "id", None)}:
                        rows.append(message_to_dict(message))
                    if len(rows) >= limit:
                        break
            return rows

    async def participants(self, entity: str, search: str | None, limit: int) -> list[dict[str, Any]]:
        async with self.client() as client:
            rows = []
            async for user in client.iter_participants(entity, search=search or "", limit=min(limit, self.settings.telethon_search_limit)):
                rows.append(obj_to_dict(user, 2))
            return rows

    async def participant_profiles(self, entity: str, search: str | None, limit: int) -> list[dict[str, Any]]:
        from telethon import functions

        async with self.client() as client:
            rows = []
            async for user in client.iter_participants(entity, search=search or "", limit=min(limit, self.settings.telethon_search_limit)):
                item = {"user": obj_to_dict(user, 2)}
                try:
                    item["full_user"] = obj_to_dict(await client(functions.users.GetFullUserRequest(user)), 3)
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(item)
            return rows

    async def common_chats(self, user: str, limit: int) -> list[dict[str, Any]]:
        from telethon import functions

        async with self.client() as client:
            input_user = await client.get_input_entity(user)
            result = await client(functions.messages.GetCommonChatsRequest(user_id=input_user, max_id=0, limit=min(limit, self.settings.telethon_search_limit)))
            return [obj_to_dict(chat, 2) for chat in result.chats]

    async def similar_channels(self, channel: str, limit: int) -> list[dict[str, Any]]:
        from telethon import functions

        async with self.client() as client:
            input_channel = await client.get_input_entity(channel)
            result = await client(functions.channels.GetChannelRecommendationsRequest(channel=input_channel))
            return [obj_to_dict(chat, 2) for chat in result.chats[:limit]]

    async def candidate_scan(self, target: str, query: str | None, limit: int) -> list[ChannelCandidate]:
        inspection = await self.inspect(target, query, min(limit, self.settings.telethon_search_limit))
        candidates = self.candidates_from_inspection(inspection, f"telethon_recursive:{target}", query or target, limit)
        return await self.filter_chat_candidates(candidates, limit)

    async def filter_chat_candidates(self, candidates: list[ChannelCandidate], limit: int) -> list[ChannelCandidate]:
        out: list[ChannelCandidate] = []
        seen: set[str] = set()
        self._load_entity_cache()
        changed = False
        async with self.client() as client:
            for candidate in candidates:
                key = candidate.username.lower()
                if key in seen:
                    continue
                cached = self._entity_cache.get(key)
                if cached is None:
                    try:
                        entity = await client.get_entity(candidate.username)
                        data = obj_to_dict(entity, 3)
                    except Exception as exc:
                        self._entity_cache[key] = {"is_chat": False, "error": f"{type(exc).__name__}: {exc}"}
                        changed = True
                        continue
                    cached = {
                        "is_chat": isinstance(data, dict) and is_chat_entity(data),
                        "entity": data,
                    }
                    self._entity_cache[key] = cached
                    changed = True
                if not cached.get("is_chat"):
                    continue
                candidate.state = "validated_channel"
                candidate.quality_score = max(candidate.quality_score, 0.35)
                candidate.quality_reasons = sorted(set(candidate.quality_reasons + ["telethon_validated", "public_chat"]))
                seen.add(key)
                out.append(candidate)
                if len(out) >= limit:
                    break
        if changed:
            self._save_entity_cache()
        return out

    async def inspect(self, target: str, query: str | None, limit: int) -> dict[str, Any]:
        out: dict[str, Any] = {"target": target}
        try:
            out["entity"] = await self.entity(target)
        except Exception as exc:
            out["entity_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["full_info"] = await self.full_info(target)
        except Exception as exc:
            out["full_info_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["messages"] = await self.messages(target, query, limit, with_sender=True)
        except Exception as exc:
            out["messages_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["forum_topics"] = await self.forum_topics(target, query, min(limit, 20))
        except Exception as exc:
            out["forum_topics_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["participants"] = await self.participants(target, query, limit)
        except Exception as exc:
            out["participants_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["participant_profiles"] = await self.participant_profiles(target, query, min(limit, 10))
        except Exception as exc:
            out["participant_profiles_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["similar_channels"] = await self.similar_channels(target, limit)
        except Exception as exc:
            out["similar_channels_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def _messages_result(self, result: Any) -> list[dict[str, Any]]:
        chat_map = {getattr(chat, "id", None): chat for chat in getattr(result, "chats", [])}
        out = []
        for message in getattr(result, "messages", []):
            item = message_to_dict(message)
            peer = getattr(message, "peer_id", None)
            channel_id = getattr(peer, "channel_id", None) or getattr(peer, "chat_id", None)
            if channel_id in chat_map:
                item["chat"] = obj_to_dict(chat_map[channel_id], 1)
            out.append(item)
        return out

    @staticmethod
    def candidates_from_messages(messages: list[dict[str, Any]], source: str, reason: str, limit: int) -> list[ChannelCandidate]:
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        for message in messages:
            chat = message.get("chat") or {}
            username = clean_username(str(chat.get("username") or ""))
            found = {username} if username else set()
            found |= set(extract_mentions(" ".join(str(x) for x in message.get("links") or [])))
            for value in sorted(found):
                if value.lower() in seen:
                    continue
                seen.add(value.lower())
                candidates.append(ChannelCandidate(username=value, source=source, reason=reason, score=0.95))
                if len(candidates) >= limit:
                    return candidates
        return candidates

    @staticmethod
    def candidates_from_inspection(inspection: dict[str, Any], source: str, reason: str, limit: int) -> list[ChannelCandidate]:
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()

        def add(username: str | None, score: float) -> None:
            clean = clean_username(username or "")
            if not clean or clean.lower() in seen or len(candidates) >= limit:
                return
            seen.add(clean.lower())
            candidates.append(ChannelCandidate(username=clean, source=source, reason=reason, score=score))

        entity = inspection.get("entity") or {}
        if entity and not is_chat_entity(entity):
            return []
        add(str(entity.get("username") or ""), 1.0)
        full_info = inspection.get("full_info") or {}
        for value in walk_strings(full_info):
            for username in extract_mentions(value):
                add(username, 0.92)
        pinned = full_info.get("pinned_message") or {}
        for username in pinned.get("mentions") or []:
            add(username, 0.95)
        for value in walk_strings(inspection.get("forum_topics") or {}, 4):
            for username in extract_mentions(value):
                add(username, 0.84)
        for message in inspection.get("messages") or []:
            chat = message.get("chat") or {}
            if is_chat_entity(chat):
                add(str(chat.get("username") or ""), 0.9)
            for username in message.get("mentions") or []:
                add(username, 0.86)
            for username in extract_mentions(" ".join(str(x) for x in message.get("links") or [])):
                add(username, 0.86)
        for channel in inspection.get("similar_channels") or []:
            if is_chat_entity(channel):
                add(str(channel.get("username") or ""), 0.82)
        return candidates[:limit]

    @staticmethod
    def candidates_from_entities(entities: list[dict[str, Any]], source: str, reason: str, limit: int, score: float = 0.9) -> list[ChannelCandidate]:
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        for entity in entities:
            if not is_chat_entity(entity):
                continue
            usernames = set()
            username = clean_username(str(entity.get("username") or ""))
            if username:
                usernames.add(username)
            for value in walk_strings(entity, 3):
                usernames |= set(extract_mentions(value))
            for value in sorted(usernames):
                if value.lower() in seen:
                    continue
                seen.add(value.lower())
                candidates.append(ChannelCandidate(username=value, source=source, reason=reason, score=score))
                if len(candidates) >= limit:
                    return candidates
        return candidates
