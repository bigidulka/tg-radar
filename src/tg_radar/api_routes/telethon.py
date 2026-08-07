from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.api_routes.deps import get_session, get_state
from tg_radar.app_state import AppState
from tg_radar.config import get_settings
from tg_radar.db import (
    add_edges_for_message,
    get_telethon_entity_cache,
    maybe_update_operator_quality,
    update_channel_quality,
    upsert_channel,
    upsert_message,
    upsert_telethon_entity_cache,
)
from tg_radar.schemas import (
    ParsedMessage,
    TelethonCandidateRequest,
    TelethonCommonChatsRequest,
    TelethonEntityRequest,
    TelethonSearchRequest,
    TelethonSimilarRequest,
    TelethonThreadRequest,
)
from tg_radar.text import content_hash


router = APIRouter()


async def telethon_call(coro):
    try:
        return await coro
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


def collect_topic_ids(value, limit: int) -> list[int]:
    out: list[int] = []

    def walk(item):
        if len(out) >= limit:
            return
        if isinstance(item, dict):
            topic_id = item.get("id") or item.get("top_message") or item.get("top_msg_id")
            title = item.get("title")
            if isinstance(topic_id, int) and (title or item.get("_") == "ForumTopic"):
                out.append(topic_id)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return sorted(set(out))[:limit]


def telethon_message_to_parsed(username: str, item: dict) -> ParsedMessage | None:
    msg_id = item.get("id")
    if not isinstance(msg_id, int):
        return None
    text = item.get("text") or ""
    date = item.get("date")
    posted_at = None
    if isinstance(date, str):
        try:
            posted_at = datetime.fromisoformat(date.replace("Z", "+00:00"))
        except ValueError:
            posted_at = None
    links = [str(x) for x in (item.get("links") or []) if x]
    mentions = [str(x) for x in (item.get("mentions") or []) if x]
    chat = item.get("chat") if isinstance(item.get("chat"), dict) else {}
    sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
    thread = item.get("thread") if isinstance(item.get("thread"), dict) else {}
    title = chat.get("title") if isinstance(chat, dict) else None
    return ParsedMessage(
        channel_username=username,
        channel_title=title,
        tg_msg_id=msg_id,
        url=f"https://t.me/{username}/{msg_id}",
        posted_at=posted_at,
        text=text,
        links=links,
        mentions=mentions,
        forward_from=str(item.get("forward") or "") or None,
        sender_id=item.get("sender_id") if isinstance(item.get("sender_id"), int) else None,
        sender_name=sender.get("username") or sender.get("first_name") or sender.get("title"),
        thread_id=thread.get("reply_to_top_id") or thread.get("reply_to_msg_id"),
        reply_to_msg_id=item.get("reply_to_msg_id") if isinstance(item.get("reply_to_msg_id"), int) else None,
        content_hash=content_hash(username, msg_id, text),
    )


@router.get("/telethon/status")
async def telethon_status(state: AppState = Depends(get_state)):
    return await telethon_call(state.telethon.status())


@router.post("/telethon/search/posts")
async def telethon_search_posts(payload: TelethonSearchRequest, state: AppState = Depends(get_state)):
    return {"messages": await telethon_call(state.telethon.search_posts(payload.query, payload.limit))}


@router.post("/telethon/search/global")
async def telethon_search_global(payload: TelethonSearchRequest, state: AppState = Depends(get_state)):
    return {
        "messages": await telethon_call(state.telethon.search_global(
            payload.query,
            payload.limit,
            broadcasts_only=payload.broadcasts_only,
            groups_only=payload.groups_only,
            users_only=payload.users_only,
        ))
    }


@router.post("/telethon/search/contacts")
async def telethon_search_contacts(payload: TelethonSearchRequest, state: AppState = Depends(get_state)):
    return await telethon_call(state.telethon.search_contacts(payload.query, payload.limit))


@router.post("/telethon/dialogs")
async def telethon_dialogs(payload: TelethonEntityRequest, state: AppState = Depends(get_state)):
    return {"dialogs": await telethon_call(state.telethon.dialogs(payload.query or payload.entity, payload.limit))}


@router.post("/telethon/entity")
async def telethon_entity(
    payload: TelethonEntityRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    clean = payload.entity.strip("@")
    cached = await get_telethon_entity_cache(session, clean)
    if cached and cached.error is None:
        return {
            "cached": True,
            "entity": {
                "username": cached.username,
                "title": cached.title,
                "is_chat": cached.is_chat,
                "type": cached.entity_type,
                "checked_at": cached.checked_at,
            },
        }
    entity = await telethon_call(state.telethon.entity(clean))
    is_chat = bool(
        isinstance(entity, dict)
        and entity.get("username")
        and (entity.get("broadcast") or entity.get("megagroup") or entity.get("gigagroup") or entity.get("title"))
    )
    await upsert_telethon_entity_cache(
        session,
        clean,
        is_chat=is_chat,
        entity_type="chat" if is_chat else "user_or_bot",
        title=entity.get("title") if isinstance(entity, dict) else None,
        error=None,
    )
    return {"cached": False, "entity": entity}


@router.post("/telethon/full-info")
async def telethon_full_info(payload: TelethonEntityRequest, state: AppState = Depends(get_state)):
    return {"full_info": await telethon_call(state.telethon.full_info(payload.entity))}


@router.post("/telethon/messages")
async def telethon_messages(payload: TelethonEntityRequest, state: AppState = Depends(get_state)):
    return {"messages": await telethon_call(state.telethon.messages(payload.entity, payload.query, payload.limit, with_sender=True))}


@router.post("/telethon/forum-topics")
async def telethon_forum_topics(payload: TelethonEntityRequest, state: AppState = Depends(get_state)):
    return {"topics": await telethon_call(state.telethon.forum_topics(payload.entity, payload.query, payload.limit))}


@router.post("/telethon/thread-messages")
async def telethon_thread_messages(payload: TelethonThreadRequest, state: AppState = Depends(get_state)):
    return {"messages": await telethon_call(state.telethon.thread_messages(payload.entity, payload.thread_id, payload.limit))}


@router.post("/telethon/forum-ingest")
async def telethon_forum_ingest(
    payload: TelethonEntityRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    username = payload.entity.strip("@")
    base_messages = await telethon_call(state.telethon.messages(username, payload.query, payload.limit, with_sender=True))
    topics = await telethon_call(state.telethon.forum_topics(username, payload.query, min(payload.limit, 50)))
    topic_ids = collect_topic_ids(topics, min(payload.limit, 50))
    thread_messages = []
    for topic_id in topic_ids[:10]:
        thread_messages.extend(await telethon_call(state.telethon.thread_messages(username, topic_id, min(payload.limit, 50))))
    parsed = [
        item
        for item in (telethon_message_to_parsed(username, row) for row in [*base_messages, *thread_messages])
        if item and item.text
    ]
    channel = await upsert_channel(session, username, None, "telethon_forum")
    saved = 0
    for item in parsed:
        message, created, changed = await upsert_message(session, channel, item)
        if not (created or changed):
            continue
        await add_edges_for_message(session, channel, message, item.mentions)
        try:
            await state.vespa.feed_message(item, channel.quality_score or 0.0)
            message.indexed_at = datetime.utcnow()
        except Exception:
            message.indexed_at = None
        saved += 1
    if parsed:
        await update_channel_quality(session, channel, parsed, "telethon_forum")
        await maybe_update_operator_quality(session, channel, parsed, get_settings())
    return {"channel": username, "topics": len(topic_ids), "messages_seen": len(parsed), "messages_saved": saved}


@router.post("/telethon/participants")
async def telethon_participants(payload: TelethonEntityRequest, state: AppState = Depends(get_state)):
    return {"participants": await telethon_call(state.telethon.participants(payload.entity, payload.query, payload.limit))}


@router.post("/telethon/participant-profiles")
async def telethon_participant_profiles(payload: TelethonEntityRequest, state: AppState = Depends(get_state)):
    return {"profiles": await telethon_call(state.telethon.participant_profiles(payload.entity, payload.query, payload.limit))}


@router.post("/telethon/common-chats")
async def telethon_common_chats(payload: TelethonCommonChatsRequest, state: AppState = Depends(get_state)):
    return {"chats": await telethon_call(state.telethon.common_chats(payload.user, payload.limit))}


@router.post("/telethon/similar")
async def telethon_similar(payload: TelethonSimilarRequest, state: AppState = Depends(get_state)):
    return {"channels": await telethon_call(state.telethon.similar_channels(payload.channel, payload.limit))}


@router.post("/telethon/inspect")
async def telethon_inspect(payload: TelethonEntityRequest, state: AppState = Depends(get_state)):
    return await telethon_call(state.telethon.inspect(payload.entity, payload.query, payload.limit))


@router.post("/telethon/candidates")
async def telethon_candidates(payload: TelethonCandidateRequest, state: AppState = Depends(get_state)):
    return {"candidates": await telethon_call(state.telethon.candidate_scan(payload.entity, payload.query, payload.limit))}
