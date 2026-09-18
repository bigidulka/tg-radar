from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.api_routes.deps import get_session, get_state
from tg_radar.app_state import AppState
from tg_radar.db import Channel, Message
from tg_radar.schemas import ChannelFetchedMessage, ChannelFetchRequest, ChannelFetchResponse
from tg_radar.service import channel_profile
from tg_radar.text import clean_username


router = APIRouter()


@router.get("/channels/{username}")
async def get_channel(username: str, session: AsyncSession = Depends(get_session)):
    profile = await channel_profile(session, username)
    if not profile:
        raise HTTPException(status_code=404, detail="channel not found")
    return profile


@router.get("/channels/{username}/messages")
async def get_channel_messages(
    username: str,
    limit: int = 50,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    clean = username.strip("@")
    rows = (
        await session.execute(
            select(Message, Channel)
            .join(Channel, Channel.id == Message.channel_id)
            .where(Channel.username == clean)
            .order_by(Message.posted_at.desc().nulls_last(), Message.id.desc())
            .limit(min(limit, 200))
        )
    ).all()
    if not rows:
        try:
            messages = await state.telethon.messages(clean, None, min(limit, 100), with_sender=True)
            return {
                "messages": [
                    {
                        "message_id": f"{clean}:{message.get('id')}",
                        "channel_username": clean,
                        "channel_title": (message.get("chat") or {}).get("title"),
                        "url": f"https://t.me/{clean}/{message.get('id')}",
                        "posted_at": message.get("date"),
                        "text": message.get("text") or "",
                        "score": 1.0,
                    }
                    for message in messages
                ]
            }
        except Exception:
            pass
    return {
        "messages": [
            {
                "message_id": f"{channel.username}:{message.tg_msg_id}",
                "channel_username": channel.username,
                "channel_title": channel.title,
                "url": message.url,
                "posted_at": message.posted_at,
                "text": message.text,
                "score": 1.0,
            }
            for message, channel in rows
        ]
    }


@router.post("/channels/{username}/fetch-messages", response_model=ChannelFetchResponse)
@router.post("/core/channels/{username}/fetch-messages", response_model=ChannelFetchResponse)
async def fetch_channel_messages(
    username: str,
    payload: ChannelFetchRequest,
    state: AppState = Depends(get_state),
):
    clean = clean_username(username)
    if not clean:
        raise HTTPException(status_code=400, detail="invalid channel username")
    pages = max(1, (payload.limit + 19) // 20)
    try:
        parsed_pages = await state.crawler.crawl_channel(clean, pages=pages)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="channel not found or not public") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    messages = [
        ChannelFetchedMessage(
            message_id=f"{item.channel_username}:{item.tg_msg_id}",
            channel_username=item.channel_username,
            channel_title=item.channel_title,
            url=item.url,
            posted_at=item.posted_at,
            text=item.text,
            views=item.views,
            links=item.links,
            mentions=item.mentions,
            forward_from=item.forward_from,
        )
        for page in parsed_pages
        for item in page.messages
        if item.text
    ][: payload.limit]
    return ChannelFetchResponse(channel=clean, messages=messages)


@router.post("/crawl/{username}")
async def crawl_channel(
    username: str,
    pages: int = 1,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    if pages < 1 or pages > 20:
        raise HTTPException(status_code=400, detail="pages must be between 1 and 20")
    result = await state.indexing.crawl_and_index(session, username, pages=pages, crawl_mode="refresh")
    return {
        "channel": username.strip("@"),
        "messages": len(result.saved),
        "seen": len(result.seen),
        "deleted_or_missing_marked": result.deleted_or_missing_marked,
    }


@router.get("/core/channels/{username}")
async def core_channel(username: str, session: AsyncSession = Depends(get_session)):
    return await get_channel(username, session)


@router.get("/core/channels/{username}/messages")
async def core_channel_messages(
    username: str,
    limit: int = 50,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    return await get_channel_messages(username, limit, state, session)


@router.post("/core/crawl/{username}")
async def core_crawl(
    username: str,
    pages: int = 1,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    return await crawl_channel(username, pages, state, session)
