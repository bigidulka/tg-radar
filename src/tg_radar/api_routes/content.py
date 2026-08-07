from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.api_routes.deps import get_session, get_state
from tg_radar.app_state import AppState
from tg_radar.content_factory import (
    article_to_out,
    create_article,
    format_telegram,
    generate_cards_from_topic,
    generate_draft,
    generate_outline,
    list_articles,
    list_cards,
    update_article,
    update_block,
    update_card_status,
    update_draft,
)
from tg_radar.db import ArticleSession
from tg_radar.schemas import (
    ArticleDraftOut,
    ArticleSessionOut,
    ContentBoard,
    ContentCardActionRequest,
    ContentCardOut,
    CreateArticleRequest,
    GenerateArticleRequest,
    GenerateContentCardsRequest,
    UpdateArticleBlockRequest,
    UpdateArticleDraftRequest,
    UpdateArticleRequest,
)


router = APIRouter()


def require_content_factory_enabled(state: AppState) -> None:
    if not state.settings.content_factory_enabled:
        raise HTTPException(status_code=404, detail="content factory disabled; use /core API")


def topic_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


@router.get("/content/board", response_model=ContentBoard)
async def content_board(topic: str, state: AppState = Depends(get_state), session: AsyncSession = Depends(get_session)):
    require_content_factory_enabled(state)
    slug = topic_slug(topic)
    return ContentBoard(
        topic_slug=slug,
        cards=await list_cards(session, slug, 160),
        articles=await list_articles(session, slug, 12),
    )


@router.post("/content/cards/generate", response_model=list[ContentCardOut])
async def content_cards_generate(
    payload: GenerateContentCardsRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    return await generate_cards_from_topic(session, payload.topic_slug, payload.limit)


@router.post("/content/cards/{card_id}/action", response_model=ContentCardOut)
async def content_card_action(
    card_id: int,
    payload: ContentCardActionRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    card = await update_card_status(session, card_id, payload.action, payload.note)
    if not card:
        raise HTTPException(status_code=404, detail="card not found")
    return card


@router.get("/content/articles", response_model=list[ArticleSessionOut])
async def content_articles(
    topic: str | None = None,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    return await list_articles(session, topic_slug(topic) if topic else None, 30)


@router.post("/content/articles", response_model=ArticleSessionOut)
async def content_article_create(
    payload: CreateArticleRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    return await create_article(
        session,
        topic_slug(payload.topic_slug),
        payload.title,
        payload.audience,
        payload.angle,
        payload.card_ids,
    )


@router.get("/content/articles/{article_id}", response_model=ArticleSessionOut)
async def content_article_get(
    article_id: int,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    article = await session.get(ArticleSession, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="article not found")
    return await article_to_out(session, article)


@router.patch("/content/articles/{article_id}", response_model=ArticleSessionOut)
async def content_article_update(
    article_id: int,
    payload: UpdateArticleRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    article = await update_article(
        session,
        article_id,
        title=payload.title,
        audience=payload.audience,
        angle=payload.angle,
        status=payload.status,
        source_card_ids=payload.source_card_ids,
    )
    if not article:
        raise HTTPException(status_code=404, detail="article not found")
    return article


@router.post("/content/articles/{article_id}/outline", response_model=ArticleSessionOut)
async def content_article_outline(
    article_id: int,
    payload: GenerateArticleRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    article = await generate_outline(session, state.settings, article_id, payload.style, payload.instruction, payload.use_llm)
    if not article:
        raise HTTPException(status_code=404, detail="article not found")
    return article


@router.post("/content/articles/{article_id}/draft", response_model=ArticleDraftOut)
async def content_article_draft(
    article_id: int,
    payload: GenerateArticleRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    draft = await generate_draft(session, state.settings, article_id, payload.style, payload.instruction, payload.use_llm)
    if not draft:
        raise HTTPException(status_code=404, detail="article not found")
    return draft


@router.post("/content/articles/{article_id}/format-telegram", response_model=ArticleDraftOut)
async def content_article_format(
    article_id: int,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    draft = await format_telegram(session, article_id)
    if not draft:
        raise HTTPException(status_code=404, detail="draft not found")
    return draft


@router.patch("/content/blocks/{block_id}", response_model=ArticleSessionOut)
async def content_block_update(
    block_id: int,
    payload: UpdateArticleBlockRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    article = await update_block(
        session,
        block_id,
        title=payload.title,
        text=payload.text,
        block_type=payload.block_type,
        status=payload.status,
        position=payload.position,
    )
    if not article:
        raise HTTPException(status_code=404, detail="block not found")
    return article


@router.patch("/content/drafts/{draft_id}", response_model=ArticleSessionOut)
async def content_draft_update(
    draft_id: int,
    payload: UpdateArticleDraftRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    require_content_factory_enabled(state)
    article = await update_draft(
        session,
        draft_id,
        text=payload.text,
        status=payload.status,
        format_name=payload.format,
    )
    if not article:
        raise HTTPException(status_code=404, detail="draft not found")
    return article
