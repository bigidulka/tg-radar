from datetime import datetime

from selectolax.parser import HTMLParser

from tg_radar.rule_loader import load_rules
from tg_radar.schemas import ParsedMessage, ParsedPage
from tg_radar.text import content_hash, extract_mentions, normalize_link, normalize_text


PARSER_RULES = load_rules("parser_rules.json")


def parse_views(value: str | None) -> int | None:
    if not value:
        return None
    raw = value.replace(",", ".").strip()
    number_chars: list[str] = []
    suffix = ""
    for char in raw:
        if char.isdigit() or char == ".":
            number_chars.append(char)
            continue
        if number_chars and char.strip():
            suffix = char.lower()
            break
    if not number_chars:
        return None
    num = float("".join(number_chars))
    num *= PARSER_RULES["view_multipliers"].get(suffix, 1)
    return int(num)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_tme_page(html: str, channel_username: str) -> ParsedPage:
    tree = HTMLParser(html)
    title_node = tree.css_first(".tgme_channel_info_header_title span, .tgme_channel_info_header_title")
    channel_title = normalize_text(title_node.text()) if title_node else None
    messages: list[ParsedMessage] = []

    for node in tree.css(".tgme_widget_message"):
        post = node.attributes.get("data-post") or ""
        parsed_post = _parse_post_ref(post)
        if not parsed_post:
            continue
        username, tg_msg_id = parsed_post

        text_node = node.css_first(".tgme_widget_message_text")
        text = normalize_text(text_node.text(separator=" ")) if text_node else ""

        # Video/voice attachments add a <time> without datetime (duration) before the meta block.
        time_node = node.css_first("time[datetime]")
        posted_at = parse_datetime(time_node.attributes.get("datetime") if time_node else None)

        view_node = node.css_first(".tgme_widget_message_views")
        views = parse_views(view_node.text() if view_node else None)

        links: list[str] = []
        for a in node.css("a"):
            href = normalize_link(a.attributes.get("href", ""))
            if href:
                links.append(href)

        forward_node = node.css_first(".tgme_widget_message_forwarded_from_name")
        forward_from = normalize_text(forward_node.text()) if forward_node else None

        url = f"https://t.me/{username}/{tg_msg_id}"
        mentions = extract_mentions(text + " " + " ".join(links))
        messages.append(
            ParsedMessage(
                channel_username=username,
                channel_title=channel_title,
                tg_msg_id=tg_msg_id,
                url=url,
                posted_at=posted_at,
                text=text,
                views=views,
                links=sorted(set(links)),
                mentions=mentions,
                forward_from=forward_from,
                content_hash=content_hash(username, tg_msg_id, text),
            )
        )

    next_before = min((m.tg_msg_id for m in messages), default=None)
    return ParsedPage(
        channel_username=channel_username,
        channel_title=channel_title,
        messages=messages,
        next_before=next_before,
    )


def _parse_post_ref(value: str) -> tuple[str, int] | None:
    parts = value.strip("/").split("/")
    if len(parts) < 2:
        return None
    username = parts[-2]
    msg_id_raw = parts[-1]
    if not username or not msg_id_raw.isdigit():
        return None
    return username, int(msg_id_raw)
