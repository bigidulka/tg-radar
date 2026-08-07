import hashlib
from urllib.parse import urlparse

from tg_radar.matching import is_handle


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def content_hash(channel_username: str, tg_msg_id: int, text: str) -> str:
    payload = f"{channel_username.lower()}:{tg_msg_id}:{normalize_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_username(value: str) -> str | None:
    value = value.strip().strip("@")
    value = value.replace("https://t.me/s/", "").replace("http://t.me/s/", "")
    value = value.replace("https://t.me/", "").replace("http://t.me/", "")
    value = value.split("/")[0].split("?")[0]
    if is_handle(value):
        return value
    return None


def extract_mentions(text: str) -> list[str]:
    values = set(_scan_mentions(text or ""))
    values |= set(_scan_tme_usernames(text or ""))
    return sorted(x for x in values if x.lower() not in {"share", "joinchat", "addlist"})


def extract_tme_usernames(text: str) -> list[str]:
    values = set(_scan_tme_usernames(text or ""))
    return sorted(x for x in values if x.lower() not in {"share", "joinchat", "addlist"})


def normalize_link(href: str) -> str | None:
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/"):
        href = "https://t.me" + href
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return href


def _scan_mentions(text: str) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "@":
            index += 1
            continue
        previous = text[index - 1] if index > 0 else ""
        if previous == "/" or previous.isalnum() or previous == "_":
            index += 1
            continue
        end = index + 1
        while end < len(text) and (text[end].isascii() and (text[end].isalnum() or text[end] == "_")):
            end += 1
        candidate = text[index + 1 : end]
        if is_handle(candidate):
            out.append(candidate)
        index = max(end, index + 1)
    return out


def _scan_tme_usernames(text: str) -> list[str]:
    out: list[str] = []
    lowered = text.lower()
    marker = "t.me/"
    index = 0
    while True:
        found = lowered.find(marker, index)
        if found < 0:
            break
        start = found + len(marker)
        if lowered[start : start + 2] == "s/":
            start += 2
        end = start
        while end < len(text) and (text[end].isascii() and (text[end].isalnum() or text[end] == "_")):
            end += 1
        candidate = text[start:end]
        if is_handle(candidate):
            out.append(candidate)
        index = max(end, found + len(marker))
    return out
