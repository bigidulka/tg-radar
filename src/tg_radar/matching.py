from __future__ import annotations


def contains_any(text: str, values: list[str]) -> bool:
    lowered = text.lower()
    return any(value and value.lower() in lowered for value in values)


def matched_groups(text: str, groups: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    return [name for name, values in groups.items() if any(value.lower() in lowered for value in values)]


def word_tokens(text: str, min_length: int = 1) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    extra = {"#", "+", ".", "-"}
    for char in text:
        if char.isalnum() or char in extra:
            current.append(char)
            continue
        if current:
            token = "".join(current)
            if len(token) >= min_length:
                out.append(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) >= min_length:
            out.append(token)
    return out


def is_handle(value: str) -> bool:
    if not 5 <= len(value) <= 32:
        return False
    return all(char.isascii() and (char.isalnum() or char == "_") for char in value)


def strip_urls_and_handles(text: str) -> str:
    parts = []
    for token in text.split():
        lowered = token.lower()
        if lowered.startswith(("http://", "https://", "t.me/")):
            parts.append(" ")
            continue
        if token.startswith("@") and is_handle(token.strip("@.,:;!?()[]{}<>")):
            parts.append(" ")
            continue
        parts.append(token)
    return " ".join(parts)


def has_email(text: str) -> bool:
    for token in text.split():
        value = token.strip(".,:;!?()[]{}<>")
        if "@" not in value or "." not in value:
            continue
        name, _, domain = value.partition("@")
        if name and "." in domain and not domain.startswith(".") and not domain.endswith("."):
            return True
    return False


def has_handle_or_tme(text: str) -> bool:
    for token in text.split():
        value = token.strip(".,:;!?()[]{}<>")
        lowered = value.lower()
        if value.startswith("@") and is_handle(value[1:]):
            return True
        if "t.me/" in lowered:
            candidate = lowered.split("t.me/", 1)[1].split("/", 1)[0].split("?", 1)[0]
            if is_handle(candidate):
                return True
    return False
