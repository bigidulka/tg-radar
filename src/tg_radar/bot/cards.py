from __future__ import annotations

from html import escape
from io import BytesIO
from math import cos, pi, sin
from urllib.parse import urlparse
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from tg_radar.bot.ratings import elo_rank, elo_visual_score
from tg_radar.bot.texts import bot_texts, text


class AirbotCardRenderer:
    def __init__(self, width: int = 1200, height: int = 1600) -> None:
        self.width = width
        self.height = height

    def render(self, result: dict[str, Any]) -> bytes:
        image = Image.new("RGB", (self.width, self.height), (244, 246, 247))
        draw = ImageDraw.Draw(image)
        channel = str(result.get("channel") or "")
        air_elo = self._plain_int(result.get("air_elo"))
        score = elo_visual_score(air_elo)
        elo_label = str(result.get("air_elo_label") or f"{air_elo} ELO")
        profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
        emoji, rank_name = _rank_bits(result, profile)
        air_title = str(profile.get("air_title") or result.get("humorous_label") or "")
        one_liner = str(profile.get("one_liner") or result.get("summary") or "")
        accent = self._accent(score)
        ink = (30, 34, 38)
        muted = (92, 95, 99)

        self._background(draw, score)
        self._header(draw, channel, accent, ink, muted)
        self._summary_card(draw, result, channel, score, elo_label, emoji, rank_name, air_title, one_liner, accent, ink, muted)
        self._tags(draw, result, 52, 430, ink, muted)
        self._fun_metrics(draw, result, 52, 550, accent, ink, muted)
        self._metric_grid(draw, result, 52, 690, accent, ink, muted)
        self._examples(draw, result, 52, 1320, ink, muted)
        self._footer_line(draw, result, muted)

        out = BytesIO()
        image.save(out, format="PNG", optimize=True)
        return out.getvalue()

    def _background(self, draw: ImageDraw.ImageDraw, score: int) -> None:
        draw.rectangle((0, 0, self.width, self.height), fill=(244, 246, 247))
        wind = min(max(score, 0), 200) / 200
        line_count = 16 + int(wind * 26)
        for index in range(line_count):
            y = 80 + index * 48
            offset = int(sin(index * 0.9) * 42 * (0.4 + wind))
            color = self._mix((226, 230, 233), (180, 190, 198), wind)
            width = 1 + int(wind * 3)
            draw.arc((-160 + offset, y - 42, self.width + 160 + offset, y + 78), 190, 350, fill=color, width=width)
        for index in range(3 + int(wind * 5)):
            cx = 980 - index * 95
            cy = 230 + index * 132
            r = 80 + int(wind * 70) + index * 18
            color = self._mix((231, 235, 238), (196, 205, 212), wind)
            draw.arc((cx - r, cy - r, cx + r, cy + r), 20, 340, fill=color, width=2 + int(wind * 3))
        draw.rectangle((0, 0, self.width, 12), fill=self._mix((205, 214, 219), (96, 105, 112), wind))
        draw.rectangle((0, self.height - 12, self.width, self.height), fill=self._mix((205, 214, 219), (96, 105, 112), wind))

    def _header(
        self,
        draw: ImageDraw.ImageDraw,
        channel: str,
        accent: tuple[int, int, int],
        ink: tuple[int, int, int],
        muted: tuple[int, int, int],
    ) -> None:
        self._text(draw, (52, 42), text("card.title"), self._font(35, True), ink)
        self._text(draw, (54, 84), text("card.bot"), self._font(22), muted)
        label = f"@{channel}"
        font = self._font(28, True)
        width = self._text_width(draw, label, font)
        self._text(draw, (self.width - 52 - width, 48), label, font, accent)
        right_label = text("card.channel").upper()
        small = self._font(17, True)
        right_width = self._text_width(draw, right_label, small)
        self._text(draw, (self.width - 52 - right_width, 86), right_label, small, muted)

    def _summary_card(
        self,
        draw: ImageDraw.ImageDraw,
        result: dict[str, Any],
        channel: str,
        score: int,
        elo_label: str,
        emoji: str,
        rank_name: str,
        air_title: str,
        one_liner: str,
        accent: tuple[int, int, int],
        ink: tuple[int, int, int],
        muted: tuple[int, int, int],
    ) -> None:
        box = (52, 138, 1148, 390)
        self._soft_panel(draw, box)
        center = (200, 264)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        self._metric_ring(draw, center, 92, metrics, accent)
        self._avatar(draw, center, 54, channel, accent)
        self._text(draw, (336, 174), elo_label, self._font(66, True), accent)
        tag = " / ".join(part for part in (air_title, rank_name) if part)[:90]
        self._wrapped(draw, tag, (340, 256), 520, self._font(23, True), ink, max_lines=2)
        rank_line = self._rank_line(result, emoji, rank_name)
        self._wrapped(draw, rank_line, (340, 318), 520, self._font(20), muted, max_lines=1)
        self._wrapped(draw, one_liner, (846, 178), 260, self._font(21), ink, max_lines=5)

    def _rank_line(self, result: dict[str, Any], emoji: str, rank_name: str) -> str:
        place = self._plain_int(result.get("leaderboard_place"))
        place_text = f"#{place}" if place > 0 else text("card.no_place")
        prefix = f"{emoji} " if emoji else ""
        rank_text = f"{prefix}{rank_name}".strip()
        measure = _rank_measure(result)
        if measure:
            rank_text = f"{rank_text} | {measure}" if rank_text else measure
        return f"{text('card.place')}: {place_text} | {rank_text}".strip()

    def _tags(
        self,
        draw: ImageDraw.ImageDraw,
        result: dict[str, Any],
        x: int,
        y: int,
        ink: tuple[int, int, int],
        muted: tuple[int, int, int],
    ) -> None:
        self._label(draw, x, y, text("card.niche_compact"), muted)
        y = self._chip_row(draw, _strings(result.get("niche_tags"), 8), x, y + 30, 520, ink)
        self._label(draw, x + 570, 430, text("card.traits_compact"), muted)
        self._chip_row(draw, _strings(result.get("core_traits"), 8), x + 570, 460, 520, ink)

    def _metric_grid(
        self,
        draw: ImageDraw.ImageDraw,
        result: dict[str, Any],
        x: int,
        y: int,
        accent: tuple[int, int, int],
        ink: tuple[int, int, int],
        muted: tuple[int, int, int],
    ) -> None:
        self._label(draw, x, y, text("card.metric_grid"), muted)
        labels = bot_texts().get("metric_labels")
        label_map = labels if isinstance(labels, dict) else {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        rows = []
        for key, value in metrics.items():
            item = value if isinstance(value, dict) else {}
            rows.append((str(label_map.get(key) or key), self._int(item.get("score"), 0, 100)))
        rows.sort(key=lambda item: item[1], reverse=True)
        col_width = 346
        row_height = 92
        start_y = y + 42
        for index, (label, value) in enumerate(rows[:18]):
            col = index % 3
            row = index // 3
            bx = x + col * col_width
            by = start_y + row * row_height
            self._metric_cell(draw, bx, by, col_width - 22, label, value, accent, ink, muted)

    def _fun_metrics(
        self,
        draw: ImageDraw.ImageDraw,
        result: dict[str, Any],
        x: int,
        y: int,
        accent: tuple[int, int, int],
        ink: tuple[int, int, int],
        muted: tuple[int, int, int],
    ) -> None:
        items = result.get("fun_metrics") if isinstance(result.get("fun_metrics"), list) else []
        if not items:
            return
        self._label(draw, x, y, text("card.joke_metrics"), muted)
        width = 258
        for index, value in enumerate(items[:4]):
            item = value if isinstance(value, dict) else {}
            bx = x + index * (width + 20)
            by = y + 36
            draw.rounded_rectangle((bx, by, bx + width, by + 76), radius=8, fill=(252, 253, 253), outline=(221, 226, 229), width=1)
            self._wrapped(draw, str(item.get("label") or ""), (bx + 12, by + 10), width - 24, self._font(15, True), ink, max_lines=1)
            self._wrapped(draw, str(item.get("display") or ""), (bx + 12, by + 38), width - 24, self._font(18, True), accent, max_lines=1)

    def _metric_cell(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        label: str,
        value: int,
        accent: tuple[int, int, int],
        ink: tuple[int, int, int],
        muted: tuple[int, int, int],
    ) -> None:
        draw.rounded_rectangle((x, y, x + width, y + 72), radius=8, fill=(252, 253, 253), outline=(221, 226, 229), width=1)
        self._wrapped(draw, label, (x + 14, y + 10), width - 92, self._font(17, True), ink, max_lines=1)
        self._text(draw, (x + width - 72, y + 10), f"{value}/100", self._font(17, True), accent)
        draw.rounded_rectangle((x + 14, y + 48, x + width - 14, y + 56), radius=4, fill=(225, 229, 232))
        fill = int((width - 28) * min(max(value, 0), 100) / 100)
        draw.rounded_rectangle((x + 14, y + 48, x + 14 + fill, y + 56), radius=4, fill=self._metric_color(value, accent))

    def _examples(
        self,
        draw: ImageDraw.ImageDraw,
        result: dict[str, Any],
        x: int,
        y: int,
        ink: tuple[int, int, int],
        muted: tuple[int, int, int],
    ) -> None:
        self._label(draw, x, y, text("card.examples"), muted)
        evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
        tactics = result.get("tactics") if isinstance(result.get("tactics"), list) else []
        items = []
        for item in evidence[:3]:
            raw = item if isinstance(item, dict) else {}
            value = str(raw.get("snippet") or raw.get("comment") or raw.get("url") or "")
            if value:
                items.append(value)
        if len(items) < 3:
            for item in tactics[: 3 - len(items)]:
                raw = item if isinstance(item, dict) else {}
                value = str(raw.get("hint") or raw.get("why_it_keeps_attention") or raw.get("name") or "")
                if value:
                    items.append(value)
        yy = y + 42
        for index, value in enumerate(items[:3], start=1):
            yy = self._wrapped(draw, f"{index}. {value}", (x + 12, yy), 1060, self._font(20), ink, max_lines=2)
            yy += 10

    def _footer_line(self, draw: ImageDraw.ImageDraw, result: dict[str, Any], muted: tuple[int, int, int]) -> None:
        confidence = self._float(result.get("confidence"), 0.0, 1.0)
        footer = self._footer(result, confidence)
        font = self._font(18)
        footer = self._trim(draw, footer, font, self.width - 104)
        self._text(draw, (52, self.height - 54), footer, font, muted)

    def _metric_ring(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        metrics: dict[str, Any],
        accent: tuple[int, int, int],
    ) -> None:
        rows = []
        for value in metrics.values():
            item = value if isinstance(value, dict) else {}
            rows.append(self._int(item.get("score"), 0, 100))
        if not rows:
            rows = [0]
        count = len(rows)
        for index, value in enumerate(rows):
            start = -90 + index * 360 / count + 2
            end = -90 + (index + 1) * 360 / count - 2
            width = 8 + int(16 * value / 100)
            color = self._metric_color(value, accent)
            bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
            draw.arc(bbox, start, end, fill=color, width=width)
        draw.ellipse((center[0] - radius + 18, center[1] - radius + 18, center[0] + radius - 18, center[1] + radius - 18), outline=(224, 229, 232), width=2)

    def _avatar(self, draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, channel: str, accent: tuple[int, int, int]) -> None:
        box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
        draw.ellipse(box, fill=(250, 251, 251), outline=accent, width=3)
        initials = self._initials(channel)
        font = self._font(28, True)
        bbox = draw.textbbox((0, 0), initials, font=font)
        self._text(draw, (center[0] - (bbox[2] - bbox[0]) // 2, center[1] - 18), initials, font, accent)

    def _chip_row(self, draw: ImageDraw.ImageDraw, values: list[str], x: int, y: int, max_width: int, ink: tuple[int, int, int]) -> int:
        if not values:
            return y
        xx = x
        yy = y
        font = self._font(18, True)
        for value in values:
            label = str(value)[:28]
            width = min(self._text_width(draw, label, font) + 24, max_width)
            if xx + width > x + max_width:
                xx = x
                yy += 38
            draw.rounded_rectangle((xx, yy, xx + width, yy + 28), radius=8, fill=(252, 253, 253), outline=(222, 227, 230))
            self._text(draw, (xx + 12, yy + 5), self._trim(draw, label, font, width - 24), font, ink)
            xx += width + 8
        return yy + 34

    def _soft_panel(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
        draw.rounded_rectangle(box, radius=16, fill=(251, 252, 252), outline=(220, 225, 228), width=1)

    def _label(self, draw: ImageDraw.ImageDraw, x: int, y: int, value: str, color: tuple[int, int, int]) -> None:
        self._text(draw, (x, y), value.upper(), self._font(17, True), color)

    def _initials(self, channel: str) -> str:
        cleaned = "".join(char for char in channel if char.isalnum())
        if not cleaned:
            return "@"
        return cleaned[:2].upper()

    def _score_panel(
        self,
        draw: ImageDraw.ImageDraw,
        score: int,
        elo_label: str,
        rank_name: str,
        air_title: str,
        accent: tuple[int, int, int],
    ) -> None:
        box = (64, 190, 1136, 520)
        draw.rounded_rectangle(box, radius=26, fill=(255, 255, 250), outline=(218, 213, 199), width=2)
        self._gauge(draw, (224, 355), 120, score, accent)
        self._text(draw, (390, 252), text("card.score").upper(), self._font(25, True), (92, 95, 99))
        self._text(draw, (390, 292), elo_label, self._font(86, True), accent)
        self._text(draw, (394, 390), rank_name, self._font(42, True), (24, 28, 32))
        self._wrapped(draw, air_title, (396, 440), 660, self._font(28), (64, 69, 72), max_lines=2)

    def _gauge(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        score: int,
        accent: tuple[int, int, int],
    ) -> None:
        bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
        draw.ellipse(bbox, outline=(229, 227, 218), width=26)
        end = -90 + int(360 * min(max(score, 0), 200) / 200)
        draw.arc(bbox, -90, end, fill=accent, width=26)
        inner = radius - 44
        draw.ellipse((center[0] - inner, center[1] - inner, center[0] + inner, center[1] + inner), fill=(246, 244, 235))

    def _section_text(self, draw: ImageDraw.ImageDraw, y: int, one_liner: str, metaphor: str) -> int:
        y = self._wrapped(draw, one_liner, (72, y), 1030, self._font(34, True), (24, 28, 32), max_lines=3)
        if metaphor:
            y = self._wrapped(draw, metaphor, (72, y + 16), 1030, self._font(26), (74, 79, 82), max_lines=3)
        return y

    def _metrics(self, draw: ImageDraw.ImageDraw, y: int, result: dict[str, Any]) -> int:
        self._text(draw, (64, y), text("card.metrics").upper(), self._font(25, True), (27, 98, 111))
        y += 48
        labels = bot_texts().get("metric_labels")
        label_map = labels if isinstance(labels, dict) else {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        rows = []
        for key, value in metrics.items():
            item = value if isinstance(value, dict) else {}
            rows.append((self._int(item.get("score"), 0, 100), str(label_map.get(key) or key)))
        rows.sort(reverse=True)
        for score, label in rows[:5]:
            y = self._bar(draw, y, label, score)
        return y

    def _tactics(self, draw: ImageDraw.ImageDraw, y: int, result: dict[str, Any]) -> int:
        self._text(draw, (64, y), text("card.tactics").upper(), self._font(25, True), (195, 69, 55))
        y += 46
        tactics = result.get("tactics") if isinstance(result.get("tactics"), list) else []
        for index, item in enumerate(tactics[:2], start=1):
            raw = item if isinstance(item, dict) else {}
            body = str(raw.get("hint") or raw.get("why_it_keeps_attention") or raw.get("name") or raw.get("code") or "")
            y = self._wrapped(draw, f"{index}. {body}", (82, y), 1020, self._font(25), (24, 28, 32), max_lines=2)
            y += 10
        return y

    def _evidence(self, draw: ImageDraw.ImageDraw, y: int, result: dict[str, Any]) -> int:
        evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
        if not evidence or y > self.height - 250:
            return y
        self._text(draw, (64, y), text("card.evidence").upper(), self._font(25, True), (70, 93, 55))
        y += 46
        for index, item in enumerate(evidence[:1], start=1):
            raw = item if isinstance(item, dict) else {}
            snippet = str(raw.get("snippet") or raw.get("comment") or raw.get("url") or "")
            y = self._wrapped(draw, f"{index}. {snippet}", (82, y), 1020, self._font(24), (46, 50, 53), max_lines=1)
            y += 8
        return y

    def _bar(self, draw: ImageDraw.ImageDraw, y: int, label: str, score: int) -> int:
        self._text(draw, (82, y), label, self._font(25, True), (24, 28, 32))
        self._text(draw, (1030, y), str(score), self._font(25, True), (24, 28, 32))
        y += 40
        draw.rounded_rectangle((82, y, 1042, y + 20), radius=10, fill=(225, 224, 215))
        fill_width = int(960 * min(max(score, 0), 100) / 100)
        draw.rounded_rectangle((82, y, 82 + fill_width, y + 20), radius=10, fill=self._accent(score * 2))
        return y + 42

    def _footer(self, result: dict[str, Any], confidence: float) -> str:
        parts = [
            f"{text('card.messages')}: {self._int(result.get('message_count'), 0, 1000000)}",
            f"{text('card.confidence')}: {int(confidence * 100)}%",
        ]
        return " | ".join(parts)

    def _pill(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: str, color: tuple[int, int, int]) -> None:
        draw.rounded_rectangle(box, radius=22, fill=color)
        font = self._font(24, True)
        bbox = draw.textbbox((0, 0), value, font=font)
        width = bbox[2] - bbox[0]
        x = box[0] + max(18, (box[2] - box[0] - width) // 2)
        self._text(draw, (x, box[1] + 12), value, font, (255, 255, 250))

    def _wrapped(
        self,
        draw: ImageDraw.ImageDraw,
        value: str,
        xy: tuple[int, int],
        max_width: int,
        font: ImageFont.ImageFont,
        fill: tuple[int, int, int],
        max_lines: int,
    ) -> int:
        lines = self._wrap(draw, value, font, max_width, max_lines)
        y = xy[1]
        step = self._line_height(font) + 7
        for line in lines:
            self._text(draw, (xy[0], y), line, font, fill)
            y += step
        return y

    def _wrap(
        self,
        draw: ImageDraw.ImageDraw,
        value: str,
        font: ImageFont.ImageFont,
        max_width: int,
        max_lines: int,
    ) -> list[str]:
        words = str(value or "").split()
        if not words:
            return []
        lines: list[str] = []
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if self._text_width(draw, trial, font) <= max_width:
                current = trial
                continue
            if current:
                lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
            lines[-1] = self._trim(draw, lines[-1], font, max_width)
        return lines

    def _trim(self, draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont, max_width: int) -> str:
        suffix = "..."
        out = value
        while out and self._text_width(draw, out + suffix, font) > max_width:
            out = out[:-1]
        return (out.rstrip() + suffix) if out else suffix

    def _text(self, draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
        draw.text(xy, str(value), font=font, fill=fill)

    def _text_width(self, draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), value, font=font)
        return bbox[2] - bbox[0]

    def _line_height(self, font: ImageFont.ImageFont) -> int:
        bbox = font.getbbox("Ag")
        return bbox[3] - bbox[1]

    def _font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        names = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        )
        for path in names:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _mix(self, first: tuple[int, int, int], second: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
        ratio = min(max(amount, 0.0), 1.0)
        return tuple(round(a + (b - a) * ratio) for a, b in zip(first, second))

    def _metric_color(self, score: int, accent: tuple[int, int, int]) -> tuple[int, int, int]:
        if score < 35:
            return (138, 154, 160)
        if score < 60:
            return (111, 134, 143)
        if score < 75:
            return self._mix((97, 116, 126), accent, 0.45)
        return accent

    def _accent(self, score: int) -> tuple[int, int, int]:
        if score < 50:
            return (88, 126, 128)
        if score < 100:
            return (107, 126, 137)
        if score < 150:
            return (112, 111, 121)
        return (78, 83, 91)

    def _int(self, value: Any, low: int, high: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = low
        return min(max(parsed, low), high)

    def _plain_int(self, value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _float(self, value: Any, low: float, high: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = low
        return min(max(parsed, low), high)


def result_card_caption(result: dict[str, Any], bot_username: str | None = None, referrer_user_id: int | None = None) -> str:
    profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
    emoji, rank_name = _rank_bits(result, profile)
    channel = str(result.get("channel") or "")
    caption = text(
        "card.caption",
        channel=channel,
        elo_label=str(result.get("air_elo_label") or ""),
        rank_name=f"{emoji} {rank_name}".strip(),
        measure=_rank_measure(result),
        one_liner=str(profile.get("one_liner") or result.get("summary") or ""),
    )
    if channel:
        caption = caption.replace(escape(f"@{channel}"), _channel_link(channel), 1)
    link = referral_link(bot_username, referrer_user_id)
    if link:
        caption = f"{caption}\n\n<a href=\"{escape(link, quote=True)}\">{escape(text('card.cta'))}</a>"
    if len(caption) <= 900:
        return caption
    return caption[:897].rstrip() + "..."


def result_profile_html(
    result: dict[str, Any],
    bot_username: str | None = None,
    referrer_user_id: int | None = None,
    avatar_url: str | None = None,
    generated_image_url: str | None = None,
) -> str:
    profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
    emoji, rank_name = _rank_bits(result, profile)
    channel = str(result.get("channel") or "")
    title = text("profile.title", channel=channel)
    title_html = escape(title)
    if channel:
        title_html = title_html.replace(escape(f"@{channel}"), _channel_link(channel), 1)
    measure = _rank_measure(result)
    score_line = text(
        "profile.score_line",
        elo=str(result.get("air_elo") or ""),
        emoji=emoji,
        rank=rank_name,
        measure=measure,
    )
    empty = text("profile.empty_value")
    parts = [
        f"<h2>{_rich_icon('channel')} {title_html}</h2>",
        _profile_place_html(result),
        f"<p><b>{escape(score_line)}</b></p>",
    ]
    avatar_block = _avatar_media_block(channel, avatar_url)
    if avatar_block:
        parts.append(avatar_block)
    generated_block = _generated_media_block(channel, generated_image_url)
    if generated_block:
        parts.append(generated_block)
    summary = str(profile.get("one_liner") or result.get("summary") or "")
    if summary:
        parts.append(f"<blockquote>{escape(summary)}</blockquote>")
    parts.append(
        "<table bordered striped>"
        f"<caption>{_rich_icon('metric')} {escape(text('profile.summary'))}</caption>"
        f"<tr><th>ELO</th><td align=\"right\"><b>{escape(str(result.get('air_elo') or empty))}</b></td></tr>"
        f"{_fight_rating_row_html(result)}"
        f"<tr><th>{escape(text('profile.rank'))}</th><td>{escape(rank_name or empty)}</td></tr>"
        f"<tr><th>{escape(text('profile.measure'))}</th><td>{escape(measure or empty)}</td></tr>"
        f"<tr><th>{escape(text('profile.messages'))}</th><td align=\"right\">{escape(str(result.get('message_count') or empty))}</td></tr>"
        f"{_fight_stats_row_html(result)}"
        "</table>"
    )
    tags = _strings(result.get("niche_tags"), 8)
    traits = _strings(result.get("core_traits"), 8)
    if tags or traits:
        parts.append("<ul>")
        if tags:
            parts.append(f"<li>{_rich_icon('tag')} <b>{escape(text('profile.niche'))}</b>: {escape(', '.join(tags))}</li>")
        if traits:
            parts.append(f"<li>{_rich_icon('tag')} <b>{escape(text('profile.traits'))}</b>: {escape(', '.join(traits))}</li>")
        parts.append("</ul>")
    fun_metrics = _fun_metrics(result)
    if fun_metrics:
        parts.append(f"<h3>{_rich_icon('equivalent')} {escape(text('profile.joke_metrics'))}</h3><ul>")
        for label, display in fun_metrics:
            parts.append(f"<li><b>{escape(label)}</b>: {escape(display)}</li>")
        parts.append("</ul>")
    rows = _metric_rows(result)
    if rows:
        parts.append(f"<h3>{_rich_icon('metric')} {escape(text('profile.metrics'))}</h3>")
        parts.append(
            "<table bordered striped>"
            f"<tr><th>{escape(text('profile.metric'))}</th><th>{escape(text('profile.value'))}</th></tr>"
        )
        for label, score in rows:
            parts.append(f"<tr><td>{escape(label)}</td><td align=\"right\">{score}</td></tr>")
        parts.append("</table>")
    tactics = result.get("tactics") if isinstance(result.get("tactics"), list) else []
    if tactics:
        parts.append(f"<h3>{_rich_icon('tactics')} {escape(text('profile.tactics'))}</h3><ol>")
        for item in tactics[:5]:
            raw = item if isinstance(item, dict) else {}
            body = str(raw.get("hint") or raw.get("why_it_keeps_attention") or raw.get("name") or raw.get("code") or "")
            if body:
                parts.append(f"<li>{escape(body)}</li>")
        parts.append("</ol>")
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    if evidence:
        parts.append(f"<details><summary>{_rich_icon('evidence')} {escape(text('profile.evidence'))}</summary><ol>")
        for item in evidence[:5]:
            raw = item if isinstance(item, dict) else {}
            url = str(raw.get("url") or "")
            snippet = str(raw.get("snippet") or raw.get("comment") or url)
            if url:
                parts.append(f"<li><a href=\"{escape(url, quote=True)}\">{escape(snippet)}</a></li>")
            elif snippet:
                parts.append(f"<li>{escape(snippet)}</li>")
        parts.append("</ol></details>")
    link = referral_link(bot_username, referrer_user_id)
    if link:
        parts.append(f"<p><a href=\"{escape(link, quote=True)}\">{escape(text('card.cta'))}</a></p>")
    return "\n".join(parts)


def result_profile_plain(
    result: dict[str, Any],
    bot_username: str | None = None,
    referrer_user_id: int | None = None,
) -> str:
    profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
    emoji, rank_name = _rank_bits(result, profile)
    lines = [
        text("profile.title", channel=str(result.get("channel") or "")),
        _profile_place_plain(result),
        text(
            "profile.score_line",
            elo=str(result.get("air_elo") or ""),
            emoji=emoji,
            rank=rank_name,
            measure=_rank_measure(result),
        ),
    ]
    summary = str(profile.get("one_liner") or result.get("summary") or "")
    if summary:
        lines.append(summary)
    fight_stats = _fight_stats_plain(result)
    if fight_stats:
        lines.append(fight_stats)
    fight_rating = _fight_rating_plain(result)
    if fight_rating:
        lines.append(fight_rating)
    rows = _metric_rows(result)
    if rows:
        lines.append("")
        lines.append(text("profile.metrics"))
        lines.extend(f"{label}: {score}" for label, score in rows)
    fun_metrics = _fun_metrics(result)
    if fun_metrics:
        lines.append("")
        lines.append(text("profile.joke_metrics"))
        lines.extend(f"{label}: {display}" for label, display in fun_metrics)
    link = referral_link(bot_username, referrer_user_id)
    if link:
        lines.append("")
        lines.append(f"{text('card.cta')}: {link}")
    return "\n".join(lines)


def _fight_stats_row_html(result: dict[str, Any]) -> str:
    stats = result.get("fight_stats") if isinstance(result.get("fight_stats"), dict) else {}
    wins = _plain_int(stats.get("wins"))
    losses = _plain_int(stats.get("losses"))
    draws = _plain_int(stats.get("draws"))
    if wins + losses + draws <= 0:
        return ""
    value = text("profile.fight_stats_value", wins=wins, losses=losses, draws=draws)
    return f"<tr><th>{escape(text('profile.fights'))}</th><td align=\"right\">{escape(value)}</td></tr>"


def _fight_rating_row_html(result: dict[str, Any]) -> str:
    rating = result.get("fight_rating") if isinstance(result.get("fight_rating"), dict) else {}
    base = _plain_int(rating.get("base") if rating else result.get("base_air_elo"))
    delta = _plain_int(rating.get("delta") if rating else result.get("fight_rating_delta"))
    current = _plain_int(rating.get("current") if rating else result.get("battle_elo"))
    if base <= 0 and delta == 0 and current <= 0:
        return ""
    value = text("profile.fight_rating_value", base=base, delta=_signed(delta), current=current)
    return f"<tr><th>{escape(text('profile.fight_rating'))}</th><td align=\"right\">{escape(value)}</td></tr>"


def _fight_stats_plain(result: dict[str, Any]) -> str:
    stats = result.get("fight_stats") if isinstance(result.get("fight_stats"), dict) else {}
    wins = _plain_int(stats.get("wins"))
    losses = _plain_int(stats.get("losses"))
    draws = _plain_int(stats.get("draws"))
    if wins + losses + draws <= 0:
        return ""
    return text("profile.fight_stats_line", wins=wins, losses=losses, draws=draws)


def _fight_rating_plain(result: dict[str, Any]) -> str:
    rating = result.get("fight_rating") if isinstance(result.get("fight_rating"), dict) else {}
    base = _plain_int(rating.get("base") if rating else result.get("base_air_elo"))
    delta = _plain_int(rating.get("delta") if rating else result.get("fight_rating_delta"))
    current = _plain_int(rating.get("current") if rating else result.get("battle_elo"))
    if base <= 0 and delta == 0 and current <= 0:
        return ""
    return text("profile.fight_rating_line", base=base, delta=_signed(delta), current=current)


def _profile_place_html(result: dict[str, Any]) -> str:
    place = _plain_int(result.get("leaderboard_place"))
    if place <= 0:
        return f"<p>{escape(text('profile.place'))}: {escape(text('profile.no_place'))}</p>"
    return f"<p>{escape(text('profile.place'))}: <b>#{place}</b></p>"


def _profile_place_plain(result: dict[str, Any]) -> str:
    place = _plain_int(result.get("leaderboard_place"))
    value = f"#{place}" if place > 0 else text("profile.no_place")
    return f"{text('profile.place')}: {value}"


def _plain_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _rating_delta(result: dict[str, Any], stats: dict[str, int]) -> int:
    if stats.get("rating_delta") is not None:
        return _plain_int(stats.get("rating_delta"))
    return _plain_int(result.get("fight_rating_delta"))


def _signed(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


def result_screen_html(
    result: dict[str, Any],
    bot_username: str | None = None,
    referrer_user_id: int | None = None,
    avatar_url: str | None = None,
    generated_image_url: str | None = None,
) -> str:
    return result_profile_html(
        result,
        bot_username=bot_username,
        referrer_user_id=referrer_user_id,
        avatar_url=avatar_url,
        generated_image_url=generated_image_url,
    )


def result_screen_plain(
    result: dict[str, Any],
    bot_username: str | None = None,
    referrer_user_id: int | None = None,
) -> str:
    return result_profile_plain(result, bot_username=bot_username, referrer_user_id=referrer_user_id)


def start_html() -> str:
    return text("screens.start_html")


def start_plain() -> str:
    return text("start")


def ask_channel_html() -> str:
    return text("screens.ask_channel_html")


def leaderboard_html(
    runs: list[Any],
    page: int,
    has_prev: bool,
    has_next: bool,
    image_urls: dict[str, str] | None = None,
    fight_stats: dict[str, dict[str, int]] | None = None,
) -> str:
    if not runs:
        return text("screens.leaderboard_empty_html")
    rows = []
    images = []
    image_urls = image_urls or {}
    fight_stats = fight_stats or {}
    for index, run in enumerate(runs, start=1 + page * 5):
        result = run.result or {}
        profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
        emoji, rank_name = _rank_bits(result, profile)
        stats = fight_stats.get(str(run.channel).lower()) or {}
        image_url = image_urls.get(str(run.channel).lower())
        if image_url:
            images.append(_generated_media_block(str(run.channel), image_url, caption=f"#{index} @{run.channel}"))
        rows.append(
            text(
                "screens.leaderboard_row_html",
                index=index,
                channel=escape(run.channel),
                elo=escape(str(result.get("air_elo") or result.get("score") or "")),
                delta=escape(_signed(_rating_delta(result, stats))),
                emoji=escape(emoji),
                rank=escape(rank_name),
                measure=escape(_rank_measure(result)),
                wins=escape(str(_plain_int(stats.get("wins")))),
                losses=escape(str(_plain_int(stats.get("losses")))),
            )
        )
    return text(
        "screens.leaderboard_html",
        page=page + 1,
        rows="\n".join([*images, *rows]),
        nav="",
    )


def referral_link(bot_username: str | None, referrer_user_id: int | None) -> str | None:
    if not bot_username or not referrer_user_id:
        return None
    username = bot_username.lstrip("@")
    return f"https://t.me/{username}?start=ref_{referrer_user_id}"


def leaderboard_text(runs: list[Any], fight_stats: dict[str, dict[str, int]] | None = None) -> str:
    if not runs:
        return text("leaderboard.empty")
    lines = [text("leaderboard.title")]
    fight_stats = fight_stats or {}
    for index, run in enumerate(runs, start=1):
        result = run.result or {}
        profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
        emoji, rank_name = _rank_bits(result, profile)
        stats = fight_stats.get(str(run.channel).lower()) or {}
        lines.append(
            text(
                "leaderboard.row",
                index=index,
                channel=run.channel,
                elo=str(result.get("air_elo") or result.get("score") or ""),
                delta=_signed(_rating_delta(result, stats)),
                emoji=emoji,
                rank=rank_name,
                measure=_rank_measure(result),
                wins=str(_plain_int(stats.get("wins"))),
                losses=str(_plain_int(stats.get("losses"))),
            )
        )
    return "\n".join(lines)


def _rich_icon(name: str) -> str:
    registry = bot_texts().get("rich_emoji")
    item = registry.get(name) if isinstance(registry, dict) else None
    data = item if isinstance(item, dict) else {}
    fallback = str(data.get("fallback") or "")
    emoji_id = str(data.get("id") or "")
    if not emoji_id:
        return escape(fallback)
    return f"<tg-emoji emoji-id=\"{escape(emoji_id, quote=True)}\">{escape(fallback)}</tg-emoji>"


def _channel_link(channel: str) -> str:
    if not channel:
        return ""
    value = escape(channel)
    url = escape(f"https://t.me/{channel}", quote=True)
    return f"<a href=\"{url}\">@{value}</a>"


def _avatar_media_block(channel: str, avatar_url: str | None) -> str:
    if not avatar_url:
        return ""
    parsed = urlparse(avatar_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    src = escape(avatar_url, quote=True)
    label = escape(f"@{channel}" if channel else text("profile.title", channel=""))
    return f"<figure><img src=\"{src}\"/><figcaption>{label}</figcaption></figure>"


def _generated_media_block(channel: str, image_url: str | None, caption: str | None = None) -> str:
    if not image_url:
        return ""
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    src = escape(image_url, quote=True)
    label = escape(caption or (f"@{channel}" if channel else text("profile.title", channel="")))
    return f"<figure><img src=\"{src}\"/><figcaption>{label}</figcaption></figure>"


def _rank_bits(result: dict[str, Any], profile: dict[str, Any]) -> tuple[str, str]:
    if result.get("air_elo") is not None:
        rank = elo_rank(_int(result.get("air_elo"), 0, 1000000))
        return str(rank.get("emoji") or result.get("air_elo_emoji") or profile.get("air_elo_emoji") or ""), str(rank.get("tag") or "")
    return str(result.get("air_elo_emoji") or profile.get("air_elo_emoji") or ""), str(profile.get("rank_name") or result.get("humorous_label") or "")


def _rank_measure(result: dict[str, Any]) -> str:
    if result.get("air_rank_measure"):
        return str(result.get("air_rank_measure"))
    profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
    if profile.get("rank_measure"):
        return str(profile.get("rank_measure"))
    if result.get("air_elo") is None:
        return ""
    rank = elo_rank(_int(result.get("air_elo"), 0, 1000000))
    value = rank.get("unit_value")
    unit = str(rank.get("unit") or "")
    if value is None or not unit:
        return ""
    return f"{value} {unit}"


def _fun_metrics(result: dict[str, Any]) -> list[tuple[str, str]]:
    items = result.get("fun_metrics") if isinstance(result.get("fun_metrics"), list) else []
    rows = []
    for value in items[:4]:
        item = value if isinstance(value, dict) else {}
        label = str(item.get("label") or "")
        display = str(item.get("display") or "")
        if label and display:
            rows.append((label, display))
    return rows


def _top_metric_rows(result: dict[str, Any], limit: int) -> list[tuple[str, int]]:
    return _metric_rows(result)[:limit]


def _metric_rows(result: dict[str, Any]) -> list[tuple[str, int]]:
    labels = bot_texts().get("metric_labels")
    label_map = labels if isinstance(labels, dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    rows = []
    for key, value in metrics.items():
        item = value if isinstance(value, dict) else {}
        score = _int(item.get("score"), 0, 100)
        rows.append((score, str(label_map.get(key) or key)))
    rows.sort(reverse=True)
    return [(label, score) for score, label in rows]


def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value[:limit] if item is not None]


def _int(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = low
    return min(max(parsed, low), high)
