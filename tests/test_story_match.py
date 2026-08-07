from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from tg_radar.config import Settings
from tg_radar.story_catalog import load_story_catalog, parse_catalog
from tg_radar.story_match import StoryMatchError, StoryMatcher, StorySignal, build_report


FIXTURE = Path(__file__).parent / "fixtures" / "article-ideas.md"
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


class FakeTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def complete_json(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


def _settings(**overrides) -> Settings:
    return Settings(llm_base_url="http://localhost:8317/v1", llm_api_key="sk-test", **overrides)


def _signal(channel: str, msg_id: int, text: str, days_ago: int = 0) -> StorySignal:
    return StorySignal(
        message_id=f"{channel}:{msg_id}",
        channel=channel,
        url=f"https://t.me/{channel}/{msg_id}",
        posted_at=NOW - timedelta(days=days_ago),
        text=text,
        pain_type="complaint",
    )


def _matcher(transport: FakeTransport, batch_size: int = 50) -> StoryMatcher:
    return StoryMatcher(
        _settings(story_match_batch_size=batch_size),
        cases=parse_catalog(FIXTURE.read_text(encoding="utf-8")),
        transport=transport,
    )


def _report(matcher: StoryMatcher, signals: list[StorySignal], pairs) -> object:
    return build_report("ai-dev-blog", 14, signals, pairs, len(matcher.cases), "medium")


def test_parse_catalog_reads_every_case_from_the_markdown():
    cases = parse_catalog(FIXTURE.read_text(encoding="utf-8"))

    assert len(cases) == 22
    assert [case.number for case in cases] == list(range(1, 23))
    assert cases[0].title == "Таймаут в 10 секунд, из-за которого движок «работал», не работая"
    assert cases[0].format == "Habr / плотный Telegram"
    assert cases[0].section == "Первая очередь"
    assert "364 → 1439" in cases[0].body
    assert cases[0].takeaway.startswith("фолбэк, который выглядит как результат")
    assert cases[13].title == "Агент заменил продукт вместо того, чтобы его расширить"
    assert cases[13].section == "Сильнейшие"


def test_case_without_takeaway_keeps_none():
    cases = {case.number: case for case in parse_catalog(FIXTURE.read_text(encoding="utf-8"))}

    assert cases[10].takeaway is None
    assert cases[10].body.startswith("`next.config.js` вычисляется при сборке")
    assert cases[11].takeaway is not None


def test_shipped_catalog_matches_the_markdown():
    shipped = load_story_catalog()

    assert len(shipped) == 22
    assert [case.number for case in shipped] == list(range(1, 23))
    assert shipped == parse_catalog(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_reference_pairs_are_reported_and_grouped_by_case():
    signals = [
        _signal("neuralshit", 7590, "Сложнее сделать так, чтобы агент не дублировал действия и переживал падение API."),
        _signal("boris_again", 4043, "Блекпилл по поводу агентов: вайбкодинг не работает, агенты производят технический долг.", 1),
        _signal("seeallochnaya", 3851, "Инцидент OpenAI: агент оставил заметки о том, как освободиться от внутренних ограничений.", 1),
    ]
    transport = FakeTransport(
        [
            json.dumps(
                {
                    "pairs": [
                        {
                            "signal_url": "https://t.me/neuralshit/7590",
                            "case_number": 6,
                            "signal_gist": "агент дублирует действия и не переживает падение API",
                            "you_add": "у автора постановка проблемы, у тебя воспроизведённый двойной постинг и статус pending_confirmation",
                            "confidence": "high",
                        },
                        {
                            "signal_url": "https://t.me/boris_again/4043",
                            "case_number": 14,
                            "signal_gist": "вайбкодинг не работает, агенты плодят техдолг",
                            "you_add": "у автора мнение, у тебя перезаписанный продукт и восстановление из be5ce86^",
                            "confidence": "high",
                        },
                        {
                            "signal_url": "https://t.me/seeallochnaya/3851",
                            "case_number": 3,
                            "signal_gist": "агент писал себе инструкции по обходу ограничений",
                            "you_add": "у автора чужой инцидент, у тебя два своих отклонённых промпта и правка, после которой прошло",
                            "confidence": "medium",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        ]
    )
    matcher = _matcher(transport)

    pairs = await matcher.match(signals)
    report = _report(matcher, signals, pairs)

    assert {pair.case.number for pair in pairs} == {6, 14, 3}
    assert all(len(pair.signals) == 1 for pair in pairs)
    assert report.signals_considered == 3
    assert report.signals_matched == 3
    assert report.notes == []


@pytest.mark.asyncio
async def test_offtopic_signals_are_shown_to_the_model_and_produce_no_pairs():
    signals = [
        _signal("cryptochan", 11, "Биткоин пробил уровень, фьючерсы на квартал, funding rate ушёл в минус."),
        _signal("jobschan", 12, "Вакансия: Senior Python разработчик, удалённо, з/п от 400к, резюме в личку."),
    ]
    transport = FakeTransport([json.dumps({"pairs": []})])
    matcher = _matcher(transport)

    pairs = await matcher.match(signals)
    report = _report(matcher, signals, pairs)

    sent = json.loads(transport.calls[0][1]["content"])
    assert [item["signal_url"] for item in sent["signals"]] == [signal.url for signal in signals]
    assert len(sent["catalog"]) == 22
    assert pairs == []
    assert report.signals_matched == 0
    assert report.notes == ["совпадений нет: ни один сигнал не покрыт кейсом с собственным доказательством"]


@pytest.mark.asyncio
async def test_pairs_the_model_invented_are_dropped():
    signals = [_signal("cryptochan", 11, "Биткоин пробил уровень, фьючерсы на квартал.")]
    transport = FakeTransport(
        [
            json.dumps(
                {
                    "pairs": [
                        {
                            "signal_url": "https://t.me/never-sent/1",
                            "case_number": 6,
                            "signal_gist": "сигнал, которого не было во входных данных",
                            "you_add": "у автора постановка проблемы, у тебя замер и коммит из рабочего дневника",
                            "confidence": "high",
                        },
                        {
                            "signal_url": "https://t.me/cryptochan/11",
                            "case_number": 99,
                            "signal_gist": "кейс, которого нет в каталоге",
                            "you_add": "у автора постановка проблемы, у тебя замер и коммит из рабочего дневника",
                            "confidence": "high",
                        },
                        {
                            "signal_url": "https://t.me/cryptochan/11",
                            "case_number": 6,
                            "signal_gist": "пустое обоснование",
                            "you_add": "есть опыт",
                            "confidence": "high",
                        },
                        {
                            "signal_url": "https://t.me/cryptochan/11",
                            "case_number": 6,
                            "signal_gist": "выдуманный уровень уверенности",
                            "you_add": "у автора постановка проблемы, у тебя замер и коммит из рабочего дневника",
                            "confidence": "очень высокая",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        ]
    )
    matcher = _matcher(transport)

    pairs = await matcher.match(signals)

    assert pairs == []


@pytest.mark.asyncio
async def test_one_case_with_five_signals_becomes_one_grouped_pair():
    signals = [
        _signal("alpha", 1, "Агент отчитался, что поправил файл, а изменений нет."),
        _signal("alpha", 2, "Снова отчёт агента разошёлся с диском.", 1),
        _signal("beta", 3, "Проверил после агента — три правки из отчёта отсутствуют.", 2),
        _signal("gamma", 4, "Приёмка по резюме агента не работает, читайте файлы.", 3),
        _signal("gamma", 5, "Отчёт агента — это утверждение, а не наблюдение.", 4),
    ]
    transport = FakeTransport(
        [
            json.dumps(
                {
                    "pairs": [
                        {
                            "signal_url": signal.url,
                            "case_number": 15,
                            "signal_gist": "отчёт агента расходится с содержимым диска",
                            "you_add": "у автора наблюдение, у тебя дважды воспроизведённый случай и правило приёмки по файлам",
                            "confidence": "high" if index == 0 else "medium",
                        }
                        for index, signal in enumerate(signals)
                    ]
                },
                ensure_ascii=False,
            )
        ]
    )
    matcher = _matcher(transport)

    pairs = await matcher.match(signals)
    report = _report(matcher, signals, pairs)

    assert len(pairs) == 1
    assert pairs[0].case.number == 15
    assert len(pairs[0].signals) == 5
    assert pairs[0].channels == ["alpha", "beta", "gamma"]
    assert pairs[0].multi_channel is True
    assert pairs[0].confidence == "high"
    assert report.signals_matched == 5


@pytest.mark.asyncio
async def test_one_signal_pairs_with_at_most_one_case():
    signals = [_signal("alpha", 1, "Агент отчитался, что поправил файл, а изменений нет.")]
    transport = FakeTransport(
        [
            json.dumps(
                {
                    "pairs": [
                        {
                            "signal_url": signals[0].url,
                            "case_number": 15,
                            "signal_gist": "первый кейс",
                            "you_add": "у автора наблюдение, у тебя дважды воспроизведённый случай и правило приёмки",
                            "confidence": "high",
                        },
                        {
                            "signal_url": signals[0].url,
                            "case_number": 14,
                            "signal_gist": "второй кейс на тот же сигнал",
                            "you_add": "у автора наблюдение, у тебя перезаписанный продукт и восстановление из git",
                            "confidence": "high",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        ]
    )
    matcher = _matcher(transport)

    pairs = await matcher.match(signals)

    assert [pair.case.number for pair in pairs] == [15]


@pytest.mark.asyncio
async def test_low_confidence_pairs_are_filtered_by_the_floor():
    signals = [_signal("alpha", 1, "Агент отчитался, что поправил файл, а изменений нет.")]
    payload = json.dumps(
        {
            "pairs": [
                {
                    "signal_url": signals[0].url,
                    "case_number": 15,
                    "signal_gist": "слабая связь",
                    "you_add": "у автора наблюдение, у тебя дважды воспроизведённый случай и правило приёмки",
                    "confidence": "low",
                }
            ]
        },
        ensure_ascii=False,
    )

    assert await _matcher(FakeTransport([payload])).match(signals, "medium") == []
    assert len(await _matcher(FakeTransport([payload])).match(signals, "low")) == 1


@pytest.mark.asyncio
async def test_signals_are_batched_and_every_batch_sees_the_whole_catalog():
    signals = [_signal("alpha", index, f"Сигнал номер {index} про поведение агента в проде.") for index in range(1, 6)]
    transport = FakeTransport([json.dumps({"pairs": []})] * 3)
    matcher = _matcher(transport, batch_size=2)

    await matcher.match(signals)

    assert len(transport.calls) == 3
    sizes = [len(json.loads(call[1]["content"])["signals"]) for call in transport.calls]
    assert sorted(sizes) == [1, 2, 2]
    assert all(len(json.loads(call[1]["content"])["catalog"]) == 22 for call in transport.calls)


@pytest.mark.asyncio
async def test_matching_without_llm_credentials_refuses_instead_of_guessing():
    matcher = StoryMatcher(
        Settings(llm_base_url=None, llm_api_key=None),
        cases=parse_catalog(FIXTURE.read_text(encoding="utf-8")),
    )

    with pytest.raises(StoryMatchError, match="llm is not configured"):
        await matcher.match([_signal("alpha", 1, "Агент отчитался, а правок не было.")])
