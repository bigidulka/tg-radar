from tg_radar.scoring import hiring_signal, source_trust, vacancy_key


def test_hiring_signal_detects_sber_ai_vacancy():
    signal = hiring_signal("Сбер ищет Python-разработчика / AI Engineer в команду GigaChat, писать @hr_sber")

    assert signal.company_match is True
    assert signal.ai_llm_match is True
    assert signal.hiring_match is True
    assert signal.direct_contact is True
    assert "company_match:sber" in signal.reasons
    assert "target_hiring_detector" in signal.reasons


def test_vacancy_key_clusters_reposts_without_urls():
    left = vacancy_key("Python-разработчик в Сбер https://t.me/a/1 писать @hr_sber")
    right = vacancy_key("Python-разработчик в Сбер https://t.me/b/2 писать @other_hr")

    assert left == right


def test_source_trust_uses_prefixes():
    assert source_trust("telethon_recursive:vibecoderchat") > source_trust("random")
