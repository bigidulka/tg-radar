from datetime import timedelta

from tg_radar.bot.analysis import AIRBOT_RESULT_SCHEMA_VERSION
from tg_radar.bot.fight import AIRBOT_FIGHT_SCHEMA_VERSION
from tg_radar.bot.store import BotRunStore, iso, utcnow


def test_bot_store_run_lifecycle_and_raw_messages(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    run = store.create_run(user_id=1, chat_id=2, channel="Demo")

    assert run.status == "queued"
    assert run.channel == "demo"
    assert store.active_run_for_user(1).run_id == run.run_id

    store.set_raw_messages(run.run_id, [{"url": "https://t.me/demo/1", "text": "hello"}])
    assert store.raw_messages(run.run_id)[0]["text"] == "hello"

    completed = store.complete_run(run.run_id, {"score": 77})
    assert completed.status == "completed"
    assert completed.result["score"] == 77
    assert store.active_run_for_user(1) is None

    cached = store.cached_completed_for_channel("demo", 86400)
    assert cached.run_id == run.run_id
    assert store.cached_completed_for_channel("demo", 86400, schema_version=AIRBOT_RESULT_SCHEMA_VERSION) is None
    assert store.latest_completed_for_user(1).run_id == run.run_id

    store.close()


def test_bot_store_cleanup_removes_old_runs_and_raw_messages(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    run = store.create_run(user_id=1, chat_id=2, channel="demo")
    store.set_raw_messages(run.run_id, [{"text": "old"}])
    old = iso(utcnow() - timedelta(days=10))
    store._conn.execute("UPDATE bot_runs SET created_at = ?, updated_at = ? WHERE run_id = ?", (old, old, run.run_id))
    store._conn.commit()

    assert store.cleanup(3600) == 1
    assert store.get_run(run.run_id) is None
    assert store.raw_messages(run.run_id) == []

    store.close()


def test_bot_store_marks_active_runs_failed(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    run = store.create_run(user_id=1, chat_id=2, channel="demo")

    assert store.fail_active("restart") == 1
    failed = store.get_run(run.run_id)
    assert failed.status == "failed"
    assert failed.error == "restart"

    store.close()


def test_bot_store_leaderboard_orders_by_elo(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    low = store.create_run(user_id=1, chat_id=2, channel="demo")
    store.complete_run(low.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 2380})
    high = store.create_run(user_id=2, chat_id=3, channel="other")
    store.complete_run(high.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 2400})

    rows = store.leaderboard()

    assert [run.channel for run in rows] == ["other", "demo"]
    assert store.leaderboard_place("other") == 1
    assert store.leaderboard_place("@demo") == 2
    assert store.leaderboard_place("missing") is None

    low = store.get_run(low.run_id)
    high = store.get_run(high.run_id)
    fight = store.create_fight_run(3, 4, low, high, "topic")
    store.complete_fight_run(
        fight.fight_id,
        {
            "schema_version": AIRBOT_FIGHT_SCHEMA_VERSION,
            "topic": "topic",
            "final": {"channel": "demo", "decisive_round": 1},
        },
    )
    rows = store.leaderboard()
    assert [run.channel for run in rows] == ["demo", "other"]
    assert rows[0].result["base_air_elo"] == 2380
    assert rows[0].result["fight_rating_delta"] > 0

    store.close()


def test_bot_store_finds_leaderboard_profiles(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    first = store.create_run(user_id=1, chat_id=2, channel="maycrypto")
    store.complete_run(first.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 3000, "niche_tags": ["crypto"]})
    second = store.create_run(user_id=2, chat_id=3, channel="builder_lfg")
    store.complete_run(second.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 1200, "summary": "bot builder"})

    assert [run.channel for run in store.find_leaderboard("may crypto")] == ["maycrypto"]
    assert [run.channel for run in store.find_leaderboard("builder")] == ["builder_lfg"]
    assert store.find_leaderboard("missing") == []

    store.close()


def test_bot_store_latest_completed_for_user_filters_schema(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    old = store.create_run(user_id=1, chat_id=2, channel="old")
    store.complete_run(old.run_id, {"schema_version": 1})
    current = store.create_run(user_id=1, chat_id=2, channel="current")
    store.complete_run(current.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 1800})

    assert store.latest_completed_for_user(1, schema_version=AIRBOT_RESULT_SCHEMA_VERSION).run_id == current.run_id
    assert store.latest_completed_for_user(1, schema_version=999) is None

    store.close()


def test_bot_store_fight_session_lifecycle(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    left = store.create_run(user_id=1, chat_id=2, channel="left")
    right = store.create_run(user_id=2, chat_id=2, channel="right")
    store.complete_run(left.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 1200})
    store.complete_run(right.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 2200})

    started = store.start_fight_session(user_id=9, chat_id=10)
    assert started.status == "picking_first"

    second_step = store.set_fight_first(9, 10, left.run_id)
    assert second_step.status == "picking_second"
    assert second_step.first_run_id == left.run_id

    awaiting = store.set_fight_second(9, 10, right.run_id)
    assert awaiting.status == "awaiting_topic"
    assert awaiting.second_run_id == right.run_id
    assert store.awaiting_fight_topic(9, 10) is True

    running = store.set_fight_running(9, 10, "who promises louder")
    assert running.status == "running"
    assert running.topic == "who promises louder"
    assert store.awaiting_fight_topic(9, 10) is False

    store.clear_fight_session(9, 10)
    assert store.fight_session(9, 10) is None
    store.close()


def test_bot_store_fight_run_cache_stats_and_matches(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    left = store.create_run(user_id=1, chat_id=2, channel="left")
    right = store.create_run(user_id=2, chat_id=2, channel="right")
    store.complete_run(left.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 1200})
    store.complete_run(right.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 2200})
    left = store.get_run(left.run_id)
    right = store.get_run(right.run_id)

    fight = store.create_fight_run(9, 10, left, right, "who promises louder")
    assert fight.status == "queued"
    assert store.fight_queue_position(fight.fight_id) == 1
    store.set_fight_status(fight.fight_id, "running")
    completed = store.complete_fight_run(
        fight.fight_id,
        {
            "schema_version": AIRBOT_FIGHT_SCHEMA_VERSION,
            "topic": "who promises louder",
            "final": {"channel": "left"},
        },
    )

    cached = store.cached_fight("left", "right", "who promises louder", 86400, schema_version=AIRBOT_FIGHT_SCHEMA_VERSION)
    assert cached.fight_id == completed.fight_id
    left_stats = store.fight_stats("left")
    right_stats = store.fight_stats("right")
    assert {key: left_stats[key] for key in ("wins", "losses", "draws")} == {"wins": 1, "losses": 0, "draws": 0}
    assert {key: right_stats[key] for key in ("wins", "losses", "draws")} == {"wins": 0, "losses": 1, "draws": 0}
    assert completed.result["rating_change"]["left"]["delta"] > 0
    assert completed.result["rating_change"]["right"]["delta"] < 0
    assert left_stats["battle_elo"] == 1200 + left_stats["rating_delta"]
    assert right_stats["battle_elo"] == 2200 + right_stats["rating_delta"]
    assert store.channel_fights("left")[0].fight_id == completed.fight_id
    assert store.fail_active_fights("restart") == 0
    store.close()


def test_bot_store_boosts_fight_k_for_large_elo_gap(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    store.configure_fight_rating(max_gap=2000, boost_start=500, boost_max=2.0)
    left = store.create_run(user_id=1, chat_id=2, channel="left")
    right = store.create_run(user_id=2, chat_id=2, channel="right")
    store.complete_run(left.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 1000})
    store.complete_run(right.run_id, {"schema_version": AIRBOT_RESULT_SCHEMA_VERSION, "air_elo": 2500})
    left = store.get_run(left.run_id)
    right = store.get_run(right.run_id)

    fight = store.create_fight_run(9, 10, left, right, "topic")
    completed = store.complete_fight_run(
        fight.fight_id,
        {
            "schema_version": AIRBOT_FIGHT_SCHEMA_VERSION,
            "topic": "topic",
            "final": {"channel": "left", "decisive_round": 3},
        },
    )

    rating = completed.result["rating_change"]
    assert rating["gap"] == 1500
    assert rating["gap_multiplier"] > 1
    assert rating["k_factor"] > 48
    store.close()
