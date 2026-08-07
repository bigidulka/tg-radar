from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4


ACTIVE_STATUSES = {"queued", "fetching", "analyzing"}
ACTIVE_FIGHT_STATUSES = {"queued", "running"}
FIGHT_RATING_MIN = 0
FIGHT_RATING_MAX = 4000
FIGHT_RATING_BASE_K = 48
FIGHT_RATING_DRAW_K = 32
FIGHT_RATING_DECISIVE_STEP = 8
FIGHT_RATING_MAX_GAP = 2400
FIGHT_RATING_GAP_BOOST_START = 800
FIGHT_RATING_GAP_BOOST_MAX = 1.75


@dataclass(frozen=True)
class BotRun:
    run_id: str
    user_id: int
    chat_id: int
    channel: str
    status: str
    source: str
    message_count: int
    result: dict | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class BotFightSession:
    user_id: int
    chat_id: int
    status: str
    first_run_id: str | None
    second_run_id: str | None
    topic: str | None
    updated_at: datetime


@dataclass(frozen=True)
class BotFightRun:
    fight_id: str
    user_id: int
    chat_id: int
    left_run_id: str
    right_run_id: str
    left_channel: str
    right_channel: str
    topic: str
    topic_key: str
    pair_key: str
    status: str
    result: dict | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utcnow()).isoformat()


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class BotRunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.fight_rating_max_gap = FIGHT_RATING_MAX_GAP
        self.fight_rating_gap_boost_start = FIGHT_RATING_GAP_BOOST_START
        self.fight_rating_gap_boost_max = FIGHT_RATING_GAP_BOOST_MAX
        self._init()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def configure_fight_rating(self, max_gap: int, boost_start: int, boost_max: float) -> None:
        self.fight_rating_max_gap = max(0, min(int(max_gap), FIGHT_RATING_MAX))
        self.fight_rating_gap_boost_start = max(0, min(int(boost_start), FIGHT_RATING_MAX))
        self.fight_rating_gap_boost_max = max(1.0, float(boost_max))

    def _init(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_runs (
                    run_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0 NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_raw_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES bot_runs(run_id) ON DELETE CASCADE,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_user_screens (
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    awaiting_channel INTEGER DEFAULT 0 NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, chat_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_referrals (
                    user_id INTEGER PRIMARY KEY,
                    referred_by_user_id INTEGER,
                    start_payload TEXT,
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_fight_sessions (
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    first_run_id TEXT,
                    second_run_id TEXT,
                    topic TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, chat_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_fight_runs (
                    fight_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    left_run_id TEXT NOT NULL,
                    right_run_id TEXT NOT NULL,
                    left_channel TEXT NOT NULL,
                    right_channel TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    topic_key TEXT NOT NULL,
                    pair_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_runs_user_status ON bot_runs (user_id, status)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_runs_channel_status ON bot_runs (channel, status, updated_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_runs_created ON bot_runs (created_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_fight_runs_status ON bot_fight_runs (status, created_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_fight_runs_user_status ON bot_fight_runs (user_id, status)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_fight_runs_cache ON bot_fight_runs (pair_key, topic_key, status, updated_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_fight_runs_left ON bot_fight_runs (left_channel, status, updated_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_fight_runs_right ON bot_fight_runs (right_channel, status, updated_at)")
            self._conn.commit()

    def create_run(self, user_id: int, chat_id: int, channel: str, source: str = "tme") -> BotRun:
        run_id = "bot_" + uuid4().hex
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bot_runs (
                    run_id, user_id, chat_id, channel, status, source, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (run_id, user_id, chat_id, channel.lower(), source, now, now),
            )
            self._conn.commit()
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("bot run was not created")
        return run

    def set_status(self, run_id: str, status: str) -> BotRun:
        with self._lock:
            self._conn.execute(
                "UPDATE bot_runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (status, iso(), run_id),
            )
            self._conn.commit()
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("bot run not found")
        return run

    def set_raw_messages(self, run_id: str, messages: list[dict]) -> None:
        now = iso()
        with self._lock:
            self._conn.execute("DELETE FROM bot_raw_messages WHERE run_id = ?", (run_id,))
            self._conn.executemany(
                "INSERT INTO bot_raw_messages (run_id, message_json, created_at) VALUES (?, ?, ?)",
                [(run_id, json.dumps(message, ensure_ascii=False), now) for message in messages],
            )
            self._conn.execute(
                "UPDATE bot_runs SET message_count = ?, updated_at = ? WHERE run_id = ?",
                (len(messages), now, run_id),
            )
            self._conn.commit()

    def complete_run(self, run_id: str, result: dict) -> BotRun:
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                UPDATE bot_runs
                SET status = 'completed',
                    result_json = ?,
                    error = NULL,
                    updated_at = ?,
                    finished_at = ?
                WHERE run_id = ?
                """,
                (json.dumps(result, ensure_ascii=False), now, now, run_id),
            )
            self._conn.commit()
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("bot run not found")
        return run

    def fail_run(self, run_id: str, error: str) -> BotRun:
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                UPDATE bot_runs
                SET status = 'failed',
                    error = ?,
                    updated_at = ?,
                    finished_at = ?
                WHERE run_id = ?
                """,
                (error[:2000], now, now, run_id),
            )
            self._conn.commit()
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("bot run not found")
        return run

    def fail_active(self, error: str) -> int:
        marks = ",".join("?" for _ in ACTIVE_STATUSES)
        now = iso()
        params = [error[:2000], now, now, *sorted(ACTIVE_STATUSES)]
        with self._lock:
            cursor = self._conn.execute(
                f"""
                UPDATE bot_runs
                SET status = 'failed',
                    error = ?,
                    updated_at = ?,
                    finished_at = ?
                WHERE status IN ({marks})
                """,
                params,
            )
            self._conn.commit()
        return int(cursor.rowcount or 0)

    def get_run(self, run_id: str) -> BotRun | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM bot_runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def raw_messages(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT message_json FROM bot_raw_messages WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [json.loads(row["message_json"]) for row in rows]

    def active_run_for_user(self, user_id: int) -> BotRun | None:
        marks = ",".join("?" for _ in ACTIVE_STATUSES)
        params = [user_id, *sorted(ACTIVE_STATUSES)]
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM bot_runs WHERE user_id = ? AND status IN ({marks}) ORDER BY created_at DESC LIMIT 1",
                params,
            ).fetchone()
        return self._row_to_run(row) if row else None

    def active_count_for_user(self, user_id: int) -> int:
        marks = ",".join("?" for _ in ACTIVE_STATUSES)
        params = [user_id, *sorted(ACTIVE_STATUSES)]
        with self._lock:
            row = self._conn.execute(
                f"SELECT count(*) AS total FROM bot_runs WHERE user_id = ? AND status IN ({marks})",
                params,
            ).fetchone()
        return int(row["total"] if row else 0)

    def active_count(self) -> int:
        marks = ",".join("?" for _ in ACTIVE_STATUSES)
        with self._lock:
            row = self._conn.execute(
                f"SELECT count(*) AS total FROM bot_runs WHERE status IN ({marks})",
                sorted(ACTIVE_STATUSES),
            ).fetchone()
        return int(row["total"] if row else 0)

    def queue_position(self, run_id: str) -> int:
        with self._lock:
            run = self._conn.execute("SELECT created_at FROM bot_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not run:
                return 0
            row = self._conn.execute(
                """
                SELECT count(*) AS total
                FROM bot_runs
                WHERE status = 'queued' AND created_at <= ?
                """,
                (run["created_at"],),
            ).fetchone()
        return int(row["total"] if row else 0)

    def create_fight_run(self, user_id: int, chat_id: int, left: BotRun, right: BotRun, topic: str) -> BotFightRun:
        fight_id = "ft_" + uuid4().hex[:20]
        now = iso()
        left_channel = left.channel.lower()
        right_channel = right.channel.lower()
        topic_key = self._fight_topic_key(topic)
        pair_key = self._fight_pair_key(left_channel, right_channel)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bot_fight_runs (
                    fight_id, user_id, chat_id, left_run_id, right_run_id,
                    left_channel, right_channel, topic, topic_key, pair_key,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    fight_id,
                    user_id,
                    chat_id,
                    left.run_id,
                    right.run_id,
                    left_channel,
                    right_channel,
                    topic[:500],
                    topic_key,
                    pair_key,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        fight = self.get_fight_run(fight_id)
        if fight is None:
            raise RuntimeError("fight run was not created")
        return fight

    def get_fight_run(self, fight_id: str) -> BotFightRun | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM bot_fight_runs WHERE fight_id = ?", (fight_id,)).fetchone()
        return self._row_to_fight_run(row) if row else None

    def set_fight_status(self, fight_id: str, status: str) -> BotFightRun:
        with self._lock:
            self._conn.execute(
                "UPDATE bot_fight_runs SET status = ?, updated_at = ? WHERE fight_id = ?",
                (status, iso(), fight_id),
            )
            self._conn.commit()
        fight = self.get_fight_run(fight_id)
        if fight is None:
            raise RuntimeError("fight run not found")
        return fight

    def complete_fight_run(self, fight_id: str, result: dict) -> BotFightRun:
        now = iso()
        with self._lock:
            row = self._conn.execute("SELECT * FROM bot_fight_runs WHERE fight_id = ?", (fight_id,)).fetchone()
            if row is None:
                raise RuntimeError("fight run not found")
            payload = dict(result or {})
            if row["status"] != "completed" and not self._has_rating_change(payload):
                payload = self._add_fight_rating_change_locked(row, payload)
            self._conn.execute(
                """
                UPDATE bot_fight_runs
                SET status = 'completed',
                    result_json = ?,
                    error = NULL,
                    updated_at = ?,
                    finished_at = ?
                WHERE fight_id = ?
                """,
                (json.dumps(payload, ensure_ascii=False), now, now, fight_id),
            )
            self._conn.commit()
        fight = self.get_fight_run(fight_id)
        if fight is None:
            raise RuntimeError("fight run not found")
        return fight

    def fail_fight_run(self, fight_id: str, error: str) -> BotFightRun:
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                UPDATE bot_fight_runs
                SET status = 'failed',
                    error = ?,
                    updated_at = ?,
                    finished_at = ?
                WHERE fight_id = ?
                """,
                (error[:2000], now, now, fight_id),
            )
            self._conn.commit()
        fight = self.get_fight_run(fight_id)
        if fight is None:
            raise RuntimeError("fight run not found")
        return fight

    def fail_active_fights(self, error: str) -> int:
        marks = ",".join("?" for _ in ACTIVE_FIGHT_STATUSES)
        now = iso()
        params = [error[:2000], now, now, *sorted(ACTIVE_FIGHT_STATUSES)]
        with self._lock:
            cursor = self._conn.execute(
                f"""
                UPDATE bot_fight_runs
                SET status = 'failed',
                    error = ?,
                    updated_at = ?,
                    finished_at = ?
                WHERE status IN ({marks})
                """,
                params,
            )
            self._conn.commit()
        return int(cursor.rowcount or 0)

    def active_fight_count_for_user(self, user_id: int) -> int:
        marks = ",".join("?" for _ in ACTIVE_FIGHT_STATUSES)
        params = [user_id, *sorted(ACTIVE_FIGHT_STATUSES)]
        with self._lock:
            row = self._conn.execute(
                f"SELECT count(*) AS total FROM bot_fight_runs WHERE user_id = ? AND status IN ({marks})",
                params,
            ).fetchone()
        return int(row["total"] if row else 0)

    def fight_queue_position(self, fight_id: str) -> int:
        with self._lock:
            fight = self._conn.execute("SELECT created_at FROM bot_fight_runs WHERE fight_id = ?", (fight_id,)).fetchone()
            if not fight:
                return 0
            row = self._conn.execute(
                """
                SELECT count(*) AS total
                FROM bot_fight_runs
                WHERE status = 'queued' AND created_at <= ?
                """,
                (fight["created_at"],),
            ).fetchone()
        return int(row["total"] if row else 0)

    def cached_fight(
        self,
        left_channel: str,
        right_channel: str,
        topic: str,
        max_age_seconds: int,
        schema_version: int | None = None,
    ) -> BotFightRun | None:
        if max_age_seconds <= 0:
            return None
        cutoff = iso(utcnow() - timedelta(seconds=max_age_seconds))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM bot_fight_runs
                WHERE pair_key = ?
                  AND topic_key = ?
                  AND status = 'completed'
                  AND updated_at >= ?
                ORDER BY updated_at DESC
                LIMIT 20
                """,
                (self._fight_pair_key(left_channel, right_channel), self._fight_topic_key(topic), cutoff),
            ).fetchall()
        for row in rows:
            fight = self._row_to_fight_run(row)
            if schema_version is None or (fight.result or {}).get("schema_version") == schema_version:
                return fight
        return None

    def channel_fights(self, channel: str, limit: int = 10, offset: int = 0) -> list[BotFightRun]:
        clean = channel.lower().strip().lstrip("@")
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM bot_fight_runs
                WHERE status = 'completed'
                  AND result_json IS NOT NULL
                  AND (left_channel = ? OR right_channel = ?)
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (clean, clean, max(1, limit), max(0, offset)),
            ).fetchall()
        return [self._row_to_fight_run(row) for row in rows]

    def fight_stats(self, channel: str) -> dict[str, int]:
        clean = channel.lower().strip().lstrip("@")
        stats = {"wins": 0, "losses": 0, "draws": 0, "rating_delta": 0, "base_air_elo": 0, "battle_elo": 0}
        if not clean:
            return stats
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT result_json
                FROM bot_fight_runs
                WHERE status = 'completed'
                  AND result_json IS NOT NULL
                  AND (left_channel = ? OR right_channel = ?)
                """,
                (clean, clean),
            ).fetchall()
            base = self._latest_base_elo_for_channel_locked(clean)
        for row in rows:
            try:
                result = json.loads(row["result_json"])
            except Exception:
                continue
            final = result.get("final") if isinstance(result, dict) else {}
            winner = str((final if isinstance(final, dict) else {}).get("channel") or "").lower()
            if not winner:
                stats["draws"] += 1
            elif winner == clean:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            stats["rating_delta"] += self._rating_delta_from_result(clean, result)
        stats["base_air_elo"] = base
        stats["battle_elo"] = self._clamp_rating(base + stats["rating_delta"])
        return stats

    def fight_rating_delta(self, channel: str) -> int:
        clean = channel.lower().strip().lstrip("@")
        if not clean:
            return 0
        with self._lock:
            return self._fight_rating_delta_locked(clean)

    def run_with_fight_rating(self, run: BotRun) -> BotRun:
        if not run.result:
            return run
        return replace(run, result=self.result_with_fight_rating(run.channel, run.result))

    def result_with_fight_rating(self, channel: str, result: dict) -> dict:
        payload = dict(result or {})
        base = self._base_elo_from_result(payload)
        delta = self.fight_rating_delta(channel)
        current = self._clamp_rating(base + delta)
        payload["base_air_elo"] = base
        payload["fight_rating_delta"] = delta
        payload["battle_elo"] = current
        payload["air_elo"] = current
        payload["air_elo_label"] = f"{current} ELO"
        payload["fight_rating"] = {"base": base, "delta": delta, "current": current}
        return payload

    def latest_completed_for_user(self, user_id: int, schema_version: int | None = None) -> BotRun | None:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM bot_runs
                WHERE user_id = ? AND status = 'completed'
                ORDER BY updated_at DESC
                LIMIT 50
                """,
                (user_id,),
            ).fetchall()
        for row in rows:
            run = self._row_to_run(row)
            if schema_version is None or (run.result or {}).get("schema_version") == schema_version:
                return run
        return None

    def leaderboard(self, limit: int = 10, offset: int = 0, scan_limit: int = 500) -> list[BotRun]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM bot_runs
                WHERE status = 'completed' AND result_json IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(limit, scan_limit),),
            ).fetchall()
        latest_by_channel: dict[str, BotRun] = {}
        for row in rows:
            run = self._row_to_run(row)
            if run.channel not in latest_by_channel:
                latest_by_channel[run.channel] = run
        runs = list(latest_by_channel.values())
        runs = [self.run_with_fight_rating(run) for run in runs]
        runs.sort(key=lambda run: self._run_rank_value(run), reverse=True)
        start = max(0, offset)
        return runs[start : start + limit]

    def find_leaderboard(self, query: str, limit: int = 10, offset: int = 0, scan_limit: int = 1000) -> list[BotRun]:
        needle = self._search_key(query)
        if not needle:
            return self.leaderboard(limit=limit, offset=offset, scan_limit=scan_limit)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM bot_runs
                WHERE status = 'completed' AND result_json IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(limit, scan_limit),),
            ).fetchall()
        latest_by_channel: dict[str, BotRun] = {}
        for row in rows:
            run = self._row_to_run(row)
            if run.channel not in latest_by_channel:
                latest_by_channel[run.channel] = run
        scored = []
        for run in latest_by_channel.values():
            run = self.run_with_fight_rating(run)
            score = self._search_score(run, needle)
            if score > 0:
                scored.append((score, self._run_rank_value(run), run))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        start = max(0, offset)
        return [run for _, _, run in scored[start : start + limit]]

    def leaderboard_place(self, channel: str, scan_limit: int = 500) -> int | None:
        clean = channel.lower().strip().lstrip("@")
        if not clean:
            return None
        for index, run in enumerate(self.leaderboard(limit=scan_limit, scan_limit=scan_limit), start=1):
            if run.channel == clean:
                return index
        return None

    def set_screen(self, user_id: int, chat_id: int, message_id: int, awaiting_channel: bool = False) -> None:
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bot_user_screens (user_id, chat_id, message_id, awaiting_channel, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, chat_id) DO UPDATE SET
                    message_id = excluded.message_id,
                    awaiting_channel = excluded.awaiting_channel,
                    updated_at = excluded.updated_at
                """,
                (user_id, chat_id, message_id, int(awaiting_channel), now),
            )
            self._conn.commit()

    def set_awaiting_channel(self, user_id: int, chat_id: int, awaiting_channel: bool) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE bot_user_screens
                SET awaiting_channel = ?, updated_at = ?
                WHERE user_id = ? AND chat_id = ?
                """,
                (int(awaiting_channel), iso(), user_id, chat_id),
            )
            self._conn.commit()

    def screen_message_id(self, user_id: int, chat_id: int) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT message_id FROM bot_user_screens WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id),
            ).fetchone()
        return int(row["message_id"]) if row else None

    def awaiting_channel(self, user_id: int, chat_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT awaiting_channel FROM bot_user_screens WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id),
            ).fetchone()
        return bool(row and row["awaiting_channel"])

    def start_fight_session(self, user_id: int, chat_id: int) -> BotFightSession:
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bot_fight_sessions (user_id, chat_id, status, updated_at)
                VALUES (?, ?, 'picking_first', ?)
                ON CONFLICT(user_id, chat_id) DO UPDATE SET
                    status = excluded.status,
                    first_run_id = NULL,
                    second_run_id = NULL,
                    topic = NULL,
                    updated_at = excluded.updated_at
                """,
                (user_id, chat_id, now),
            )
            self._conn.commit()
        session = self.fight_session(user_id, chat_id)
        if session is None:
            raise RuntimeError("fight session was not created")
        return session

    def set_fight_first(self, user_id: int, chat_id: int, run_id: str) -> BotFightSession:
        self._upsert_fight_session(user_id, chat_id, "picking_second", first_run_id=run_id)
        session = self.fight_session(user_id, chat_id)
        if session is None:
            raise RuntimeError("fight session not found")
        return session

    def set_fight_second(self, user_id: int, chat_id: int, run_id: str) -> BotFightSession:
        current = self.fight_session(user_id, chat_id)
        self._upsert_fight_session(
            user_id,
            chat_id,
            "awaiting_topic",
            first_run_id=current.first_run_id if current else None,
            second_run_id=run_id,
        )
        session = self.fight_session(user_id, chat_id)
        if session is None:
            raise RuntimeError("fight session not found")
        return session

    def set_fight_running(self, user_id: int, chat_id: int, topic: str) -> BotFightSession:
        current = self.fight_session(user_id, chat_id)
        self._upsert_fight_session(
            user_id,
            chat_id,
            "running",
            first_run_id=current.first_run_id if current else None,
            second_run_id=current.second_run_id if current else None,
            topic=topic,
        )
        session = self.fight_session(user_id, chat_id)
        if session is None:
            raise RuntimeError("fight session not found")
        return session

    def clear_fight_session(self, user_id: int, chat_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM bot_fight_sessions WHERE user_id = ? AND chat_id = ?", (user_id, chat_id))
            self._conn.commit()

    def fight_session(self, user_id: int, chat_id: int) -> BotFightSession | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM bot_fight_sessions WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id),
            ).fetchone()
        return self._row_to_fight_session(row) if row else None

    def awaiting_fight_topic(self, user_id: int, chat_id: int) -> bool:
        session = self.fight_session(user_id, chat_id)
        return bool(session and session.status == "awaiting_topic")

    def record_referral(self, user_id: int, referred_by_user_id: int | None, start_payload: str | None) -> None:
        if referred_by_user_id == user_id:
            referred_by_user_id = None
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bot_referrals (user_id, referred_by_user_id, start_payload, first_seen_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    referred_by_user_id = COALESCE(bot_referrals.referred_by_user_id, excluded.referred_by_user_id),
                    start_payload = excluded.start_payload,
                    updated_at = excluded.updated_at
                """,
                (user_id, referred_by_user_id, start_payload, now, now),
            )
            self._conn.commit()

    def cached_completed_for_channel(
        self,
        channel: str,
        max_age_seconds: int,
        schema_version: int | None = None,
    ) -> BotRun | None:
        if max_age_seconds <= 0:
            return None
        cutoff = iso(utcnow() - timedelta(seconds=max_age_seconds))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM bot_runs
                WHERE channel = ? AND status = 'completed' AND updated_at >= ?
                ORDER BY updated_at DESC
                LIMIT 20
                """,
                (channel.lower(), cutoff),
            ).fetchall()
        for row in rows:
            run = self._row_to_run(row)
            if schema_version is None or (run.result or {}).get("schema_version") == schema_version:
                return run
        return None

    def user_success_in_cooldown(self, user_id: int, cooldown_seconds: int) -> BotRun | None:
        if cooldown_seconds <= 0:
            return None
        cutoff = iso(utcnow() - timedelta(seconds=cooldown_seconds))
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM bot_runs
                WHERE user_id = ? AND status = 'completed' AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id, cutoff),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def cleanup(self, ttl_seconds: int) -> int:
        cutoff = iso(utcnow() - timedelta(seconds=ttl_seconds))
        with self._lock:
            old = self._conn.execute("SELECT run_id FROM bot_runs WHERE created_at < ?", (cutoff,)).fetchall()
            run_ids = [row["run_id"] for row in old]
            if run_ids:
                marks = ",".join("?" for _ in run_ids)
                self._conn.execute(f"DELETE FROM bot_runs WHERE run_id IN ({marks})", run_ids)
            self._conn.execute("DELETE FROM bot_fight_sessions WHERE updated_at < ?", (cutoff,))
            self._conn.execute("DELETE FROM bot_fight_runs WHERE created_at < ?", (cutoff,))
            self._conn.commit()
        return len(run_ids)

    def _run_rank_value(self, run: BotRun) -> tuple[int, str]:
        result = run.result or {}
        value = result.get("battle_elo")
        if value is None:
            value = result.get("air_elo")
        if value is None:
            value = result.get("score")
        try:
            parsed = int(value)
        except Exception:
            parsed = 0
        return parsed, run.updated_at.isoformat()

    def _search_score(self, run: BotRun, needle: str) -> int:
        channel = self._search_key(run.channel)
        if channel == needle:
            return 100
        if channel.startswith(needle):
            return 90
        if needle in channel:
            return 80
        blob = self._search_key(json.dumps(run.result or {}, ensure_ascii=False))
        if needle in blob:
            return 40
        return 0

    def _search_key(self, value: object) -> str:
        return "".join(char.lower() for char in str(value or "") if char.isalnum())

    def _add_fight_rating_change_locked(self, row: sqlite3.Row, result: dict) -> dict:
        left_channel = str(row["left_channel"])
        right_channel = str(row["right_channel"])
        left_before = self._channel_current_rating_locked(left_channel, row["fight_id"], result)
        right_before = self._channel_current_rating_locked(right_channel, row["fight_id"], result)
        final = result.get("final") if isinstance(result.get("final"), dict) else {}
        winner = str(final.get("winner") or "").lower()
        winner_channel = str(final.get("channel") or "").lower().lstrip("@")
        if winner == "left" or winner_channel == left_channel:
            left_score = 1.0
        elif winner == "right" or winner_channel == right_channel:
            left_score = 0.0
        else:
            left_score = 0.5
        expected_left = 1 / (1 + 10 ** ((right_before - left_before) / 400))
        expected_right = 1 - expected_left
        gap = abs(left_before - right_before)
        k_factor = self._fight_k_factor(final, left_score, left_before, right_before)
        gap_multiplier = round(k_factor / max(1, self._fight_base_k_factor(final, left_score)), 3)
        raw_left_delta = round(k_factor * (left_score - expected_left))
        raw_right_delta = round(k_factor * ((1 - left_score) - expected_right))
        left_after = self._clamp_rating(left_before + raw_left_delta)
        right_after = self._clamp_rating(right_before + raw_right_delta)
        payload = dict(result)
        payload["rating_change"] = {
            "left": {
                "channel": left_channel,
                "before": left_before,
                "after": left_after,
                "delta": left_after - left_before,
                "outcome": self._fight_outcome(left_score),
            },
            "right": {
                "channel": right_channel,
                "before": right_before,
                "after": right_after,
                "delta": right_after - right_before,
                "outcome": self._fight_outcome(1 - left_score),
            },
            "expected": {"left": round(expected_left, 3), "right": round(expected_right, 3)},
            "gap": gap,
            "gap_multiplier": gap_multiplier,
            "k_factor": k_factor,
        }
        return payload

    def _has_rating_change(self, result: dict) -> bool:
        return isinstance((result or {}).get("rating_change"), dict)

    def _fight_base_k_factor(self, final: dict, left_score: float) -> int:
        if left_score == 0.5:
            return FIGHT_RATING_DRAW_K
        decisive_round = self._safe_int(final.get("decisive_round"), 3)
        if decisive_round <= 0:
            decisive_round = 3
        return FIGHT_RATING_BASE_K + max(0, 3 - min(decisive_round, 3)) * FIGHT_RATING_DECISIVE_STEP

    def _fight_k_factor(self, final: dict, left_score: float, left_before: int, right_before: int) -> int:
        base = self._fight_base_k_factor(final, left_score)
        if left_score == 0.5:
            return base
        gap = abs(left_before - right_before)
        start = min(self.fight_rating_gap_boost_start, self.fight_rating_max_gap)
        if gap <= start or self.fight_rating_gap_boost_max <= 1.0:
            return base
        span = max(1, self.fight_rating_max_gap - start)
        ratio = min(1.0, (gap - start) / span)
        multiplier = 1 + (self.fight_rating_gap_boost_max - 1) * ratio
        return max(1, round(base * multiplier))

    def _fight_outcome(self, score: float) -> str:
        if score >= 1.0:
            return "win"
        if score <= 0.0:
            return "loss"
        return "draw"

    def _channel_current_rating_locked(self, channel: str, exclude_fight_id: str | None = None, result: dict | None = None) -> int:
        base = self._latest_base_elo_for_channel_locked(channel)
        if base <= 0 and result:
            base = self._fighter_base_elo(result, channel)
        return self._clamp_rating(base + self._fight_rating_delta_locked(channel, exclude_fight_id))

    def _latest_base_elo_for_channel_locked(self, channel: str) -> int:
        row = self._conn.execute(
            """
            SELECT result_json
            FROM bot_runs
            WHERE channel = ? AND status = 'completed' AND result_json IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (channel.lower().strip().lstrip("@"),),
        ).fetchone()
        if not row:
            return 0
        try:
            result = json.loads(row["result_json"])
        except Exception:
            return 0
        return self._base_elo_from_result(result if isinstance(result, dict) else {})

    def _fight_rating_delta_locked(self, channel: str, exclude_fight_id: str | None = None) -> int:
        params: list[str] = [channel, channel]
        extra = ""
        if exclude_fight_id:
            extra = " AND fight_id != ?"
            params.append(exclude_fight_id)
        rows = self._conn.execute(
            f"""
            SELECT result_json
            FROM bot_fight_runs
            WHERE status = 'completed'
              AND result_json IS NOT NULL
              AND (left_channel = ? OR right_channel = ?)
              {extra}
            ORDER BY finished_at, updated_at
            """,
            params,
        ).fetchall()
        total = 0
        for row in rows:
            try:
                result = json.loads(row["result_json"])
            except Exception:
                continue
            if isinstance(result, dict):
                total += self._rating_delta_from_result(channel, result)
        return total

    def _rating_delta_from_result(self, channel: str, result: dict) -> int:
        rating = result.get("rating_change") if isinstance(result.get("rating_change"), dict) else {}
        for side in ("left", "right"):
            item = rating.get(side) if isinstance(rating, dict) else None
            data = item if isinstance(item, dict) else {}
            if str(data.get("channel") or "").lower().lstrip("@") == channel:
                return self._safe_int(data.get("delta"), 0)
        return 0

    def _fighter_base_elo(self, result: dict, channel: str) -> int:
        fighters = result.get("fighters") if isinstance(result.get("fighters"), list) else []
        clean = channel.lower().strip().lstrip("@")
        for value in fighters:
            item = value if isinstance(value, dict) else {}
            if str(item.get("channel") or "").lower().lstrip("@") == clean:
                return self._safe_int(item.get("air_elo"), 0)
        return 0

    def _base_elo_from_result(self, result: dict) -> int:
        value = result.get("base_air_elo")
        if value is None:
            value = result.get("air_elo")
        if value is None:
            value = result.get("score")
        return self._clamp_rating(self._safe_int(value, 0))

    def _clamp_rating(self, value: int) -> int:
        return min(max(self._safe_int(value, 0), FIGHT_RATING_MIN), FIGHT_RATING_MAX)

    def _safe_int(self, value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return default

    def _row_to_run(self, row: sqlite3.Row) -> BotRun:
        result_raw = row["result_json"]
        return BotRun(
            run_id=str(row["run_id"]),
            user_id=int(row["user_id"]),
            chat_id=int(row["chat_id"]),
            channel=str(row["channel"]),
            status=str(row["status"]),
            source=str(row["source"]),
            message_count=int(row["message_count"] or 0),
            result=json.loads(result_raw) if result_raw else None,
            error=row["error"],
            created_at=parse_dt(row["created_at"]) or utcnow(),
            updated_at=parse_dt(row["updated_at"]) or utcnow(),
            finished_at=parse_dt(row["finished_at"]),
        )

    def _row_to_fight_run(self, row: sqlite3.Row) -> BotFightRun:
        result_raw = row["result_json"]
        return BotFightRun(
            fight_id=str(row["fight_id"]),
            user_id=int(row["user_id"]),
            chat_id=int(row["chat_id"]),
            left_run_id=str(row["left_run_id"]),
            right_run_id=str(row["right_run_id"]),
            left_channel=str(row["left_channel"]),
            right_channel=str(row["right_channel"]),
            topic=str(row["topic"]),
            topic_key=str(row["topic_key"]),
            pair_key=str(row["pair_key"]),
            status=str(row["status"]),
            result=json.loads(result_raw) if result_raw else None,
            error=row["error"],
            created_at=parse_dt(row["created_at"]) or utcnow(),
            updated_at=parse_dt(row["updated_at"]) or utcnow(),
            finished_at=parse_dt(row["finished_at"]),
        )

    def _upsert_fight_session(
        self,
        user_id: int,
        chat_id: int,
        status: str,
        first_run_id: str | None = None,
        second_run_id: str | None = None,
        topic: str | None = None,
    ) -> None:
        now = iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bot_fight_sessions (
                    user_id, chat_id, status, first_run_id, second_run_id, topic, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, chat_id) DO UPDATE SET
                    status = excluded.status,
                    first_run_id = excluded.first_run_id,
                    second_run_id = excluded.second_run_id,
                    topic = excluded.topic,
                    updated_at = excluded.updated_at
                """,
                (user_id, chat_id, status, first_run_id, second_run_id, topic, now),
            )
            self._conn.commit()

    def _row_to_fight_session(self, row: sqlite3.Row) -> BotFightSession:
        return BotFightSession(
            user_id=int(row["user_id"]),
            chat_id=int(row["chat_id"]),
            status=str(row["status"]),
            first_run_id=row["first_run_id"],
            second_run_id=row["second_run_id"],
            topic=row["topic"],
            updated_at=parse_dt(row["updated_at"]) or utcnow(),
        )

    def _fight_topic_key(self, topic: str) -> str:
        return " ".join(str(topic or "").lower().split())[:300]

    def _fight_pair_key(self, left_channel: str, right_channel: str) -> str:
        return f"{left_channel.lower().strip().lstrip('@')}|{right_channel.lower().strip().lstrip('@')}"
