# TG Radar

Agent-ready index for public Telegram channels.

## What it does

- Discovers public Telegram channels from keywords, seed channels, links, and mentions.
- Crawls public `https://t.me/s/<channel>` pages without Telethon in the default path.
- Parses messages, links, mentions, views, dates, and forward/repost hints.
- Stores crawl state in PostgreSQL.
- Indexes messages into Vespa for hybrid BM25 + vector search.
- Exposes FastAPI endpoints for agents.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

API:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/core/health/deep
curl -X POST http://localhost:8080/discover \
  -H 'content-type: application/json' \
  -d '{"keywords":["Яндекс нанимает AI агентов"],"seed_channels":["aostrikov_ai_agents"],"depth":1,"limit":20}'
```

Fast keyword ingest:

```bash
curl -X POST http://localhost:18081/ingest \
  -H 'content-type: application/json' \
  -d '{
    "keywords":["Яндекс нанимает AI агентов","тимлид из Яндекса ищет инженеров"],
    "seed_channels":["aostrikov_ai_agents","ya_jobs","yandexforml"],
    "depth":1,
    "limit":30,
    "pages_per_channel":1,
    "crawl":true
  }'
```

`/ingest` runs external `site:t.me/s "<keyword>"` discovery, graph expansion from seeds, stores channel candidates, crawls public `t.me/s` pages, writes messages to PostgreSQL, and feeds Vespa when it is available. If Vespa is down, `/search` falls back to PostgreSQL full-text search.

Autonomous keyword search task:

```bash
curl -X POST http://localhost:18081/tasks \
  -H 'content-type: application/json' \
  -d '{
    "name":"yandex-ai-hiring",
    "keywords":["Яндекс нанимает разработчиков AI агентов","тимлид Яндекс ищет инженеров"],
    "seed_channels":["aostrikov_ai_agents"],
    "depth":2,
    "limit":50,
    "pages_per_channel":1,
    "crawl":true,
    "interval_seconds":300,
    "enabled":true
  }'

curl -X POST http://localhost:18081/engine/start
curl http://localhost:18081/tasks/yandex-ai-hiring
```

The engine loop keeps running in the API process. Each enabled task is editable and consists of keywords, optional seed channels, crawl limits, graph depth, and an interval. On every due run it discovers new channel candidates first, then crawls messages as secondary processing.
In Docker Compose, `api` only serves HTTP and `worker` owns recurring tasks plus queued agent runs.

Collection agent:

```bash
export TG_RADAR_LLM_BASE_URL=http://localhost:8317/v1
export TG_RADAR_LLM_API_KEY=sk-local-dev-key
export TG_RADAR_LLM_MODEL=gpt-5.5
export TG_RADAR_AGENT_RUNTIME_PROVIDER=openai_compatible
export TG_RADAR_AGENT_COLLECTION_SYSTEM_PROMPT=collection_agent_system.md
export TG_RADAR_AGENT_CONTEXT_BUDGET=120000
export TG_RADAR_AGENT_ENABLED_TOOLS=core_status,discover,ingest

curl -X POST http://localhost:18081/agent/run \
  -H 'content-type: application/json' \
  -d '{
    "goal":"собери каналы про AI agent hiring",
    "keywords":["AI agent hiring","разработка AI агентов вакансии"],
    "seed_channels":["aostrikov_ai_agents"],
    "mode":"auto",
    "limit":40,
    "pages_per_channel":1
  }'

tg-radar-agent "мониторь AI agent hiring" \
  --keyword "AI agent hiring" \
  --seed-channel aostrikov_ai_agents \
  --task-name ai-agent-hiring \
  --mode create_task

tg-radar-agent-shell
```

`/agent/run` uses the shared agent kernel. Runtime, prompts, tools, phase policy, checkpoints, memory, and verification are configured through `TG_RADAR_AGENT_*` and `src/tg_radar/config_data/*.json`.
`POST /core/agent/runs` enqueues a DB-backed run. Poll `GET /core/agent/runs/{run_id}`, cancel with `POST /core/agent/runs/{run_id}/cancel`, retry terminal runs with `POST /core/agent/runs/{run_id}/retry`.

Runtime, prompt file, context budget, recent step/task limits, enabled tools, model temperature, and model output token limit are configured through `TG_RADAR_AGENT_*`.
If `TG_RADAR_LLM_BASE_URL` is empty or `TG_RADAR_AGENT_RUNTIME_PROVIDER` is not `openai_compatible`, the collection agent uses the deterministic runtime.
`tg-radar-agent-shell` is an interactive CLI for local testing. Use `/help`, `/keyword ...`, `/seed ...`, `/run ...`, `/status`, `/config`, `/quit`.

Content factory prompts live in `src/tg_radar/prompts/` and can be overridden with `TG_RADAR_CONTENT_PROMPT_DIR`, `TG_RADAR_CONTENT_SYSTEM_PROMPT`, `TG_RADAR_CONTENT_OUTLINE_TASK_PROMPT`, and `TG_RADAR_CONTENT_DRAFT_TASK_PROMPT`.
Domain rules live in `src/tg_radar/config_data/*.json`; override all rule JSON files with `TG_RADAR_RULES_DIR`.

Operator quality classification (off by default):

```bash
TG_RADAR_OPERATOR_QUALITY_ENABLED=true
TG_RADAR_OPERATOR_QUALITY_MESSAGE_LIMIT=40
TG_RADAR_OPERATOR_QUALITY_MIN_MESSAGES=5
TG_RADAR_OPERATOR_QUALITY_RECHECK_INTERVAL_DAYS=7
```

When enabled and `TG_RADAR_LLM_BASE_URL`/`TG_RADAR_LLM_API_KEY` are set, every crawl runs a single LLM call (see `src/tg_radar/operator_quality.py`) that reads a channel's recent messages and tags a niche category (`novel`/`saturated`/`hobby`/`mixed`/`unclear`) and an income-authenticity verdict, instead of keyword matching. Results are cached on the channel until `TG_RADAR_OPERATOR_QUALITY_RECHECK_INTERVAL_DAYS` elapses and surface on `GET /channels/{username}` (full detail with evidence) and `GET /candidates` (category/tags/confidence summary).

Speed and dedupe knobs:

```bash
TG_RADAR_MAX_KEYWORD_EXPANSIONS=6
TG_RADAR_DISCOVERY_QUERY_CACHE_TTL_SECONDS=3600
TG_RADAR_EXTERNAL_SEARCH_TIER_CONCURRENCY=4
TG_RADAR_OPEN_WEBSEARCH_URL=http://open-websearch:3000
TG_RADAR_OPEN_WEBSEARCH_ENGINES=startpage,duckduckgo
TG_RADAR_CRAWL_COOLDOWN_SECONDS=1800
TG_RADAR_INGEST_CONCURRENCY=8
TG_RADAR_REQUESTS_PER_SECOND=2
```

- Query cache prevents repeated external searches for the same expanded keyword.
- Tiered search queries fast public/API sources first and only falls back to slow Google/Jina/DDG if nothing is found.
- Crawl cooldown prevents recrawling the same channel on every task loop.
- Existing messages are skipped before edge extraction and Vespa feed.
- `crawl_mode=refresh` updates recent message text/views/links and marks missing recent messages as `deleted_or_missing`; `crawl_mode=backfill` walks history.
- News/digest agent requests default to `freshness_days=14` unless explicitly set.
- For fast production discovery, use `TG_RADAR_BRAVE_API_KEY`, `TG_RADAR_SERPAPI_KEY`, or `TG_RADAR_SEARXNG_ENDPOINT`; keyless Google/Jina/DDG fallback is slower.

Auth:

```bash
TG_RADAR_API_TOKEN=change-me
TG_RADAR_PUBLIC_METRICS=false
VITE_TG_RADAR_API_TOKEN=change-me
```

When `TG_RADAR_API_TOKEN` is set, mutating endpoints require `Authorization: Bearer <token>`. `/health` stays public; `/metrics` is public only when `TG_RADAR_PUBLIC_METRICS=true`.

## Prod smoke

Run the prod-like smoke before deploy:

```bash
scripts/prod_smoke.sh
```

The script exports `TG_RADAR_API_TOKEN=smoke-token`, `TG_RADAR_PUBLIC_METRICS=false`, `VITE_TG_RADAR_API_TOKEN=smoke-token`, starts Docker Compose, runs `alembic upgrade head` in the API container, then checks auth, worker queue, eval, refresh crawl, and frontend proxy.
If `18081` or `18082` is busy, it picks a free host port and prints `api_url` / `ui_url`.

Expected compact output:

```text
OK health=200 api_url=http://127.0.0.1:... ui_url=http://127.0.0.1:... migration=head deep_health=ok metrics_no_token=401 post_no_token=401 post_bad_token=401 run_start=queued run_completed=run_... eval_score=1.0 refresh=candidates=...,crawled=...,messages=...,cached=...,deleted=...,freshness=14 ui_token=injected ui_status_proxy=200
```

Manual UI smoke:

```bash
TG_RADAR_API_TOKEN=smoke-token VITE_TG_RADAR_API_TOKEN=smoke-token docker compose up --build -d postgres api worker frontend
open http://127.0.0.1:18082
```

In the UI, run `/status`. It should enqueue a run and finish without `401`; timeline should show `queued -> running -> completed`.

Migration discipline:

- Prod path is always `alembic upgrade head`.
- `init_db()` is only a local/dev compatibility guard for existing bootstrap databases; it does not replace migrations.
- `0001_prod_hardening` is intentionally broad for this release-hardening step; split into incremental migrations later.

Troubleshooting:

- DB schema mismatch: run `alembic upgrade head`.
- `worker=false` in `/core/health/deep`: check `docker compose logs worker`.
- UI `401`: set `VITE_TG_RADAR_API_TOKEN=smoke-token` and restart `frontend`.

Discovery sources:

- `telemetr_public`: anonymous Telemetr/tgadsspy public API, no key, default on.
- `lyzem`: public Lyzem HTML search, no key, default on.
- `open_websearch`: Aas-ee/open-webSearch MCP/local daemon, no key, `POST /search`.
- `telethon`: optional read-only user-account OSINT adapter, off by default.
- `tgstat`: TGStat posts search API, requires `TG_RADAR_TGSTAT_TOKEN`.
- `brave`, `serpapi`, `searxng`: optional faster search backends.
- `jina_google`, `duckduckgo`, `google`, `bing`: keyless fallback.

Open WebSearch:

- GitHub: https://github.com/Aas-ee/open-webSearch
- Docker compose exposes it on `http://127.0.0.1:3210` and the API uses `http://open-websearch:3000`.
- Direct local check:

```bash
curl http://127.0.0.1:3210/health
curl -X POST http://127.0.0.1:3210/search \
  -H 'content-type: application/json' \
  -d '{"query":"site:t.me web3 job","limit":5,"engines":["startpage","duckduckgo"]}'
```

Telethon user-channel OSINT:

```bash
TG_RADAR_TELETHON_ENABLED=true
TG_RADAR_TELETHON_HOME=/home/fsdf1234/Projects/pi-telethon/.pi-telethon
TG_RADAR_TELETHON_PROFILE=bigidulka2
TG_RADAR_TELETHON_USER_SEARCH_ENABLED=true
```

Available read-only paths:

- `POST /telethon/search/posts`: official user-channel `channels.searchPosts`.
- `POST /telethon/search/global`: global MTProto search with `broadcasts_only`, `groups_only`, `users_only`.
- `POST /telethon/search/contacts`: public Telegram contacts search for channels, chats, and users.
- `POST /telethon/dialogs`: account dialogs lookup.
- `POST /telethon/entity`: resolve user/channel/chat entity.
- `POST /telethon/full-info`: full channel/chat/user info, including about and pinned message when accessible.
- `POST /telethon/messages`: channel/chat message search with sender enrichment.
- `POST /telethon/forum-topics`: forum topic list/search for groups with topics.
- `POST /telethon/thread-messages`: messages from a forum/comment thread by top message/topic id.
- `POST /telethon/participants`: participant lookup where account access allows it.
- `POST /telethon/participant-profiles`: participant profile/bio scan where account access allows it.
- `POST /telethon/common-chats`: common chats with a user.
- `POST /telethon/similar`: official similar channel recommendations.
- `POST /telethon/inspect`: combined entity/messages/participants/similar probe.
- `POST /telethon/candidates`: recursive candidate extraction from about, pinned, messages, hidden URLs, participants, user bios, similar channels, and linked chats.

When `TG_RADAR_TELETHON_USER_SEARCH_ENABLED=true`, Telethon user-channel/chat search is also part of `/discover`, `/ingest`, `/expand`, and the autonomous task engine.

UI:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:18082`.

## Local tests

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Useful focused checks:

```bash
.venv/bin/python -m pytest tests/test_agent_core.py tests/test_native_tools.py
.venv/bin/python -m pytest tests/test_agent_factory.py tests/test_openai_transport.py tests/test_langgraph_runtime.py
.venv/bin/python -m pytest tests/test_api_routes.py tests/test_cli.py
.venv/bin/python -m pytest tests/test_architecture.py
```

Agent shell smoke test:

```bash
docker compose up -d postgres
tg-radar-agent-shell
/status
/quit
```

Agent eval smoke test:

```bash
cat > tmp/agent-eval.json <<'JSON'
{
  "cases": [
    {
      "name": "status",
      "request": {"goal": "status", "mode": "status", "max_steps": 2},
      "expected_tools": ["core_status"]
    }
  ]
}
JSON

tg-radar-agent-eval tmp/agent-eval.json
curl -X POST http://localhost:18081/core/agent/evals/run -H 'content-type: application/json' -d '{}'
```

Architecture guards are in `tests/test_architecture.py`; they verify external prompt/config files, no runtime Cyrillic constants, no project regex usage, core/storage separation, and runtime transport boundaries.

Migrations and backup:

```bash
alembic upgrade head
docker compose exec postgres pg_dump -U tg_radar -d tg_radar -Fc -f /tmp/tg_radar.dump
docker compose exec postgres pg_restore -U tg_radar -d tg_radar --clean --if-exists /tmp/tg_radar.dump
curl http://localhost:18081/core/health/deep
```

For production, run `alembic upgrade head` before starting traffic. Do not rely on `init_db()` as a migration runner.

## Project structure

```text
src/tg_radar/
  agent_core/      typed actions, loop, phase policy, approvals, compaction, verification
  agent_context/   context modules: task state, repo map, retrieval, diff, memory
  agent_runtime/   rule-based, OpenAI chat/responses, LangGraph-style adapters
  agent_tools/     collection tools, native dev tools, MCP bridge
  agent_state/     persistent agent event store
  agent_eval/      JSON regression eval runner
  api_routes/      FastAPI routers grouped by surface
  config_data/     JSON rules/manifests
  prompts/         prompt files
```

Top-level compatibility shims like `tg_radar.api_admin` still re-export the new `tg_radar.api_routes.*` modules.

## Notes

- `curl_cffi` is used for browser-compatible public HTTP fetches, not aggressive bypassing.
- Telethon/MTProto is optional, read-only, and disabled in the default crawler.
- Vespa is the source of truth for search ranking; PostgreSQL is the source of truth for crawl state.
