#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOKEN="${TG_RADAR_API_TOKEN:-smoke-token}"
TMP_DIR="${TG_RADAR_SMOKE_TMP_DIR:-$ROOT_DIR/tmp/prod-smoke}"
SEED_CHANNEL="${TG_RADAR_SMOKE_SEED_CHANNEL:-telegram}"
KEYWORD="${TG_RADAR_SMOKE_KEYWORD:-telegram}"
WAIT_SECONDS="${TG_RADAR_SMOKE_WAIT_SECONDS:-180}"

choose_port() {
  local requested="$1"
  python3 - "$requested" <<'PY'
import socket
import sys

requested = int(sys.argv[1])

def can_bind(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()

if can_bind(requested):
    print(requested)
else:
    sock = socket.socket()
    sock.bind(("0.0.0.0", 0))
    print(sock.getsockname()[1])
    sock.close()
PY
}

export TG_RADAR_API_PORT="${TG_RADAR_API_PORT:-$(choose_port 18081)}"
export TG_RADAR_FRONTEND_PORT="${TG_RADAR_FRONTEND_PORT:-$(choose_port 18082)}"

API_URL="${TG_RADAR_SMOKE_API_URL:-http://127.0.0.1:$TG_RADAR_API_PORT}"
UI_URL="${TG_RADAR_SMOKE_UI_URL:-http://127.0.0.1:$TG_RADAR_FRONTEND_PORT}"

export TG_RADAR_API_TOKEN="$TOKEN"
export TG_RADAR_PUBLIC_METRICS="${TG_RADAR_PUBLIC_METRICS:-false}"
export VITE_TG_RADAR_API_TOKEN="${VITE_TG_RADAR_API_TOKEN:-$TOKEN}"
export TG_RADAR_AGENT_RUNTIME_PROVIDER="${TG_RADAR_AGENT_RUNTIME_PROVIDER:-rule_based}"
export TG_RADAR_LLM_BASE_URL="${TG_RADAR_LLM_BASE_URL:-}"
export TG_RADAR_LLM_API_KEY="${TG_RADAR_LLM_API_KEY:-}"

mkdir -p "$TMP_DIR"

COMPOSE=(docker compose)
AUTH_HEADER=(-H "authorization: Bearer $TOKEN")
JSON_HEADER=(-H "content-type: application/json")

SUMMARY=()

fail() {
  echo "FAIL $*" >&2
  exit 1
}

summary() {
  SUMMARY+=("$1=$2")
}

http_code() {
  local output="$1"
  shift
  curl -sS -o "$output" -w "%{http_code}" "$@"
}

json_value() {
  local file="$1"
  local path="$2"
  python3 - "$file" "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)
for part in sys.argv[2].split("."):
    if isinstance(data, list):
        data = data[int(part)]
    else:
        data = data.get(part)
print("" if data is None else data)
PY
}

json_check() {
  local file="$1"
  local expr="$2"
  python3 - "$file" "$expr" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)
safe = {"data": data, "bool": bool, "float": float, "int": int, "len": len}
if not eval(sys.argv[2], {"__builtins__": {}}, safe):
    raise SystemExit(1)
PY
}

wait_http_code() {
  local url="$1"
  local expect="$2"
  local out="$3"
  local deadline=$((SECONDS + WAIT_SECONDS))
  local code
  while (( SECONDS < deadline )); do
    code="$(http_code "$out" "$url" || true)"
    if [[ "$code" == "$expect" ]]; then
      return 0
    fi
    sleep 2
  done
  fail "$url expected $expect, got ${code:-none}"
}

wait_deep_health() {
  local out="$TMP_DIR/deep-health.json"
  local deadline=$((SECONDS + WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -fsS "$API_URL/core/health/deep" -o "$out" >/dev/null 2>&1 \
      && json_check "$out" "bool(data.get('ok') and data.get('db') and data.get('worker') and data.get('crawler'))"; then
      summary "deep_health" "ok"
      return 0
    fi
    sleep 2
  done
  cat "$out" >&2 2>/dev/null || true
  fail "deep health not ok"
}

post_json() {
  local url="$1"
  local data="$2"
  local out="$3"
  http_code "$out" -X POST "${JSON_HEADER[@]}" "${AUTH_HEADER[@]}" "$url" -d "$data"
}

echo "prod-smoke: compose up"
echo "prod-smoke: api=$API_URL ui=$UI_URL"
"${COMPOSE[@]}" up --build -d postgres api worker frontend

wait_http_code "$API_URL/health" "200" "$TMP_DIR/health.json"
summary "health" "200"
summary "api_url" "$API_URL"
summary "ui_url" "$UI_URL"

echo "prod-smoke: alembic upgrade head"
"${COMPOSE[@]}" exec -T api alembic upgrade head >"$TMP_DIR/alembic.log"
summary "migration" "head"

wait_deep_health

metrics_code="$(http_code "$TMP_DIR/metrics-no-token.txt" "$API_URL/metrics" || true)"
[[ "$metrics_code" == "401" ]] || fail "metrics without token expected 401, got $metrics_code"
summary "metrics_no_token" "$metrics_code"

unauth_payload='{"goal":"status","mode":"status","max_steps":2}'
unauth_code="$(http_code "$TMP_DIR/run-no-token.json" -X POST "${JSON_HEADER[@]}" "$API_URL/core/agent/runs" -d "$unauth_payload" || true)"
[[ "$unauth_code" == "401" ]] || fail "POST without token expected 401, got $unauth_code"
summary "post_no_token" "$unauth_code"

bad_code="$(http_code "$TMP_DIR/run-bad-token.json" -X POST "${JSON_HEADER[@]}" -H "authorization: Bearer bad-token" "$API_URL/core/agent/runs" -d "$unauth_payload" || true)"
[[ "$bad_code" == "401" ]] || fail "POST bad token expected 401, got $bad_code"
summary "post_bad_token" "$bad_code"

start_code="$(post_json "$API_URL/core/agent/runs" "$unauth_payload" "$TMP_DIR/run-start.json")"
[[ "$start_code" == "200" ]] || fail "start agent run expected 200, got $start_code"
run_id="$(json_value "$TMP_DIR/run-start.json" "run_id")"
run_start_status="$(json_value "$TMP_DIR/run-start.json" "status")"
[[ -n "$run_id" ]] || fail "missing run_id"
[[ "$run_start_status" == "queued" ]] || fail "agent run expected queued, got $run_start_status"
summary "run_start" "$run_start_status"

run_status=""
deadline=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < deadline )); do
  curl -fsS "${AUTH_HEADER[@]}" "$API_URL/core/agent/runs/$run_id" -o "$TMP_DIR/run-status.json"
  run_status="$(json_value "$TMP_DIR/run-status.json" "status")"
  [[ "$run_status" == "completed" ]] && break
  [[ "$run_status" == "failed" || "$run_status" == "cancelled" || "$run_status" == "timeout" ]] && break
  sleep 1
done
[[ "$run_status" == "completed" ]] || fail "agent run $run_id not completed, status=$run_status"
summary "run_completed" "$run_id"

eval_payload='{"cases":[{"name":"status","request":{"goal":"status","mode":"status","max_steps":2},"expected_tools":["core_status"]}]}'
eval_code="$(post_json "$API_URL/core/agent/evals/run" "$eval_payload" "$TMP_DIR/eval.json")"
[[ "$eval_code" == "200" ]] || fail "eval expected 200, got $eval_code"
json_check "$TMP_DIR/eval.json" "float(data.get('score') or 0) >= 1.0"
eval_score="$(json_value "$TMP_DIR/eval.json" "score")"
summary "eval_score" "$eval_score"

ingest_payload="$(python3 - "$SEED_CHANNEL" "$KEYWORD" <<'PY'
import json
import sys

print(json.dumps({
    "task_name": "prod-smoke-refresh",
    "keywords": [sys.argv[2]],
    "seed_channels": [sys.argv[1]],
    "depth": 0,
    "limit": 1,
    "pages_per_channel": 1,
    "crawl": True,
    "crawl_mode": "refresh",
    "freshness_days": 14,
    "max_live_crawl": 1,
}))
PY
)"
ingest_code="$(post_json "$API_URL/core/ingest" "$ingest_payload" "$TMP_DIR/ingest.json")"
[[ "$ingest_code" == "200" ]] || fail "refresh ingest expected 200, got $ingest_code"
refresh_stats="$(python3 - "$TMP_DIR/ingest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)
print(
    "candidates={candidates_found},crawled={channels_crawled},messages={messages},cached={cached},deleted={deleted},freshness={freshness}".format(
        candidates_found=data.get("candidates_found", 0),
        channels_crawled=data.get("channels_crawled", 0),
        messages=(data.get("messages_saved", 0) or 0),
        cached=(data.get("cached_messages_used", 0) or 0),
        deleted=(data.get("deleted_or_missing_marked", 0) or 0),
        freshness=data.get("freshness_days"),
    )
)
PY
)"
summary "refresh" "$refresh_stats"

wait_http_code "$UI_URL" "200" "$TMP_DIR/ui.html"
curl -fsS "$UI_URL/src/main.tsx" -o "$TMP_DIR/ui-main.tsx"
if grep -Fq "$VITE_TG_RADAR_API_TOKEN" "$TMP_DIR/ui-main.tsx"; then
  summary "ui_token" "injected"
else
  fail "VITE_TG_RADAR_API_TOKEN not visible in Vite transform"
fi
ui_proxy_code="$(http_code "$TMP_DIR/ui-status.json" "${AUTH_HEADER[@]}" "$UI_URL/api/core/status" || true)"
[[ "$ui_proxy_code" == "200" ]] || fail "frontend proxy /api/core/status expected 200, got $ui_proxy_code"
summary "ui_status_proxy" "$ui_proxy_code"

printf 'OK'
for item in "${SUMMARY[@]}"; do
  printf ' %s' "$item"
done
printf '\n'
