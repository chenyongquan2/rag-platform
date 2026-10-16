#!/usr/bin/env bash
# Start the local services of docker-compose.yml and wait until each one is healthy.
# Usage: scripts/dev_up.sh [timeout_seconds]    (default 300; works in Git Bash on Windows)
# A cold start takes one to two and a half minutes on the development machine (ES is slowest).
set -euo pipefail
cd "$(dirname "$0")/.."

timeout_s="${1:-300}"
deadline=$((SECONDS + timeout_s))

mkdir -p data/otel # the Collector writes traces.jsonl here
docker compose up -d

status_of() {
  local id
  id="$(docker compose ps -q "$1")"
  if [ -z "$id" ]; then
    echo missing
    return
  fi
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$id"
}

give_up() {
  echo "timeout after ${timeout_s}s: $1" >&2
  docker compose logs --tail 30 "$2" >&2
  exit 1
}

wait_healthy() {
  until [ "$(status_of "$1")" = healthy ]; do
    ((SECONDS < deadline)) || give_up "$1 is $(status_of "$1")" "$1"
    sleep 2
  done
  echo "healthy  $1"
}

wait_http() {
  until curl -fsS -m 3 -o /dev/null "$2"; do
    ((SECONDS < deadline)) || give_up "$1 does not answer at $2" "$1"
    sleep 2
  done
  echo "healthy  $1"
}

wait_healthy elasticsearch
wait_healthy redis-cache
wait_healthy jaeger
# The Collector image has no shell for a container healthcheck; ask its health_check extension.
wait_http otel-collector http://127.0.0.1:33133/

cat <<'INFO'
Elasticsearch  http://127.0.0.1:29200   (ES_URL)
Redis          redis://127.0.0.1:26379/0 (REDIS_URL)
OTLP gRPC      http://127.0.0.1:24317   (OTLP_ENDPOINT; OTLP HTTP on 24318)
Jaeger UI      http://127.0.0.1:36686
Trace file     data/otel/traces.jsonl
INFO
