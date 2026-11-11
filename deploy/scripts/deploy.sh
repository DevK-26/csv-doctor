#!/usr/bin/env bash
# CSV Doctor — deploy / update on the VM. Used by hand AND by GitHub Actions.
#   bash deploy/scripts/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "==> Pull latest code"
git pull --ff-only

echo "==> Build + start (detached)"
docker compose up -d --build

echo "==> Wait for engine health via Caddy"
for i in $(seq 1 30); do
  if curl -fsS http://localhost/api/health >/dev/null 2>&1; then
    echo "HEALTH OK:"
    curl -s http://localhost/api/health
    echo
    docker compose ps
    exit 0
  fi
  sleep 2
done

echo "HEALTH FAILED — last logs:"
docker compose logs --tail=80
exit 1
