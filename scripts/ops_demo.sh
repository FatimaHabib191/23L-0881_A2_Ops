#!/usr/bin/env bash
# Task 6.5 and the 6.2/6.3 evidence commands. Run from the project root after docker compose up -d --build.
export MSYS_NO_PATHCONV=1   # stops Git Bash on Windows rewriting container paths
set -x
docker compose ps
docker compose stop web
docker compose start web
docker compose restart web
docker compose logs --tail=20 web
docker compose logs --tail=20 db
docker compose exec web sh -c 'id; pwd; python --version'
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT id, created_at, status, label, confidence, model_version FROM request_log ORDER BY id DESC LIMIT 5;"'

# Task 6.3: the published-port list is empty, and a host connection to 5432 fails.
docker compose port db 5432
docker compose ps db --format '{{.Name}} ports={{.Ports}}'
timeout 3 bash -c '</dev/tcp/127.0.0.1/5432' && echo "PORT OPEN" || echo "Port 5432 is closed from the host"
# The web container still reaches the database by service name:
docker compose exec web python -c "import socket; socket.create_connection(('db', 5432), 3); print('db:5432 reachable inside the network')"
