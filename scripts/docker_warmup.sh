#!/usr/bin/env bash
# Task 6.1: run each block, and take one screenshot per step.
export MSYS_NO_PATHCONV=1   # stops Git Bash on Windows rewriting container paths
set -x
mkdir -p "$HOME/warmup_html"
echo "<h1>23L-0881 Docker warm-up</h1>" > "$HOME/warmup_html/index.html"

# Run a container, map host port 8081 to container port 80, mount a volume.
docker run -d --name warmup-nginx -p 8081:80 -v "$HOME/warmup_html:/usr/share/nginx/html:ro" nginx:alpine
sleep 2
curl -s http://localhost:8081

# Read its logs.
docker logs warmup-nginx

# Run a command through a shell inside it. For an interactive shell: docker exec -it warmup-nginx sh
docker exec warmup-nginx sh -c 'hostname; whoami; ls /usr/share/nginx/html'

# List, then remove containers and images.
docker ps -a
docker stop warmup-nginx
docker rm warmup-nginx
docker images
docker rmi nginx:alpine
docker ps -a
