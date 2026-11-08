#!/usr/bin/env bash
# Task 6.4: compare the unoptimised and final image sizes and list the largest layers.
set -euo pipefail
docker build -q -f Dockerfile.unoptimized -t visionguard:before .
docker build -q -t visionguard:after .
docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' | grep -E 'REPOSITORY|visionguard:(before|after)'
for tag in before after; do
  echo; echo "Largest layers of visionguard:$tag"
  docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "visionguard:$tag" | sort -hr | head -5 | cut -c1-140
done
