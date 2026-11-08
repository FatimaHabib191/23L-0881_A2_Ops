#!/usr/bin/env bash
# Task 1.4 toy part. Without arguments it uses a local folder as the remote.
# For DagsHub, create an empty repository and run with:
#   DAGSHUB_USER=you DAGSHUB_REPO=dvc-toy DAGSHUB_TOKEN=token bash scripts/dvc_toy.sh
set -euo pipefail
TOY_DIR="${TOY_DIR:-$HOME/dvc_toy_23L-0881}"
rm -rf "$TOY_DIR"
mkdir -p "$TOY_DIR" && cd "$TOY_DIR"
step() { printf '\n$ %s\n' "$*"; "$@"; }

step git init -b main
step dvc init
git config user.name >/dev/null || git config user.name "Fatima Habib"
git config user.email >/dev/null || git config user.email "23l-0881@students.local"
git commit -qm "Initialise DVC"

mkdir data
printf 'id,value\n1,10\n2,20\n' > data/toy.csv
step dvc add data/toy.csv
step cat data/toy.csv.dvc
step git add data/toy.csv.dvc data/.gitignore
step git commit -m "Data v1: two rows"

printf '3,30\n' >> data/toy.csv
step dvc add data/toy.csv
step git add data/toy.csv.dvc
step git commit -m "Data v2: three rows"
step cat data/toy.csv

if [[ -n "${DAGSHUB_USER:-}" ]]; then
  git remote add origin "https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.git"
  git -c "http.extraheader=Authorization: Basic $(printf '%s:%s' "$DAGSHUB_USER" "$DAGSHUB_TOKEN" | base64 -w0)" push -u origin main
  dvc remote add -d origin "https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc"
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USER"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
else
  mkdir -p "$TOY_DIR-remote"
  step dvc remote add -d localremote "$TOY_DIR-remote"
fi
git add .dvc/config
git commit -qm "Configure DVC remote" || true
step dvc push

echo; echo "=== Return to the older version ==="
step git checkout HEAD~2 -- data/toy.csv.dvc
step dvc checkout data/toy.csv
step cat data/toy.csv

echo; echo "=== Back to the latest version, then delete local data and pull ==="
step git checkout HEAD -- data/toy.csv.dvc
step dvc checkout data/toy.csv
step rm -f data/toy.csv
step rm -rf .dvc/cache
step dvc pull
step cat data/toy.csv
step git log --oneline
