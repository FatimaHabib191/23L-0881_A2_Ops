#!/usr/bin/env bash
# Task 1.4 apply part. Run once at the project root, with the dataset in data/raw.
# Values are read from the git-ignored .env file:
# DAGSHUB_USER (or MLFLOW_TRACKING_USERNAME) and MLFLOW_TRACKING_PASSWORD (your token).
set -euo pipefail
set -a; source <(tr -d '\r' < .env); set +a   # tr handles Windows line endings
USER_NAME="${DAGSHUB_USER:-$MLFLOW_TRACKING_USERNAME}"
REPO="23L-0881_A2_Ops"

[[ -d .dvc ]] || dvc init
dvc remote add -f -d origin "https://dagshub.com/${USER_NAME}/${REPO}.dvc"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$USER_NAME"
dvc remote modify origin --local password "$MLFLOW_TRACKING_PASSWORD"
dvc add data/raw
git add data/raw.dvc .dvc/config .dvcignore
git commit -m "Track dataset with DVC" || true
dvc push
echo "Pointer file:"; cat data/raw.dvc
