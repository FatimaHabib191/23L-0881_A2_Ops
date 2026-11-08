# VisionGuard: AI vs Real Image Classification

![CI](https://github.com/FatimaHabib191/23L-0881_A2_Ops/actions/workflows/ci.yml/badge.svg)

VisionGuard tells whether an uploaded image is AI-generated or real. It covers a TensorFlow/Keras model, MLflow tracking on DagsHub, DVC data versioning, a Flask site with a PostgreSQL request trail, Docker Compose, GitHub Actions, and a cloud model API.

Roll number 23L-0881. Project name `23L-0881_A2_Ops`.

| Link | Value |
| --- | --- |
| GitHub | https://github.com/FatimaHabib191/23L-0881_A2_Ops |
| DagsHub | https://dagshub.com/FatimaHabib191/23L-0881_A2_Ops |
| Model API | http://YOUR_VM_PUBLIC_IP:8000 |

## Quick start

```bash
cp .env.example .env              # then fill in the values
# put the champion at models/champion/model.keras (scripts/fetch_champion.py)
docker compose up -d --build
```

Open http://localhost:5000. Pages: `/` upload, `/result/<id>`, `/history`, `/health`.

## Architecture

```
Browser -> Flask web (localhost:5000) -> PostgreSQL (internal network, no published port)
                  |
                  | INFERENCE_MODE=remote: multipart POST with X-API-Key
                  v
           Model API on the cloud VM (port 8000) -> Keras champion, loaded once at start-up
```

The full step-by-step guide is in [RUNBOOK.md](RUNBOOK.md), and cloud steps are in [docs/DEPLOY_OCI.md](docs/DEPLOY_OCI.md).

## Layout

| Path | Purpose |
| --- | --- |
| `visionguard/` | Shared preprocessing, tf.data pipeline, models, training, evaluation, tracking |
| `scripts/` | One script per assignment step |
| `webapp/` | Flask site, validation, inference backends, database |
| `model_api/` | Key-protected inference API |
| `tests/` | Pytest suite that runs in CI |
| `ci_toy/` | Throwaway repository contents for the Actions tool lab |
