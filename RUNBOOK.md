# VisionGuard runbook

Run these in order. Each block says which sub-task and screenshot it produces. Keep your username, the date and the project name visible in every screenshot. Use Python 3.12.

## 0. Accounts and project

1. Create a GitHub repository named `23L-0881_A2_Ops`.
2. On DagsHub choose **Create > Connect a repository > GitHub** and pick it. Use the same name.
3. On DagsHub open **User settings > Tokens** and create a token.

```bash
cd 23L-0881_A2_Ops
git init -b main
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-serve.txt -r requirements-train.txt -r requirements-dev.txt
cp .env.example .env        # fill in the DagsHub lines
git add . && git commit -m "VisionGuard project"
git remote add origin https://github.com/FatimaHabib191/23L-0881_A2_Ops.git
git push -u origin main
```

Download the Kaggle dataset and extract it so that you have `data/raw/AiArtData/...` and `data/raw/RealArt/...`.

**GPU option:** on Kaggle or Colab, clone the repository, keep the preinstalled TensorFlow, run `pip install -r requirements-train.txt`, and set the three `MLFLOW_*` values as secrets. Training on a laptop CPU also works, but the tuner and seven experiments will take hours.

## Task 1

| Sub-task | Command | Evidence |
| --- | --- | --- |
| 1.1 | `python scripts/check_env.py` | Screenshot of the terminal |
| 1.2 | `python scripts/audit_dataset.py` | `outputs/audit/summary.txt` |
| 1.2, splits | `python scripts/make_splits.py` | Terminal output |
| 1.3, 2.1 figure | `python scripts/eda.py` | `outputs/eda/*.png` |
| 1.4 toy | `DAGSHUB_USER=you DAGSHUB_REPO=dvc-toy DAGSHUB_TOKEN=token bash scripts/dvc_toy.sh` (create the empty `dvc-toy` repository on DagsHub first) | Terminal, and the DagsHub commits page |
| 1.4 apply | `bash scripts/dvc_apply.sh` then `git push` | Terminal, and the DVC files in DagsHub |

## Task 3.1 and 3.2 tool labs

```bash
python scripts/mlflow_toy.py --backend local --lr 0.001
python scripts/mlflow_toy.py --backend local --lr 0.01
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db      # open http://127.0.0.1:5000
python scripts/mlflow_toy.py --lr 0.001                        # goes to DagsHub through .env
git check-ignore -v .env                                       # proof for 3.2
```

Select both runs in the `toy-mnist` experiment and click **Compare** for the screenshot.

## Task 2 and 3: training

Commit first so the `git_commit` tag is meaningful. Do not evaluate on the test set until the model choice is final.

```bash
python scripts/train.py --run-name B0-cnn --model cnn        # 2.2, 2.4, baseline
python scripts/run_experiments.py                            # 2.3, 2.4, 3.3 manual, 3.5
python scripts/tuner_toy.py                                  # 2.7 toy
python scripts/tune_visionguard.py --trials 10               # 2.7 apply
python scripts/compare_runs.py                               # 3.5 table and chart
python scripts/train.py --run-name AUTO-autolog --logging autolog   # 3.3 autolog
```

Evidence is in `outputs/runs/<run>/`: `model_summary.txt`, `layer_table.csv`, `curves_all.png`, `curves_stage1.png`, `curves_stage2.png`, `trainable_layers.json`, and `val_result.json` (best epoch and threshold).

Replace `E1-base` below with the best run printed by `compare_runs.py`. Each command refuses to run twice for the same run.

```bash
python scripts/evaluate_final.py --run-name B0-cnn           # 2.5
python scripts/evaluate_final.py --run-name E1-base          # 2.5
python scripts/evaluate_final.py --run-name AUTO-autolog     # final metrics for 3.3 only
python scripts/error_analysis.py --run-name E1-base          # 2.6, then correct each hypothesis by eye
python scripts/gradcam.py --run-name E1-base                 # bonus
```

## Task 3.6 registry

```bash
python scripts/register_model.py --run-name B0-cnn                 # version 1
python scripts/register_model.py --run-name E1-base --champion     # version 2, alias champion
python scripts/load_champion.py path/to/an_image.jpg               # terminal output for the report
python scripts/fetch_champion.py                                   # writes models/champion/model.keras
```

Copy the `threshold` value from `outputs/runs/E1-base/val_result.json` to `MODEL_THRESHOLD` in `.env`, and set `MODEL_VERSION=VisionGuard:v2`.

## Task 4 and 5: Flask

```bash
export INFERENCE_MODE=local DATABASE_URL=sqlite:///visionguard.db FLASK_DEBUG=0
flask --app webapp:create_app run --no-debug
```

For the 4.1 console evidence, run this with one known AI image and one known real image:

```bash
python -c "
from webapp.inference import LocalPredictor
import os, sys
p = LocalPredictor('models/champion/model.keras', float(os.environ['MODEL_THRESHOLD']), os.environ['MODEL_VERSION'])
for f in sys.argv[1:]: print(f, p.predict(open(f,'rb').read()))
" known_ai.jpg known_real.jpg
```

Screenshots for 4.2: a `.txt` file (415), a file over 8 MB (413), and a text file renamed to `.jpg` (422). Screenshots for 4.4 and 5: `/health` and `/history`.

## Task 6: Docker

```bash
bash scripts/docker_warmup.sh                 # 6.1
docker compose up -d --build                  # 6.2, 6.3
bash scripts/ops_demo.sh                      # 6.2 user, 6.3 closed port, 6.5 operations
bash scripts/image_report.sh                  # 6.4
```

**6.5 failure demo: wrong database password**

```bash
# Edit POSTGRES_PASSWORD in .env to a wrong value
docker compose up -d --no-deps --force-recreate web
docker compose ps
docker compose logs web | grep -E "FATAL|password authentication"
# Restore the real password in .env
docker compose up -d --no-deps --force-recreate web
```

The database keeps the password from its first start, so only the web service sees the changed value.

**6.6 Actions:** copy the contents of `ci_toy/`, including its hidden `.github` folder, to a new throwaway GitHub repository and push. Follow the comment in `test_calculator.py` for a red run, then a green run. The project workflow runs when you push this repository. Replace `FatimaHabib191` in the README badge.

## Task 7 and 8

Follow [docs/DEPLOY_OCI.md](docs/DEPLOY_OCI.md).

## Before submitting

- Replace every `[FILL]` in the report, and put your real results in it.
- Record the graded commit: `git rev-parse HEAD`.
- Name the PDF `23L-0881_VisionGuard.pdf`.
- Make the DagsHub project public or share it with your instructor.
- Check the API from a network other than your own.
