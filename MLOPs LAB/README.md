
# MLOps-Lab

Assignment 1 — Software Engineering with ML FlowLab: Introduction to MLOps and Development Environment Setup.

This repository documents a self-configured MLOps development environment on Ubuntu, a standardized project structure, and a config-driven MLflow experiment tracking pipeline.

---

## Project structure

```
MLOps-Lab/
├── data/               # raw / processed datasets
├── notebooks/          # exploratory notebooks
├── src/                # training and pipeline code
│   ├── __init__.py
│   └── train.py
├── models/             # local model artifacts (gitignored)
├── reports/            # lab report, screenshots
│   └── screenshots/
├── configs/
│   └── config.yaml      # experiment/model/data hyperparameters
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Environment setup

Built and verified on Ubuntu (Linux-native, no Windows/WSL dependencies).

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
sudo snap install --classic code
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Project virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify tooling:

```bash
python3.11 --version
git --version
code --version
docker --version
mlflow --version
```

---

## Running the experiment

`src/train.py` trains a `LogisticRegression` model on the Iris dataset, with all hyperparameters externalized to `configs/config.yaml` — nothing hardcoded in the script.

```bash
source venv/bin/activate
python src/train.py
```

Each run logs to MLflow:

- **Parameters**: `model_name`, `C`, `max_iter`, `random_state`, `test_size`
- **Metrics**: `accuracy`, `precision`, `recall`, `f1_score`
- **Artifacts**: trained model (`model/`), `config.yaml`, `confusion_matrix.png`

To sweep hyperparameters, edit `C` in `configs/config.yaml` and rerun — each run is tracked separately under the same experiment for side-by-side comparison.

---

## Viewing results in MLflow UI

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Use the Runs tab to compare multiple runs' parameters and metrics; use Artifacts to inspect the logged model and confusion matrix per run.

---

## Deliverables

- [X] GitHub repository (this repo)
- [X] Lab report (PDF) — `reports/`
- [X] Installation verification screenshots — `reports/screenshots/`
- [X] Reflection document
- [X] Conceptual and challenge activity answers — included in lab report

---

## Tech stack

`Python 3.11` · `MLflow` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `Docker` · `Git`
