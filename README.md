# Customer Churn MLOps

[![CI](https://github.com/anveshsvemuri/customer-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/anveshsvemuri/customer-churn-mlops/actions/workflows/ci.yml)

A production-style, PII-free reference platform for telecom churn prediction. It demonstrates the full path from deterministic data generation and validation to model promotion, batch scoring, drift signals, and online inference.

## Architecture

```mermaid
flowchart LR
    A[Synthetic telecom data] --> B[Validation]
    B --> C[Deterministic training]
    C --> D[MLflow experiment]
    D --> E{ROC-AUC gate}
    E -->|pass| F[Model registry version]
    F --> G[Batch scoring]
    F --> H[FastAPI inference]
    G --> I[Drift monitoring]
```

## What this demonstrates

- Reproducible PII-free dataset generation
- Explicit feature and target validation
- Scikit-learn pipeline with deterministic split and evaluation
- MLflow experiment lineage for parameters, metrics, tags, and model artifacts
- Promotion gate before optional model-registry version creation
- Batch prediction and lightweight drift monitoring
- Validated FastAPI request/response contracts
- Docker packaging and GitHub Actions quality gates
- Responsible-use model card

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
churn-mlops train
uvicorn churn_mlops.api:app --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/predict -H 'content-type: application/json' -d '{
  "tenure_months": 6,
  "monthly_charges": 105,
  "support_tickets": 4,
  "contract_months": 1,
  "autopay": 0,
  "streaming_services": 2
}'
```

Generated datasets and model artifacts are intentionally excluded from Git. Run `make train` before building the Docker image.

## Track and register a model

MLflow is optional so the core API image stays small. A local SQLite backend gives
reproducible experiment history and registry support:

```bash
pip install -e '.[dev,tracking]'
churn-mlops train \
  --tracking-uri sqlite:///mlflow.db \
  --experiment-name telecom-churn \
  --registered-model-name telecom-churn
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The CLI validates the ROC-AUC promotion gate before it calls the tracker. Each run
records the seed, row count, feature contract, decision threshold, evaluation
metrics, PII classification, and serialized pipeline. Supplying a registered model
name creates a new model version only for a candidate that passed the gate. Local
MLflow databases and run artifacts are ignored by Git.

## Roadmap

- Evidently monitoring report and delayed-label evaluation
- Cloud object storage and Terraform deployment module
- Grounded retention-strategy copilot powered by the separate Data Platform Copilot project
