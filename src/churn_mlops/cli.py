from __future__ import annotations

import argparse
from pathlib import Path

from .data import generate_customers
from .model import promote, save_artifacts, train
from .tracking import track_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reproducible churn model")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--tracking-uri", help="MLflow URI, for example sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="telecom-churn")
    parser.add_argument(
        "--registered-model-name",
        help="Create a model version after the promotion gate passes (requires --tracking-uri)",
    )
    args = parser.parse_args()
    if args.registered_model_name and not args.tracking_uri:
        parser.error("--registered-model-name requires --tracking-uri")
    frame = generate_customers(args.rows, args.seed)
    model, metrics = train(frame, args.seed)
    promote(metrics)
    save_artifacts(model, metrics, args.output)
    tracking = None
    if args.tracking_uri:
        tracking = track_training(
            model,
            metrics,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            registered_model_name=args.registered_model_name,
            rows=len(frame),
            seed=args.seed,
        )
    message = f"trained rows={len(frame)} roc_auc={metrics.roc_auc} output={args.output}"
    if tracking:
        message += f" mlflow_run={tracking.run_id} model_uri={tracking.model_uri}"
    print(message)


if __name__ == "__main__":
    main()
