from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .batch import score_csv
from .data import generate_customers
from .model import promote, save_artifacts, train
from .monitoring import create_drift_report, save_drift_report
from .tracking import track_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Operate the reproducible churn platform")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and optionally track a model")
    training.add_argument("--rows", type=int, default=2000)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--output", type=Path, default=Path("artifacts"))
    training.add_argument("--tracking-uri", help="MLflow URI, for example sqlite:///mlflow.db")
    training.add_argument("--experiment-name", default="telecom-churn")
    training.add_argument(
        "--registered-model-name",
        help="Create a model version after the promotion gate passes (requires --tracking-uri)",
    )
    monitoring = commands.add_parser("monitor", help="Compare current features with a reference")
    monitoring.add_argument("--reference", type=Path, required=True)
    monitoring.add_argument("--current", type=Path, required=True)
    monitoring.add_argument("--output", type=Path, default=Path("artifacts/drift-report.json"))
    monitoring.add_argument("--threshold", type=float, default=0.25)
    monitoring.add_argument("--fail-on-drift", action="store_true")
    scoring = commands.add_parser("score", help="Score a CSV with bounded memory")
    scoring.add_argument("--model", type=Path, required=True)
    scoring.add_argument("--input", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--manifest", type=Path, required=True)
    scoring.add_argument("--threshold", type=float, default=0.5)
    scoring.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    if args.command == "score":
        manifest = score_csv(
            args.model,
            args.input,
            args.output,
            args.manifest,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        print(
            f"scored rows={manifest.rows_scored} churners={manifest.predicted_churners} "
            f"output={args.output} manifest={args.manifest}"
        )
        return
    if args.command == "monitor":
        report = create_drift_report(
            pd.read_csv(args.reference), pd.read_csv(args.current), args.threshold
        )
        save_drift_report(report, args.output)
        print(
            f"monitor status={report.status} drifted_features={report.drifted_features} "
            f"output={args.output}"
        )
        if args.fail_on_drift and report.status == "alert":
            raise SystemExit(2)
        return
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
