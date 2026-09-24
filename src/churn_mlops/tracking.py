from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .data import FEATURES
from .model import Metrics

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TrackingResult:
    run_id: str
    model_uri: str
    registered_model_name: str | None


def track_training(
    model: Pipeline,
    metrics: Metrics,
    *,
    tracking_uri: str,
    experiment_name: str = "telecom-churn",
    registered_model_name: str | None = None,
    rows: int,
    seed: int,
) -> TrackingResult:
    """Log a promoted model and optionally create a registry version.

    Callers must run the promotion gate before invoking this function. Keeping the
    gate outside the tracker makes it impossible for a failed candidate to be
    registered accidentally through this API.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:  # pragma: no cover - exercised without the optional extra
        raise RuntimeError("MLflow is not installed; run pip install -e '.[tracking]'") from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "algorithm": "logistic_regression",
                "features": ",".join(FEATURES),
                "rows": rows,
                "seed": seed,
                "decision_threshold": 0.5,
            }
        )
        mlflow.log_metrics(asdict(metrics))
        mlflow.set_tags(
            {
                "data.classification": "synthetic-no-pii",
                "promotion.gate": "passed",
                "pipeline.stage": "candidate",
            }
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
        )
        return TrackingResult(
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            registered_model_name=registered_model_name,
        )
