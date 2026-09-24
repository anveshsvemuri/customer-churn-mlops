from pathlib import Path

import mlflow

from churn_mlops.data import generate_customers
from churn_mlops.model import promote, train
from churn_mlops.tracking import track_training


def test_tracks_promoted_model_and_metrics(tmp_path: Path):
    model, metrics = train(generate_customers(1000, 42), seed=42)
    promote(metrics)

    result = track_training(
        model,
        metrics,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="test-churn",
        registered_model_name="test-telecom-churn",
        rows=1000,
        seed=42,
    )

    run = mlflow.get_run(result.run_id)
    assert run.data.metrics["roc_auc"] == metrics.roc_auc
    assert run.data.params["seed"] == "42"
    assert run.data.tags["data.classification"] == "synthetic-no-pii"
    assert result.model_uri.startswith("models:/m-") or result.model_uri.startswith("runs:/")
    versions = mlflow.MlflowClient().search_model_versions("name='test-telecom-churn'")
    assert len(versions) == 1
    assert versions[0].run_id == result.run_id
