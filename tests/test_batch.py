import json
from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.batch import score_csv
from churn_mlops.data import generate_customers
from churn_mlops.model import save_artifacts, train


def _trained_files(tmp_path: Path) -> tuple[Path, Path]:
    frame = generate_customers(500, 42)
    model, metrics = train(frame)
    save_artifacts(model, metrics, tmp_path / "artifacts")
    input_path = tmp_path / "customers.csv"
    frame.drop(columns=["churned"]).assign(email="not-exported@example.com").to_csv(
        input_path, index=False
    )
    return tmp_path / "artifacts" / "model.joblib", input_path


def test_batch_scoring_is_chunked_atomic_and_privacy_safe(tmp_path: Path):
    model_path, input_path = _trained_files(tmp_path)
    output_path = tmp_path / "nested" / "predictions.csv"
    manifest_path = tmp_path / "nested" / "manifest.json"

    manifest = score_csv(
        model_path, input_path, output_path, manifest_path, batch_size=73, threshold=0.4
    )
    output = pd.read_csv(output_path)
    saved_manifest = json.loads(manifest_path.read_text())

    assert output.columns.tolist() == [
        "customer_id",
        "churn_probability",
        "churn_prediction",
    ]
    assert len(output) == manifest.rows_scored == 500
    assert output["churn_prediction"].sum() == manifest.predicted_churners
    assert saved_manifest["input_sha256"] == manifest.input_sha256
    assert saved_manifest["model_sha256"] == manifest.model_sha256
    assert saved_manifest["threshold"] == 0.4
    assert not output_path.with_suffix(".csv.tmp").exists()


def test_batch_scoring_rejects_invalid_configuration(tmp_path: Path):
    model_path, input_path = _trained_files(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        score_csv(model_path, input_path, tmp_path / "out.csv", tmp_path / "run.json", batch_size=0)
    with pytest.raises(ValueError, match="threshold"):
        score_csv(
            model_path,
            input_path,
            tmp_path / "out.csv",
            tmp_path / "run.json",
            threshold=1.1,
        )
