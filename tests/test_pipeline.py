from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.data import FEATURES, generate_customers, validate_frame
from churn_mlops.model import predict, promote, save_artifacts, train
from churn_mlops.monitoring import population_stability


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_customers(200, 7), generate_customers(200, 7))


def test_training_prediction_and_artifacts(tmp_path: Path):
    frame = generate_customers(1500, 42)
    model, metrics = train(frame)
    promote(metrics, minimum_auc=0.65)
    scored = predict(model, frame.head(3))
    assert scored["churn_probability"].between(0, 1).all()
    save_artifacts(model, metrics, tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_validation_rejects_missing_feature():
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(generate_customers(100).drop(columns=[FEATURES[0]]))


def test_drift_detects_shift():
    reference = generate_customers(500, 1)
    current = generate_customers(500, 2)
    current["monthly_charges"] += 40
    assert population_stability(reference, current)["monthly_charges"] > 1

