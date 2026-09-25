from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.data import FEATURES, generate_customers, validate_frame
from churn_mlops.model import predict, promote, save_artifacts, train
from churn_mlops.monitoring import create_drift_report, population_stability, save_drift_report


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


def test_drift_report_flags_distribution_shift(tmp_path: Path):
    reference = generate_customers(1000, 1)
    current = generate_customers(1000, 2)
    current["monthly_charges"] += 50
    report = create_drift_report(reference, current)

    assert report.status == "alert"
    assert report.drifted_features >= 1
    charges = next(item for item in report.features if item.feature == "monthly_charges")
    assert charges.severity == "significant"
    assert charges.psi >= report.threshold

    output = tmp_path / "nested" / "drift-report.json"
    save_drift_report(report, output)
    assert '"status": "alert"' in output.read_text()


def test_drift_report_is_healthy_for_identical_data():
    reference = generate_customers(1000, 1)
    report = create_drift_report(reference, reference.copy())
    assert report.status == "healthy"
    assert report.drifted_features == 0
    assert all(feature.psi == 0 for feature in report.features)
