from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET, validate_frame


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float


def train(frame: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, Metrics]:
    validate_frame(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.25, random_state=seed, stratify=frame[TARGET]
    )
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=seed))]
    )
    pipeline.fit(x_train, y_train)
    probability = pipeline.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = Metrics(
        roc_auc=round(roc_auc_score(y_test, probability), 4),
        f1=round(f1_score(y_test, prediction), 4),
        precision=round(precision_score(y_test, prediction, zero_division=0), 4),
        recall=round(recall_score(y_test, prediction, zero_division=0), 4),
        accuracy=round(accuracy_score(y_test, prediction), 4),
    )
    return pipeline, metrics


def promote(metrics: Metrics, minimum_auc: float = 0.70) -> None:
    if metrics.roc_auc < minimum_auc:
        raise ValueError(f"promotion rejected: ROC-AUC {metrics.roc_auc} < {minimum_auc}")


def save_artifacts(model: Pipeline, metrics: Metrics, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    (directory / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2) + "\n")


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def predict(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    validate_frame(frame, require_target=False)
    result = frame.copy()
    result["churn_probability"] = model.predict_proba(frame[FEATURES])[:, 1].round(4)
    result["churn_prediction"] = (result["churn_probability"] >= 0.5).astype(int)
    return result

