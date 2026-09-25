from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, validate_frame


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    mean_shift: float
    severity: str
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    schema_version: str
    reference_rows: int
    current_rows: int
    threshold: float
    status: str
    drifted_features: int
    features: list[FeatureDrift]


def population_stability(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """Return standardized mean shift as a lightweight deterministic drift signal."""
    validate_frame(reference, require_target=False)
    validate_frame(current, require_target=False)
    drift: dict[str, float] = {}
    for feature in FEATURES:
        scale = float(reference[feature].std()) or 1.0
        drift[feature] = round(
            abs(float(current[feature].mean() - reference[feature].mean())) / scale, 4
        )
    return drift


def _proportions(reference: pd.Series, current: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    combined_unique = np.union1d(reference.unique(), current.unique())
    if len(combined_unique) <= bins:
        reference_counts = reference.value_counts(normalize=True).reindex(combined_unique, fill_value=0)
        current_counts = current.value_counts(normalize=True).reindex(combined_unique, fill_value=0)
        return reference_counts.to_numpy(), current_counts.to_numpy()

    boundaries = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(boundaries) < 3:
        return np.array([1.0]), np.array([1.0])
    boundaries[0], boundaries[-1] = -math.inf, math.inf
    reference_counts, _ = np.histogram(reference, bins=boundaries)
    current_counts, _ = np.histogram(current, bins=boundaries)
    return reference_counts / len(reference), current_counts / len(current)


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10, epsilon: float = 1e-6
) -> float:
    """Calculate PSI using reference quantiles or categorical buckets."""
    reference_share, current_share = _proportions(reference, current, bins)
    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)
    psi = np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    return round(float(psi), 4)


def _severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def create_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.25
) -> DriftReport:
    if threshold <= 0:
        raise ValueError("drift threshold must be positive")
    validate_frame(reference, require_target=False)
    validate_frame(current, require_target=False)
    mean_shifts = population_stability(reference, current)
    features = []
    for feature in FEATURES:
        psi = population_stability_index(reference[feature], current[feature])
        features.append(
            FeatureDrift(
                feature=feature,
                psi=psi,
                mean_shift=mean_shifts[feature],
                severity=_severity(psi),
                drifted=psi >= threshold,
            )
        )
    drifted_features = sum(item.drifted for item in features)
    return DriftReport(
        schema_version="1.0",
        reference_rows=len(reference),
        current_rows=len(current),
        threshold=threshold,
        status="alert" if drifted_features else "healthy",
        drifted_features=drifted_features,
        features=features,
    )


def save_drift_report(report: DriftReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
