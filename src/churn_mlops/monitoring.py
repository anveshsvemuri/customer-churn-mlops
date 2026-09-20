from __future__ import annotations

import pandas as pd

from .data import FEATURES, validate_frame


def population_stability(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """Return standardized mean shift as a lightweight deterministic drift signal."""
    validate_frame(reference, require_target=False)
    validate_frame(current, require_target=False)
    drift: dict[str, float] = {}
    for feature in FEATURES:
        scale = float(reference[feature].std()) or 1.0
        drift[feature] = round(abs(float(current[feature].mean() - reference[feature].mean())) / scale, 4)
    return drift

