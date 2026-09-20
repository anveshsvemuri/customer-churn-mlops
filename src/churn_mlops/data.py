from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_months",
    "autopay",
    "streaming_services",
]
TARGET = "churned"


def generate_customers(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic, PII-free telecom customers with a learnable churn signal."""
    if rows < 100:
        raise ValueError("rows must be at least 100")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    charges = rng.normal(72, 24, rows).clip(20, 150).round(2)
    tickets = rng.poisson(1.4, rows).clip(0, 8)
    contract = rng.choice([1, 12, 24], rows, p=[0.55, 0.3, 0.15])
    autopay = rng.binomial(1, 0.62, rows)
    streaming = rng.integers(0, 4, rows)
    logit = (
        -1.2
        - 0.045 * tenure
        + 0.025 * (charges - 70)
        + 0.50 * tickets
        - 0.060 * contract
        - 0.70 * autopay
        + 0.12 * streaming
    )
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "customer_id": [f"SYN-{i:06d}" for i in range(rows)],
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_tickets": tickets,
            "contract_months": contract,
            "autopay": autopay,
            "streaming_services": streaming,
            TARGET: churned,
        }
    )


def validate_frame(frame: pd.DataFrame, require_target: bool = True) -> None:
    required = set(FEATURES) | ({TARGET} if require_target else set())
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("dataset cannot be empty")
    if frame[FEATURES].isna().any().any():
        raise ValueError("features cannot contain null values")
    if (frame["tenure_months"] < 0).any() or (frame["monthly_charges"] <= 0).any():
        raise ValueError("tenure and charges must be valid")
    if require_target and not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("target must be binary")
