from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .model import load_model, predict


@dataclass(frozen=True)
class BatchManifest:
    input_sha256: str
    model_sha256: str
    rows_scored: int
    predicted_churners: int
    mean_churn_probability: float
    threshold: float
    output_columns: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, destination)


def score_csv(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    threshold: float = 0.5,
    batch_size: int = 10_000,
) -> BatchManifest:
    """Score a CSV in bounded batches and write only identifiers plus predictions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if input_path.resolve() in {output_path.resolve(), manifest_path.resolve()}:
        raise ValueError("input, output, and manifest paths must be different")

    model = load_model(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = positives = 0
    probability_sum = 0.0
    output_columns: list[str] | None = None
    try:
        for chunk_number, frame in enumerate(pd.read_csv(input_path, chunksize=batch_size)):
            scored = predict(model, frame, threshold)
            columns = ["customer_id"] if "customer_id" in scored.columns else []
            columns += ["churn_probability", "churn_prediction"]
            safe_output = scored[columns]
            safe_output.to_csv(temporary, mode="a", index=False, header=chunk_number == 0)
            output_columns = columns
            rows += len(safe_output)
            positives += int(safe_output["churn_prediction"].sum())
            probability_sum += float(safe_output["churn_probability"].sum())
        if rows == 0 or output_columns is None:
            raise ValueError("input dataset cannot be empty")
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = BatchManifest(
        input_sha256=_sha256(input_path),
        model_sha256=_sha256(model_path),
        rows_scored=rows,
        predicted_churners=positives,
        mean_churn_probability=round(probability_sum / rows, 4),
        threshold=threshold,
        output_columns=output_columns,
    )
    _atomic_json(asdict(manifest), manifest_path)
    return manifest
