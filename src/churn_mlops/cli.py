from __future__ import annotations

import argparse
from pathlib import Path

from .data import generate_customers
from .model import promote, save_artifacts, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reproducible churn model")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    frame = generate_customers(args.rows, args.seed)
    model, metrics = train(frame, args.seed)
    promote(metrics)
    save_artifacts(model, metrics, args.output)
    print(f"trained rows={len(frame)} roc_auc={metrics.roc_auc} output={args.output}")


if __name__ == "__main__":
    main()

