#!/usr/bin/env python3
"""Normalize a MoleculeSTM retrieval CSV into the shared JSON metric schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to MoleculeSTM results CSV")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv).expanduser()
    metrics: dict[str, float] = {}
    direction_map = {
        "Given Structure": "s2t",
        "Given Text": "t2s",
    }

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            direction = row["direction"].strip()
            t_value = int(row["T"])
            accuracy = float(row["accuracy"])
            prefix = direction_map.get(direction)
            if prefix is None:
                raise ValueError(f"Unexpected direction: {direction}")
            metrics[f"{prefix}_t{t_value}"] = accuracy * 100.0 if accuracy <= 1.5 else accuracy

    expected = {"s2t_t4", "s2t_t10", "s2t_t20", "t2s_t4", "t2s_t10", "t2s_t20"}
    missing = sorted(expected - metrics.keys())
    if missing:
        raise ValueError(f"Missing metrics in {path}: {missing}")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
