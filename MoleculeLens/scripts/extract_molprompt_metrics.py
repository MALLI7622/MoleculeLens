#!/usr/bin/env python3
"""Normalize MolPrompt retrieval embeddings from NPZ into the shared JSON metric schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = ("s2t_t4", "s2t_t10", "s2t_t20", "t2s_t4", "t2s_t10", "t2s_t20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Path to MolPrompt best-metrics NPZ")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for T-choose-one evaluation")
    return parser.parse_args()


def retrieval_accuracy(query_repr: np.ndarray, key_repr: np.ndarray, t_values=(4, 10, 20), seed: int = 42) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    sample_count = query_repr.shape[0]
    similarity = query_repr @ key_repr.T
    results: dict[int, float] = {}
    for t_value in t_values:
        if t_value > sample_count:
            raise ValueError(f"T={t_value} exceeds sample count {sample_count}")
        correct = 0
        choices = np.arange(sample_count)
        for index in range(sample_count):
            negatives = rng.choice(choices[choices != index], size=t_value - 1, replace=False)
            candidates = np.concatenate(([index], negatives))
            if candidates[np.argmax(similarity[index, candidates])] == index:
                correct += 1
        results[t_value] = correct / sample_count * 100.0
    return results


def main() -> None:
    args = parse_args()
    path = Path(args.npz).expanduser()
    archive = np.load(path)
    if "test_graph" not in archive or "test_text" not in archive:
        raise ValueError(f"{path} does not contain test_graph and test_text arrays")

    graph_repr = archive["test_graph"]
    text_repr = archive["test_text"]
    s2t = retrieval_accuracy(graph_repr, text_repr, seed=args.seed)
    t2s = retrieval_accuracy(text_repr, graph_repr, seed=args.seed)

    metrics = {
        "s2t_t4": s2t[4],
        "s2t_t10": s2t[10],
        "s2t_t20": s2t[20],
        "t2s_t4": t2s[4],
        "t2s_t10": t2s[10],
        "t2s_t20": t2s[20],
    }

    missing = sorted(set(METRIC_KEYS) - metrics.keys())
    if missing:
        raise ValueError(f"Missing metrics: {missing}")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
