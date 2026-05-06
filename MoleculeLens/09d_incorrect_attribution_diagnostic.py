#!/usr/bin/env python3
"""Summarize attribution fidelity for correct vs incorrect retrievals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_eval_utils import diagonal_ranks_from_similarity


ROOT = Path(__file__).resolve().parent


def metric_block(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "pearson_vs_exact_grad_mean": float(frame["pearson_vs_exact_grad"].mean()),
        "pearson_vs_bit_drop_mean": float(frame["pearson_vs_bit_drop"].mean()),
        "top10_jaccard_vs_exact_grad_mean": float(frame["top10_jaccard_vs_exact_grad"].mean()),
        "top10_jaccard_vs_bit_drop_mean": float(frame["top10_jaccard_vs_bit_drop"].mean()),
    }


def main() -> None:
    bt = torch.load(ROOT / "outputs/Bt_test.pt", map_location="cpu")
    bm = torch.load(ROOT / "outputs/Bm_test.pt", map_location="cpu")
    sim = (bt @ bm.T).numpy()
    ranks = diagonal_ranks_from_similarity(sim)
    retrieved = np.argmax(sim, axis=1)

    test_df = pd.read_csv(ROOT / "outputs/test_df.csv").reset_index(drop=True)
    val_df = pd.read_csv(ROOT / "outputs/track1/attribution_validation_per_pair.csv")
    val_df = val_df.merge(
        test_df[["pref_name", "target_name", "action_type"]],
        left_on="drug_idx",
        right_index=True,
        how="left",
        suffixes=("", "_metadata"),
    )
    val_df["retrieval_rank"] = val_df["drug_idx"].map(lambda idx: int(ranks[int(idx)]))
    val_df["retrieved_idx"] = val_df["drug_idx"].map(lambda idx: int(retrieved[int(idx)]))
    val_df["retrieved_name"] = val_df["retrieved_idx"].map(lambda idx: test_df.loc[idx, "pref_name"])
    val_df["hit_r1"] = val_df["retrieval_rank"] == 1
    val_df["hit_r10"] = val_df["retrieval_rank"] <= 10

    correct = val_df[val_df["hit_r1"]]
    incorrect = val_df[~val_df["hit_r1"]]
    hard_misses = val_df[val_df["retrieval_rank"] > 10]

    summary = {
        "semantics": (
            "Attribution approximation fidelity stratified by retrieval outcome. "
            "Metrics compare Equation 2 against exact gradients and exact present-bit score drops."
        ),
        "all_pairs": metric_block(val_df),
        "correct_at_rank1": metric_block(correct),
        "incorrect_at_rank1": metric_block(incorrect),
        "rank_gt_10": metric_block(hard_misses),
        "incorrect_count": int(len(incorrect)),
        "rank_gt_10_count": int(len(hard_misses)),
    }

    out_dir = ROOT / "outputs/track1"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "incorrect_attribution_summary.csv"
    cases_csv = out_dir / "incorrect_attribution_cases.csv"
    json_path = ROOT / "results/incorrect_attribution_summary.json"

    rows = []
    for label, frame in [
        ("correct_at_rank1", correct),
        ("incorrect_at_rank1", incorrect),
        ("rank_gt_10", hard_misses),
    ]:
        row = {"subset": label}
        row.update(metric_block(frame))
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    case_cols = [
        "drug_idx",
        "drug_name",
        "retrieval_rank",
        "retrieved_name",
        "target_name",
        "action_type",
        "pearson_vs_bit_drop",
        "top10_jaccard_vs_bit_drop",
        "pearson_vs_exact_grad",
        "top10_jaccard_vs_exact_grad",
    ]
    incorrect.sort_values(["retrieval_rank", "pearson_vs_bit_drop"], ascending=[False, False])[
        case_cols
    ].head(30).to_csv(cases_csv, index=False)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {cases_csv}")
    print(f"Wrote {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
