#!/usr/bin/env python3
"""Validate the linear saliency approximation used in Section 7.

For each held-out scaffold test pair, this script compares Equation 2 against:
  1. the exact gradient of the normalized similarity score w.r.t. the molecule
     fingerprint bits, and
  2. the exact leave-one-present-bit-out score drop.

It also measures drug specificity under identical no-drug text queries by
comparing exact top-10 bit-drop sets across drugs that share the same text after
removing the `Drug: X.` field.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

DRUG_FIELD_RE = re.compile(r"\s*Drug:\s*[^.]+\.")


def strip_drug_field(text: str) -> str:
    return DRUG_FIELD_RE.sub("", str(text)).strip()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = np.linalg.norm(x_center) * np.linalg.norm(y_center)
    if denom == 0.0:
        return float("nan")
    return float(x_center.dot(y_center) / denom)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0:
        return float("nan")
    return float(x.dot(y) / denom)


def topk_jaccard(a: np.ndarray, b: np.ndarray, indices: np.ndarray, k: int = 10) -> float:
    top_a = set(indices[np.argsort(np.abs(a))[-k:]].tolist())
    top_b = set(indices[np.argsort(np.abs(b))[-k:]].tolist())
    return float(len(top_a & top_b) / len(top_a | top_b))


def exact_bit_drop_scores(
    u: np.ndarray,
    score: float,
    bt: np.ndarray,
    wk: np.ndarray,
) -> np.ndarray:
    u_drop = u[None, :] - wk
    norms = np.linalg.norm(u_drop, axis=1, keepdims=True)
    dropped = (u_drop / norms) @ bt
    return score - dropped


def metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "std": float(np.nanstd(values)),
    }


print("Loading model state and test artifacts ...")
wm_state = torch.load("outputs/proj_mol.pt", map_location="cpu")
wt_state = torch.load("outputs/proj_text.pt", map_location="cpu")
z_text = torch.load("outputs/Z_text_test.pt", map_location="cpu").float()
z_text_nodrug = torch.load("outputs/Z_text_test_nodrug.pt", map_location="cpu").float()
x_mol = np.load("outputs/X_mol_test.npy").astype(np.float32)
df = pd.read_csv("outputs/test_df.csv").reset_index(drop=True)

w_m = wm_state["weight"].float()
b_m = wm_state["bias"].float()
w_t = wt_state["weight"].float()
b_t = wt_state["bias"].float()


print("Computing per-pair validation metrics ...")
rows: list[dict[str, object]] = []
exact_top10_by_pair: dict[int, set[int]] = {}
approx_top10_by_pair: dict[int, set[int]] = {}

for i in range(len(df)):
    x = torch.tensor(x_mol[i], dtype=torch.float32, requires_grad=True)
    z_i = z_text[i]

    u = w_m @ x + b_m
    bt_i = F.normalize(w_t @ z_i + b_t, dim=0)
    score = torch.dot(F.normalize(u, dim=0), bt_i)
    score.backward()

    grad = x.grad.detach().numpy()
    approx = (w_m.T @ bt_i).detach().numpy()
    present_idx = np.flatnonzero(x_mol[i] > 0.5)
    if present_idx.size < 5:
        continue

    grad_present = grad[present_idx].astype(np.float64)
    approx_present = approx[present_idx].astype(np.float64)

    u_np = u.detach().numpy().astype(np.float64)
    bt_np = bt_i.detach().numpy().astype(np.float64)
    wk = w_m[:, present_idx].detach().numpy().T.astype(np.float64)
    delta_scores = exact_bit_drop_scores(u_np, float(score.detach()), bt_np, wk)

    exact_top10_by_pair[i] = set(present_idx[np.argsort(np.abs(delta_scores))[-10:]].tolist())
    approx_top10_by_pair[i] = set(present_idx[np.argsort(np.abs(approx_present))[-10:]].tolist())

    rows.append(
        {
            "drug_idx": i,
            "drug_name": df.loc[i, "pref_name"],
            "n_present_bits": int(present_idx.size),
            "pearson_vs_exact_grad": pearson(approx_present, grad_present),
            "cosine_vs_exact_grad": cosine(approx_present, grad_present),
            "top10_jaccard_vs_exact_grad": topk_jaccard(
                approx_present, grad_present, present_idx, k=10
            ),
            "pearson_vs_bit_drop": pearson(approx_present, delta_scores),
            "cosine_vs_bit_drop": cosine(approx_present, delta_scores),
            "top10_jaccard_vs_bit_drop": topk_jaccard(
                approx_present, delta_scores, present_idx, k=10
            ),
        }
    )

per_pair_df = pd.DataFrame(rows)
per_pair_path = ROOT / "outputs/track1/attribution_validation_per_pair.csv"
per_pair_path.parent.mkdir(parents=True, exist_ok=True)
per_pair_df.to_csv(per_pair_path, index=False)
print(f"Wrote {per_pair_path}")


print("Computing same-text drug-specificity diagnostic ...")
same_text_groups: dict[str, list[int]] = defaultdict(list)
for idx, text in enumerate(df["text_rich"].tolist()):
    same_text_groups[strip_drug_field(text)].append(idx)

pairwise_exact = []
pairwise_approx = []
for group in same_text_groups.values():
    if len(group) < 2:
        continue
    valid = [idx for idx in group if idx in exact_top10_by_pair and idx in approx_top10_by_pair]
    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            top_exact_a = exact_top10_by_pair[valid[a]]
            top_exact_b = exact_top10_by_pair[valid[b]]
            top_approx_a = approx_top10_by_pair[valid[a]]
            top_approx_b = approx_top10_by_pair[valid[b]]
            pairwise_exact.append(
                len(top_exact_a & top_exact_b) / len(top_exact_a | top_exact_b)
            )
            pairwise_approx.append(
                len(top_approx_a & top_approx_b) / len(top_approx_a | top_approx_b)
            )

pairwise_exact_arr = np.array(pairwise_exact, dtype=np.float64)
pairwise_approx_arr = np.array(pairwise_approx, dtype=np.float64)

summary = {
    "notes": {
        "approximation": (
            "Equation 2 uses the linear proxy W_m^T b_t and omits the molecule bias "
            "term and normalization denominator."
        ),
        "exact_score": "normalize(W_m x + b_m) dot normalize(W_t z + b_t)",
    },
    "per_pair_csv": str(per_pair_path.relative_to(ROOT)),
    "evaluated_pairs": int(len(per_pair_df)),
    "approx_vs_exact_gradient": metric_summary(
        per_pair_df["pearson_vs_exact_grad"].to_numpy(dtype=np.float64)
    ),
    "approx_vs_bit_drop": metric_summary(
        per_pair_df["pearson_vs_bit_drop"].to_numpy(dtype=np.float64)
    ),
    "top10_jaccard_vs_exact_gradient": metric_summary(
        per_pair_df["top10_jaccard_vs_exact_grad"].to_numpy(dtype=np.float64)
    ),
    "top10_jaccard_vs_bit_drop": metric_summary(
        per_pair_df["top10_jaccard_vs_bit_drop"].to_numpy(dtype=np.float64)
    ),
    "same_nodrug_text_drug_specificity": {
        "group_count": int(sum(1 for group in same_text_groups.values() if len(group) >= 2)),
        "pair_count": int(len(pairwise_exact_arr)),
        "exact_top10_jaccard": metric_summary(pairwise_exact_arr),
        "approx_top10_jaccard": metric_summary(pairwise_approx_arr),
        "fraction_exact_not_identical": float(np.mean(pairwise_exact_arr < 1.0))
        if len(pairwise_exact_arr)
        else 0.0,
    },
}

summary_path = ROOT / "results/attribution_validation.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {summary_path}")
