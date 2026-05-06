"""
Sync paper-facing artifacts to the current paper semantics.

Updates:
  outputs/track1/leakage_per_pair.csv
  outputs/track1/wrong_close_analysis.csv
  outputs/robustness/bootstrap_ci.csv

Track 1 leakage now uses a same-model text ablation: the rich molecule and text
projection heads are held fixed, and only the `Drug: X.` span is removed from
the text at inference time. The robustness/bootstrap CSV retains the earlier
retrained no-drug comparison so the paper can report both semantics explicitly.

Same-drug filtering uses a standardized parent-structure key derived from
RDKit FragmentParent, which collapses salts, hydrates, and simple formulation
fragments before comparing molecules.
"""

import csv
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from paper_eval_utils import diagonal_ranks_from_similarity, structural_parent_key


os.chdir(os.path.dirname(os.path.abspath(__file__)))


def bootstrap_ci(scores: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    means = np.array([scores[rng.integers(0, n, n)].mean() for _ in range(n_boot)], dtype=np.float64)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def diagonal_hits_and_ranks(sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute diagonal hits with the canonical optimistic tie-breaking rule."""
    ranks = diagonal_ranks_from_similarity(sim)
    hits = (ranks == 1).astype(int)
    return hits, ranks.astype(int)


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    return {
        "R1": float(np.mean(ranks == 1)),
        "R5": float(np.mean(ranks <= 5)),
        "R10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
    }


def attr_vec(query_idx: int, w_m: np.ndarray, bt: np.ndarray) -> np.ndarray:
    return w_m.T @ bt[query_idx]


print("Loading embeddings and metadata ...")
bm = torch.load("outputs/Bm_test.pt", map_location="cpu")
bt_r = torch.load("outputs/Bt_test.pt", map_location="cpu")
bt_nd_retrained = torch.load("outputs/Bt_test_nodrug.pt", map_location="cpu")
z_nd = torch.load("outputs/Z_text_test_nodrug.pt", map_location="cpu")
wt_state = torch.load("outputs/proj_text.pt", map_location="cpu")
w_m_state = torch.load("outputs/proj_mol.pt", map_location="cpu")

bm_np = F.normalize(bm, dim=1).numpy().astype(np.float32)
bt_r_np = F.normalize(bt_r, dim=1).numpy().astype(np.float32)
bt_nd_retrained_np = F.normalize(bt_nd_retrained, dim=1).numpy().astype(np.float32)

w_t = wt_state["weight"]
b_t = wt_state["bias"]
bt_nd_same = F.normalize(z_nd @ w_t.T + b_t, dim=1)
bt_nd_same_np = bt_nd_same.numpy().astype(np.float32)

df = pd.read_csv("outputs/test_df.csv").reset_index(drop=True)
n = len(df)

sim_r = bt_r_np @ bm_np.T
sim_nd_same = bt_nd_same_np @ bm_np.T
sim_nd_retrained = bt_nd_retrained_np @ bm_np.T

hits_r, ranks_r = diagonal_hits_and_ranks(sim_r)
hits_nd_same, ranks_nd_same = diagonal_hits_and_ranks(sim_nd_same)
hits_nd_retrained, ranks_nd_retrained = diagonal_hits_and_ranks(sim_nd_retrained)

metrics_r = metrics_from_ranks(ranks_r)
metrics_nd_same = metrics_from_ranks(ranks_nd_same)
metrics_nd_retrained = metrics_from_ranks(ranks_nd_retrained)

print("Diagonal metrics:")
print(
    f"  text_rich   R@1={metrics_r['R1']:.4f}  R@5={metrics_r['R5']:.4f}  "
    f"R@10={metrics_r['R10']:.4f}  MRR={metrics_r['MRR']:.4f}"
)
print(
    f"  text_nodrug (same-model ablation)   R@1={metrics_nd_same['R1']:.4f}  "
    f"R@5={metrics_nd_same['R5']:.4f}  R@10={metrics_nd_same['R10']:.4f}  "
    f"MRR={metrics_nd_same['MRR']:.4f}"
)
print(
    f"  text_nodrug (retrained control)     R@1={metrics_nd_retrained['R1']:.4f}  "
    f"R@5={metrics_nd_retrained['R5']:.4f}  R@10={metrics_nd_retrained['R10']:.4f}  "
    f"MRR={metrics_nd_retrained['MRR']:.4f}"
)
leakage_abs = metrics_r["R1"] - metrics_nd_same["R1"]
leakage_pct = leakage_abs / metrics_r["R1"] * 100.0
print(f"  same-model leakage abs={leakage_abs:.4f}  pct={leakage_pct:.1f}%")


print("\nUpdating outputs/track1/leakage_per_pair.csv ...")
leakage_csv = "outputs/track1/leakage_per_pair.csv"
w_m = w_m_state["weight"].numpy()
x_mol = np.load("outputs/X_mol_test.npy")

leakage_rows: List[dict] = []
for i in range(n):
    fp_i = x_mol[i].astype(bool)
    present = np.flatnonzero(fp_i)
    if present.size < 10:
        jaccard = float("nan")
    else:
        attr_r = attr_vec(i, w_m, bt_r_np)
        attr_nd = attr_vec(i, w_m, bt_nd_same_np)
        top_r = set(present[np.argsort(np.abs(attr_r[present]))[-10:]].tolist())
        top_nd = set(present[np.argsort(np.abs(attr_nd[present]))[-10:]].tolist())
        jaccard = float(len(top_r & top_nd) / len(top_r | top_nd))

    leakage_rows.append(
        {
            "molecule_chembl_id": df.loc[i, "molecule_chembl_id"],
            "pref_name": df.loc[i, "pref_name"],
            "mechanism_of_action": df.loc[i, "mechanism_of_action"],
            "action_type": df.loc[i, "action_type"],
            "attr_jaccard_overlap": jaccard,
            "rank_rich": int(ranks_r[i]),
            "rank_nodrug": int(ranks_nd_same[i]),
            "hit_rich": int(hits_r[i]),
            "hit_nodrug": int(hits_nd_same[i]),
            "nodrug_semantics": "same_model_rich_head_text_ablation",
        }
    )

with open(leakage_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(leakage_rows[0].keys()))
    writer.writeheader()
    writer.writerows(leakage_rows)

j_vals = np.array([float(row["attr_jaccard_overlap"]) for row in leakage_rows], dtype=np.float64)
leakage_candidates = [
    row
    for row in leakage_rows
    if row["hit_rich"] == 1 and row["hit_nodrug"] == 0 and float(row["attr_jaccard_overlap"]) < 0.2
]
print(f"  Mean J = {np.nanmean(j_vals):.3f} ± {np.nanstd(j_vals):.3f}")
print(f"  J < 0.2: {int(np.nansum(j_vals < 0.2))}/{n} = {np.nanmean(j_vals < 0.2) * 100:.1f}%")
print(f"  J > 0.6: {int(np.nansum(j_vals > 0.6))}/{n} = {np.nanmean(j_vals > 0.6) * 100:.1f}%")
print(f"  leakage candidates: {len(leakage_candidates)}/{n} = {len(leakage_candidates) / n * 100:.1f}%")


print("\nUpdating outputs/track1/wrong_close_analysis.csv ...")
entity_keys = [structural_parent_key(smiles) for smiles in df["smiles"].tolist()]

wrong_close_rows: List[dict] = []
for i in range(n):
    if ranks_r[i] not in (2, 3):
        continue
    retrieved_idx = int(np.argmax(sim_r[i]))
    if entity_keys[i] == entity_keys[retrieved_idx]:
        continue

    attr_i = attr_vec(i, w_m, bt_r_np)
    attr_ret = attr_vec(retrieved_idx, w_m, bt_r_np)
    fp_i = x_mol[i].astype(bool)
    fp_ret = x_mol[retrieved_idx].astype(bool)
    shared = fp_i & fp_ret

    if shared.sum() > 1:
        corr = float(np.corrcoef(attr_i[shared], attr_ret[shared])[0, 1])
    else:
        corr = float("nan")

    moa_correct = str(df.loc[i, "mechanism_of_action"])
    moa_retrieved = str(df.loc[retrieved_idx, "mechanism_of_action"])
    wrong_close_rows.append(
        {
            "drug_idx": i,
            "retrieved_idx": retrieved_idx,
            "drug_name": df.loc[i, "pref_name"],
            "retrieved_name": df.loc[retrieved_idx, "pref_name"],
            "rank": int(ranks_r[i]),
            "shared_bits": int(shared.sum()),
            "attr_correlation": corr,
            "entity_key": entity_keys[i],
            "retrieved_entity_key": entity_keys[retrieved_idx],
            "moa_correct": moa_correct,
            "moa_retrieved": moa_retrieved,
            "moa_correct_match": int(moa_correct.strip().upper() == moa_retrieved.strip().upper()),
        }
    )

wrong_close_csv = "outputs/track1/wrong_close_analysis.csv"
wrong_close_fields = [
    "drug_idx",
    "retrieved_idx",
    "drug_name",
    "retrieved_name",
    "rank",
    "shared_bits",
    "attr_correlation",
    "entity_key",
    "retrieved_entity_key",
    "moa_correct",
    "moa_retrieved",
    "moa_correct_match",
]
with open(wrong_close_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=wrong_close_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(wrong_close_rows)

corrs = np.array(
    [row["attr_correlation"] for row in wrong_close_rows if not np.isnan(row["attr_correlation"])],
    dtype=np.float64,
)
same_moa = sum(int(row["moa_correct_match"]) for row in wrong_close_rows)
high_corr = int(np.sum(corrs > 0.5))
print(f"  rows: {len(wrong_close_rows)}")
print(f"  mean rho: {corrs.mean():.3f} ± {corrs.std():.3f}")
print(f"  rho > 0.5: {high_corr}/{len(corrs)} = {high_corr / len(corrs) * 100:.0f}%")
print(f"  same MOA: {same_moa}/{len(wrong_close_rows)}")


print("\nUpdating outputs/robustness/bootstrap_ci.csv ...")
r1_mean_r, r1_lo_r, r1_hi_r = bootstrap_ci(hits_r.astype(np.float64))
r1_mean_nd_same, r1_lo_nd_same, r1_hi_nd_same = bootstrap_ci(hits_nd_same.astype(np.float64))
r1_mean_nd_retrained, r1_lo_nd_retrained, r1_hi_nd_retrained = bootstrap_ci(
    hits_nd_retrained.astype(np.float64)
)
mrr_mean_r, mrr_lo_r, mrr_hi_r = bootstrap_ci((1.0 / ranks_r).astype(np.float64))
mrr_mean_nd_same, mrr_lo_nd_same, mrr_hi_nd_same = bootstrap_ci(
    (1.0 / ranks_nd_same).astype(np.float64)
)

ci_rows = [
    {
        "model": "MoleculeLens (text_rich)",
        "N": n,
        "R1": round(r1_mean_r, 4),
        "R1_lo": round(r1_lo_r, 4),
        "R1_hi": round(r1_hi_r, 4),
        "R5": round(metrics_r["R5"], 4),
        "R5_lo": "",
        "R5_hi": "",
        "R10": round(metrics_r["R10"], 4),
        "R10_lo": "",
        "R10_hi": "",
        "MRR": round(metrics_r["MRR"], 4),
        "MRR_lo": round(mrr_lo_r, 4),
        "MRR_hi": round(mrr_hi_r, 4),
        "note": "diagonal bootstrap 10k",
    },
    {
        "model": "MoleculeLens (text_nodrug)",
        "N": n,
        "R1": round(r1_mean_nd_retrained, 4),
        "R1_lo": round(r1_lo_nd_retrained, 4),
        "R1_hi": round(r1_hi_nd_retrained, 4),
        "R5": round(metrics_nd_retrained["R5"], 4),
        "R5_lo": "",
        "R5_hi": "",
        "R10": round(metrics_nd_retrained["R10"], 4),
        "R10_lo": "",
        "R10_hi": "",
        "MRR": round(metrics_nd_retrained["MRR"], 4),
        "MRR_lo": "",
        "MRR_hi": "",
        "note": "retrained no-drug head; diagonal bootstrap 10k",
    },
    {
        "model": "MoleculeLens (text_nodrug, same_model)",
        "N": n,
        "R1": round(r1_mean_nd_same, 4),
        "R1_lo": round(r1_lo_nd_same, 4),
        "R1_hi": round(r1_hi_nd_same, 4),
        "R5": round(metrics_nd_same["R5"], 4),
        "R5_lo": "",
        "R5_hi": "",
        "R10": round(metrics_nd_same["R10"], 4),
        "R10_lo": "",
        "R10_hi": "",
        "MRR": round(metrics_nd_same["MRR"], 4),
        "MRR_lo": round(mrr_lo_nd_same, 4),
        "MRR_hi": round(mrr_hi_nd_same, 4),
        "note": "rich model fixed; Drug: X. removed at inference",
    },
    {
        "model": "MolPrompt",
        "N": n,
        "R1": 0.129,
        "R1_lo": 0.1007,
        "R1_hi": 0.1638,
        "R5": "",
        "R5_lo": "",
        "R5_hi": "",
        "R10": "",
        "R10_lo": "",
        "R10_hi": "",
        "MRR": "",
        "MRR_lo": "",
        "MRR_hi": "",
        "note": "Wilson CI (diagonal eval; no per-example scores)",
    },
    {
        "model": "KV-PLM",
        "N": n,
        "R1": 0.055,
        "R1_lo": 0.0372,
        "R1_hi": 0.0806,
        "R5": "",
        "R5_lo": "",
        "R5_hi": "",
        "R10": "",
        "R10_lo": "",
        "R10_hi": "",
        "MRR": "",
        "MRR_lo": "",
        "MRR_hi": "",
        "note": "Wilson CI (diagonal eval; no per-example scores)",
    },
    {
        "model": "Graphormer (0-shot)",
        "N": n,
        "R1": 0.002,
        "R1_lo": 0.0003,
        "R1_hi": 0.0124,
        "R5": "",
        "R5_lo": "",
        "R5_hi": "",
        "R10": "",
        "R10_lo": "",
        "R10_hi": "",
        "MRR": "",
        "MRR_lo": "",
        "MRR_hi": "",
        "note": "Wilson CI (diagonal eval; no per-example scores)",
    },
]
ci_fields = [
    "model",
    "N",
    "R1",
    "R1_lo",
    "R1_hi",
    "R5",
    "R5_lo",
    "R5_hi",
    "R10",
    "R10_lo",
    "R10_hi",
    "MRR",
    "MRR_lo",
    "MRR_hi",
    "note",
]
with open("outputs/robustness/bootstrap_ci.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ci_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(ci_rows)

print(f"  text_rich                    R@1 = {r1_mean_r:.3f} [{r1_lo_r:.3f}, {r1_hi_r:.3f}]")
print(
    f"  text_nodrug (retrained)     R@1 = {r1_mean_nd_retrained:.3f} "
    f"[{r1_lo_nd_retrained:.3f}, {r1_hi_nd_retrained:.3f}]"
)
print(
    f"  text_nodrug (same-model)    R@1 = {r1_mean_nd_same:.3f} "
    f"[{r1_lo_nd_same:.3f}, {r1_hi_nd_same:.3f}]"
)
print("\nDone.")
