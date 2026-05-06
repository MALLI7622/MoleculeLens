"""
Historical multi-label evaluation over parent-structure groups.

For each text query i, a retrieval is counted as correct if any molecule row
with the same RDKit FragmentParent appears in the top-k list. This script is
kept for reference only; it writes to outputs/multilabel/ and does not update
the camera-ready paper artifacts.

Outputs:
  outputs/multilabel/metrics.json
  outputs/multilabel/leakage_per_pair.csv
  outputs/multilabel/wrong_close_analysis.csv
  outputs/multilabel/bootstrap_ci.csv
"""

import os, json, csv, copy
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict

from paper_eval_utils import structural_parent_key

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("outputs/multilabel", exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def build_drug_groups(df):
    """Return dict: parent-structure key -> sorted list of row indices."""
    groups = defaultdict(list)
    for i, row in df.iterrows():
        key = structural_parent_key(str(row["smiles"]))
        groups[key].append(i)
    return groups


def multilabel_metrics(S, drug_groups_by_idx):
    """
    S: [N, N] numpy similarity matrix (text rows × mol cols)
    drug_groups_by_idx: for row i, the set of all mol-column indices that
                        belong to the same physical drug.
    Returns: r1, r5, r10, mrr  (scalars, 0–1)
    """
    N = S.shape[0]
    r1 = r5 = r10 = 0
    mrr_vals = []
    for i in range(N):
        positives = drug_groups_by_idx[i]      # set of correct col indices
        order = np.argsort(-S[i])              # cols ranked best→worst
        best_rank = N                          # fallback
        for rank_0, j in enumerate(order):
            if j in positives:
                best_rank = rank_0 + 1
                break
        if best_rank == 1:  r1  += 1
        if best_rank <= 5:  r5  += 1
        if best_rank <= 10: r10 += 1
        mrr_vals.append(1.0 / best_rank)
    return r1/N, r5/N, r10/N, float(np.mean(mrr_vals))


def bootstrap_ci(scores_vec, n_boot=10000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n = len(scores_vec)
    means = np.array([scores_vec[rng.integers(0, n, n)].mean()
                      for _ in range(n_boot)])
    return float(means.mean()), float(np.percentile(means, 2.5)), \
           float(np.percentile(means, 97.5))


# ── load ─────────────────────────────────────────────────────────────────────

print("Loading embeddings …")
Bm      = torch.load("outputs/Bm_test.pt",          map_location="cpu")
Bt_r    = torch.load("outputs/Bt_test.pt",           map_location="cpu")
Bt_nd   = torch.load("outputs/Bt_test_nodrug.pt",    map_location="cpu")

Bm_np   = F.normalize(Bm,    dim=1).numpy().astype(np.float32)
Bt_r_np = F.normalize(Bt_r,  dim=1).numpy().astype(np.float32)
Bt_nd_np= F.normalize(Bt_nd, dim=1).numpy().astype(np.float32)

df     = pd.read_csv("outputs/test_df.csv").reset_index(drop=True)
df_nd  = pd.read_csv("outputs/test_df_nodrug.csv").reset_index(drop=True)

N = len(df)
print(f"N = {N}")

# similarity matrices (text-query direction: rows=text, cols=mol)
S_r  = Bt_r_np  @ Bm_np.T    # [N, N]
S_nd = Bt_nd_np @ Bm_np.T

# ── parent-structure groups ──────────────────────────────────────────────────

groups = build_drug_groups(df)
# For each query row i, positive column set = all rows sharing the same parent
# structure after RDKit FragmentParent standardization.
drug_groups_by_idx = {}
for i, row in df.iterrows():
    key = structural_parent_key(str(row["smiles"]))
    drug_groups_by_idx[i] = set(groups[key])

print(f"Unique parent structures in test set: {len(groups)}")
print(f"Parent structures with >1 row: {sum(1 for v in groups.values() if len(v)>1)}")

# ── multi-label headline metrics ──────────────────────────────────────────────

r1_r,  r5_r,  r10_r,  mrr_r  = multilabel_metrics(S_r,  drug_groups_by_idx)
r1_nd, r5_nd, r10_nd, mrr_nd = multilabel_metrics(S_nd, drug_groups_by_idx)

leakage_abs = r1_r - r1_nd
leakage_pct = leakage_abs / r1_r * 100 if r1_r > 0 else 0.0

print(f"\n=== Multi-label metrics (text-query direction) ===")
print(f"text_rich  : R@1={r1_r:.4f}  R@5={r5_r:.4f}  R@10={r10_r:.4f}  MRR={mrr_r:.4f}")
print(f"text_nodrug: R@1={r1_nd:.4f}  R@5={r5_nd:.4f}  R@10={r10_nd:.4f}  MRR={mrr_nd:.4f}")
print(f"Leakage drop: abs={leakage_abs:.3f}  pct={leakage_pct:.1f}%")

metrics = dict(
    N=N,
    eval="multilabel",
    text_rich  =dict(R1=round(r1_r,4),  R5=round(r5_r,4),  R10=round(r10_r,4),  MRR=round(mrr_r,4)),
    text_nodrug=dict(R1=round(r1_nd,4), R5=round(r5_nd,4), R10=round(r10_nd,4), MRR=round(mrr_nd,4)),
    leakage_abs=round(leakage_abs,4),
    leakage_pct=round(leakage_pct,1),
)
with open("outputs/multilabel/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

# ── per-pair hit vector (for attribution & leakage analysis) ──────────────────

def per_pair_hits(S, drug_groups_by_idx):
    N = S.shape[0]
    hits_r1 = np.zeros(N, dtype=int)
    ranks   = np.zeros(N, dtype=int)
    for i in range(N):
        positives = drug_groups_by_idx[i]
        order = np.argsort(-S[i])
        for rank_0, j in enumerate(order):
            if j in positives:
                ranks[i] = rank_0 + 1
                break
        hits_r1[i] = int(ranks[i] == 1)
    return hits_r1, ranks

hits_r,  ranks_r  = per_pair_hits(S_r,  drug_groups_by_idx)
hits_nd, ranks_nd = per_pair_hits(S_nd, drug_groups_by_idx)

print(f"\nMulti-label hits: rich={hits_r.sum()} nodrug={hits_nd.sum()}")

# ── update leakage_per_pair.csv ───────────────────────────────────────────────

leakage_source = "outputs/track1/leakage_per_pair_raw.csv"
if not os.path.exists(leakage_source):
    leakage_source = "outputs/track1/leakage_per_pair.csv"
rows_lp = list(csv.DictReader(open(leakage_source)))
leakage_csv = "outputs/multilabel/leakage_per_pair.csv"

# Rebuild attribution Jaccard values (unchanged — attribution formula
# doesn't depend on evaluation protocol, only ranking does)
for i, r in enumerate(rows_lp):
    r["hit_rich"]   = str(int(hits_r[i]))
    r["hit_nodrug"] = str(int(hits_nd[i]))
    r["rank_rich"]  = str(int(ranks_r[i]))
    r["rank_nodrug"]= str(int(ranks_nd[i]))

fieldnames = list(rows_lp[0].keys())
with open(leakage_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_lp)

print(f"Updated {leakage_csv}")

# Re-compute Jaccard leakage stats
j_vals = [float(r["attr_jaccard_overlap"]) for r in rows_lp]
n_hit_r  = sum(int(r["hit_rich"])   for r in rows_lp)
n_hit_nd = sum(int(r["hit_nodrug"]) for r in rows_lp)
j_mean = np.mean(j_vals)
j_std  = np.std(j_vals)
j_low02  = sum(1 for j in j_vals if j < 0.2)
j_high06 = sum(1 for j in j_vals if j > 0.6)

# Leakage candidates: correct rich, wrong nodrug, J < 0.2
leakage_cands = [r for r in rows_lp
                 if r["hit_rich"]=="1" and r["hit_nodrug"]=="0"
                 and float(r["attr_jaccard_overlap"])<0.2]

print(f"\nJaccard stats (multi-label hits):")
print(f"  Mean J = {j_mean:.3f} ± {j_std:.3f}")
print(f"  J < 0.2: {j_low02}/{N} = {j_low02/N*100:.1f}%")
print(f"  J > 0.6: {j_high06}/{N} = {j_high06/N*100:.1f}%")
print(f"  Leakage candidates: {len(leakage_cands)}/{N} = {len(leakage_cands)/N*100:.1f}%")

# ── update wrong_close_analysis.csv (filter same-parent-structure pairs) ─────

wc_csv = "outputs/multilabel/wrong_close_analysis.csv"

# A pair is "wrong-close" only if:
#   rank is 2 or 3 AND the retrieved row belongs to a different parent
#   structure than the query row.
def get_drug(idx):
    return structural_parent_key(str(df.loc[idx, "smiles"]))

# First: collect ALL rank-2/3 retrieval pairs under multi-label eval
# A rank-2/3 means: the first correct drug is NOT at rank-1
# We want: query i where multilabel-rank ∈ {2,3}, and the rank-1 col
# is a DIFFERENT drug than query i

wrong_close_rows = []
for i in range(N):
    if ranks_r[i] not in (2, 3):
        continue
    positives = drug_groups_by_idx[i]
    order = np.argsort(-S_r[i])
    retrieved_idx = int(order[0])       # rank-1 col
    drug_i   = get_drug(i)
    drug_ret = get_drug(retrieved_idx)
    if drug_i == drug_ret:
        # Same physical drug at rank-1 → should NOT be in wrong-close
        # (but wait: if same drug is at rank-1, multilabel rank would be 1,
        #  not 2/3 — so this case shouldn't occur here)
        continue

    # Compute attribution correlation
    # Load W_m weight for correlation if available
    wrong_close_rows.append({
        "drug_idx":       i,
        "retrieved_idx":  retrieved_idx,
        "drug_name":      df.loc[i, "pref_name"],
        "retrieved_name": df.loc[retrieved_idx, "pref_name"],
        "rank":           int(ranks_r[i]),
        "moa_correct":    df.loc[i, "mechanism_of_action"],
        "moa_retrieved":  df.loc[retrieved_idx, "mechanism_of_action"],
    })

print(f"\nWrong-but-close (multi-label, different drug): {len(wrong_close_rows)}")

# ── rebuild wrong-close with attribution correlations ─────────────────────────

# Load saved fingerprints and attribution weights
X_mol = np.load("outputs/X_mol_test.npy")    # [N, 2048]

# Load W_m weight matrix
Wm_state = torch.load("outputs/proj_mol.pt", map_location="cpu")
W_m = Wm_state["weight"].numpy()   # [256, 2048]

# Reload Bt (normalized text embeddings)
Bt_r_norm = F.normalize(Bt_r, dim=1).numpy()  # [N, 256]

def attr_vec(j, W_m, Bt):
    """Linear saliency: W_m.T @ bt_j  (shape [2048])"""
    return W_m.T @ Bt[j]

for row in wrong_close_rows:
    i   = row["drug_idx"]
    ret = row["retrieved_idx"]

    attr_i   = attr_vec(i,   W_m, Bt_r_norm)
    attr_ret = attr_vec(ret, W_m, Bt_r_norm)

    fp_i   = X_mol[i].astype(bool)
    fp_ret = X_mol[ret].astype(bool)
    shared = fp_i & fp_ret

    if shared.sum() > 1:
        corr = float(np.corrcoef(attr_i[shared], attr_ret[shared])[0, 1])
    else:
        corr = float("nan")

    row["shared_bits"]       = int(shared.sum())
    row["attr_correlation"]  = corr
    row["moa_correct_match"] = int(
        str(row["moa_correct"]).strip().upper() ==
        str(row["moa_retrieved"]).strip().upper()
    )

# Stats
corrs = [r["attr_correlation"] for r in wrong_close_rows
         if not np.isnan(r.get("attr_correlation", float("nan")))]
same_moa = sum(r["moa_correct_match"] for r in wrong_close_rows)
high_corr = sum(1 for c in corrs if c > 0.5)

print(f"  Mean ρ = {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
print(f"  ρ > 0.5: {high_corr}/{len(corrs)} = {high_corr/max(len(corrs),1)*100:.0f}%")
print(f"  Same MOA: {same_moa}/{len(wrong_close_rows)}")

# Overwrite CSV
fieldnames_wc = ["drug_idx","retrieved_idx","drug_name","retrieved_name",
                 "rank","shared_bits","attr_correlation",
                 "moa_correct","moa_retrieved","moa_correct_match"]
with open(wc_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames_wc, extrasaction="ignore")
    w.writeheader()
    w.writerows(wrong_close_rows)

print(f"Updated {wc_csv}")

# ── bootstrap CI on multi-label R@1 ──────────────────────────────────────────

rich_scores  = hits_r.astype(float)
nodrug_scores = hits_nd.astype(float)

r1_mean_r,  r1_lo_r,  r1_hi_r  = bootstrap_ci(rich_scores)
r1_mean_nd, r1_lo_nd, r1_hi_nd = bootstrap_ci(nodrug_scores)

# MRR bootstrap
mrr_scores_r = np.array([1.0/ranks_r[i] for i in range(N)])
mrr_mean_r, mrr_lo_r, mrr_hi_r = bootstrap_ci(mrr_scores_r)

print(f"\nBootstrap 95% CI (multi-label):")
print(f"  text_rich   R@1 = {r1_mean_r:.3f} [{r1_lo_r:.3f}, {r1_hi_r:.3f}]")
print(f"  text_nodrug R@1 = {r1_mean_nd:.3f} [{r1_lo_nd:.3f}, {r1_hi_nd:.3f}]")

ci_rows = [
    dict(model="MoleculeLens (text_rich)",   N=N,
         R1=round(r1_mean_r,4),  R1_lo=round(r1_lo_r,4),  R1_hi=round(r1_hi_r,4),
         R5=round(r5_r,4), R5_lo="", R5_hi="",
         R10=round(r10_r,4), R10_lo="", R10_hi="",
         MRR=round(mrr_r,4), MRR_lo=round(mrr_lo_r,4), MRR_hi=round(mrr_hi_r,4),
         note="multilabel bootstrap 10k"),
    dict(model="MoleculeLens (text_nodrug)", N=N,
         R1=round(r1_mean_nd,4), R1_lo=round(r1_lo_nd,4), R1_hi=round(r1_hi_nd,4),
         R5=round(r5_nd,4), R5_lo="", R5_hi="",
         R10=round(r10_nd,4), R10_lo="", R10_hi="",
         MRR=round(mrr_nd,4), MRR_lo="", MRR_hi="",
         note="multilabel bootstrap 10k"),
    dict(model="MolPrompt", N=N,
         R1=0.129, R1_lo=0.1007, R1_hi=0.1638,
         R5="", R5_lo="", R5_hi="",
         R10="", R10_lo="", R10_hi="",
         MRR="", MRR_lo="", MRR_hi="",
         note="Wilson CI (diagonal eval; no per-example scores)"),
    dict(model="KV-PLM", N=N,
         R1=0.055, R1_lo=0.0372, R1_hi=0.0806,
         R5="", R5_lo="", R5_hi="",
         R10="", R10_lo="", R10_hi="",
         MRR="", MRR_lo="", MRR_hi="",
         note="Wilson CI (diagonal eval; no per-example scores)"),
    dict(model="Graphormer (0-shot)", N=N,
         R1=0.002, R1_lo=0.0003, R1_hi=0.0124,
         R5="", R5_lo="", R5_hi="",
         R10="", R10_lo="", R10_hi="",
         MRR="", MRR_lo="", MRR_hi="",
         note="Wilson CI (diagonal eval; no per-example scores)"),
]
ci_fieldnames = ["model","N","R1","R1_lo","R1_hi",
                 "R5","R5_lo","R5_hi","R10","R10_lo","R10_hi",
                 "MRR","MRR_lo","MRR_hi","note"]
with open("outputs/multilabel/bootstrap_ci.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=ci_fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(ci_rows)

print("Updated outputs/multilabel/bootstrap_ci.csv")
print("\nDone.")
