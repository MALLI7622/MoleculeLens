"""
Raw robustness experiments for MoleculeLens.

Experiment A — Bootstrap confidence intervals (10,000 resamples)
    For MoleculeLens (rich + nodrug): bootstrap from saved per-pair
    diagonal retrieval outcomes.
    For other models (MolPrompt, KV-PLM, Graphormer): analytical Wilson CI
    from known aggregate R@1 (no per-example scores saved).

Experiment B — Multi-seed scaffold split variance (seeds 0–4)
    Pre-encode all 2,699 texts ONCE with frozen RoBERTa, then for each seed
    re-split and re-train only the two linear projection heads.
    Gives mean ± std across 5 independent scaffold splits.

Outputs:
    outputs/robustness/bootstrap_ci_raw.csv
    outputs/robustness/multiseed_results.csv
    outputs/robustness/fig_robustness.pdf/.png

This script no longer rewrites MoleculeLens-paper/figures/; the publication
robustness figure is generated separately by 12b_robustness_figures.py.
"""

import os, copy, random, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

from paper_eval_utils import (
    DEFAULT_TEXT_MAX_LENGTH,
    diagonal_ranks_from_similarity,
)

os.makedirs("outputs/robustness", exist_ok=True)

# ── constants (must match 03_train_scaffold_split.py) ─────────────────────
TEXT_MODEL  = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
SHARED_D    = 256
ECFP_BITS   = 2048
ECFP_RADIUS = 2
MAX_TXT_LEN = DEFAULT_TEXT_MAX_LENGTH
BATCH_SIZE  = 512
EPOCHS      = 100
LR          = 1e-3
WEIGHT_DECAY= 1e-4
TEMP        = 0.07
MARGIN      = 0.1
N_BOOTSTRAP = 10_000
SEEDS       = [0, 1, 2, 3, 4]


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)

def recall_mrr(sim, k_vals=(1,5,10)):
    """
    sim: [N,N] similarity matrix with the correct match on the diagonal.
    Returns per-example hit vectors for each Recall@k and reciprocal rank.
    """
    sim_np = sim.detach().cpu().numpy() if hasattr(sim, "detach") else np.asarray(sim)
    ranks = diagonal_ranks_from_similarity(sim_np)
    out = {}
    for k in k_vals:
        out[f"R@{k}"] = (ranks <= k).astype(np.float64)
    out["MRR"] = 1.0 / ranks.astype(np.float64)
    return out


def bootstrap_ci(arr, n=N_BOOTSTRAP, ci=95):
    """Bootstrap CI for mean of arr."""
    means = np.array([arr[np.random.randint(0, len(arr), len(arr))].mean()
                      for _ in range(n)])
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(arr.mean()), float(lo), float(hi)


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for a proportion p observed over n trials."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return float(centre - half), float(centre + half)


def murcko_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
    except Exception:
        return None


def ecfp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(ECFP_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, ECFP_RADIUS, nBits=ECFP_BITS)
    arr = np.zeros(ECFP_BITS, dtype=np.float32)
    arr[list(fp.GetOnBits())] = 1.0
    return arr


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("EXPERIMENT A — Bootstrap 95% CI")
print("=" * 65)

Bm = torch.load("outputs/Bm_test.pt", map_location="cpu")          # [435,256]
Bt = torch.load("outputs/Bt_test.pt", map_location="cpu")          # [435,256]
Bm_nd = torch.load("outputs/Bm_test_nodrug.pt", map_location="cpu")
Bt_nd = torch.load("outputs/Bt_test_nodrug.pt", map_location="cpu")
N_TEST = Bm.shape[0]

sim_rich   = F.normalize(Bt, dim=1)    @ F.normalize(Bm, dim=1).T
sim_nodrug = F.normalize(Bt_nd, dim=1) @ F.normalize(Bm_nd, dim=1).T

bootstrap_rows = []

for label, sim in [("MoleculeLens (text_rich)",   sim_rich),
                   ("MoleculeLens (text_nodrug)",  sim_nodrug)]:
    per_ex = recall_mrr(sim)
    row = {"model": label, "N": N_TEST}
    for metric, arr in per_ex.items():
        mean, lo, hi = bootstrap_ci(arr)
        row[metric]         = round(mean, 4)
        row[f"{metric}_lo"] = round(lo, 4)
        row[f"{metric}_hi"] = round(hi, 4)
        print(f"  {label:35s}  {metric}: {mean:.3f}  95%CI [{lo:.3f}, {hi:.3f}]")
    bootstrap_rows.append(row)

# Analytical Wilson CI for other models (aggregate R@1 known)
other_models = {
    "MolPrompt":           (0.129, N_TEST),
    "KV-PLM":              (0.055, N_TEST),
    "Graphormer (0-shot)": (0.002, N_TEST),
}
for model, (r1, n) in other_models.items():
    lo, hi = wilson_ci(r1, n)
    row = {"model": model, "N": n,
           "R@1": r1, "R@1_lo": round(lo,4), "R@1_hi": round(hi,4),
           "note": "Wilson CI (no per-example scores available)"}
    bootstrap_rows.append(row)
    print(f"  {model:35s}  R@1: {r1:.3f}  95%CI [{lo:.3f}, {hi:.3f}]  (Wilson)")

ci_df = pd.DataFrame(bootstrap_rows)
ci_df.to_csv("outputs/robustness/bootstrap_ci_raw.csv", index=False)
print("\nSaved: outputs/robustness/bootstrap_ci_raw.csv")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Multi-seed variance
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("EXPERIMENT B — Multi-seed scaffold split variance (seeds 0–4)")
print("=" * 65)

# ── 1. Load full dataset ──────────────────────────────────────────────────
df_all = pd.read_csv("chembl_mechanisms.csv")
print(f"  Full dataset: {len(df_all)} rows")

# ── 2. Pre-encode ALL texts ONCE (frozen RoBERTa) ─────────────────────────
print("\n  Pre-encoding all texts with frozen RoBERTa (done ONCE)...")
tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
rob = AutoModel.from_pretrained(TEXT_MODEL)
rob.eval()

def cls_encode(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_TXT_LEN)
        with torch.no_grad():
            h = rob(**enc).last_hidden_state[:, 0, :]
        out.append(h)
    return torch.cat(out, dim=0)

t0 = time.time()
# text_rich (with drug name)
texts_rich = df_all["text_rich"].tolist()
Z_all_rich = cls_encode(texts_rich)

# text_nodrug (strip "Drug: X." from text_rich)
def strip_drug(t):
    import re
    return re.sub(r'\s*Drug:\s+[^.]+\.?\s*$', '', t).strip()
texts_nodrug = [strip_drug(t) for t in texts_rich]
Z_all_nodrug = cls_encode(texts_nodrug)

print(f"  Encoded {len(df_all)} texts × 2 conditions in {time.time()-t0:.0f}s")
print(f"  Z_all_rich:   {Z_all_rich.shape}")
print(f"  Z_all_nodrug: {Z_all_nodrug.shape}")

# ── 3. Pre-compute all ECFP4 fingerprints ─────────────────────────────────
print("  Computing ECFP4 fingerprints...")
X_all = np.stack([ecfp(s) for s in df_all["smiles"].tolist()]).astype(np.float32)
print(f"  X_all: {X_all.shape}")

# ── 4. Training infrastructure ────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, Z, X):
        self.Z = torch.tensor(Z, dtype=torch.float32)
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return len(self.Z)
    def __getitem__(self, i): return self.Z[i], self.X[i]


def make_projections(d_text, d_mol=ECFP_BITS, d_shared=SHARED_D):
    pt = nn.Linear(d_text, d_shared, bias=True)
    pm = nn.Linear(d_mol,  d_shared, bias=True)
    nn.init.xavier_uniform_(pt.weight); nn.init.zeros_(pt.bias)
    nn.init.xavier_uniform_(pm.weight); nn.init.zeros_(pm.bias)
    return pt, pm


def contrastive_loss(Bt, Bm, temp=TEMP, margin=MARGIN):
    """Symmetric InfoNCE + same-target margin (simplified: no group labels)."""
    N = Bt.shape[0]
    sim = Bt @ Bm.T / temp
    labels = torch.arange(N)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


def train_projections(Z_train, X_train, d_text, seed_val):
    set_seed(seed_val)
    proj_t, proj_m = make_projections(d_text)
    opt = torch.optim.Adam(
        list(proj_t.parameters()) + list(proj_m.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    ds  = PairDataset(Z_train, X_train)
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(1, EPOCHS + 1):
        proj_t.train(); proj_m.train()
        for Zb, Xb in dl:
            bt = F.normalize(proj_t(Zb), dim=1)
            bm = F.normalize(proj_m(Xb), dim=1)
            loss = contrastive_loss(bt, bm)
            opt.zero_grad(); loss.backward(); opt.step()
    return proj_t, proj_m


def evaluate(proj_t, proj_m, Z_test, X_test):
    proj_t.eval(); proj_m.eval()
    with torch.no_grad():
        Zt = torch.tensor(Z_test, dtype=torch.float32)
        Xm = torch.tensor(X_test, dtype=torch.float32)
        Bt = F.normalize(proj_t(Zt), dim=1)
        Bm = F.normalize(proj_m(Xm), dim=1)
    sim = Bm @ Bt.T
    per_ex = recall_mrr(sim)
    return {m: float(arr.mean()) for m, arr in per_ex.items()}, Bm, Bt


# ── 5. Run seeds ──────────────────────────────────────────────────────────
seed_results = []

for seed in SEEDS:
    print(f"\n  --- Seed {seed} ---")
    set_seed(seed)

    # scaffold split
    df_s = df_all.copy()
    df_s["scaffold"] = df_s["smiles"].apply(murcko_scaffold)
    df_s = df_s.dropna(subset=["scaffold"]).reset_index(drop=True)

    scaffolds = df_s["scaffold"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    cut = int(0.9 * len(scaffolds))
    train_idx = df_s[df_s["scaffold"].isin(set(scaffolds[:cut]))].index.tolist()
    test_idx  = df_s[df_s["scaffold"].isin(set(scaffolds[cut:]))].index.tolist()

    Z_tr_r  = Z_all_rich[train_idx].numpy()
    Z_te_r  = Z_all_rich[test_idx].numpy()
    Z_tr_nd = Z_all_nodrug[train_idx].numpy()
    Z_te_nd = Z_all_nodrug[test_idx].numpy()
    X_tr    = X_all[train_idx]
    X_te    = X_all[test_idx]
    d_text  = Z_all_rich.shape[1]

    print(f"    train={len(train_idx)}  test={len(test_idx)}")

    # train both conditions
    t0 = time.time()
    pt_r,  pm_r  = train_projections(Z_tr_r,  X_tr, d_text, seed)
    pt_nd, pm_nd = train_projections(Z_tr_nd, X_tr, d_text, seed + 100)
    elapsed = time.time() - t0

    metrics_r,  Bm_s, Bt_s    = evaluate(pt_r,  pm_r,  Z_te_r,  X_te)
    metrics_nd, Bm_nd, Bt_nd_ = evaluate(pt_nd, pm_nd, Z_te_nd, X_te)

    row = {"seed": seed,
           "n_train": len(train_idx), "n_test": len(test_idx),
           "train_s": elapsed}
    for k, v in metrics_r.items():
        row[f"{k}_rich"]   = round(v, 4)
    for k, v in metrics_nd.items():
        row[f"{k}_nodrug"] = round(v, 4)
    row["leakage_drop_abs"] = round(metrics_r["R@1"] - metrics_nd["R@1"], 4)
    row["leakage_drop_pct"] = round(
        100 * (metrics_r["R@1"] - metrics_nd["R@1"]) / metrics_r["R@1"], 1)

    seed_results.append(row)
    print(f"    R@1 rich={metrics_r['R@1']:.3f}  nodrug={metrics_nd['R@1']:.3f}  "
          f"drop={row['leakage_drop_pct']:.1f}%  ({elapsed:.0f}s)")

ms_df = pd.DataFrame(seed_results)
ms_df.to_csv("outputs/robustness/multiseed_results.csv", index=False)
print("\nSaved: outputs/robustness/multiseed_results.csv")

# ── summary stats ─────────────────────────────────────────────────────────
print("\n  Multi-seed summary:")
for col in ["R@1_rich", "R@5_rich", "R@10_rich", "MRR_rich",
            "R@1_nodrug", "leakage_drop_pct"]:
    vals = ms_df[col].values
    print(f"    {col:22s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"[{vals.min():.3f}, {vals.max():.3f}]")


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating robustness figures...")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

# ── Panel A: Bootstrap CI forest plot ─────────────────────────────────────
ax = fig.add_subplot(gs[0])
models = ["MoleculeLens\n(text_rich)", "MoleculeLens\n(text_nodrug)",
          "MolPrompt", "KV-PLM", "Graphormer\n(0-shot)"]
r1_means = [ci_df.loc[ci_df["model"].str.contains(m.replace("\n",""), case=False),
                       "R@1"].values[0]
            for m in ["MoleculeLens (text_rich)", "MoleculeLens (text_nodrug)",
                      "MolPrompt", "KV-PLM", "Graphormer"]]
r1_lo    = [ci_df.loc[ci_df["model"].str.contains(m, case=False), "R@1_lo"].values[0]
            for m in ["MoleculeLens \\(text_rich\\)", "MoleculeLens \\(text_nodrug\\)",
                      "MolPrompt", "KV-PLM", "Graphormer"]]
r1_hi    = [ci_df.loc[ci_df["model"].str.contains(m, case=False), "R@1_hi"].values[0]
            for m in ["MoleculeLens \\(text_rich\\)", "MoleculeLens \\(text_nodrug\\)",
                      "MolPrompt", "KV-PLM", "Graphormer"]]

# re-query cleanly
rows_for_plot = []
for tag in ["text_rich", "text_nodrug", "MolPrompt", "KV-PLM", "Graphormer"]:
    hit = ci_df[ci_df["model"].str.contains(tag, case=False)]
    if len(hit):
        rows_for_plot.append(hit.iloc[0])

ys = range(len(rows_for_plot))
colours = ["#2166ac", "#d6604d", "#4dac26", "#f1a340", "#888888"]
for i, (r, col) in enumerate(zip(rows_for_plot, colours)):
    mean = r["R@1"]; lo = r["R@1_lo"]; hi = r["R@1_hi"]
    ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
    ax.plot(mean, i, "o", color=col, ms=7, zorder=4)
    ax.text(hi + 0.004, i, f"{mean:.3f}", va="center", fontsize=7, color=col)

ax.set_yticks(range(len(rows_for_plot)))
ax.set_yticklabels([r["model"].replace("MoleculeLens (","ML\n(")
                    .replace(")","") for r in rows_for_plot], fontsize=7)
ax.set_xlabel("Recall@1")
ax.set_title("(A) Bootstrap 95% CI\n(Recall@1, N=435 test pairs)", fontweight="bold")
ax.axvline(0, color="#cccccc", lw=0.7)

# ── Panel B: multi-seed R@1 strip chart ───────────────────────────────────
ax = fig.add_subplot(gs[1])
rich_vals   = ms_df["R@1_rich"].values
nodrug_vals = ms_df["R@1_nodrug"].values
seeds_x     = ms_df["seed"].values

ax.plot(seeds_x, rich_vals,   "o-", color="#2166ac", lw=2, ms=7,
        label=f"text_rich   ({rich_vals.mean():.3f}±{rich_vals.std():.3f})")
ax.plot(seeds_x, nodrug_vals, "s--", color="#d6604d", lw=2, ms=7,
        label=f"text_nodrug ({nodrug_vals.mean():.3f}±{nodrug_vals.std():.3f})")
ax.axhline(rich_vals.mean(),   color="#2166ac", lw=1, ls=":", alpha=0.7)
ax.axhline(nodrug_vals.mean(), color="#d6604d", lw=1, ls=":", alpha=0.7)
ax.fill_between(seeds_x,
                rich_vals.mean() - rich_vals.std(),
                rich_vals.mean() + rich_vals.std(),
                alpha=0.12, color="#2166ac")
ax.fill_between(seeds_x,
                nodrug_vals.mean() - nodrug_vals.std(),
                nodrug_vals.mean() + nodrug_vals.std(),
                alpha=0.12, color="#d6604d")
ax.set_xlabel("Scaffold split seed")
ax.set_ylabel("Recall@1")
ax.set_xticks(SEEDS)
ax.set_title("(B) Multi-seed stability\n(5 independent scaffold splits)", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

# ── Panel C: leakage drop across seeds ────────────────────────────────────
ax = fig.add_subplot(gs[2])
drops = ms_df["leakage_drop_pct"].values
ax.bar(seeds_x, drops, color="#9467bd", edgecolor="white", width=0.6, alpha=0.85)
ax.axhline(drops.mean(), color="#333333", lw=1.5, ls="--",
           label=f"mean = {drops.mean():.1f}%")
ax.fill_between([-0.5, 4.5],
                drops.mean() - drops.std(),
                drops.mean() + drops.std(),
                alpha=0.15, color="#9467bd",
                label=f"±1 std = {drops.std():.1f}pp")
ax.set_xlabel("Scaffold split seed")
ax.set_ylabel("Leakage drop (%)")
ax.set_xticks(SEEDS)
ax.set_title("(C) Leakage drop stability\nacross scaffold splits", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

fig.suptitle(
    "Raw robustness summary — bootstrap CI and multi-seed scaffold split variance",
    fontsize=10, fontweight="bold", y=1.02
)
fig.savefig("outputs/robustness/fig_robustness.pdf",
            bbox_inches="tight", format="pdf")
fig.savefig("outputs/robustness/fig_robustness.png",
            bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved: outputs/robustness/fig_robustness.pdf/.png")

# ══════════════════════════════════════════════════════════════════════════
# PAPER-READY TABLES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("RAW ROBUSTNESS SUMMARY")
print("=" * 65)

print("\n--- Bootstrap CI table (for Table 1 / Table 2 footnote) ---")
ml_rich = ci_df[ci_df["model"].str.contains("text_rich")].iloc[0]
ml_nd   = ci_df[ci_df["model"].str.contains("text_nodrug")].iloc[0]
for metric in ["R@1", "R@5", "R@10", "MRR"]:
    if metric in ml_rich:
        print(f"  MoleculeLens {metric}: "
              f"{ml_rich[metric]:.3f} [{ml_rich[f'{metric}_lo']:.3f}, "
              f"{ml_rich[f'{metric}_hi']:.3f}]")

print("\n--- Multi-seed summary table ---")
header = f"{'Seed':>6} {'n_test':>7} {'R@1_rich':>10} {'MRR_rich':>10} "
header += f"{'R@1_nd':>8} {'Drop%':>7}"
print(header)
for _, r in ms_df.iterrows():
    print(f"  {int(r.seed):4d}   {int(r.n_test):5d}   "
          f"{r['R@1_rich']:8.3f}   {r['MRR_rich']:8.3f}   "
          f"{r['R@1_nodrug']:6.3f}   {r['leakage_drop_pct']:5.1f}%")
print(f"  {'Mean':>4}           "
      f"{ms_df['R@1_rich'].mean():8.3f}   {ms_df['MRR_rich'].mean():8.3f}   "
      f"{ms_df['R@1_nodrug'].mean():6.3f}   {ms_df['leakage_drop_pct'].mean():5.1f}%")
print(f"  {'Std':>4}           "
      f"{ms_df['R@1_rich'].std():8.3f}   {ms_df['MRR_rich'].std():8.3f}   "
      f"{ms_df['R@1_nodrug'].std():6.3f}   {ms_df['leakage_drop_pct'].std():5.1f}%")
