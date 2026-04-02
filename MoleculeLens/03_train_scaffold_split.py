# -*- coding: utf-8 -*-
"""
03_train_scaffold_split.py
============================
Trains the thin contrastive bridge (ECFP4 + S-Biomed-RoBERTa) under a
rigorous Bemis-Murcko scaffold split, with hard-negative weighting and
margin loss — exactly as described in the NeurIPS 2025 paper.

Fixes applied vs original notebook:
  - BUG #3 : `import copy` added at top (was NameError at proj_text_0 deepcopy)
  - BUG #7 : Unified SEED=0 throughout (was inconsistently 0 and 42)
  - BUG #8 : EPOCHS=100, loop range(1, EPOCHS+1) → 100 epochs not 101
  - BUG #6 : grouped_recall_at1 defined once with min_group parameter
  - BUG #9 : cls_encode defined once
  - Removed duplicate class/function definitions

NEW:
  - --remove_drug_name strips "Drug: X." from text_rich before encoding
  - All output filenames get a "_nodrug" suffix when flag is set, so both
    conditions coexist in the same --outdir without overwriting each other

Outputs (saved to --outdir):
    Default (with drug name):
        proj_text.pt, proj_mol.pt, Bt_test.pt, Bm_test.pt,
        Z_text_test.pt, X_mol_test.npy, test_df.csv,
        epoch_losses.npy, training_loss.png

    With --remove_drug_name:
        proj_text_nodrug.pt, proj_mol_nodrug.pt, Bt_test_nodrug.pt,
        Bm_test_nodrug.pt, Z_text_test_nodrug.pt, X_mol_test_nodrug.npy,
        test_df_nodrug.csv, epoch_losses_nodrug.npy, training_loss_nodrug.png

Usage:
    # Paper result (with drug name, default)
    python 03_train_scaffold_split.py --csv chembl_mechanisms.csv --outdir outputs

    # Leakage ablation (drug name stripped)
    python 03_train_scaffold_split.py --csv chembl_mechanisms.csv --outdir outputs \\
        --remove_drug_name
"""

import os
import copy
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config (matches paper Section 3)
# ---------------------------------------------------------------------------
TEXT_MODEL   = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
MAX_TXT_LEN  = 96
BATCH_SIZE   = 512
EPOCHS       = 100        # FIX #8: was 101 in original (EPOCHS=101, range 1..101)
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TEMP         = 0.07
MARGIN       = 0.1
SHARED_D     = 256
ECFP_BITS    = 2048
ECFP_RADIUS  = 2
SEED         = 0          # FIX #7: unified seed (was 0 and 42 mixed in original)
MIN_GROUP_SIZE = 3
LOG_EVERY    = 10


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Molecular features
# ---------------------------------------------------------------------------
def ecfp4_bitvect(smi, nbits=2048, radius=2):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(nbits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def murcko_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(core) if core else None


# ---------------------------------------------------------------------------
# Text encoding  (FIX #9: single definition, no duplicates)
# ---------------------------------------------------------------------------
@torch.no_grad()
def cls_encode(texts, tok, model, device, max_length=96, bs=64):
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="Encode text (CLS)"):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state[:, 0, :]
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, Z_text, X_mol, group_labels):
        self.Zt  = torch.tensor(Z_text, dtype=torch.float32)
        self.Xm  = torch.tensor(X_mol,  dtype=torch.float32)
        self.grp = np.array(group_labels).astype(str)

    def __len__(self):
        return len(self.Zt)

    def __getitem__(self, i):
        return self.Zt[i], self.Xm[i], self.grp[i], i


# ---------------------------------------------------------------------------
# Loss: weighted InfoNCE + margin
# ---------------------------------------------------------------------------
def build_same_target_weights(grps, device):
    B = len(grps)
    W = torch.ones((B, B), device=device)
    for i in range(B):
        for j in range(B):
            if i != j and grps[i] == grps[j]:
                W[i, j] = 2.0
    return W


def hardest_same_target_neg(logits, groups):
    B = logits.size(0)
    hard = torch.full((B,), float("-inf"), device=logits.device)
    for i in range(B):
        mask = torch.tensor(
            [(groups[i] == groups[j] and j != i) for j in range(B)],
            device=logits.device)
        if mask.any():
            hard[i] = logits[i][mask].max()
    return hard


def contrastive_loss_with_weights(bt, bm, groups, device, T=0.07, margin=0.1):
    bt = F.normalize(bt, dim=1)
    bm = F.normalize(bm, dim=1)
    logits = (bt @ bm.T) / T
    labels = torch.arange(bt.size(0), device=bt.device)

    W = build_same_target_weights(groups, device)
    ce_row = F.cross_entropy(logits,   labels, reduction="none")
    ce_col = F.cross_entropy(logits.T, labels, reduction="none")
    with torch.no_grad():
        B = logits.size(0)
        avg_w_row = (W.sum(dim=1) - torch.diag(W)) / (B - 1)
        avg_w_col = (W.sum(dim=0) - torch.diag(W)) / (B - 1)
    loss_nce = 0.5 * ((ce_row * avg_w_row).mean() + (ce_col * avg_w_col).mean())

    pos       = logits.diag()
    hard_same = hardest_same_target_neg(logits, groups)
    mask      = torch.isfinite(hard_same)
    margin_loss = (
        torch.clamp(margin - (pos[mask] - hard_same[mask]), min=0).mean()
        if mask.any() else torch.tensor(0.0, device=logits.device)
    )
    return loss_nce + margin_loss, logits


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_side(Z_text, X_mol, proj_text, proj_mol, device):
    Bt = F.normalize(proj_text(torch.tensor(Z_text, dtype=torch.float32).to(device)), dim=1).cpu()
    Bm = F.normalize(proj_mol (torch.tensor(X_mol,  dtype=torch.float32).to(device)), dim=1).cpu()
    return Bt, Bm


def recall_mrr(S_np):
    N = S_np.shape[0]
    gt   = np.arange(N)
    top1 = S_np.argmax(axis=1)
    recall1 = (top1 == gt).mean()
    ranks = []
    for i in range(N):
        order    = np.argsort(-S_np[i])
        rank_pos = int(np.where(order == i)[0][0]) + 1
        ranks.append(1.0 / rank_pos)
    return float(recall1), float(np.mean(ranks))


# FIX #6: single definition with min_group parameter (was defined 3x in original)
def grouped_recall_at1(S_np, meta_df, group_col="target_chembl_id", min_group=3):
    """Recall@1 computed within each target group, micro-averaged."""
    assert S_np.shape[0] == S_np.shape[1] == len(meta_df)
    groups = meta_df[group_col].fillna("UNK").astype(str).values
    idx_by_group = {}
    for i, g in enumerate(groups):
        idx_by_group.setdefault(g, []).append(i)
    hits = total = 0
    for g, idxs in idx_by_group.items():
        if len(idxs) < min_group:
            continue
        sub  = S_np[np.ix_(idxs, idxs)]
        top1 = sub.argmax(axis=1)
        hits  += (top1 == np.arange(len(idxs))).sum()
        total += len(idxs)
    return hits / total if total else 0.0


def recall_at_k(S_np, k_list=(1, 5, 10)):
    """Cumulative Recall@k for multiple k values."""
    N = S_np.shape[0]
    results = {}
    for k in k_list:
        topk  = np.argsort(-S_np, axis=1)[:, :k]
        hits  = sum(i in topk[i] for i in range(N))
        results[k] = hits / N
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    set_seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    need_cols = {"smiles", "text_rich", "target_chembl_id"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Run 01_download_data.py first.")
    df = df.dropna(subset=["smiles", "text_rich"]).copy()

    if args.remove_drug_name:
        df["text_rich"] = df["text_rich"].str.replace(
            r"Drug:\s*[^.]+\. ?", "", regex=True).str.strip()
        print("WARNING: Drug names stripped from text_rich (leakage ablation)")

    # File suffix so both conditions coexist in the same --outdir
    sfx = "_nodrug" if args.remove_drug_name else ""

    print("Sample text [0]:", df["text_rich"].iloc[0])
    print("Sample text [1]:", df["text_rich"].iloc[1], "\n")

    # ------------------------------------------------------------------
    # 2. Scaffold split
    # ------------------------------------------------------------------
    print("Computing Murcko scaffolds ...")
    df["scaffold"] = df["smiles"].apply(murcko_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    scaffolds = df["scaffold"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(scaffolds)
    cut         = int(0.9 * len(scaffolds))
    train_scafs = set(scaffolds[:cut])
    test_scafs  = set(scaffolds[cut:])

    train_df = df[df["scaffold"].isin(train_scafs)].reset_index(drop=True)
    test_df  = df[df["scaffold"].isin(test_scafs)].reset_index(drop=True)
    print(f"Split → train: {len(train_df)} | test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 3. Features
    # ------------------------------------------------------------------
    print("Computing ECFP4 fingerprints ...")
    X_mol_train = np.stack([ecfp4_bitvect(s, ECFP_BITS, ECFP_RADIUS)
                            for s in tqdm(train_df["smiles"].tolist())])
    X_mol_test  = np.stack([ecfp4_bitvect(s, ECFP_BITS, ECFP_RADIUS)
                            for s in tqdm(test_df["smiles"].tolist())])

    print("Encoding text with S-Biomed-RoBERTa ...")
    text_tok   = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    text_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    Z_text_train = cls_encode(train_df["text_rich"].tolist(), text_tok, text_model,
                              device, max_length=MAX_TXT_LEN, bs=64)
    Z_text_test  = cls_encode(test_df["text_rich"].tolist(),  text_tok, text_model,
                              device, max_length=MAX_TXT_LEN, bs=64)

    # ------------------------------------------------------------------
    # 4. DataLoader
    # ------------------------------------------------------------------
    train_ds = PairDataset(Z_text_train.numpy(), X_mol_train,
                           train_df["target_chembl_id"].tolist())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ------------------------------------------------------------------
    # 5. Projection heads
    # ------------------------------------------------------------------
    d_text = Z_text_train.shape[1]
    d_fp   = X_mol_train.shape[1]

    proj_text = nn.Linear(d_text, SHARED_D).to(device)
    proj_mol  = nn.Linear(d_fp,   SHARED_D).to(device)

    # FIX #3: import copy at top — deepcopy now works without NameError
    proj_text_0 = copy.deepcopy(proj_text)
    proj_mol_0  = copy.deepcopy(proj_mol)

    opt = torch.optim.AdamW(
        list(proj_text.parameters()) + list(proj_mol.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)

    # ------------------------------------------------------------------
    # 6. Training  (FIX #8: EPOCHS=100, range(1, 101) = 100 iterations)
    # ------------------------------------------------------------------
    print(f"\nTraining for {EPOCHS} epochs ...")
    epoch_losses = []
    for ep in range(1, EPOCHS + 1):
        proj_text.train(); proj_mol.train()
        running = n = 0
        for Zt_b, Xm_b, grp_b, _ in train_dl:
            Zt_b = Zt_b.to(device); Xm_b = Xm_b.to(device)
            Bt = proj_text(Zt_b); Bm = proj_mol(Xm_b)
            loss, _ = contrastive_loss_with_weights(Bt, Bm, grp_b, device,
                                                    T=TEMP, margin=MARGIN)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n += 1
        avg = running / max(n, 1)
        epoch_losses.append(avg)
        if ep % LOG_EVERY == 0 or ep == 1:
            print(f"  Epoch {ep:>3}/{EPOCHS}: loss={avg:.4f}")

    # ------------------------------------------------------------------
    # 7. Evaluate on test split
    # ------------------------------------------------------------------
    print("\n=== TEST (Scaffold split) ===")
    Bt_test, Bm_test = embed_side(Z_text_test.numpy(), X_mol_test,
                                  proj_text, proj_mol, device)
    S_np = (Bt_test @ Bm_test.T).numpy()

    rec1, mrr = recall_mrr(S_np)
    gR1 = grouped_recall_at1(S_np, test_df, group_col="target_chembl_id",
                             min_group=MIN_GROUP_SIZE)
    rk  = recall_at_k(S_np, k_list=[1, 5, 10])

    print(f"Global  Recall@1 : {rec1:.3f}")
    print(f"Global  MRR      : {mrr:.3f}")
    print(f"Grouped Recall@1 : {gR1:.3f}  (min_group={MIN_GROUP_SIZE})")
    for k, v in rk.items():
        print(f"Recall@{k:<2}        : {v:.3f}")

    # ------------------------------------------------------------------
    # 8. Save checkpoints  (sfx="" or "_nodrug" depending on --remove_drug_name)
    # ------------------------------------------------------------------
    torch.save(proj_text.state_dict(),   os.path.join(args.outdir, f"proj_text{sfx}.pt"))
    torch.save(proj_mol.state_dict(),    os.path.join(args.outdir, f"proj_mol{sfx}.pt"))
    torch.save(proj_text_0.state_dict(), os.path.join(args.outdir, f"proj_text_0{sfx}.pt"))
    torch.save(proj_mol_0.state_dict(),  os.path.join(args.outdir, f"proj_mol_0{sfx}.pt"))

    # Save test embeddings and metadata for visualisation script
    torch.save(Bt_test,     os.path.join(args.outdir, f"Bt_test{sfx}.pt"))
    torch.save(Bm_test,     os.path.join(args.outdir, f"Bm_test{sfx}.pt"))
    torch.save(Z_text_test, os.path.join(args.outdir, f"Z_text_test{sfx}.pt"))
    np.save(os.path.join(args.outdir, f"X_mol_test{sfx}.npy"), X_mol_test)
    test_df.to_csv(os.path.join(args.outdir, f"test_df{sfx}.csv"), index=False)
    np.save(os.path.join(args.outdir, f"epoch_losses{sfx}.npy"), np.array(epoch_losses))

    # ------------------------------------------------------------------
    # 9. Loss curve plot
    # ------------------------------------------------------------------
    cond_label = "no drug name" if args.remove_drug_name else "with drug name"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), epoch_losses, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
    ax.set_title(f"Contrastive Bridge Training Loss — Scaffold Split ({cond_label})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"training_loss{sfx}.png"), dpi=120)
    plt.close()
    print(f"\nAll outputs saved to {args.outdir}/ (suffix=\"{sfx}\")")
    if args.remove_drug_name:
        print("Compare scaffold results:")
        print(f"  With drug name : test_df.csv / proj_text.pt ...")
        print(f"  No drug name   : test_df_nodrug.csv / proj_text_nodrug.pt ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, default="chembl_mechanisms_enriched.csv")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--remove_drug_name", action="store_true",
                        help="Strip 'Drug: X.' from text_rich for leakage ablation.")
    main(parser.parse_args())