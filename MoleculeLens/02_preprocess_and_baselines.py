# -*- coding: utf-8 -*-
"""
02_preprocess_and_baselines.py
================================
Loads chembl_mechanisms.csv, builds frozen ChemBERTa + PubMedBERT embeddings,
evaluates frozen-encoder Recall@1 / MRR baselines, and saves embeddings to disk
for use by the training script.

Fixes applied vs original notebook:
  - BUG #2 : N = max(...) → N = min(...)  (was crashing when len(df) < N)
  - BUG #4 : encode_smiles hardcoded max_length=128, now uses the parameter
  - BUG #9 : cls_encode defined once (was duplicated 4 times)
  - Consistent SEED=42 throughout this file (frozen baseline section)
  - Removed duplicate imports

Outputs (saved to --outdir):
    frozen_E_text.pt      — [N, H_t] frozen PubMedBERT CLS embeddings
    frozen_E_smiles.pt    — [N, H_s] frozen ChemBERTa CLS embeddings
    baseline_df.csv       — the N-row dataframe used (with text_rich)

Usage:
    python 02_preprocess_and_baselines.py --csv chembl_mechanisms.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text_rich(row):
    parts = [row["mechanism_of_action"]]
    if isinstance(row.get("target_name"), str):
        parts.append(f"Target: {row['target_name']}.")
    if isinstance(row.get("action_type"), str):
        parts.append(f"Action: {row['action_type']}.")
    if isinstance(row.get("pref_name"), str):
        parts.append(f"Drug: {row['pref_name']}.")
    return " ".join(parts)


@torch.no_grad()
def cls_encode(texts, tok, model, device, max_length=128, bs=64):
    """CLS-token encoder — defined ONCE (was duplicated 4x in original)."""
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="CLS encode"):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state[:, 0, :]   # [CLS]
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


def recall_mrr(sim):
    """Given [N,N] cosine-sim tensor, return Recall@1 and MRR."""
    N = sim.shape[0]
    gt = np.arange(N)
    top1 = sim.argmax(dim=1).numpy()
    recall1 = (top1 == gt).mean()
    ranks = []
    for i in range(N):
        order = torch.argsort(sim[i], descending=True).numpy()
        rank_pos = int(np.where(order == i)[0][0]) + 1
        ranks.append(1.0 / rank_pos)
    return float(recall1), float(np.mean(ranks))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    set_seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["smiles", "mechanism_of_action"])
    df = df[df["mechanism_of_action"].str.len() > 15].copy()

    # Build text_rich if not already present
    if "text_rich" not in df.columns:
        df["text_rich"] = df.apply(build_text_rich, axis=1)

    # FIX #2: was max(1000, len(df)) — crashes when len(df) < 1000
    # Correct intent is to cap at N, not upsample
    N = min(args.n_samples, len(df))
    df = df.sample(N, random_state=SEED).reset_index(drop=True)
    print(f"Using {len(df)} pairs (seed={SEED})")

    smiles_list = df["smiles"].tolist()
    text_list   = df["text_rich"].tolist()

    # ------------------------------------------------------------------
    # 2. Load encoders
    # ------------------------------------------------------------------
    print("\nLoading ChemBERTa ...")
    chem_tok   = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", use_fast=True)
    chem_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device).eval()

    print("Loading PubMedBERT ...")
    text_tok   = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_fast=True)
    text_model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract").to(device).eval()

    # ------------------------------------------------------------------
    # 3. Encode  (FIX #4: max_length parameter now properly passed through)
    # ------------------------------------------------------------------
    print("\nEncoding SMILES with ChemBERTa ...")
    E_smiles = cls_encode(smiles_list, chem_tok, chem_model, device,
                          max_length=128, bs=64)

    print("Encoding text with PubMedBERT ...")
    E_text = cls_encode(text_list, text_tok, text_model, device,
                        max_length=96, bs=64)

    print(f"  E_smiles: {E_smiles.shape}  E_text: {E_text.shape}")

    # ------------------------------------------------------------------
    # 4. Frozen baseline metrics
    # ------------------------------------------------------------------
    E_text_n   = F.normalize(E_text,   dim=1)
    E_smiles_n = F.normalize(E_smiles, dim=1)
    sim = E_text_n @ E_smiles_n.T

    rec1, mrr = recall_mrr(sim)
    print(f"\n[Frozen ChemBERTa + PubMedBERT] Recall@1={rec1:.3f}  MRR={mrr:.3f}")

    # Shuffled sanity baseline
    perm = np.random.permutation(len(df))
    sim_shuf = E_text_n @ E_smiles_n[perm].T
    rec1_shuf = (sim_shuf.argmax(dim=1).numpy() == np.arange(len(df))).mean()
    print(f"[Shuffled baseline]              Recall@1={rec1_shuf:.3f}")

    # ------------------------------------------------------------------
    # 5. Heatmap — frozen baseline
    # ------------------------------------------------------------------
    K = min(40, len(df))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(sim[:K, :K].numpy(), vmin=-1, vmax=1, cmap="viridis", ax=ax)
    ax.set_title("Frozen encoders: cosine similarity (first K pairs)")
    ax.set_xlabel("SMILES index"); ax.set_ylabel("Text index")
    plt.tight_layout()
    heatmap_path = os.path.join(args.outdir, "frozen_baseline_heatmap.png")
    plt.savefig(heatmap_path, dpi=120)
    plt.close()
    print(f"Heatmap saved → {heatmap_path}")

    # ------------------------------------------------------------------
    # 6. Save embeddings and dataframe for training script
    # ------------------------------------------------------------------
    torch.save(E_text,   os.path.join(args.outdir, "frozen_E_text.pt"))
    torch.save(E_smiles, os.path.join(args.outdir, "frozen_E_smiles.pt"))
    df.to_csv(os.path.join(args.outdir, "baseline_df.csv"), index=False)
    print(f"\nSaved embeddings and dataframe to {args.outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      type=str, default="chembl_mechanisms.csv")
    parser.add_argument("--outdir",   type=str, default="outputs")
    parser.add_argument("--n_samples", type=int, default=3000,
                        help="Max pairs to use (default 3000). FIX: was min→max in original.")
    main(parser.parse_args())