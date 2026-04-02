# -*- coding: utf-8 -*-
"""
05_compare_moleculestm_vs_thinbridges.py
==========================================
Head-to-head comparison of MoleculeSTM (zero-shot) vs Thin Bridges (trained)
on the SAME ChEMBL test pairs, under the SAME evaluation metrics.

FIXES vs previous broken version:
  - FIXED import: mega_mol_bart.MegaMolBART  (not mega_molbart.MegaMolBART)
  - FIXED model init: MegaMolBART(vocab_path=..., input_dir=None, output_dir=None)
                      molecule_model = MegaMolBART_wrapper.model
  - FIXED checkpoint: mol2latent_model.pth   (not molecule2latent.pth)
  - FIXED checkpoint: text2latent_model.pth  (not text2latent.pth)
  - FIXED mol repr: get_molecule_repr_MoleculeSTM(smi, mol2latent=None,
                        molecule_type="SMILES", MegaMolBART_wrapper=...)
  - FIXED text repr: pooler_output  (not last_hidden_state[:,0,:])
  - FIXED prepare_text_tokens returns (input_ids, attention_mask) not a dict

NEW vs previous version:
  - ADDED: --strip_drug_name flag  (leakage ablation: removes "Drug: X." from text)
  - ADDED: sample text preview so you can verify what is being encoded
  - ADDED: plot_score_distributions()  (positive vs negative separation)
  - ADDED: title_suffix on all plots to label with/without drug name condition

Usage:
    # Default — with drug name in text (tests for name-based shortcut)
    python 05_compare_moleculestm_vs_thinbridges.py \\
        --moleculestm_dir /workspace/MoleculeSTM \\
        --bridge_outdir   /workspace/MoleculeLens/outputs \\
        --outdir          /workspace/MoleculeLens/outputs/comparison_withdrug

    # Leakage ablation — strip drug name (fair structural alignment test)
    python 05_compare_moleculestm_vs_thinbridges.py \\
        --moleculestm_dir /workspace/MoleculeSTM \\
        --bridge_outdir   /workspace/MoleculeLens/outputs \\
        --outdir          /workspace/MoleculeLens/outputs/comparison_nodrug \\
        --strip_drug_name
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42


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
# Shared evaluation metrics
# ---------------------------------------------------------------------------
def recall_mrr(S_np):
    """Recall@1 and MRR from [N,N] similarity matrix (diagonal = ground truth)."""
    N       = S_np.shape[0]
    top1    = S_np.argmax(axis=1)
    recall1 = (top1 == np.arange(N)).mean()
    ranks   = []
    for i in range(N):
        order    = np.argsort(-S_np[i])
        rank_pos = int(np.where(order == i)[0][0]) + 1
        ranks.append(1.0 / rank_pos)
    return float(recall1), float(np.mean(ranks))


def recall_at_k(S_np, k_list=(1, 5, 10)):
    N    = S_np.shape[0]
    topk = np.argsort(-S_np, axis=1)
    return {k: sum(i in topk[i, :k] for i in range(N)) / N for k in k_list}


def t_choose_one(S_np, T_list=(4, 10, 20), n_trials=1000, seed=42):
    """
    T-choose-one accuracy (MoleculeSTM paper protocol).
    For each of n_trials queries, sample T-1 random negatives and check
    if the positive is top-ranked. Both S->T and T->S directions.
    """
    rng    = np.random.default_rng(seed)
    N      = S_np.shape[0]
    trials = min(n_trials, N)
    results = {}
    for T in T_list:
        if T > N:
            results[T] = {"S->T": float("nan"), "T->S": float("nan")}
            continue
        s2t_hits = t2s_hits = 0
        for i in range(trials):
            negs  = rng.choice([j for j in range(N) if j != i],
                               size=T - 1, replace=False)
            cands = np.concatenate([[i], negs])
            if cands[np.argmax(S_np[i,      cands])] == i:
                s2t_hits += 1
            if cands[np.argmax(S_np[cands,  i    ])] == i:
                t2s_hits += 1
        results[T] = {"S->T": s2t_hits / trials, "T->S": t2s_hits / trials}
    return results


# ---------------------------------------------------------------------------
# MoleculeSTM inference
# ---------------------------------------------------------------------------
def load_moleculestm(moleculestm_dir, model_dir, device):
    """
    Load pretrained MoleculeSTM SMILES checkpoint.

    Import pattern and checkpoint filenames taken directly from the working
    downstream_01_retrieval_ChEMBL.py fix (document attached by user):

      from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
      MegaMolBART_wrapper = MegaMolBART(vocab_path=..., input_dir=None, output_dir=None)
      molecule_model      = MegaMolBART_wrapper.model
      mol2latent.pth      → mol2latent_model.pth
      text2latent.pth     → text2latent_model.pth
      text repr           → pooler_output
      mol repr            → get_molecule_repr_MoleculeSTM(smi, mol2latent=None,
                                molecule_type="SMILES", MegaMolBART_wrapper=wrapper)
    """
    sys.path.insert(0, moleculestm_dir)

    # CORRECT import — mega_mol_bart (underscore), not MegaMolBART (camel)
    from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
    from MoleculeSTM.utils import prepare_text_tokens, get_molecule_repr_MoleculeSTM

    # Molecule encoder — wrapper pattern, then extract .model
    vocab_path          = os.path.join(moleculestm_dir, "MoleculeSTM", "bart_vocab.txt")
    MegaMolBART_wrapper = MegaMolBART(vocab_path=vocab_path,
                                      input_dir=None, output_dir=None)
    molecule_model      = MegaMolBART_wrapper.model
    mol2latent          = nn.Linear(256, 256)

    molecule_model.load_state_dict(
        torch.load(os.path.join(model_dir, "molecule_model.pth"), map_location="cpu"))
    # CORRECT filename: mol2latent_model.pth
    mol2latent.load_state_dict(
        torch.load(os.path.join(model_dir, "mol2latent_model.pth"), map_location="cpu"))
    molecule_model = molecule_model.to(device).eval()
    mol2latent     = mol2latent.to(device).eval()

    # Text encoder
    pretrained_SciBERT_folder = os.path.join(moleculestm_dir, "data", "pretrained_SciBERT")
    text_tokenizer = AutoTokenizer.from_pretrained(
        "allenai/scibert_scivocab_uncased", cache_dir=pretrained_SciBERT_folder)
    text_model  = AutoModel.from_pretrained(
        "allenai/scibert_scivocab_uncased", cache_dir=pretrained_SciBERT_folder)
    text2latent = nn.Linear(768, 256)

    text_model.load_state_dict(
        torch.load(os.path.join(model_dir, "text_model.pth"), map_location="cpu"))
    # CORRECT filename: text2latent_model.pth
    text2latent.load_state_dict(
        torch.load(os.path.join(model_dir, "text2latent_model.pth"), map_location="cpu"))
    text_model  = text_model.to(device).eval()
    text2latent = text2latent.to(device).eval()

    return (MegaMolBART_wrapper, mol2latent,
            text_tokenizer, text_model, text2latent,
            get_molecule_repr_MoleculeSTM, prepare_text_tokens)


@torch.no_grad()
def moleculestm_embeddings(smiles_list, text_list, components, device,
                            batch_size=32, max_seq_len=512):
    (MegaMolBART_wrapper, mol2latent,
     text_tokenizer, text_model, text2latent,
     get_molecule_repr_MoleculeSTM, prepare_text_tokens) = components

    mol_reprs, txt_reprs = [], []
    for start in tqdm(range(0, len(smiles_list), batch_size),
                      desc="MoleculeSTM embeddings"):
        smi_batch  = smiles_list[start: start + batch_size]
        text_batch = text_list[start:   start + batch_size]

        # Molecule side — mol2latent=None means wrapper handles projection internally
        mol_repr = get_molecule_repr_MoleculeSTM(
            smi_batch, mol2latent=None,
            molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper)
        mol_repr = F.normalize(mol2latent(mol_repr), dim=-1)
        mol_reprs.append(mol_repr.cpu())

        # Text side — prepare_text_tokens returns (input_ids, attention_mask)
        text_tokens_ids, text_masks = prepare_text_tokens(
            device=device, description=text_batch,
            tokenizer=text_tokenizer, max_seq_len=max_seq_len)
        text_out  = text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        # CORRECT: pooler_output (not last_hidden_state[:,0,:])
        text_repr = text_out["pooler_output"]
        text_repr = F.normalize(text2latent(text_repr), dim=-1)
        txt_reprs.append(text_repr.cpu())

    return torch.cat(mol_reprs, dim=0), torch.cat(txt_reprs, dim=0)


# ---------------------------------------------------------------------------
# Thin Bridges inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def cls_encode(texts, tok, model, device, max_length=96, bs=64):
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="Encoding text (S-BiomedRoBERTa)"):
        batch = texts[i:i + bs]
        enc   = tok(batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        out   = model(**enc).last_hidden_state[:, 0, :]
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


def ecfp4_bitvect(smi, nbits=2048, radius=2):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(nbits, dtype=np.float32)
    bv  = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


@torch.no_grad()
def thinbridges_embeddings(smiles_list, text_list, bridge_outdir, device):
    """Load saved Thin Bridges projection heads and project the given pairs."""
    TEXT_MODEL = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
    text_tok   = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    text_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    Z_text = cls_encode(text_list, text_tok, text_model, device,
                        max_length=96, bs=64)

    X_mol = np.stack([ecfp4_bitvect(s)
                      for s in tqdm(smiles_list, desc="ECFP4 fingerprints")])

    SHARED_D  = 256
    proj_text = nn.Linear(Z_text.shape[1], SHARED_D).to(device)
    proj_mol  = nn.Linear(X_mol.shape[1],  SHARED_D).to(device)
    proj_text.load_state_dict(torch.load(
        os.path.join(bridge_outdir, "global_proj_text.pt"), map_location=device))
    proj_mol.load_state_dict(torch.load(
        os.path.join(bridge_outdir, "global_proj_mol.pt"),  map_location=device))
    proj_text.eval(); proj_mol.eval()

    Bt = F.normalize(proj_text(Z_text.to(device)), dim=1).cpu()
    Bm = F.normalize(proj_mol(
        torch.tensor(X_mol, dtype=torch.float32).to(device)), dim=1).cpu()
    return Bt, Bm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_heatmaps(S_mstm, S_bridge, outdir, title_suffix="", K=40):
    K = min(K, S_mstm.shape[0], S_bridge.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(S_mstm[:K,   :K], vmin=-1, vmax=1, cmap="viridis", ax=axes[0])
    axes[0].set_title("MoleculeSTM (zero-shot)", fontsize=12)
    axes[0].set_xlabel("Molecule index"); axes[0].set_ylabel("Text index")
    sns.heatmap(S_bridge[:K, :K], vmin=-1, vmax=1, cmap="viridis", ax=axes[1])
    axes[1].set_title("Thin Bridges (trained)", fontsize=12)
    axes[1].set_xlabel("Molecule index"); axes[1].set_ylabel("Text index")
    plt.suptitle(
        f"Cosine similarity: MoleculeSTM vs Thin Bridges{title_suffix}",
        fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "comparison_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")


def plot_recall_at_k(rk_mstm, rk_bridge, outdir, title_suffix=""):
    k_list = sorted(rk_mstm.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_list, [rk_mstm[k]   for k in k_list], "o--",
            label="MoleculeSTM (zero-shot)", linewidth=2)
    ax.plot(k_list, [rk_bridge[k] for k in k_list], "s-",
            label="Thin Bridges (trained)",  linewidth=2)
    ax.axhline(y=1/k_list[0], color="gray", linestyle=":", label="Random@1")
    ax.set_xlabel("k"); ax.set_ylabel("Recall@k")
    ax.set_title(f"Recall@k: MoleculeSTM vs Thin Bridges{title_suffix}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "comparison_recall_at_k.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def plot_t_choose_one(tco_mstm, tco_bridge, outdir, title_suffix=""):
    T_list = sorted(tco_mstm.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, direction in zip(axes, ["S->T", "T->S"]):
        vals_mstm   = [tco_mstm[T][direction]  for T in T_list]
        vals_bridge = [tco_bridge[T][direction] for T in T_list]
        vals_random = [1/T                      for T in T_list]
        x = np.arange(len(T_list)); w = 0.25
        ax.bar(x - w, vals_mstm,   width=w, label="MoleculeSTM",  color="steelblue",  alpha=0.85)
        ax.bar(x,     vals_bridge, width=w, label="Thin Bridges",  color="darkorange", alpha=0.85)
        ax.bar(x + w, vals_random, width=w, label="Random",        color="gray",       alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f"T={T}" for T in T_list])
        ax.set_ylabel("Accuracy"); ax.set_title(f"T-choose-one ({direction})")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.suptitle(
        f"T-choose-one: MoleculeSTM vs Thin Bridges{title_suffix}", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "comparison_t_choose_one.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")


def plot_score_distributions(S_mstm, S_bridge, outdir, title_suffix=""):
    """Histogram of positive vs negative cosine scores — shows separation quality."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, S_np, title in zip(
            axes,
            [S_mstm,           S_bridge],
            ["MoleculeSTM (zero-shot)", "Thin Bridges (trained)"]):
        N    = S_np.shape[0]
        mask = ~np.eye(N, dtype=bool)
        pos  = np.diag(S_np)
        neg  = S_np[mask]
        ax.hist(neg, bins=80, alpha=0.5, density=True, label="Negatives", color="steelblue")
        ax.hist(pos, bins=30, alpha=0.8, density=True, label="Positives", color="orange")
        ax.set_title(title); ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density"); ax.legend(); ax.set_xlim(-1, 1)
    plt.suptitle(
        f"Positive vs Negative similarity distributions{title_suffix}", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "comparison_score_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    set_seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 1. Load shared test pairs
    # ------------------------------------------------------------------
    df_path = os.path.join(args.bridge_outdir, "global_df.csv")
    print(f"Loading pairs from {df_path} ...")
    df = pd.read_csv(df_path).dropna(subset=["smiles", "text_rich"])
    df = df[df["text_rich"].str.len() > 20].reset_index(drop=True)

    if args.max_pairs and len(df) > args.max_pairs:
        df = df.sample(args.max_pairs, random_state=SEED).reset_index(drop=True)

    # --strip_drug_name: remove "Drug: aspirin." from text_rich
    # This tests whether MoleculeSTM's performance is driven by recognising
    # drug names in SciBERT (name-based shortcut) vs true structural alignment
    if args.strip_drug_name:
        df["text_rich"] = df["text_rich"].str.replace(
            r"Drug:\s*[^.]+\.\s*", "", regex=True).str.strip()
        print("WARNING: Drug names stripped from text_rich (leakage ablation mode)")

    title_suffix = " (no drug name)" if args.strip_drug_name else " (with drug name)"
    print(f"Evaluating on {len(df)} pairs  [{title_suffix.strip()}]")
    print(f"Sample text [0]: {df['text_rich'].iloc[0]}")
    print(f"Sample text [1]: {df['text_rich'].iloc[1]}\n")

    smiles_list = df["smiles"].tolist()
    text_list   = df["text_rich"].tolist()

    # ------------------------------------------------------------------
    # 2. MoleculeSTM embeddings (zero-shot)
    # ------------------------------------------------------------------
    print("=" * 55)
    print("Loading MoleculeSTM ...")
    model_dir = os.path.join(
        args.moleculestm_dir, "data", "pretrained_MoleculeSTM", "SMILES")
    sys.path.insert(0, args.moleculestm_dir)
    components = load_moleculestm(args.moleculestm_dir, model_dir, device)
    mol_mstm, txt_mstm = moleculestm_embeddings(
        smiles_list, text_list, components, device, batch_size=32)
    S_mstm = (txt_mstm @ mol_mstm.T).numpy()
    print("MoleculeSTM done.\n")

    # ------------------------------------------------------------------
    # 3. Thin Bridges embeddings (trained)
    # ------------------------------------------------------------------
    print("=" * 55)
    print("Loading Thin Bridges ...")
    Bt, Bm   = thinbridges_embeddings(
        smiles_list, text_list, args.bridge_outdir, device)
    S_bridge = (Bt @ Bm.T).numpy()
    print("Thin Bridges done.\n")

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    print("=" * 55)
    print("Computing metrics ...")

    rec1_mstm,   mrr_mstm   = recall_mrr(S_mstm)
    rec1_bridge, mrr_bridge = recall_mrr(S_bridge)
    rk_mstm    = recall_at_k(S_mstm,   k_list=[1, 5, 10])
    rk_bridge  = recall_at_k(S_bridge, k_list=[1, 5, 10])
    tco_mstm   = t_choose_one(S_mstm,   T_list=[4, 10, 20], seed=SEED)
    tco_bridge = t_choose_one(S_bridge, T_list=[4, 10, 20], seed=SEED)
    random_r1  = 1 / len(df)

    # ------------------------------------------------------------------
    # 5. Print table
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"  COMPARISON TABLE  {title_suffix.strip()}")
    print(f"  Same ChEMBL pairs  N={len(df)}")
    print("=" * 68)
    print(f"{'Metric':<34} {'MoleculeSTM':>11} {'ThinBridges':>11} {'Random':>8}")
    print("-" * 68)
    print(f"{'Recall@1':<34} {rec1_mstm:>11.3f} {rec1_bridge:>11.3f} {random_r1:>8.4f}")
    print(f"{'MRR':<34} {mrr_mstm:>11.3f} {mrr_bridge:>11.3f} {'—':>8}")
    print(f"{'Recall@5':<34} {rk_mstm[5]:>11.3f} {rk_bridge[5]:>11.3f} {'—':>8}")
    print(f"{'Recall@10':<34} {rk_mstm[10]:>11.3f} {rk_bridge[10]:>11.3f} {'—':>8}")
    print("-" * 68)
    for T in [4, 10, 20]:
        r = 1/T
        s2t_m, s2t_b = tco_mstm[T]["S->T"], tco_bridge[T]["S->T"]
        t2s_m, t2s_b = tco_mstm[T]["T->S"], tco_bridge[T]["T->S"]
        print(f"{'T='+str(T)+' S->T':<34} {s2t_m:>11.3f} {s2t_b:>11.3f} {r:>8.3f}")
        print(f"{'T='+str(T)+' T->S':<34} {t2s_m:>11.3f} {t2s_b:>11.3f} {r:>8.3f}")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 6. Save CSV
    # ------------------------------------------------------------------
    rows = [
        {"Metric": "Recall@1",  "MoleculeSTM": rec1_mstm,
         "ThinBridges": rec1_bridge,  "Random": random_r1},
        {"Metric": "MRR",       "MoleculeSTM": mrr_mstm,
         "ThinBridges": mrr_bridge,   "Random": None},
        {"Metric": "Recall@5",  "MoleculeSTM": rk_mstm[5],
         "ThinBridges": rk_bridge[5], "Random": None},
        {"Metric": "Recall@10", "MoleculeSTM": rk_mstm[10],
         "ThinBridges": rk_bridge[10],"Random": None},
    ]
    for T in [4, 10, 20]:
        rows += [
            {"Metric": f"T={T} S->T", "MoleculeSTM": tco_mstm[T]["S->T"],
             "ThinBridges": tco_bridge[T]["S->T"], "Random": 1/T},
            {"Metric": f"T={T} T->S", "MoleculeSTM": tco_mstm[T]["T->S"],
             "ThinBridges": tco_bridge[T]["T->S"], "Random": 1/T},
        ]
    out_csv = os.path.join(args.outdir, "comparison_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nResults CSV → {out_csv}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    plot_heatmaps(S_mstm, S_bridge, args.outdir,
                  title_suffix=title_suffix)
    plot_recall_at_k(rk_mstm, rk_bridge, args.outdir,
                     title_suffix=title_suffix)
    plot_t_choose_one(tco_mstm, tco_bridge, args.outdir,
                      title_suffix=title_suffix)
    plot_score_distributions(S_mstm, S_bridge, args.outdir,
                             title_suffix=title_suffix)

    print(f"\nAll outputs saved to {args.outdir}/")
    if not args.strip_drug_name:
        print("\nNEXT STEP: Run again with --strip_drug_name to check whether")
        print("MoleculeSTM's T-choose-one score drops — that drop reveals")
        print("name-based shortcut vs true structural alignment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moleculestm_dir", type=str,
                        default="/workspace/MoleculeSTM",
                        help="Root directory of MoleculeSTM repo")
    parser.add_argument("--bridge_outdir",   type=str,
                        default="/workspace/MoleculeLens/outputs",
                        help="outputs/ dir from 03b_train_global_bridge.py")
    parser.add_argument("--outdir",          type=str,
                        default="/workspace/MoleculeLens/outputs/comparison",
                        help="Where to save comparison outputs")
    parser.add_argument("--max_pairs",       type=int, default=None,
                        help="Cap number of pairs (default: use all 2699)")
    parser.add_argument("--strip_drug_name", action="store_true",
                        help="Remove 'Drug: X.' from text_rich to test whether "
                             "MoleculeSTM relies on name recognition rather than "
                             "structural alignment (leakage ablation)")
    main(parser.parse_args())
