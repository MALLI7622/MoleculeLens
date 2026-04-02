# -*- coding: utf-8 -*-
"""
06_graphormer_retrieval.py
===========================
Benchmarks Graphormer-base + SciBERT on the ChEMBL drug-mechanism retrieval
task using the identical scaffold split and evaluation metrics as Thin Bridges
(03_train_scaffold_split.py) and MoleculeSTM (05_compare_moleculestm_vs_thinbridges.py).

Three conditions are evaluated and clearly distinguished:

  Condition 1 — Zero-shot
    Graph  : Graphormer-base CLS token (768-d), no training
    Text   : SciBERT pooler output (768-d), no training
    Eval   : cosine similarity of raw embeddings
    Compare: analogous to MoleculeSTM zero-shot

  Condition 2 — MolPrompt-style (faithful replication)
    Graph  : Graphormer-base, UNFROZEN, trained end-to-end
    Text   : SciBERT, UNFROZEN, trained end-to-end
    Heads  : 2-layer MLP  768→768→256  (exactly as in contrastive_GraST_gp.py)
    Loss   : symmetric NT-Xent only  (no margin term — matches MolPrompt paper)
    Compare: the closest achievable approximation to MolPrompt without its
             proprietary 200k-step pretrained checkpoint

  Condition 3 — Thin-Bridges-style (frozen encoders, linear projectors)
    Graph  : Graphormer-base, FROZEN
    Text   : SciBERT, FROZEN
    Heads  : single linear layer  768→256  (matches Thin Bridges architecture)
    Loss   : NT-Xent + hard-negative margin  (matches Thin Bridges loss)
    Compare: same objective as Thin Bridges; shows effect of swapping
             ECFP4→Graphormer and S-BiomedRoBERTa→SciBERT

Key architecture differences (source: model/contrastive_GraST_gp.py):
    MolPrompt         → encoders unfrozen, MLP proj heads, pure NT-Xent
    Thin Bridges      → frozen text enc, linear proj heads, NT-Xent + margin
    This script C2    → encoders unfrozen, MLP proj heads, pure NT-Xent  ← faithful
    This script C3    → frozen encoders,  linear proj heads, NT-Xent+margin ← Thin Bridges swap

Prerequisites
-------------
  export LD_LIBRARY_PATH=~/miniconda3/envs/MoleculeSTM/lib:$LD_LIBRARY_PATH
  cd /home/cheriearjun/MolPrompt/Molprop_dataset && compile myalgos.pyx once

Usage
-----
# All three conditions (standard, with drug name)
python 06_graphormer_retrieval.py --csv chembl_mechanisms.csv --outdir outputs/graphormer

# Drug name leakage ablation
python 06_graphormer_retrieval.py --csv chembl_mechanisms.csv \\
    --outdir outputs/graphormer --remove_drug_name

# Zero-shot only (fast, no training)
python 06_graphormer_retrieval.py --csv chembl_mechanisms.csv \\
    --outdir outputs/graphormer --zero_shot_only
"""

import os
import sys
import copy
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MOLPROMPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "MolPrompt"))

SCIBERT_PATH = os.environ.get(
    "SCIBERT_PATH",
    os.path.expanduser(
        "~/data/pretrained_SciBERT/models--allenai--scibert_scivocab_uncased"
        "/snapshots/24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
    )
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
SEED        = 0          # same as 03_train_scaffold_split.py
GRAPH_DIM   = 768        # Graphormer hidden dim
TEXT_DIM    = 768        # SciBERT hidden dim
SHARED_D    = 256        # projection output dim (matches both MolPrompt and Thin Bridges)
MAX_TXT_LEN = 96
MAX_NODE    = 512
MAX_HOP     = 20
SPATIAL_MAX = 20
LOG_EVERY   = 10

# Condition 2 (MolPrompt-style)
C2_EPOCHS       = 100
C2_LR           = 2e-5   # lower LR because encoders are unfrozen (avoids catastrophic forgetting)
C2_WEIGHT_DECAY = 0.0
C2_TEMP         = 0.1    # MolPrompt default temperature
C2_BATCH_SIZE   = 16     # smaller batch because full encoders are in the gradient graph

# Condition 3 (Thin-Bridges-style)
C3_EPOCHS       = 100
C3_LR           = 1e-3
C3_WEIGHT_DECAY = 1e-4
C3_TEMP         = 0.07
C3_MARGIN       = 0.1
C3_BATCH_SIZE   = 128    # larger batch is fine — frozen encoders, pre-cached embeddings


# ===========================================================================
# Reproducibility
# ===========================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Graph featurisation (MolPrompt pipeline)
# ===========================================================================
def _setup_molprompt():
    if MOLPROMPT_DIR not in sys.path:
        sys.path.insert(0, MOLPROMPT_DIR)
    from Molprop_dataset.MoleculeNet_Graph import mol_to_graph_data_obj_simple
    from Molprop_dataset.wrapper import preprocess_item
    return mol_to_graph_data_obj_simple, preprocess_item


def smiles_to_graph(smi, mol_to_graph, preprocess):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return preprocess(mol_to_graph(mol))


# Padding helpers (mirror collator_prop.py, no prompt fields needed)
def _pad1d(x, n):
    x = x + 1
    if x.size(0) < n:
        t = x.new_zeros([n]); t[:x.size(0)] = x; x = t
    return x.unsqueeze(0)

def _pad2d(x, n):
    x = x + 1
    if x.size(0) < n:
        t = x.new_zeros([n, x.size(1)]); t[:x.size(0)] = x; x = t
    return x.unsqueeze(0)

def _pad_attn_bias(x, n):
    if x.size(0) < n:
        t = x.new_zeros([n, n]).fill_(float("-inf"))
        t[:x.size(0), :x.size(0)] = x; t[x.size(0):, :x.size(0)] = 0; x = t
    return x.unsqueeze(0)

def _pad_edge_type(x, n):
    if x.size(0) < n:
        t = x.new_zeros([n, n, x.size(-1)]); t[:x.size(0), :x.size(0)] = x; x = t
    return x.unsqueeze(0)

def _pad_spatial_pos(x, n):
    x = x + 1
    if x.size(0) < n:
        t = x.new_zeros([n, n]); t[:x.size(0), :x.size(0)] = x; x = t
    return x.unsqueeze(0)

def _pad3d(x, n1, n2, n3):
    x = x + 1
    l1, l2, l3, l4 = x.size()
    if l1 < n1 or l2 < n2 or l3 < n3:
        t = x.new_zeros([n1, n2, n3, l4]); t[:l1, :l2, :l3] = x; x = t
    return x.unsqueeze(0)


def collate_graphs(graphs, multi_hop_max_dist=MAX_HOP, spatial_pos_max=SPATIAL_MAX):
    graphs = [g for g in graphs if g is not None and g.x.size(0) <= MAX_NODE]
    attn_biases     = [g.attn_bias      for g in graphs]
    attn_edge_types = [g.attn_edge_type for g in graphs]
    spatial_poses   = [g.spatial_pos    for g in graphs]
    in_degrees      = [g.in_degree      for g in graphs]
    out_degrees     = [g.out_degree     for g in graphs]
    xs              = [g.x              for g in graphs]
    edge_inputs     = [g.edge_input[:, :, :multi_hop_max_dist, :] for g in graphs]

    for i in range(len(attn_biases)):
        attn_biases[i][1:, 1:][spatial_poses[i] >= spatial_pos_max] = float("-inf")

    max_n = max(x.size(0) for x in xs)
    max_d = max(e.size(-2) for e in edge_inputs)
    return dict(
        x              = torch.cat([_pad2d(i, max_n)              for i in xs]),
        edge_input     = torch.cat([_pad3d(i, max_n, max_n, max_d) for i in edge_inputs]),
        attn_bias      = torch.cat([_pad_attn_bias(i, max_n + 1)   for i in attn_biases]),
        attn_edge_type = torch.cat([_pad_edge_type(i, max_n)        for i in attn_edge_types]),
        spatial_pos    = torch.cat([_pad_spatial_pos(i, max_n)      for i in spatial_poses]),
        in_degree      = torch.cat([_pad1d(i, max_n)               for i in in_degrees]),
        out_degree     = torch.cat([_pad1d(i, max_n)               for i in out_degrees]),
    )


# ===========================================================================
# Encoders
# ===========================================================================
def load_graph_encoder(device):
    if MOLPROMPT_DIR not in sys.path:
        sys.path.insert(0, MOLPROMPT_DIR)
    from model.graphormer_baseline import GraphEncoderBaseline
    enc = GraphEncoderBaseline(pretrained=True).to(device)
    return enc


def load_text_encoder(device):
    from transformers import BertTokenizer, BertModel
    print(f"Loading SciBERT from {SCIBERT_PATH} ...")
    tok   = BertTokenizer.from_pretrained(SCIBERT_PATH)
    model = BertModel.from_pretrained(SCIBERT_PATH).to(device)
    return tok, model


@torch.no_grad()
def encode_graphs_frozen(graphs, encoder, device, batch_size=32):
    """Encode graphs with frozen encoder. Returns [N, 768] tensor."""
    encoder.eval()
    reps = []
    for i in tqdm(range(0, len(graphs), batch_size), desc="Encode graphs"):
        b  = collate_graphs(graphs[i: i + batch_size])
        rep = encoder(b["x"].to(device), b["attn_bias"].to(device),
                      b["attn_edge_type"].to(device), b["spatial_pos"].to(device),
                      b["in_degree"].to(device), b["out_degree"].to(device),
                      b["edge_input"].to(device))
        reps.append(rep.cpu())
    return torch.cat(reps, dim=0)


@torch.no_grad()
def encode_texts_frozen(texts, tok, model, device, bs=64):
    """Encode texts with frozen SciBERT. Returns [N, 768] pooler_output tensor."""
    model.eval()
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="Encode texts"):
        enc = tok(texts[i: i + bs], padding=True, truncation=True,
                  max_length=MAX_TXT_LEN, return_tensors="pt").to(device)
        outs.append(model(**enc)["pooler_output"].cpu())
    return torch.cat(outs, dim=0)


# ===========================================================================
# Projection heads
# ===========================================================================
def make_mlp_head(in_dim, hidden_dim, out_dim):
    """2-layer MLP with ReLU — identical to MolPrompt's graph_proj_head / text_proj_head."""
    return nn.Sequential(
        nn.Linear(in_dim,    hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


# ===========================================================================
# Losses
# ===========================================================================
def nt_xent_loss(z1, z2, T):
    """Pure symmetric NT-Xent (MolPrompt paper loss). No margin term."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N  = z1.size(0)
    logits_12 = (z1 @ z2.T) / T
    logits_21 = logits_12.T
    labels = torch.arange(N, device=z1.device)
    return 0.5 * (F.cross_entropy(logits_12, labels) +
                  F.cross_entropy(logits_21, labels))


def _build_weights(grps, device):
    B = len(grps)
    W = torch.ones((B, B), device=device)
    for i in range(B):
        for j in range(B):
            if i != j and grps[i] == grps[j]:
                W[i, j] = 2.0
    return W


def _hardest_neg(logits, groups):
    B = logits.size(0)
    hard = torch.full((B,), float("-inf"), device=logits.device)
    for i in range(B):
        mask = torch.tensor([(groups[i] == groups[j] and j != i)
                             for j in range(B)], device=logits.device)
        if mask.any():
            hard[i] = logits[i][mask].max()
    return hard


def nt_xent_margin_loss(bt, bm, groups, device, T, margin):
    """NT-Xent + hard-negative margin loss (Thin Bridges loss)."""
    bt = F.normalize(bt, dim=1)
    bm = F.normalize(bm, dim=1)
    logits = (bt @ bm.T) / T
    labels = torch.arange(bt.size(0), device=bt.device)

    W = _build_weights(groups, device)
    ce_r = F.cross_entropy(logits,   labels, reduction="none")
    ce_c = F.cross_entropy(logits.T, labels, reduction="none")
    with torch.no_grad():
        B = logits.size(0)
        wr = (W.sum(1) - W.diag()) / (B - 1)
        wc = (W.sum(0) - W.diag()) / (B - 1)
    loss_nce = 0.5 * ((ce_r * wr).mean() + (ce_c * wc).mean())

    pos  = logits.diag()
    hard = _hardest_neg(logits, groups)
    mask = torch.isfinite(hard)
    m_loss = (torch.clamp(margin - (pos[mask] - hard[mask]), min=0).mean()
              if mask.any() else torch.tensor(0.0, device=logits.device))
    return loss_nce + m_loss


# ===========================================================================
# Datasets
# ===========================================================================
class EmbedPairDataset(Dataset):
    """Pre-cached embedding pairs (for C3 frozen-encoder training)."""
    def __init__(self, Z_text, Z_mol, groups):
        self.Zt  = Z_text.float()
        self.Zm  = Z_mol.float()
        self.grp = np.array(groups).astype(str)
    def __len__(self): return len(self.Zt)
    def __getitem__(self, i): return self.Zt[i], self.Zm[i], self.grp[i], i


class GraphTextPairDataset(Dataset):
    """Raw graph + text pairs (for C2 end-to-end training)."""
    def __init__(self, graphs, texts, groups):
        self.graphs = graphs
        self.texts  = texts
        self.grp    = np.array(groups).astype(str)
    def __len__(self): return len(self.graphs)
    def __getitem__(self, i): return self.graphs[i], self.texts[i], self.grp[i], i


# ===========================================================================
# Evaluation metrics (identical to 05_compare_moleculestm_vs_thinbridges.py)
# ===========================================================================
def recall_mrr(S_np):
    N    = S_np.shape[0]
    top1 = S_np.argmax(axis=1)
    r1   = (top1 == np.arange(N)).mean()
    mrr  = np.mean([1.0 / (int(np.where(np.argsort(-S_np[i]) == i)[0][0]) + 1)
                    for i in range(N)])
    return float(r1), float(mrr)


def recall_at_k(S_np, k_list=(1, 5, 10)):
    topk = np.argsort(-S_np, axis=1)
    return {k: sum(i in topk[i, :k] for i in range(S_np.shape[0])) / S_np.shape[0]
            for k in k_list}


def t_choose_one(S_np, T_list=(4, 10, 20), n_trials=1000, seed=SEED):
    rng, N, trials = np.random.default_rng(seed), S_np.shape[0], min(n_trials, S_np.shape[0])
    results = {}
    for T in T_list:
        if T > N:
            results[T] = {"S->T": float("nan"), "T->S": float("nan")}; continue
        s2t = t2s = 0
        for i in range(trials):
            negs  = rng.choice([j for j in range(N) if j != i], size=T-1, replace=False)
            cands = np.concatenate([[i], negs])
            if cands[np.argmax(S_np[i,     cands])] == i: s2t += 1
            if cands[np.argmax(S_np[cands, i    ])] == i: t2s += 1
        results[T] = {"S->T": s2t / trials, "T->S": t2s / trials}
    return results


def compute_metrics(S_np):
    rec1, mrr = recall_mrr(S_np)
    rk        = recall_at_k(S_np, k_list=[1, 5, 10])
    tco       = t_choose_one(S_np, T_list=[4, 10, 20])
    return rec1, mrr, rk, tco


def print_table(label, rec1, mrr, rk, tco, N):
    print(f"\n{'='*65}\n  {label}  (N={N})\n{'='*65}")
    print(f"  {'Recall@1':<28} {rec1:>8.3f}")
    print(f"  {'MRR':<28} {mrr:>8.3f}")
    print(f"  {'Recall@5':<28} {rk[5]:>8.3f}")
    print(f"  {'Recall@10':<28} {rk[10]:>8.3f}")
    for T in [4, 10, 20]:
        print(f"  {'T='+str(T)+' S->T':<28} {tco[T]['S->T']:>8.3f}")
        print(f"  {'T='+str(T)+' T->S':<28} {tco[T]['T->S']:>8.3f}")
    print(f"{'='*65}")


# ===========================================================================
# Training loops
# ===========================================================================
def train_c2_molprompt_style(graph_enc, text_enc, text_tok,
                              train_graphs, train_texts, train_groups,
                              device, epochs, lr, weight_decay, temp, batch_size):
    """
    Condition 2: MolPrompt-style end-to-end training.
    Both encoders are unfrozen. Projection heads are 2-layer MLPs.
    Loss: pure symmetric NT-Xent (no margin).
    """
    proj_graph = make_mlp_head(GRAPH_DIM, GRAPH_DIM, SHARED_D).to(device)
    proj_text  = make_mlp_head(TEXT_DIM,  TEXT_DIM,  SHARED_D).to(device)

    # Train everything jointly
    opt = torch.optim.AdamW(
        list(graph_enc.parameters()) +
        list(text_enc.parameters()) +
        list(proj_graph.parameters()) +
        list(proj_text.parameters()),
        lr=lr, weight_decay=weight_decay)

    epoch_losses = []
    print(f"\nCondition 2 — MolPrompt-style: training {epochs} epochs "
          f"(batch={batch_size}, encoders UNFROZEN, MLP heads, NT-Xent only) ...")

    for ep in range(1, epochs + 1):
        graph_enc.train(); text_enc.train()
        proj_graph.train(); proj_text.train()
        running = n = 0

        # Shuffle indices
        idx = list(range(len(train_graphs)))
        random.shuffle(idx)

        for start in range(0, len(idx) - batch_size + 1, batch_size):
            batch_idx = idx[start: start + batch_size]
            batch_graphs = [train_graphs[i] for i in batch_idx]
            batch_texts  = [train_texts[i]  for i in batch_idx]
            batch_groups = [train_groups[i] for i in batch_idx]

            # Graph forward
            b = collate_graphs(batch_graphs)
            graph_rep = graph_enc(
                b["x"].to(device), b["attn_bias"].to(device),
                b["attn_edge_type"].to(device), b["spatial_pos"].to(device),
                b["in_degree"].to(device), b["out_degree"].to(device),
                b["edge_input"].to(device))
            graph_proj = proj_graph(graph_rep)

            # Text forward
            enc = text_tok(batch_texts, padding=True, truncation=True,
                           max_length=MAX_TXT_LEN, return_tensors="pt").to(device)
            text_rep  = text_enc(**enc)["pooler_output"]
            text_proj = proj_text(text_rep)

            loss = nt_xent_loss(text_proj, graph_proj, T=temp)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n += 1

        avg = running / max(n, 1)
        epoch_losses.append(avg)
        if ep % LOG_EVERY == 0 or ep == 1:
            print(f"  Epoch {ep:>3}/{epochs}: loss={avg:.4f}")

    return proj_graph, proj_text, epoch_losses


def train_c3_thinbridges_style(Z_text_train, Z_graph_train, train_groups,
                                device, epochs, lr, weight_decay, temp, margin, batch_size):
    """
    Condition 3: Thin-Bridges-style — pre-cached embeddings, linear projectors,
    NT-Xent + hard-negative margin loss. Encoders are NOT updated.
    """
    proj_graph = nn.Linear(GRAPH_DIM, SHARED_D).to(device)
    proj_text  = nn.Linear(TEXT_DIM,  SHARED_D).to(device)

    opt = torch.optim.AdamW(
        list(proj_graph.parameters()) + list(proj_text.parameters()),
        lr=lr, weight_decay=weight_decay)

    ds = EmbedPairDataset(Z_text_train, Z_graph_train, train_groups)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    epoch_losses = []
    print(f"\nCondition 3 — Thin-Bridges-style: training {epochs} epochs "
          f"(batch={batch_size}, encoders FROZEN, linear heads, NT-Xent+margin) ...")

    for ep in range(1, epochs + 1):
        proj_graph.train(); proj_text.train()
        running = n = 0
        for Zt, Zg, grps, _ in dl:
            Zt = Zt.to(device); Zg = Zg.to(device)
            Bt = proj_text(Zt);  Bg = proj_graph(Zg)
            loss = nt_xent_margin_loss(Bt, Bg, grps, device, T=temp, margin=margin)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n += 1
        avg = running / max(n, 1)
        epoch_losses.append(avg)
        if ep % LOG_EVERY == 0 or ep == 1:
            print(f"  Epoch {ep:>3}/{epochs}: loss={avg:.4f}")

    return proj_graph, proj_text, epoch_losses


# ===========================================================================
# Plots
# ===========================================================================
def plot_three_recall(rk_zs, rk_c2, rk_c3, outdir, sfx=""):
    k_list = sorted(rk_zs.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_list, [rk_zs[k] for k in k_list], "o:", label="Zero-shot",
            linewidth=2, color="gray")
    ax.plot(k_list, [rk_c2[k] for k in k_list], "s-",
            label="MolPrompt-style (unfrozen, MLP, NT-Xent)",
            linewidth=2, color="royalblue")
    ax.plot(k_list, [rk_c3[k] for k in k_list], "^--",
            label="Thin-Bridges-style (frozen, linear, +margin)",
            linewidth=2, color="darkorange")
    ax.set_xlabel("k"); ax.set_ylabel("Recall@k")
    ax.set_title(f"Graphormer+SciBERT Recall@k on ChEMBL{sfx}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, f"graphormer_recall_at_k{sfx}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved → {path}")


def plot_loss_curves(losses_c2, losses_c3, outdir, sfx=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    for ax, losses, title, color in zip(
            axes,
            [losses_c2, losses_c3],
            ["MolPrompt-style (NT-Xent, unfrozen encoders)",
             "Thin-Bridges-style (NT-Xent+margin, frozen encoders)"],
            ["royalblue", "darkorange"]):
        ax.plot(range(1, len(losses) + 1), losses, linewidth=1.5, color=color)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle(f"Training Loss Curves{sfx}", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, f"graphormer_training_loss{sfx}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved → {path}")


def plot_score_dist(S_zs, S_c2, S_c3, outdir, sfx=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, S_np, title in zip(
            axes, [S_zs, S_c2, S_c3],
            ["Zero-shot",
             "MolPrompt-style\n(unfrozen, MLP, NT-Xent)",
             "Thin-Bridges-style\n(frozen, linear, +margin)"]):
        N = S_np.shape[0]
        ax.hist(S_np[~np.eye(N, dtype=bool)], bins=80, alpha=0.5,
                density=True, label="Negatives", color="steelblue")
        ax.hist(np.diag(S_np), bins=30, alpha=0.8,
                density=True, label="Positives", color="orange")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Cosine similarity"); ax.legend(); ax.set_xlim(-1, 1)
    plt.suptitle(f"Positive vs Negative distributions{sfx}", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, f"graphormer_score_dist{sfx}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"Saved → {path}")


# ===========================================================================
# Save results CSV
# ===========================================================================
def save_csv(outdir, sfx, N,
             rec1_zs, mrr_zs, rk_zs, tco_zs,
             rec1_c2, mrr_c2, rk_c2, tco_c2,
             rec1_c3, mrr_c3, rk_c3, tco_c3):
    rows = [
        {"Metric": "Recall@1",
         "ZeroShot": rec1_zs, "MolPromptStyle": rec1_c2, "ThinBridgesStyle": rec1_c3,
         "Random": 1/N},
        {"Metric": "MRR",
         "ZeroShot": mrr_zs,  "MolPromptStyle": mrr_c2,  "ThinBridgesStyle": mrr_c3,
         "Random": None},
    ]
    for k in [5, 10]:
        rows.append({"Metric": f"Recall@{k}",
                     "ZeroShot": rk_zs[k], "MolPromptStyle": rk_c2[k] if rk_c2 else None,
                     "ThinBridgesStyle": rk_c3[k] if rk_c3 else None, "Random": None})
    for T in [4, 10, 20]:
        for d in ["S->T", "T->S"]:
            rows.append({"Metric": f"T={T} {d}",
                         "ZeroShot": tco_zs[T][d],
                         "MolPromptStyle":    tco_c2[T][d] if tco_c2 else None,
                         "ThinBridgesStyle":  tco_c3[T][d] if tco_c3 else None,
                         "Random": 1/T})
    path = os.path.join(outdir, f"graphormer_results{sfx}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nResults CSV → {path}")


# ===========================================================================
# Main
# ===========================================================================
def main(args):
    set_seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sfx    = "_nodrug" if args.remove_drug_name else ""
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load + clean data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.csv).dropna(subset=["smiles", "text_rich"]).copy()
    if args.remove_drug_name:
        df["text_rich"] = df["text_rich"].str.replace(
            r"Drug:\s*[^.]+\.\s*", "", regex=True).str.strip()
        print("WARNING: Drug names stripped from text_rich (leakage ablation)")
    print(f"Sample text [0]: {df['text_rich'].iloc[0]}")

    # ------------------------------------------------------------------
    # 2. Scaffold split (SEED=0, 90/10 — identical to 03_train_scaffold_split.py)
    # ------------------------------------------------------------------
    def _scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core) if core else None

    df["scaffold"] = df["smiles"].apply(_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    scaffolds = df["scaffold"].unique()
    rng = np.random.default_rng(SEED); rng.shuffle(scaffolds)
    cut = int(0.9 * len(scaffolds))
    train_df = df[df["scaffold"].isin(set(scaffolds[:cut]))].reset_index(drop=True)
    test_df  = df[df["scaffold"].isin(set(scaffolds[cut:]))].reset_index(drop=True)
    print(f"Scaffold split → train: {len(train_df)} | test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 3. Featurise all SMILES → Graphormer graph objects
    # ------------------------------------------------------------------
    print("\nFeaturising SMILES ...")
    mol_to_graph, preprocess = _setup_molprompt()

    def make_graphs(smiles_series, desc):
        gs, ok = [], []
        for i, s in enumerate(tqdm(smiles_series, desc=desc)):
            g = smiles_to_graph(s, mol_to_graph, preprocess)
            gs.append(g)
            if g is not None: ok.append(i)
        return gs, ok

    train_graphs_raw, tr_ok = make_graphs(train_df["smiles"], "Featurise train")
    test_graphs_raw,  te_ok = make_graphs(test_df["smiles"],  "Featurise test")

    train_df     = train_df.iloc[tr_ok].reset_index(drop=True)
    test_df      = test_df.iloc[te_ok].reset_index(drop=True)
    train_graphs = [train_graphs_raw[i] for i in tr_ok]
    test_graphs  = [test_graphs_raw[i]  for i in te_ok]
    print(f"After validation → train: {len(train_df)} | test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 4. Load encoders
    # ------------------------------------------------------------------
    print("\nLoading Graphormer encoder ...")
    graph_enc = load_graph_encoder(device)
    print("Loading SciBERT text encoder ...")
    text_tok, text_enc = load_text_encoder(device)

    # ------------------------------------------------------------------
    # 5. Pre-cache frozen embeddings (used for zero-shot and C3)
    # ------------------------------------------------------------------
    print("\nPre-caching embeddings (frozen encoders) ...")
    graph_enc.eval(); text_enc.eval()
    Z_graph_train = encode_graphs_frozen(train_graphs, graph_enc, device,
                                         batch_size=args.graph_batch_size)
    Z_graph_test  = encode_graphs_frozen(test_graphs,  graph_enc, device,
                                         batch_size=args.graph_batch_size)
    Z_text_train = encode_texts_frozen(train_df["text_rich"].tolist(),
                                       text_tok, text_enc, device)
    Z_text_test  = encode_texts_frozen(test_df["text_rich"].tolist(),
                                       text_tok, text_enc, device)

    # ------------------------------------------------------------------
    # 6. Condition 1 — Zero-shot
    # ------------------------------------------------------------------
    print("\n" + "="*65 + "\nCONDITION 1 — ZERO-SHOT\n" + "="*65)
    S_zs = (F.normalize(Z_text_test.float(), dim=1) @
            F.normalize(Z_graph_test.float(), dim=1).T).numpy()
    rec1_zs, mrr_zs, rk_zs, tco_zs = compute_metrics(S_zs)
    print_table("Zero-shot (no training)", rec1_zs, mrr_zs, rk_zs, tco_zs, len(test_df))

    if args.zero_shot_only:
        save_csv(args.outdir, sfx, len(test_df),
                 rec1_zs, mrr_zs, rk_zs, tco_zs,
                 None, None, None, None,
                 None, None, None, None)
        test_df.to_csv(os.path.join(args.outdir, f"graphormer_test_df{sfx}.csv"), index=False)
        return

    # ------------------------------------------------------------------
    # 7. Condition 2 — MolPrompt-style (end-to-end, MLP heads, NT-Xent only)
    # ------------------------------------------------------------------
    print("\n" + "="*65 + "\nCONDITION 2 — MOLPROMPT-STYLE\n" + "="*65)
    print("  Encoders: UNFROZEN | Heads: 2-layer MLP | Loss: NT-Xent only")

    # Reload fresh copies so C2 and C3 start from identical initial weights
    graph_enc_c2 = load_graph_encoder(device)
    text_enc_c2  = load_text_encoder(device)[1]   # re-use same tokenizer

    proj_graph_c2, proj_text_c2, losses_c2 = train_c2_molprompt_style(
        graph_enc_c2, text_enc_c2, text_tok,
        train_graphs, train_df["text_rich"].tolist(),
        train_df["target_chembl_id"].tolist(),
        device, C2_EPOCHS, C2_LR, C2_WEIGHT_DECAY, C2_TEMP, C2_BATCH_SIZE)

    # Evaluate C2
    graph_enc_c2.eval(); text_enc_c2.eval()
    proj_graph_c2.eval(); proj_text_c2.eval()
    with torch.no_grad():
        Zg_te = encode_graphs_frozen(test_graphs, graph_enc_c2, device,
                                     batch_size=args.graph_batch_size)
        Zt_te = encode_texts_frozen(test_df["text_rich"].tolist(),
                                    text_tok, text_enc_c2, device)
        Bg_c2 = F.normalize(proj_graph_c2(Zg_te.to(device)), dim=1).cpu()
        Bt_c2 = F.normalize(proj_text_c2(Zt_te.to(device)),  dim=1).cpu()
    S_c2 = (Bt_c2 @ Bg_c2.T).numpy()
    rec1_c2, mrr_c2, rk_c2, tco_c2 = compute_metrics(S_c2)
    print_table("MolPrompt-style (unfrozen enc, MLP, NT-Xent)",
                rec1_c2, mrr_c2, rk_c2, tco_c2, len(test_df))

    # ------------------------------------------------------------------
    # 8. Condition 3 — Thin-Bridges-style (frozen, linear, NT-Xent+margin)
    # ------------------------------------------------------------------
    print("\n" + "="*65 + "\nCONDITION 3 — THIN-BRIDGES-STYLE\n" + "="*65)
    print("  Encoders: FROZEN | Heads: linear | Loss: NT-Xent + margin")

    proj_graph_c3, proj_text_c3, losses_c3 = train_c3_thinbridges_style(
        Z_text_train, Z_graph_train,
        train_df["target_chembl_id"].tolist(),
        device, C3_EPOCHS, C3_LR, C3_WEIGHT_DECAY, C3_TEMP, C3_MARGIN, C3_BATCH_SIZE)

    proj_graph_c3.eval(); proj_text_c3.eval()
    with torch.no_grad():
        Bg_c3 = F.normalize(proj_graph_c3(Z_graph_test.to(device)), dim=1).cpu()
        Bt_c3 = F.normalize(proj_text_c3(Z_text_test.to(device)),  dim=1).cpu()
    S_c3 = (Bt_c3 @ Bg_c3.T).numpy()
    rec1_c3, mrr_c3, rk_c3, tco_c3 = compute_metrics(S_c3)
    print_table("Thin-Bridges-style (frozen enc, linear, NT-Xent+margin)",
                rec1_c3, mrr_c3, rk_c3, tco_c3, len(test_df))

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    save_csv(args.outdir, sfx, len(test_df),
             rec1_zs, mrr_zs, rk_zs, tco_zs,
             rec1_c2, mrr_c2, rk_c2, tco_c2,
             rec1_c3, mrr_c3, rk_c3, tco_c3)

    test_df.to_csv(os.path.join(args.outdir, f"graphormer_test_df{sfx}.csv"), index=False)
    torch.save(proj_graph_c2.state_dict(),
               os.path.join(args.outdir, f"graphormer_c2_proj_graph{sfx}.pt"))
    torch.save(proj_text_c2.state_dict(),
               os.path.join(args.outdir, f"graphormer_c2_proj_text{sfx}.pt"))
    torch.save(proj_graph_c3.state_dict(),
               os.path.join(args.outdir, f"graphormer_c3_proj_graph{sfx}.pt"))
    torch.save(proj_text_c3.state_dict(),
               os.path.join(args.outdir, f"graphormer_c3_proj_text{sfx}.pt"))
    np.save(os.path.join(args.outdir, f"graphormer_c2_losses{sfx}.npy"), np.array(losses_c2))
    np.save(os.path.join(args.outdir, f"graphormer_c3_losses{sfx}.npy"), np.array(losses_c3))

    plot_three_recall(rk_zs, rk_c2, rk_c3, args.outdir, sfx=sfx)
    plot_loss_curves(losses_c2, losses_c3, args.outdir, sfx=sfx)
    plot_score_dist(S_zs, S_c2, S_c3, args.outdir, sfx=sfx)
    print(f"\nAll outputs saved to {args.outdir}/  (suffix=\"{sfx}\")")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graphormer+SciBERT ChEMBL retrieval — 3 conditions")
    parser.add_argument("--csv",    default="chembl_mechanisms.csv")
    parser.add_argument("--outdir", default="outputs/graphormer")
    parser.add_argument("--remove_drug_name", action="store_true",
                        help="Strip 'Drug: X.' from text_rich (leakage ablation)")
    parser.add_argument("--zero_shot_only", action="store_true",
                        help="Only evaluate zero-shot condition (no training)")
    parser.add_argument("--graph_batch_size", type=int, default=32,
                        help="Batch size for frozen graph encoding (reduce if OOM)")
    main(parser.parse_args())
