# -*- coding: utf-8 -*-
"""
03b_train_global_bridge.py
===========================
Reproduces the ECFP4 + text_rich contrastive bridge result from the
NeurIPS 2025 paper (Table 1, row 4):

    ECFP4 + text_rich bridge   Recall@1 = 0.762   MRR = 0.863

This uses a GLOBAL split (random train/test on the full dataset),
NOT the scaffold split. This is the simpler of the two training settings.
For scaffold-split generalisation results (Recall@1=0.150), use
03_train_scaffold_split.py instead.

Fixes applied vs original notebook cell:
  - BUG #3 : `import copy` added (was NameError on copy.deepcopy)
  - BUG #7 : SEED=42 set explicitly and applied to all random ops
  - Epoch losses saved for visualisation
  - Checkpoints saved to --outdir

Usage:
    python 03b_train_global_bridge.py \
        --csv chembl_mechanisms_enriched.csv \
        --outdir outputs
"""

import os
import copy
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
warnings.filterwarnings("ignore")

SEED        = 42
TEXT_MODEL  = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
MAX_TXT_LEN = 96
BATCH_SIZE  = 512
EPOCHS      = 100
LR          = 1e-3
WEIGHT_DECAY= 1e-4
TEMP        = 0.07
SHARED_D    = 256
ECFP_BITS   = 2048
ECFP_RADIUS = 2


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ecfp4_bitvect(smi, nbits=2048, radius=2):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(nbits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


@torch.no_grad()
def cls_encode(texts, tok, model, device, max_length=96, bs=64):
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="Encoding text"):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state[:, 0, :]
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


class PairDS(Dataset):
    def __init__(self, Zt, Xm):
        self.Zt = torch.tensor(Zt, dtype=torch.float32)
        self.Xm = torch.tensor(Xm, dtype=torch.float32)
    def __len__(self): return len(self.Zt)
    def __getitem__(self, i): return self.Zt[i], self.Xm[i]


def contrastive_step(bt, bm, proj_text, proj_mol, device, T=0.07):
    bt = F.normalize(proj_text(bt.to(device)), dim=1)
    bm = F.normalize(proj_mol (bm.to(device)), dim=1)
    logits = bt @ bm.T / T
    labels = torch.arange(bt.size(0), device=bt.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


@torch.no_grad()
def eval_recall_mrr(Zt, Xm, proj_text, proj_mol, device):
    Bt = F.normalize(proj_text(torch.tensor(Zt, dtype=torch.float32).to(device)), dim=1).cpu()
    Bm = F.normalize(proj_mol (torch.tensor(Xm, dtype=torch.float32).to(device)), dim=1).cpu()
    S    = Bt @ Bm.T
    S_np = S.numpy()
    gt   = np.arange(S_np.shape[0])
    top1 = S_np.argmax(axis=1)
    recall1 = (top1 == gt).mean()
    ranks = []
    for i in range(S_np.shape[0]):
        order = np.argsort(-S_np[i])
        ranks.append(1.0 / (int(np.where(order == i)[0][0]) + 1))
    return float(recall1), float(np.mean(ranks)), S_np


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
    df = pd.read_csv(args.csv).dropna(subset=["smiles", "text_rich"])
    df = df[df["text_rich"].str.len() > 20].reset_index(drop=True)
    print(f"Loaded {len(df)} pairs")

    # ------------------------------------------------------------------
    # 2. Molecular fingerprints
    # ------------------------------------------------------------------
    print("Computing ECFP4 fingerprints ...")
    X_mol = np.stack([ecfp4_bitvect(s, ECFP_BITS, ECFP_RADIUS)
                      for s in tqdm(df["smiles"].tolist(), desc="ECFP4")])

    # ------------------------------------------------------------------
    # 3. Text embeddings (frozen)
    # ------------------------------------------------------------------
    print(f"Loading text encoder: {TEXT_MODEL} ...")
    text_tok   = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    text_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    Z_text = cls_encode(df["text_rich"].tolist(), text_tok, text_model,
                        device, max_length=MAX_TXT_LEN, bs=64)
    print(f"  Z_text: {Z_text.shape}   X_mol: {X_mol.shape}")

    # ------------------------------------------------------------------
    # 4. DataLoader
    # ------------------------------------------------------------------
    ds = PairDS(Z_text.numpy(), X_mol)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ------------------------------------------------------------------
    # 5. Projection heads
    # ------------------------------------------------------------------
    d_text = Z_text.shape[1]
    d_fp   = X_mol.shape[1]
    proj_text = nn.Linear(d_text, SHARED_D).to(device)
    proj_mol  = nn.Linear(d_fp,   SHARED_D).to(device)

    # FIX #3: import copy at top — was NameError in original notebook
    proj_text_0 = copy.deepcopy(proj_text)
    proj_mol_0  = copy.deepcopy(proj_mol)

    opt = optim.AdamW(
        list(proj_text.parameters()) + list(proj_mol.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {EPOCHS} epochs ...")
    epoch_losses = []
    for ep in range(1, EPOCHS + 1):
        proj_text.train(); proj_mol.train()
        tot = n = 0
        for bt, bm in dl:
            opt.zero_grad()
            loss = contrastive_step(bt, bm, proj_text, proj_mol, device, T=TEMP)
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
        avg = tot / max(n, 1)
        epoch_losses.append(avg)
        print(f"Epoch {ep:>3}/{EPOCHS}  loss={avg:.4f}")

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    print("\n=== Final Evaluation (global split) ===")
    rec1, mrr, S_np = eval_recall_mrr(
        Z_text.numpy(), X_mol, proj_text, proj_mol, device)
    print(f"Recall@1 : {rec1:.3f}")
    print(f"MRR      : {mrr:.3f}")

    # ------------------------------------------------------------------
    # 8. Save artefacts
    # ------------------------------------------------------------------
    torch.save(proj_text.state_dict(),   os.path.join(args.outdir, "global_proj_text.pt"))
    torch.save(proj_mol.state_dict(),    os.path.join(args.outdir, "global_proj_mol.pt"))
    torch.save(proj_text_0.state_dict(), os.path.join(args.outdir, "global_proj_text_0.pt"))
    torch.save(proj_mol_0.state_dict(),  os.path.join(args.outdir, "global_proj_mol_0.pt"))
    torch.save(Z_text,                   os.path.join(args.outdir, "global_Z_text.pt"))
    np.save(os.path.join(args.outdir, "global_X_mol.npy"), X_mol)
    np.save(os.path.join(args.outdir, "global_epoch_losses.npy"), np.array(epoch_losses))
    df.to_csv(os.path.join(args.outdir, "global_df.csv"), index=False)

    print(f"\nAll outputs saved to {args.outdir}/")
    print("\nTo visualise: open 04_results_visualisation.ipynb and set MODE='global'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, default="chembl_mechanisms_enriched.csv")
    parser.add_argument("--outdir", type=str, default="outputs")
    main(parser.parse_args())