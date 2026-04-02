# -*- coding: utf-8 -*-
"""
04_train_cloob_experiments.py
==============================
CLOOB-enhanced Thin Bridges: Full ablation suite (Variants A–D)
with optional drug-name leakage ablation.

Variants (controlled by --variant flag):
  A  Thin Bridges baseline      InfoNCE   + No Hopfield   (reproduces paper)
  B  InfoLOOB only              InfoLOOB  + No Hopfield
  C  Hopfield + InfoNCE         InfoNCE   + Hopfield
  D  Hopfield + InfoLOOB        InfoLOOB  + Hopfield       (full CLOOB — proposed)

Each variant is run with and without --remove_drug_name for leakage ablation,
giving 8 conditions total.

Architecture
------------
  - Encoders: ECFP4 (2048-bit, frozen) + S-BiomedRoBERTa (frozen)
  - Projection heads: Linear(d→256) for each modality  [ONLY trained weights]
  - Hopfield enrichment: softmax-weighted retrieval over minibatch embeddings
    stored in two Hopfield memories (U=mol, V=text), as in CLOOB (NeurIPS 2022)
  - InfoLOOB: Leave-One-Out Bound — excludes positive from denominator,
    avoids InfoNCE saturation (gradient scale factor = 1 vs (1-p1) in InfoNCE)
  - Loss temperature τ is FIXED (not learnable) per CLOOB recommendation

Key hyperparameters (CLOOB-tuned for small-data regime, ~2699 pairs)
----------------------------------------------------------------------
  BETA      = 5.0     # Hopfield inverse temperature (β < ln(512)≈6.2 → metastable)
  TAU       = 1/15    # InfoLOOB temperature (fixed, τ⁻¹=15)
  EPOCHS    = 300     # More epochs than baseline (cosine-annealed LR, fast per-epoch)
  LR        = 5e-4    # Slightly lower than baseline (InfoLOOB needs stable early training)

Usage
-----
  # Run all 4 variants sequentially (recommended for full paper table):
  python 04_train_cloob_experiments.py --csv chembl_mechanisms.csv --outdir outputs_cloob --variant all

  # Run a single variant:
  python 04_train_cloob_experiments.py --csv chembl_mechanisms.csv --outdir outputs_cloob --variant D

  # Run with drug name stripped (leakage ablation):
  python 04_train_cloob_experiments.py --csv chembl_mechanisms.csv --outdir outputs_cloob --variant D --remove_drug_name

  # Run full 2x4 matrix (all variants x both drug-name conditions):
  python 04_train_cloob_experiments.py --csv chembl_mechanisms.csv --outdir outputs_cloob --variant all --run_leakage

Outputs (per variant, per drug-name condition):
  outputs_cloob/
    variant_A/           proj_text.pt, proj_mol.pt, results.json, loss_curve.png
    variant_A_nodrug/    ... (leakage condition)
    variant_B/
    variant_B_nodrug/
    variant_C/
    variant_D/
    variant_D_nodrug/
    summary_table.csv    (all variants side by side, printed at end)
    summary_table.png    (bar chart)
"""

import os
import copy
import json
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
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
TEXT_MODEL     = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
MAX_TXT_LEN    = 96
BATCH_SIZE     = 512
EPOCHS         = 300        # More epochs — InfoLOOB converges slower than InfoNCE
LR             = 5e-4       # Slightly lower than baseline for stability
WEIGHT_DECAY   = 0.1        # Strong regularisation (CLOOB recommendation)
TEMP_INFONCE   = 0.07       # InfoNCE temperature (matches baseline)
TAU_INFOLOOB   = 1.0 / 15   # InfoLOOB temperature τ (FIXED, τ⁻¹=15)
BETA_HOPFIELD  = 5.0        # Hopfield inverse temperature β  (sweep: 3,5,7,10)
MARGIN         = 0.1        # Hard-negative margin (same as baseline)
SHARED_D       = 256        # Shared embedding dimension
ECFP_BITS      = 2048
ECFP_RADIUS    = 2
SEED           = 0
MIN_GROUP_SIZE = 3
LOG_EVERY      = 30
T_CHOOSE_ONE_TRIALS = 1000

VARIANT_DESCRIPTIONS = {
    "A": "InfoNCE  + No Hopfield  (Thin Bridges baseline)",
    "B": "InfoLOOB + No Hopfield",
    "C": "InfoNCE  + Hopfield",
    "D": "InfoLOOB + Hopfield     (full CLOOB — proposed)",
}

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# MOLECULAR FEATURES
# =============================================================================
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


# =============================================================================
# TEXT ENCODING (frozen)
# =============================================================================
@torch.no_grad()
def cls_encode(texts, tok, model, device, max_length=96, bs=64):
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="Encode text (CLS)", leave=False):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state[:, 0, :]
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


# =============================================================================
# DATASET
# =============================================================================
class PairDataset(Dataset):
    def __init__(self, Z_text, X_mol, group_labels):
        self.Zt  = torch.tensor(Z_text, dtype=torch.float32)
        self.Xm  = torch.tensor(X_mol,  dtype=torch.float32)
        self.grp = np.array(group_labels).astype(str)

    def __len__(self):
        return len(self.Zt)

    def __getitem__(self, i):
        return self.Zt[i], self.Xm[i], self.grp[i]


# =============================================================================
# HOPFIELD ENRICHMENT MODULE
# =============================================================================
class HopfieldEnrich(nn.Module):
    """
    Modern Hopfield network retrieval layer (CLOOB, Fürst et al. NeurIPS 2022).

    Given a batch of query embeddings X (shape [B, d]) and a memory bank
    stored_patterns (shape [M, d]), computes:

        retrieved = stored_patterns^T @ softmax(β · stored_patterns @ X^T)

    Both X and stored_patterns must be L2-normalized before calling.

    By default stores the minibatch itself (U=X, V=Y) per the CLOOB paper.
    β is the inverse temperature:
      - β → 0   : returns mean of all stored patterns (no discrimination)
      - β → ∞   : nearest-neighbour lookup (sharp retrieval, kills covariance)
      - β ≈ 5   : metastable regime — enriches covariance structure

    Notes
    -----
    - Re-normalises retrieved embeddings to unit norm after retrieval.
    - No trainable parameters — pure softmax attention.
    - Memory cost: O(B²·d) — negligible for B=512, d=256 on A5000.
    """
    def __init__(self, beta: float = 5.0):
        super().__init__()
        self.beta = beta

    def forward(self, queries: torch.Tensor,
                stored: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries : [B, d]  L2-normalised query embeddings
            stored  : [M, d]  L2-normalised memory bank embeddings

        Returns:
            retrieved : [B, d]  L2-normalised enriched embeddings
        """
        # [B, M] — similarity of each query against all stored patterns
        sim = self.beta * (queries @ stored.T)          # [B, M]
        weights = torch.softmax(sim, dim=1)             # [B, M]
        retrieved = weights @ stored                    # [B, d]
        return F.normalize(retrieved, dim=1)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def info_nce_loss(bt: torch.Tensor, bm: torch.Tensor,
                  T: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style).
    Both bt and bm must be L2-normalised, shape [B, d].
    """
    logits = (bt @ bm.T) / T                            # [B, B]
    labels = torch.arange(bt.size(0), device=bt.device)
    loss_t2m = F.cross_entropy(logits,   labels)
    loss_m2t = F.cross_entropy(logits.T, labels)
    return (loss_t2m + loss_m2t) / 2


def info_loob_loss(bt: torch.Tensor, bm: torch.Tensor,
                   tau: float = 1.0 / 15) -> torch.Tensor:
    """
    Symmetric InfoLOOB (Leave-One-Out Bound, Poole et al. 2019 / CLOOB 2022).

    Key difference from InfoNCE: the positive sample is EXCLUDED from the
    denominator. This avoids the (1-p1) gradient scaling factor that causes
    InfoNCE saturation when positive pairs become highly aligned.

    Loss = -1/B Σ_i  log [exp(τ⁻¹ bt_i·bm_i) / Σ_{j≠i} exp(τ⁻¹ bt_i·bm_j)]
         + -1/B Σ_i  log [exp(τ⁻¹ bt_i·bm_i) / Σ_{j≠i} exp(τ⁻¹ bt_j·bm_i)]

    IMPORTANT: τ must be FIXED (not learnable) — learnable τ with InfoLOOB
    causes pathological optimization (CLOOB paper, Appendix A.1.4).

    Both bt and bm must be L2-normalised, shape [B, d].
    """
    B = bt.size(0)
    # Scale by τ⁻¹
    inv_tau = 1.0 / tau
    logits = inv_tau * (bt @ bm.T)                      # [B, B]

    # Mask out diagonal (positive) from denominator
    mask = ~torch.eye(B, dtype=torch.bool, device=bt.device)

    loss = 0.0
    for logits_i, direction in [(logits, "row"), (logits.T, "col")]:
        # For each anchor i: log( exp(pos_i) / sum_{j≠i} exp(neg_j) )
        # = pos_i - logsumexp(neg_j for j≠i)
        pos_scores = logits_i.diag()                    # [B]
        # Collect negatives only (exclude diagonal)
        neg_logits = logits_i.masked_fill(~mask, float("-inf"))  # [B, B]
        lse_neg = torch.logsumexp(neg_logits, dim=1)   # [B]
        loss += -(pos_scores - lse_neg).mean()

    # Scale by τ (matching CLOOB Eq.5 τ·L_InfoLOOB to remove τ⁻¹ from gradients)
    return tau * (loss / 2)


def build_same_target_weights(grps, device):
    """Weight matrix W[i,j] = 2.0 if same target, else 1.0 (from baseline)."""
    B = len(grps)
    W = torch.ones((B, B), device=device)
    for i in range(B):
        for j in range(B):
            if i != j and grps[i] == grps[j]:
                W[i, j] = 2.0
    return W


def hardest_same_target_neg(logits, groups):
    """Find hardest negative among same-target pairs (from baseline)."""
    B = logits.size(0)
    hard = torch.full((B,), float("-inf"), device=logits.device)
    for i in range(B):
        mask = torch.tensor(
            [(groups[i] == groups[j] and j != i) for j in range(B)],
            device=logits.device)
        if mask.any():
            hard[i] = logits[i][mask].max()
    return hard


def margin_loss_term(logits: torch.Tensor,
                     groups,
                     margin: float = 0.1) -> torch.Tensor:
    """
    Hard-negative margin loss from the Thin Bridges baseline.
    Applies to same-target pairs to improve within-target discrimination.
    Preserved across ALL variants to keep the only difference being
    Hopfield enrichment and loss objective.
    """
    pos       = logits.diag()
    hard_same = hardest_same_target_neg(logits, groups)
    mask      = torch.isfinite(hard_same)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)
    return torch.clamp(margin - (pos[mask] - hard_same[mask]), min=0).mean()


# =============================================================================
# COMBINED TRAINING STEP (dispatches to correct variant)
# =============================================================================
def compute_loss(bt_raw: torch.Tensor,
                 bm_raw: torch.Tensor,
                 groups,
                 hopfield_mol: HopfieldEnrich,
                 hopfield_txt: HopfieldEnrich,
                 use_hopfield: bool,
                 use_infoloob: bool,
                 device: str,
                 margin: float = 0.1) -> tuple:
    """
    Unified loss computation for all four variants.

    Step 1: L2-normalise projected embeddings
    Step 2 (optional): Hopfield enrichment using minibatch as memory
    Step 3: Compute contrastive loss (InfoNCE or InfoLOOB)
    Step 4: Add hard-negative margin loss
    Step 5: Return (total_loss, raw_logits_for_monitoring)

    Returns
    -------
    loss   : scalar tensor
    logits : [B, B] similarity matrix (pre-Hopfield, for monitoring)
    """
    # Step 1 — normalise
    bt = F.normalize(bt_raw, dim=1)
    bm = F.normalize(bm_raw, dim=1)

    # Raw logits for margin loss and monitoring (always InfoNCE scale for comparability)
    raw_logits = (bt @ bm.T) / TEMP_INFONCE

    # Step 2 — optional Hopfield enrichment
    if use_hopfield:
        # CLOOB architecture: U=mol memory, V=text memory
        # U_bt = text queries molecule store, U_bm = mol queries molecule store
        # V_bt = text queries text store,     V_bm = mol queries text store
        U_bt = hopfield_mol(bt, bm)   # text-retrieved mol embeddings  [B,d]
        U_bm = hopfield_mol(bm, bm)   # mol-retrieved mol embeddings   [B,d]
        V_bt = hopfield_txt(bt, bt)   # text-retrieved text embeddings [B,d]
        V_bm = hopfield_txt(bm, bt)   # mol-retrieved text embeddings  [B,d]
        # Use enriched embeddings for loss (CLOOB Eq.5: U-U and V-V dot products)
        bt_eff = U_bt   # enriched text anchor
        bm_eff = U_bm   # enriched mol  anchor
        # Second term uses V store
        bt_v   = V_bt
        bm_v   = V_bm
    else:
        bt_eff = bt; bm_eff = bm
        bt_v   = bt; bm_v   = bm

    # Step 3 — contrastive loss
    if use_infoloob:
        # CLOOB: two InfoLOOB terms, one per Hopfield store
        loss_contrastive = (
            info_loob_loss(bt_eff, bm_eff, tau=TAU_INFOLOOB) +
            info_loob_loss(bm_v,   bt_v,   tau=TAU_INFOLOOB)
        ) / 2
    else:
        # Weighted InfoNCE (same as baseline, using hard-negative weights)
        W = build_same_target_weights(groups, device)
        ce_row = F.cross_entropy(raw_logits,   torch.arange(bt.size(0), device=device),
                                 reduction="none")
        ce_col = F.cross_entropy(raw_logits.T, torch.arange(bt.size(0), device=device),
                                 reduction="none")
        B = raw_logits.size(0)
        avg_w_row = (W.sum(dim=1) - torch.diag(W)) / (B - 1)
        avg_w_col = (W.sum(dim=0) - torch.diag(W)) / (B - 1)
        loss_contrastive = 0.5 * (
            (ce_row * avg_w_row).mean() + (ce_col * avg_w_col).mean()
        )

    # Step 4 — margin loss (same for all variants, uses raw logits)
    loss_margin = margin_loss_term(raw_logits, groups, margin=margin)

    return loss_contrastive + loss_margin, raw_logits


# =============================================================================
# EVALUATION HELPERS (unchanged from baseline)
# =============================================================================
@torch.no_grad()
def embed_side(Z_text, X_mol, proj_text, proj_mol, device):
    Bt = F.normalize(proj_text(
        torch.tensor(Z_text, dtype=torch.float32).to(device)), dim=1).cpu()
    Bm = F.normalize(proj_mol(
        torch.tensor(X_mol,  dtype=torch.float32).to(device)), dim=1).cpu()
    return Bt, Bm


def recall_mrr(S_np):
    N     = S_np.shape[0]
    top1  = S_np.argmax(axis=1)
    rec1  = (top1 == np.arange(N)).mean()
    ranks = [1.0 / (int(np.where(np.argsort(-S_np[i]) == i)[0][0]) + 1)
             for i in range(N)]
    return float(rec1), float(np.mean(ranks))


def recall_at_k(S_np, k_list=(1, 5, 10)):
    N    = S_np.shape[0]
    topk = np.argsort(-S_np, axis=1)
    return {k: sum(i in topk[i, :k] for i in range(N)) / N for k in k_list}


def grouped_recall_at1(S_np, meta_df,
                       group_col="target_chembl_id", min_group=3):
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


def t_choose_one(S_np, T_list=(4, 10, 20),
                 n_trials=T_CHOOSE_ONE_TRIALS, seed=SEED):
    """T-choose-one accuracy (MoleculeSTM paper protocol)."""
    rng    = np.random.default_rng(seed)
    N      = S_np.shape[0]
    trials = min(n_trials, N)
    results = {}
    for T in T_list:
        if T > N:
            results[T] = {"S->T": float("nan"), "T->S": float("nan")}
            continue
        s2t = t2s = 0
        for i in range(trials):
            negs  = rng.choice([j for j in range(N) if j != i],
                               size=T - 1, replace=False)
            cands = np.concatenate([[i], negs])
            if cands[np.argmax(S_np[i,     cands])] == i:
                s2t += 1
            if cands[np.argmax(S_np[cands, i    ])] == i:
                t2s += 1
        results[T] = {"S->T": s2t / trials, "T->S": t2s / trials}
    return results


# =============================================================================
# TRAIN + EVAL ONE VARIANT
# =============================================================================
def run_variant(variant: str,
                Z_text_train, X_mol_train, train_df,
                Z_text_test,  X_mol_test,  test_df,
                outdir: str,
                remove_drug_name: bool,
                device: str):
    """
    Train and evaluate a single variant (A/B/C/D).

    variant : one of "A", "B", "C", "D"
    Returns : dict of all metrics
    """
    use_hopfield  = variant in ("C", "D")
    use_infoloob  = variant in ("B", "D")
    sfx           = "_nodrug" if remove_drug_name else ""
    run_name      = f"variant_{variant}{sfx}"
    run_dir       = os.path.join(outdir, f"variant_{variant}{sfx}")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(SEED)
    print(f"\n{'='*65}")
    print(f"  VARIANT {variant}: {VARIANT_DESCRIPTIONS[variant]}")
    print(f"  Condition: {'no drug name' if remove_drug_name else 'with drug name'}")
    print(f"  use_hopfield={use_hopfield}  use_infoloob={use_infoloob}")
    print(f"  Output dir: {run_dir}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    train_ds = PairDataset(Z_text_train, X_mol_train,
                           train_df["target_chembl_id"].tolist())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True)

    # ------------------------------------------------------------------
    # Projection heads (only trainable parameters)
    # ------------------------------------------------------------------
    d_text = Z_text_train.shape[1]
    d_fp   = X_mol_train.shape[1]
    proj_text = nn.Linear(d_text, SHARED_D).to(device)
    proj_mol  = nn.Linear(d_fp,   SHARED_D).to(device)

    # ------------------------------------------------------------------
    # Hopfield modules (no parameters — pure retrieval)
    # ------------------------------------------------------------------
    hopfield_mol = HopfieldEnrich(beta=BETA_HOPFIELD).to(device)
    hopfield_txt = HopfieldEnrich(beta=BETA_HOPFIELD).to(device)

    # ------------------------------------------------------------------
    # Optimiser
    # Note: only projection heads have parameters
    # Hopfield has no parameters, InfoLOOB has no parameters
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(
        list(proj_text.parameters()) + list(proj_mol.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts (CLOOB training schedule)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=1, eta_min=1e-6)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epoch_losses = []
    print(f"\nTraining {EPOCHS} epochs  (batch={BATCH_SIZE}, lr={LR})...")
    for ep in range(1, EPOCHS + 1):
        proj_text.train(); proj_mol.train()
        running = n = 0
        for Zt_b, Xm_b, grp_b in train_dl:
            Zt_b = Zt_b.to(device)
            Xm_b = Xm_b.to(device)

            # Project (raw, not normalised yet — normalisation inside compute_loss)
            bt_raw = proj_text(Zt_b)
            bm_raw = proj_mol(Xm_b)

            loss, _ = compute_loss(
                bt_raw, bm_raw, grp_b,
                hopfield_mol, hopfield_txt,
                use_hopfield=use_hopfield,
                use_infoloob=use_infoloob,
                device=device,
                margin=MARGIN)

            opt.zero_grad()
            loss.backward()
            # Gradient clipping for stability (especially InfoLOOB early training)
            torch.nn.utils.clip_grad_norm_(
                list(proj_text.parameters()) + list(proj_mol.parameters()),
                max_norm=1.0)
            opt.step()
            running += loss.item(); n += 1

        scheduler.step()
        avg = running / max(n, 1)
        epoch_losses.append(avg)
        if ep % LOG_EVERY == 0 or ep == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {ep:>4}/{EPOCHS}: loss={avg:.4f}  lr={lr_now:.2e}")

    # ------------------------------------------------------------------
    # Evaluate on test scaffold split
    # ------------------------------------------------------------------
    print(f"\n--- Evaluating Variant {variant} on test split ---")
    proj_text.eval(); proj_mol.eval()
    Bt_test, Bm_test = embed_side(Z_text_test, X_mol_test,
                                  proj_text, proj_mol, device)
    S_np = (Bt_test @ Bm_test.T).numpy()

    rec1, mrr   = recall_mrr(S_np)
    rk          = recall_at_k(S_np, k_list=[1, 5, 10])
    gR1         = grouped_recall_at1(S_np, test_df,
                                     group_col="target_chembl_id",
                                     min_group=MIN_GROUP_SIZE)
    tco         = t_choose_one(S_np, T_list=[4, 10, 20])

    print(f"  Recall@1       : {rec1:.3f}")
    print(f"  MRR            : {mrr:.3f}")
    print(f"  Recall@5       : {rk[5]:.3f}")
    print(f"  Recall@10      : {rk[10]:.3f}")
    print(f"  Grouped R@1    : {gR1:.3f}  (min_group={MIN_GROUP_SIZE})")
    for T in [4, 10, 20]:
        print(f"  T={T} S->T     : {tco[T]['S->T']:.3f}")
        print(f"  T={T} T->S     : {tco[T]['T->S']:.3f}")

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    torch.save(proj_text.state_dict(),
               os.path.join(run_dir, "proj_text.pt"))
    torch.save(proj_mol.state_dict(),
               os.path.join(run_dir, "proj_mol.pt"))
    torch.save(Bt_test, os.path.join(run_dir, "Bt_test.pt"))
    torch.save(Bm_test, os.path.join(run_dir, "Bm_test.pt"))
    np.save(os.path.join(run_dir, "epoch_losses.npy"),
            np.array(epoch_losses))
    np.save(os.path.join(run_dir, "S_test.npy"), S_np)

    # Results dict
    results = {
        "variant":          variant,
        "description":      VARIANT_DESCRIPTIONS[variant],
        "remove_drug_name": remove_drug_name,
        "use_hopfield":     use_hopfield,
        "use_infoloob":     use_infoloob,
        "recall_at_1":      rec1,
        "mrr":              mrr,
        "recall_at_5":      rk[5],
        "recall_at_10":     rk[10],
        "grouped_recall_1": gR1,
        "T4_S2T":  tco[4]["S->T"],  "T4_T2S":  tco[4]["T->S"],
        "T10_S2T": tco[10]["S->T"], "T10_T2S": tco[10]["T->S"],
        "T20_S2T": tco[20]["S->T"], "T20_T2S": tco[20]["T->S"],
        "epochs":     EPOCHS,
        "beta":       BETA_HOPFIELD,
        "tau":        TAU_INFOLOOB,
        "lr":         LR,
        "batch_size": BATCH_SIZE,
        "test_size":  len(test_df),
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Loss curve
    _plot_loss_curve(epoch_losses, variant, remove_drug_name, run_dir)

    return results


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def _plot_loss_curve(epoch_losses, variant, remove_drug_name, run_dir):
    cond = "no drug name" if remove_drug_name else "with drug name"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(
        f"Variant {variant}: {VARIANT_DESCRIPTIONS[variant]}\n({cond})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=120)
    plt.close()


def plot_summary_table(all_results: list, outdir: str):
    """
    Side-by-side bar chart of key metrics across all variants.
    Produces: summary_table.csv  +  summary_table.png
    """
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(outdir, "summary_table.csv"), index=False)

    # Select rows: with drug name only for main comparison
    df_main   = df[~df["remove_drug_name"]].copy()
    df_nodrug = df[ df["remove_drug_name"]].copy()

    metrics = ["recall_at_1", "mrr", "recall_at_5",
               "T4_S2T", "T4_T2S", "T10_S2T", "T10_T2S",
               "T20_S2T", "T20_T2S"]
    labels  = ["R@1", "MRR", "R@5",
               "T4 S→T", "T4 T→S", "T10 S→T", "T10 T→S",
               "T20 S→T", "T20 T→S"]

    variants  = ["A", "B", "C", "D"]
    colors    = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    x         = np.arange(len(metrics))
    bar_width = 0.2

    # ---- Main figure (with drug name) ----
    fig, ax = plt.subplots(figsize=(15, 5))
    for k, (var, col) in enumerate(zip(variants, colors)):
        row = df_main[df_main["variant"] == var]
        if row.empty:
            continue
        vals = [float(row[m].values[0]) for m in metrics]
        ax.bar(x + k * bar_width, vals, bar_width,
               label=f"Variant {var}", color=col, alpha=0.85)

    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Thin Bridges CLOOB Ablation — With Drug Name\n"
                 "A: InfoNCE+NoHop | B: InfoLOOB+NoHop | "
                 "C: InfoNCE+Hop | D: InfoLOOB+Hop (CLOOB)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_withdrug.png"), dpi=130)
    plt.close()
    print(f"Saved → {os.path.join(outdir, 'summary_withdrug.png')}")

    # ---- Leakage ablation figure (with vs without drug name for each variant) ----
    if not df_nodrug.empty:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        leak_metrics = ["recall_at_1", "T4_S2T", "T10_S2T", "T20_S2T"]
        leak_labels  = ["Recall@1", "T=4 S→T", "T=10 S→T", "T=20 S→T"]
        for ax, var, col in zip(axes, variants, colors):
            row_w = df_main[df_main["variant"] == var]
            row_n = df_nodrug[df_nodrug["variant"] == var]
            if row_w.empty:
                ax.set_title(f"Variant {var}\n(no data)")
                continue
            vals_w = [float(row_w[m].values[0]) for m in leak_metrics]
            vals_n = [float(row_n[m].values[0]) for m in leak_metrics] \
                if not row_n.empty else [0] * len(leak_metrics)
            drop_pct = [(w - n) / w * 100 if w > 0 else 0
                        for w, n in zip(vals_w, vals_n)]
            xi = np.arange(len(leak_metrics))
            ax.bar(xi - 0.2, vals_w, 0.35, label="With drug name",
                   color=col,   alpha=0.85)
            ax.bar(xi + 0.2, vals_n, 0.35, label="No drug name",
                   color=col,   alpha=0.4)
            ax.set_xticks(xi)
            ax.set_xticklabels(leak_labels, rotation=15, ha="right")
            ax.set_title(
                f"Variant {var}\n"
                f"Avg drop: {np.mean(drop_pct):.1f}%", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("Score")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle("Drug Name Leakage Ablation — All Variants", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "leakage_ablation.png"),
                    dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved → {os.path.join(outdir, 'leakage_ablation.png')}")


def print_summary_table(all_results: list):
    """Print a clean terminal table of all results."""
    rows_w = [r for r in all_results if not r["remove_drug_name"]]
    rows_n = [r for r in all_results if r["remove_drug_name"]]

    cols = ["recall_at_1", "mrr", "recall_at_5", "recall_at_10",
            "T4_S2T", "T4_T2S", "T10_S2T", "T10_T2S",
            "T20_S2T", "T20_T2S"]
    col_hdr = ["R@1", "MRR", "R@5", "R@10",
               "T4 S→T", "T4 T→S", "T10 S→T", "T10 T→S",
               "T20 S→T", "T20 T→S"]

    print("\n" + "=" * 95)
    print("  CLOOB ABLATION SUMMARY  —  WITH DRUG NAME")
    print("=" * 95)
    hdr = f"{'Variant':<12}" + "".join(f"{h:>8}" for h in col_hdr)
    print(hdr)
    print("-" * 95)
    for r in rows_w:
        row = f"  {r['variant']} ({['NoH+NCE','NoH+LOB','H+NCE','H+LOB']['ABCD'.index(r['variant'])]}) "
        row += "".join(f"{r[c]:>8.3f}" for c in cols)
        print(row)

    if rows_n:
        print("\n" + "=" * 95)
        print("  CLOOB ABLATION SUMMARY  —  NO DRUG NAME  (Leakage Ablation)")
        print("=" * 95)
        print(hdr)
        print("-" * 95)
        for r in rows_n:
            row = f"  {r['variant']} ({['NoH+NCE','NoH+LOB','H+NCE','H+LOB']['ABCD'.index(r['variant'])]}) "
            row += "".join(f"{r[c]:>8.3f}" for c in cols)
            print(row)

        print("\n" + "=" * 95)
        print("  DRUG NAME LEAKAGE  —  % DROP  (with→no drug name)")
        print("=" * 95)
        print(hdr)
        print("-" * 95)
        for rw in rows_w:
            rn_list = [r for r in rows_n if r["variant"] == rw["variant"]]
            if not rn_list:
                continue
            rn  = rn_list[0]
            row = f"  {rw['variant']} ({['NoH+NCE','NoH+LOB','H+NCE','H+LOB']['ABCD'.index(rw['variant'])]}) "
            for c in cols:
                drop = (rw[c] - rn[c]) / rw[c] * 100 if rw[c] > 0 else 0
                row += f"{drop:>7.1f}%"
            print(row)
    print("=" * 95)


# =============================================================================
# DATA LOADING & SCAFFOLD SPLIT (identical to baseline 03_train_scaffold_split.py)
# =============================================================================
def load_and_split(csv_path: str, remove_drug_name: bool, device: str):
    """
    Load ChEMBL data, apply scaffold split, encode features.
    Text encoder and ECFP4 encoder are BOTH frozen.
    Returns numpy arrays + DataFrames.
    """
    df = pd.read_csv(csv_path)
    need_cols = {"smiles", "text_rich", "target_chembl_id"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. "
                         "Run 01_download_data.py first.")
    df = df.dropna(subset=["smiles", "text_rich"]).copy()

    if remove_drug_name:
        df["text_rich"] = df["text_rich"].str.replace(
            r"Drug:\s*[^.]+\. ?", "", regex=True).str.strip()
        print("  [Leakage ablation] Drug names stripped from text_rich")

    # Scaffold split (Bemis-Murcko, 90/10, seed=0)
    print("  Computing Murcko scaffolds ...")
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
    print(f"  Scaffold split → train: {len(train_df)} | test: {len(test_df)}")

    # ECFP4 fingerprints (frozen)
    print("  Computing ECFP4 fingerprints ...")
    X_mol_train = np.stack([ecfp4_bitvect(s, ECFP_BITS, ECFP_RADIUS)
                            for s in tqdm(train_df["smiles"].tolist(), leave=False)])
    X_mol_test  = np.stack([ecfp4_bitvect(s, ECFP_BITS, ECFP_RADIUS)
                            for s in tqdm(test_df["smiles"].tolist(),  leave=False)])

    # S-BiomedRoBERTa embeddings (frozen)
    print(f"  Encoding text with {TEXT_MODEL} ...")
    text_tok   = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    text_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    Z_text_train = cls_encode(train_df["text_rich"].tolist(),
                              text_tok, text_model, device,
                              max_length=MAX_TXT_LEN, bs=64).numpy()
    Z_text_test  = cls_encode(test_df["text_rich"].tolist(),
                              text_tok, text_model, device,
                              max_length=MAX_TXT_LEN, bs=64).numpy()

    # Free text model VRAM (it's frozen, we don't need it anymore)
    del text_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Z_text_train: {Z_text_train.shape}  "
          f"X_mol_train: {X_mol_train.shape}")
    return (Z_text_train, X_mol_train, train_df,
            Z_text_test,  X_mol_test,  test_df)


# =============================================================================
# MAIN
# =============================================================================
def main(args):
    set_seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"CLOOB Thin Bridges — Experiment Suite")
    print(f"Variants to run : {args.variant}")
    print(f"Run leakage     : {args.run_leakage}")
    print(f"Output dir      : {args.outdir}\n")

    variants_to_run = (["A", "B", "C", "D"]
                       if args.variant == "all"
                       else [args.variant.upper()])

    # Conditions: always run with drug name; optionally also without
    conditions = [False]
    if args.run_leakage or args.remove_drug_name:
        if args.remove_drug_name:
            conditions = [True]          # only nodrug if flag set explicitly
        else:
            conditions = [False, True]   # both conditions

    all_results = []

    for remove_drug in conditions:
        cond_label = "no drug name" if remove_drug else "with drug name"
        print(f"\n{'#'*65}")
        print(f"# CONDITION: {cond_label.upper()}")
        print(f"{'#'*65}")

        # Load data once per condition (text encoding differs if drug name stripped)
        print(f"\nLoading and encoding data ({cond_label}) ...")
        (Z_text_train, X_mol_train, train_df,
         Z_text_test,  X_mol_test,  test_df) = load_and_split(
            args.csv, remove_drug_name=remove_drug, device=device)

        for variant in variants_to_run:
            results = run_variant(
                variant=variant,
                Z_text_train=Z_text_train, X_mol_train=X_mol_train,
                train_df=train_df,
                Z_text_test=Z_text_test,   X_mol_test=X_mol_test,
                test_df=test_df,
                outdir=args.outdir,
                remove_drug_name=remove_drug,
                device=device)
            all_results.append(results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_results:
        print_summary_table(all_results)
        plot_summary_table(all_results, args.outdir)

        # Save combined CSV
        summary_csv = os.path.join(args.outdir, "summary_table.csv")
        pd.DataFrame(all_results).to_csv(summary_csv, index=False)
        print(f"\nSummary CSV saved → {summary_csv}")

    print("\nAll experiments complete.")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLOOB-enhanced Thin Bridges ablation suite (Variants A-D)")

    parser.add_argument(
        "--csv", type=str, default="chembl_mechanisms.csv",
        help="Path to ChEMBL CSV (output of 01_download_data.py)")

    parser.add_argument(
        "--outdir", type=str, default="outputs_cloob",
        help="Directory for all experiment outputs")

    parser.add_argument(
        "--variant", type=str, default="all",
        choices=["A", "B", "C", "D", "a", "b", "c", "d", "all"],
        help=("Which variant to run:\n"
              "  A = InfoNCE  + No Hopfield  (baseline)\n"
              "  B = InfoLOOB + No Hopfield\n"
              "  C = InfoNCE  + Hopfield\n"
              "  D = InfoLOOB + Hopfield (full CLOOB, proposed)\n"
              "  all = run all four sequentially"))

    parser.add_argument(
        "--remove_drug_name", action="store_true",
        help="Strip 'Drug: X.' from text_rich. Runs ONLY the no-drug condition.")

    parser.add_argument(
        "--run_leakage", action="store_true",
        help="Run both with-drug and no-drug conditions for full leakage ablation.")

    parser.add_argument(
        "--beta", type=float, default=BETA_HOPFIELD,
        help=f"Hopfield inverse temperature β (default {BETA_HOPFIELD}). "
             "Sweep: 3, 5, 7, 10")

    parser.add_argument(
        "--tau_inv", type=float, default=15.0,
        help="InfoLOOB τ⁻¹ temperature (default 15). FIXED, not learnable. "
             "Sweep: 10, 15, 20, 30")

    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Training epochs (default {EPOCHS})")

    args = parser.parse_args()

    # Allow CLI overrides of global constants
    BETA_HOPFIELD      = args.beta
    TAU_INFOLOOB       = 1.0 / args.tau_inv
    EPOCHS             = args.epochs

    main(args)