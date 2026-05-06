#!/usr/bin/env python3
"""Run a full-loss multi-seed check for MoleculeLens.

This complements ``12_robustness.py``. The existing five-seed robustness table
is a fast proxy with plain InfoNCE. This script uses the same same-target
weighting and hardest-same-target margin loss as ``03_train_scaffold_split.py``
while reusing cached global frozen embeddings.

Outputs:
    outputs/robustness/full_loss_multiseed_results.csv
    results/full_loss_multiseed_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader, Dataset

from paper_eval_utils import diagonal_metric_summary


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs/robustness"
RESULTS_DIR = ROOT / "results"

SHARED_D = 256
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEMP = 0.07
MARGIN = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


class PairDataset(Dataset):
    def __init__(self, z_text: np.ndarray, x_mol: np.ndarray, groups: list[str]):
        self.z_text = torch.tensor(z_text, dtype=torch.float32)
        self.x_mol = torch.tensor(x_mol, dtype=torch.float32)
        self.groups = np.asarray(groups).astype(str)

    def __len__(self) -> int:
        return len(self.z_text)

    def __getitem__(self, idx: int):
        return self.z_text[idx], self.x_mol[idx], self.groups[idx]


def build_same_target_weights(groups, device: torch.device) -> torch.Tensor:
    groups = list(map(str, groups))
    same = torch.tensor(
        [[i != j and groups[i] == groups[j] for j in range(len(groups))] for i in range(len(groups))],
        device=device,
    )
    weights = torch.ones((len(groups), len(groups)), device=device)
    weights[same] = 2.0
    return weights


def hardest_same_target_neg(logits: torch.Tensor, groups) -> torch.Tensor:
    groups = list(map(str, groups))
    hard = torch.full((logits.size(0),), float("-inf"), device=logits.device)
    for i, group in enumerate(groups):
        mask = torch.tensor(
            [i != j and group == groups[j] for j in range(len(groups))],
            device=logits.device,
        )
        if mask.any():
            hard[i] = logits[i][mask].max()
    return hard


def full_loss(bt: torch.Tensor, bm: torch.Tensor, groups, device: torch.device) -> torch.Tensor:
    bt = F.normalize(bt, dim=1)
    bm = F.normalize(bm, dim=1)
    logits = (bt @ bm.T) / TEMP
    labels = torch.arange(bt.size(0), device=device)

    weights = build_same_target_weights(groups, device)
    ce_row = F.cross_entropy(logits, labels, reduction="none")
    ce_col = F.cross_entropy(logits.T, labels, reduction="none")
    with torch.no_grad():
        batch = logits.size(0)
        avg_w_row = (weights.sum(dim=1) - torch.diag(weights)) / max(batch - 1, 1)
        avg_w_col = (weights.sum(dim=0) - torch.diag(weights)) / max(batch - 1, 1)
    loss_nce = 0.5 * ((ce_row * avg_w_row).mean() + (ce_col * avg_w_col).mean())

    pos = logits.diag()
    hard_same = hardest_same_target_neg(logits, groups)
    mask = torch.isfinite(hard_same)
    margin_loss = (
        torch.clamp(MARGIN - (pos[mask] - hard_same[mask]), min=0).mean()
        if mask.any()
        else torch.tensor(0.0, device=device)
    )
    return loss_nce + margin_loss


def scaffold_split(df: pd.DataFrame, seed: int) -> tuple[list[int], list[int]]:
    df_s = df.copy()
    df_s["scaffold"] = df_s["smiles"].apply(murcko_scaffold)
    df_s = df_s.dropna(subset=["scaffold"])
    scaffolds = df_s["scaffold"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    cut = int(0.9 * len(scaffolds))
    train_scaffolds = set(scaffolds[:cut])
    test_scaffolds = set(scaffolds[cut:])
    train_idx = df_s[df_s["scaffold"].isin(train_scaffolds)].index.tolist()
    test_idx = df_s[df_s["scaffold"].isin(test_scaffolds)].index.tolist()
    return train_idx, test_idx


def train_and_eval(
    z_all: torch.Tensor,
    x_all: np.ndarray,
    df: pd.DataFrame,
    seed: int,
    epochs: int,
    device: torch.device,
) -> dict[str, float | int]:
    set_seed(seed)
    train_idx, test_idx = scaffold_split(df, seed)
    z_train = z_all[train_idx].numpy()
    z_test = z_all[test_idx].numpy()
    x_train = x_all[train_idx]
    x_test = x_all[test_idx]
    groups = df.iloc[train_idx]["target_chembl_id"].fillna("UNK").astype(str).tolist()

    proj_text = nn.Linear(z_all.shape[1], SHARED_D).to(device)
    proj_mol = nn.Linear(x_all.shape[1], SHARED_D).to(device)
    opt = torch.optim.AdamW(
        list(proj_text.parameters()) + list(proj_mol.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loader = DataLoader(
        PairDataset(z_train, x_train, groups),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    for _ in range(epochs):
        proj_text.train()
        proj_mol.train()
        for z_batch, x_batch, group_batch in loader:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)
            loss = full_loss(proj_text(z_batch), proj_mol(x_batch), group_batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()

    proj_text.eval()
    proj_mol.eval()
    with torch.no_grad():
        bt = F.normalize(proj_text(torch.tensor(z_test, dtype=torch.float32, device=device)), dim=1)
        bm = F.normalize(proj_mol(torch.tensor(x_test, dtype=torch.float32, device=device)), dim=1)
    metrics = diagonal_metric_summary((bt @ bm.T).detach().cpu().numpy(), ks=(1, 5, 10))

    return {
        "seed": seed,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "epochs": epochs,
        "R@1": round(metrics["Recall@1"], 4),
        "R@5": round(metrics["Recall@5"], 4),
        "R@10": round(metrics["Recall@10"], 4),
        "MRR": round(metrics["MRR"], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(ROOT / "outputs/global_df.csv")
    z_all = torch.load(ROOT / "outputs/global_Z_text.pt", map_location="cpu").float()
    x_all = np.load(ROOT / "outputs/global_X_mol.npy").astype(np.float32)
    if not (len(df) == len(z_all) == len(x_all)):
        raise ValueError("Cached metadata, text embeddings, and fingerprints are not aligned.")

    rows = []
    for seed in args.seeds:
        row = train_and_eval(z_all, x_all, df, seed, args.epochs, device)
        rows.append(row)
        print(
            f"seed={seed} n_test={row['n_test']} R@1={row['R@1']:.3f} "
            f"R@5={row['R@5']:.3f} R@10={row['R@10']:.3f} MRR={row['MRR']:.3f}"
        )

    out_csv = OUT_DIR / "full_loss_multiseed_results.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "semantics": "full MoleculeLens loss: same-target weighting plus hardest-same-target margin",
        "notes": "Uses cached frozen text embeddings and 90/10 scaffold splits. Seeds 0-4 were selected as the same five consecutive seeds used by the proxy multi-seed analysis before evaluating full-loss results.",
        "csv": str(out_csv.relative_to(ROOT)),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "mean_R@1": float(out_df["R@1"].mean()),
        "std_R@1": float(out_df["R@1"].std(ddof=1)) if len(out_df) > 1 else 0.0,
        "mean_MRR": float(out_df["MRR"].mean()),
        "std_MRR": float(out_df["MRR"].std(ddof=1)) if len(out_df) > 1 else 0.0,
    }
    summary_path = RESULTS_DIR / "full_loss_multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
