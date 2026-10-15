"""Shared core for the NMI revision experiments (see NEXT_STEPS.md).

Reuses the paper's model exactly (03_train_scaffold_split.py):
ECFP fingerprint -> Linear(2048,256), frozen S-Biomed-RoBERTa CLS -> Linear(768,256),
same-target-weighted symmetric InfoNCE + hardest-same-target margin, AdamW, 100 epochs.

Additions:
  * text encodings cached once per text variant, at all 13 layers (for E14);
  * fingerprint options: radius, nbits, parent-structure (salt stripping), counts;
  * split builders: canonical manifest, scaffold partitions, target-disjoint, joint;
  * vectorised loss (numerically identical to the paper's per-element loops);
  * cluster bootstrap over parent structures.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]          # MoleculeLens/
sys.path.insert(0, str(ROOT))
from paper_eval_utils import diagonal_metric_summary, structural_parent_key  # noqa: E402

CACHE = ROOT / "nmi_cache"
RESULTS = ROOT / "nmi_results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

TEXT_MODEL = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
MAX_TXT_LEN = 96
CANON_CSV = ROOT / "chembl_mechanisms.csv"
CANON_MANIFEST = ROOT / "splits" / "chembl_scaffold_seed0_train80_val10_test10.json"

HP = dict(batch_size=512, epochs=100, lr=1e-3, weight_decay=1e-4, temp=0.07,
          margin=0.1, dim=256)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_df(csv=CANON_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv)
    return df.dropna(subset=["smiles", "text_rich"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# text variants
# ---------------------------------------------------------------------------
DRUG_RE = r"Drug:\s*[^.]+\. ?"


def text_variant(df: pd.DataFrame, variant: str, seed: int = 0) -> list[str]:
    """Return model text for each row.

    rich       : paper text_rich (mechanism + target + action + drug name)
    nodrug     : text_rich with 'Drug: X.' removed (paper's no-drug condition)
    mech       : mechanism_of_action only
    stem       : text_rich with the drug name replaced by its last 5 letters
                 (INN-stem proxy: '-tinib', '-azole', ...)
    shuffled   : text_rich with drug names permuted across rows (name kept, identity broken)
    nameonly   : 'Drug: X.' only
    """
    rich = df["text_rich"].astype(str)
    if variant == "rich":
        return rich.tolist()
    if variant == "nodrug":
        return rich.str.replace(DRUG_RE, "", regex=True).str.strip().tolist()
    if variant == "mech":
        return df["mechanism_of_action"].astype(str).tolist()
    names = df["pref_name"].fillna("").astype(str)
    base = rich.str.replace(DRUG_RE, "", regex=True).str.strip()
    if variant == "stem":
        stems = names.map(lambda n: ("-" + n.split()[0][-5:].lower()) if n else "")
        return [f"{b} Drug: {s}." if s else b for b, s in zip(base, stems)]
    if variant == "shuffled":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(names))
        sh = names.values[perm]
        return [f"{b} Drug: {s}." if s else b for b, s in zip(base, sh)]
    if variant == "nameonly":
        return [f"Drug: {n}." for n in names]
    raise ValueError(variant)


# ---------------------------------------------------------------------------
# text encoding (cached, all layers)
# ---------------------------------------------------------------------------
_TOK = _ENC = None


def _encoder():
    global _TOK, _ENC
    if _ENC is None:
        from transformers import AutoModel, AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
        _ENC = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE).eval()
    return _TOK, _ENC


@torch.no_grad()
def encode_texts_meanpool(texts: list[str], bs: int = 64) -> torch.Tensor:
    """Final-layer mean-pooled token embeddings (masked). Cached."""
    key = hashlib.md5(("\n".join(texts) + f"|{MAX_TXT_LEN}|mean").encode()).hexdigest()[:16]
    path = CACHE / f"textmean_{key}.pt"
    if path.exists():
        return torch.load(path)
    tok, enc = _encoder()
    outs = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_TXT_LEN,
                return_tensors="pt").to(DEVICE)
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        outs.append(((h * m).sum(1) / m.sum(1)).cpu())
    Z = torch.cat(outs, 0)
    torch.save(Z, path)
    return Z


@torch.no_grad()
def encode_texts(texts: list[str], all_layers: bool = False, bs: int = 64) -> torch.Tensor:
    """CLS embeddings. Returns [N,768] (final layer) or [N,13,768] if all_layers.
    Cached on disk by content hash."""
    key = hashlib.md5(("\n".join(texts) + f"|{MAX_TXT_LEN}").encode()).hexdigest()[:16]
    path = CACHE / f"text_{key}_L{'all' if all_layers else '12'}.pt"
    if path.exists():
        return torch.load(path)
    alt = CACHE / f"text_{key}_Lall.pt"
    if not all_layers and alt.exists():
        return torch.load(alt)[:, 12, :].contiguous()
    tok, enc = _encoder()
    outs = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_TXT_LEN,
                return_tensors="pt").to(DEVICE)
        if all_layers:
            hs = enc(**b, output_hidden_states=True).hidden_states
            outs.append(torch.stack([h[:, 0, :] for h in hs], dim=1).cpu())
        else:
            outs.append(enc(**b).last_hidden_state[:, 0, :].cpu())
    Z = torch.cat(outs, 0)
    torch.save(Z, path)
    return Z


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------
def parent_smiles(smi: str) -> str:
    return structural_parent_key(smi)


def fingerprint_matrix(smiles: list[str], radius=2, nbits=2048, parent=False,
                       counts=False) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")
    key = hashlib.md5(("\n".join(smiles) + f"|{radius}|{nbits}|{parent}|{counts}").encode()).hexdigest()[:16]
    path = CACHE / f"fp_{key}.npy"
    if path.exists():
        return np.load(path)
    X = np.zeros((len(smiles), nbits), dtype=np.float32)
    for i, s in enumerate(smiles):
        s2 = parent_smiles(s) if parent else s
        m = Chem.MolFromSmiles(s2)
        if m is None:
            continue
        if counts:
            fp = AllChem.GetHashedMorganFingerprint(m, radius, nBits=nbits)
            for k, v in fp.GetNonzeroElements().items():
                X[i, k] = v
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
            arr = np.zeros((nbits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            X[i] = arr
    np.save(path, X)
    return X


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def canonical_split() -> dict:
    m = json.load(open(CANON_MANIFEST))
    return {k: list(v) for k, v in m["splits"].items()}


def murcko(smi: str):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(core) if core else None


def _group_split(groups: np.ndarray, seed: int, frac_test=0.1, frac_val=0.1) -> dict:
    """Random group-level split with group shuffling (same recipe as 07_make_chembl_split_manifest,
    i.e. split by unique-group count)."""
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test, n_val = int(round(frac_test * n)), int(round(frac_val * n))
    test_g, val_g = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    idx = np.arange(len(groups))
    return {"train": [int(i) for i in idx if groups[i] not in test_g and groups[i] not in val_g],
            "val": [int(i) for i in idx if groups[i] in val_g],
            "test": [int(i) for i in idx if groups[i] in test_g]}


def scaffold_split(df, seed, frac_train=0.8, frac_val=0.1) -> dict:
    """Exact recipe of 07_make_chembl_split_manifest.py (seed 0 == canonical manifest):
    first-appearance scaffold order, acyclic molecules share the empty scaffold '', int() cuts."""
    sc = df["smiles"].map(murcko)
    ok = sc.notna().values
    scaffolds = sc[ok].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    n = len(scaffolds)
    c1, c2 = int(frac_train * n), int((frac_train + frac_val) * n)
    sets = {"train": set(scaffolds[:c1]), "val": set(scaffolds[c1:c2]), "test": set(scaffolds[c2:])}
    return {k: [int(i) for i in np.where(ok & sc.isin(v).values)[0]] for k, v in sets.items()}


def scaffold_of_rows(df) -> np.ndarray:
    return df["smiles"].map(lambda s: murcko(s) if murcko(s) is not None else s).values


def target_split(df, seed) -> dict:
    return _group_split(df["target_chembl_id"].fillna("UNK").astype(str).values, seed)


def joint_split(df, seed, frac=0.1) -> dict:
    """Scaffold- AND target-disjoint: choose held-out scaffolds and targets; test = rows whose
    scaffold and target are both held out; rows touching exactly one held-out set are dropped."""
    sc = df["smiles"].map(lambda s: murcko(s) or s).values
    tg = df["target_chembl_id"].fillna("UNK").astype(str).values
    rng = np.random.default_rng(seed)
    us, ut = np.array(sorted(set(sc))), np.array(sorted(set(tg)))
    rng.shuffle(us); rng.shuffle(ut)
    hs = set(us[:int(round(0.3 * len(us)))])
    ht = set(ut[:int(round(0.3 * len(ut)))])
    test = [i for i in range(len(df)) if sc[i] in hs and tg[i] in ht]
    train = [i for i in range(len(df)) if sc[i] not in hs and tg[i] not in ht]
    return {"train": train, "val": [], "test": test,
            "dropped": len(df) - len(train) - len(test)}


# ---------------------------------------------------------------------------
# model + loss
# ---------------------------------------------------------------------------
def full_loss(bt, bm, groups, T=0.07, margin=0.1):
    """Vectorised version of 03_train_scaffold_split.contrastive_loss_with_weights."""
    bt = F.normalize(bt, dim=1)
    bm = F.normalize(bm, dim=1)
    logits = (bt @ bm.T) / T
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)
    g = torch.as_tensor(groups, device=logits.device)
    same = (g[:, None] == g[None, :]) & ~torch.eye(B, dtype=torch.bool, device=logits.device)
    W = 1.0 + same.float()
    ce_row = F.cross_entropy(logits, labels, reduction="none")
    ce_col = F.cross_entropy(logits.T, labels, reduction="none")
    with torch.no_grad():
        avg_w_row = (W.sum(1) - 1.0) / (B - 1)
        avg_w_col = (W.sum(0) - 1.0) / (B - 1)
    loss_nce = 0.5 * ((ce_row * avg_w_row).mean() + (ce_col * avg_w_col).mean())
    hard = logits.masked_fill(~same, float("-inf")).max(1).values
    mask = torch.isfinite(hard)
    pos = logits.diag()
    m_loss = (torch.clamp(margin - (pos[mask] - hard[mask]), min=0).mean()
              if mask.any() else logits.new_zeros(()))
    return loss_nce + m_loss


def group_ids(labels) -> np.ndarray:
    _, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return inv


def train_bridge(Zt: np.ndarray, Xm: np.ndarray, groups, seed=0, dim=None, epochs=None,
                 loss="full", verbose=False):
    """Train the two linear heads. Returns (proj_text, proj_mol) on CPU, eval mode."""
    dim = dim or HP["dim"]
    epochs = epochs or HP["epochs"]
    set_seed(seed)
    Zt = torch.as_tensor(np.asarray(Zt), dtype=torch.float32)
    Xm = torch.as_tensor(np.asarray(Xm), dtype=torch.float32)
    gid = torch.as_tensor(group_ids(groups))
    pt = nn.Linear(Zt.shape[1], dim).to(DEVICE)
    pm = nn.Linear(Xm.shape[1], dim).to(DEVICE)
    opt = torch.optim.AdamW(list(pt.parameters()) + list(pm.parameters()),
                            lr=HP["lr"], weight_decay=HP["weight_decay"])
    N = len(Zt)
    bs = HP["batch_size"]
    Ztd, Xmd, gd = Zt.to(DEVICE), Xm.to(DEVICE), gid.to(DEVICE)
    for ep in range(epochs):
        # same sampling as DataLoader(shuffle=True, drop_last=True) under torch RNG
        perm = torch.randperm(N)
        nb = N // bs if N >= bs else 1
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs].to(DEVICE)
            bt, bm = pt(Ztd[idx]), pm(Xmd[idx])
            if loss == "full":
                l = full_loss(bt, bm, gd[idx], HP["temp"], HP["margin"])
            else:  # plain symmetric InfoNCE
                bt_, bm_ = F.normalize(bt, dim=1), F.normalize(bm, dim=1)
                lg = bt_ @ bm_.T / HP["temp"]
                lab = torch.arange(len(idx), device=DEVICE)
                l = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
            opt.zero_grad(); l.backward(); opt.step()
        if verbose and (ep % 20 == 0):
            print(f"  ep {ep} loss {l.item():.4f}")
    return pt.cpu().eval(), pm.cpu().eval()


@torch.no_grad()
def similarity(pt, pm, Zt, Xm) -> np.ndarray:
    bt = F.normalize(pt(torch.as_tensor(np.asarray(Zt), dtype=torch.float32)), dim=1)
    bm = F.normalize(pm(torch.as_tensor(np.asarray(Xm), dtype=torch.float32)), dim=1)
    return (bt @ bm.T).numpy()


def metrics(S) -> dict:
    m = diagonal_metric_summary(S, ks=(1, 5, 10))
    return {"R@1": m["Recall@1"], "R@5": m["Recall@5"], "R@10": m["Recall@10"],
            "MRR": m["MRR"], "ranks": m["ranks"]}


def strip(m: dict) -> dict:
    return {k: float(v) for k, v in m.items() if k != "ranks"}


def structure_r1(S, smiles_test) -> float:
    """R@1 against a structure-deduplicated gallery (unique SMILES), random tie-break (expected).
    Recommended main protocol (see E1): multi-mechanism drugs are not double-counted and
    identical-fingerprint ties cannot inflate the score."""
    g = group_ids(smiles_test)
    G = g.max() + 1
    rep = np.array([np.where(g == k)[0][0] for k in range(G)])
    Sg = np.asarray(S)[:, rep]
    own = Sg[np.arange(len(g)), g]
    gt = np.sum(Sg > own[:, None] + 1e-7, axis=1)
    eq = np.sum(np.abs(Sg - own[:, None]) <= 1e-7, axis=1)
    return float(np.mean((gt == 0) / eq))


# ---------------------------------------------------------------------------
# attribution (paper Eq. 1 restricted to present bits, as in Appendix D)
# ---------------------------------------------------------------------------
@torch.no_grad()
def saliency(pt, pm, z_text: np.ndarray, x_mol: np.ndarray) -> np.ndarray:
    """attr_k = x_k * [W_m^T b_t_hat]_k for one (text, molecule) pair; zeros for absent bits."""
    bt = F.normalize(pt(torch.as_tensor(z_text, dtype=torch.float32)[None]), dim=1)[0]
    w = (pm.weight.T @ bt).numpy()
    return np.asarray(x_mol) * w


def bit_environments(smi: str, radius=2, nbits=2048, parent=False) -> dict:
    """{bit: [(center_atom, radius), ...]} for a folded Morgan fingerprint."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(parent_smiles(smi) if parent else smi)
    info = {}
    if m is not None:
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits, bitInfo=info)
    return info


def env_atoms(mol, center, rad) -> set:
    from rdkit import Chem
    if rad == 0:
        return {center}
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
    atoms = {center}
    for b in bonds:
        bd = mol.GetBondWithIdx(b)
        atoms.update([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()])
    return atoms


def env_smiles(mol, center, rad) -> str:
    from rdkit import Chem
    if rad == 0:
        return mol.GetAtomWithIdx(center).GetSymbol()
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
    sub = Chem.PathToSubmol(mol, bonds)
    return Chem.MolToSmiles(sub)


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
def cluster_bootstrap(ranks: np.ndarray, clusters, n_boot=10000, seed=0) -> dict:
    """Percentile CIs for R@1/5/10/MRR, resampling whole clusters (e.g. parent structures)."""
    ranks = np.asarray(ranks)
    cl = group_ids(clusters)
    members = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
    rng = np.random.default_rng(seed)
    nc = len(members)
    stats = {k: [] for k in ("R@1", "R@5", "R@10", "MRR")}
    for _ in range(n_boot):
        pick = rng.integers(0, nc, nc)
        idx = np.concatenate([members[p] for p in pick])
        r = ranks[idx]
        stats["R@1"].append(np.mean(r <= 1)); stats["R@5"].append(np.mean(r <= 5))
        stats["R@10"].append(np.mean(r <= 10)); stats["MRR"].append(np.mean(1.0 / r))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def row_bootstrap(ranks, n_boot=10000, seed=0) -> dict:
    return cluster_bootstrap(ranks, np.arange(len(ranks)), n_boot, seed)


def fmt_ci(v, ci) -> str:
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _C:
            def __enter__(self_):
                self_.s = time.time()

            def __exit__(self_, *a):
                timer.t[name] = timer.t.get(name, 0) + time.time() - self_.s
        return _C()


# ---------------------------------------------------------------------------
# canonical experiment helper
# ---------------------------------------------------------------------------
def run_split(df, split: dict, text: str = "rich", seed=0, radius=2, nbits=2048, parent=False,
              counts=False, dim=None, loss="full", Zall=None, Xall=None, return_model=False):
    """Train on split['train'], evaluate on split['test']. Returns dict with metrics + ranks."""
    if Zall is None:
        Zall = encode_texts(text_variant(df, text)).numpy()
    if Xall is None:
        Xall = fingerprint_matrix(df["smiles"].tolist(), radius, nbits, parent, counts)
    tr, te = np.array(split["train"]), np.array(split["test"])
    groups = df["target_chembl_id"].fillna("UNK").astype(str).values
    pt, pm = train_bridge(Zall[tr], Xall[tr], groups[tr], seed=seed, dim=dim, loss=loss)
    S = similarity(pt, pm, Zall[te], Xall[te])
    m = metrics(S)
    mm = strip(m)
    mm["R@1_struct"] = structure_r1(S, df["smiles"].values[te])
    out = {"metrics": mm, "ranks": m["ranks"].tolist(), "n_test": int(len(te)),
           "n_train": int(len(tr))}
    if return_model:
        out["model"] = (pt, pm)
        out["S"] = S
    return out
