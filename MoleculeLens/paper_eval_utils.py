"""Shared evaluation and grouping helpers for the camera-ready paper artifacts."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable

import numpy as np


DEFAULT_TEXT_MAX_LENGTH = 96


def to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def diagonal_ranks_from_similarity(similarity) -> np.ndarray:
    """
    Canonical diagonal retrieval rank with optimistic tie-breaking.

    rank(i) = 1 + #{j : S[i, j] > S[i, i]}
    """
    sim = to_numpy(similarity).astype(np.float64, copy=False)
    correct = np.diag(sim)
    return (1 + np.sum(sim > correct[:, None], axis=1)).astype(int)


def diagonal_metric_summary(similarity, ks: Iterable[int] = (1, 5, 10)) -> dict:
    ranks = diagonal_ranks_from_similarity(similarity)
    summary = {"ranks": ranks, "MRR": float(np.mean(1.0 / ranks))}
    for k in ks:
        summary[f"Recall@{k}"] = float(np.mean(ranks <= k))
    return summary


def grouped_diagonal_recall_at1(similarity, group_labels, min_group: int = 3) -> float:
    sim = to_numpy(similarity).astype(np.float64, copy=False)
    idx_by_group = defaultdict(list)
    for idx, label in enumerate(group_labels):
        idx_by_group[str(label)].append(idx)

    hits = 0
    total = 0
    for idxs in idx_by_group.values():
        if len(idxs) < min_group:
            continue
        sub = sim[np.ix_(idxs, idxs)]
        ranks = diagonal_ranks_from_similarity(sub)
        hits += int(np.sum(ranks == 1))
        total += len(idxs)
    return hits / total if total else 0.0


@lru_cache(maxsize=None)
def structural_parent_key(smiles: str) -> str:
    """
    Canonical parent-structure key that collapses salts, solvates, and simple
    formulation fragments before comparing molecules.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    parent = rdMolStandardize.FragmentParent(mol)
    return Chem.MolToSmiles(parent, isomericSmiles=True)
