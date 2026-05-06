#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a reproducible scaffold split manifest for ChEMBL retrieval experiments.

The manifest stores source-row indices from the original CSV so sibling repos can
consume exactly the same train/val/test split even if their internal dataset
pipelines differ.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core else None


def build_manifest(df, seed, frac_train, frac_val, frac_test):
    if not np.isclose(frac_train + frac_val + frac_test, 1.0):
        raise ValueError("split fractions must sum to 1.0")

    scaffolds = df["scaffold"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)

    n_scaffolds = len(scaffolds)
    cut_train = int(frac_train * n_scaffolds)
    cut_val = int((frac_train + frac_val) * n_scaffolds)

    split_scaffolds = {
        "train": set(scaffolds[:cut_train]),
        "val": set(scaffolds[cut_train:cut_val]),
        "test": set(scaffolds[cut_val:]),
    }

    split_rows = {}
    split_counts = {}
    scaffold_counts = {}
    for split_name, scaffold_set in split_scaffolds.items():
        rows = (
            df.loc[df["scaffold"].isin(scaffold_set), "source_row"]
            .astype(int)
            .sort_values()
            .tolist()
        )
        split_rows[split_name] = rows
        split_counts[split_name] = len(rows)
        scaffold_counts[split_name] = len(scaffold_set)

    return {
        "schema_version": 1,
        "split_method": "random_scaffold_by_unique_scaffold_count",
        "seed": int(seed),
        "fractions": {
            "train": float(frac_train),
            "val": float(frac_val),
            "test": float(frac_test),
        },
        "eligible_rows": int(len(df)),
        "eligible_scaffolds": int(df["scaffold"].nunique()),
        "split_row_counts": split_counts,
        "split_scaffold_counts": scaffold_counts,
        "splits": split_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Create ChEMBL split manifest")
    parser.add_argument("--csv", default="chembl_mechanisms.csv")
    parser.add_argument(
        "--out",
        default="splits/chembl_scaffold_seed0_train80_val10_test10.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frac_train", type=float, default=0.8)
    parser.add_argument("--frac_val", type=float, default=0.1)
    parser.add_argument("--frac_test", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).reset_index().rename(columns={"index": "source_row"})
    df = df.dropna(subset=["smiles", "text_rich"]).copy()
    df["scaffold"] = df["smiles"].apply(scaffold_from_smiles)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    manifest = build_manifest(df, args.seed, args.frac_train, args.frac_val, args.frac_test)
    manifest["csv"] = os.path.abspath(args.csv)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Saved {}".format(out_path))
    print("Eligible rows: {}".format(manifest["eligible_rows"]))
    print(
        "Split counts: train={} val={} test={}".format(
            manifest["split_row_counts"]["train"],
            manifest["split_row_counts"]["val"],
            manifest["split_row_counts"]["test"],
        )
    )


if __name__ == "__main__":
    main()
