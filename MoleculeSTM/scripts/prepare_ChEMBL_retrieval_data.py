"""
prepare_ChEMBL_retrieval_data.py

Builds a structure-text retrieval dataset from the ChEMBL files already in
the MoleculeSTM HuggingFace dataset repo:

    ChEMBL_data/degree_threshold_0/raw/
        assay.tsv           - assay descriptions (text side)
        smiles_test.csv     - molecule SMILES, test split
        smiles_train.csv    - molecule SMILES, train split
        labels_test.npz     - binary activity matrix (molecules x assays)
        labels_train.npz    - binary activity matrix (molecules x assays)

Scale context:
    DrugBank retrieval sets used in the paper: ~1K-3K pairs (Table 7).
    Raw ChEMBL positive pairs: ~385K — far too large for T-choose-one eval.

    Strategy: one pair per molecule (the molecule's highest-connectivity
    assay), giving a dataset of comparable size to DrugBank.
    --max_pairs controls the cap (default 3000).

Usage:
    python prepare_ChEMBL_retrieval_data.py \\
        --dataspace ../data \\
        --skip_pubchem_filter

    # With PubChemSTM filtering (if CID2SMILES.csv is available):
    python prepare_ChEMBL_retrieval_data.py \\
        --dataspace ../data \\
        --pubchemstm_smiles_path ../data/PubChemSTM_data/raw/CID2SMILES.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonicalize(smi: str):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def load_pubchemstm_smiles(path: str) -> set:
    print(f"Loading PubChemSTM SMILES from {path} ...")
    df = pd.read_csv(path, header=0)
    smiles_col = df.columns[1]
    canon = set()
    for smi in tqdm(df[smiles_col].dropna().tolist(), desc="Canonicalising PubChemSTM"):
        c = canonicalize(smi)
        if c:
            canon.add(c)
    print(f"  {len(canon)} unique canonical SMILES in PubChemSTM")
    return canon


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_retrieval_pairs(smiles_csv, labels_array, assay_df, max_pairs, seed):
    """
    For each molecule pick its single most-active assay (highest column sum =
    most-tested assay, to mirror the paper's DrugBank selection logic of
    keeping the most informative description).  This gives one pair per
    molecule, keeping dataset size comparable to DrugBank (~1K-3K pairs).

    Falls back to all positive pairs if max_pairs is None.
    """
    smiles_df  = pd.read_csv(smiles_csv)
    smiles_col = [c for c in smiles_df.columns if c.lower() == "smiles"][0]
    smiles_list = smiles_df[smiles_col].tolist()

    n_mol, n_assay = labels_array.shape
    print(f"  Label matrix : {n_mol} molecules x {n_assay} assays")
    assert len(smiles_list) == n_mol, \
        f"SMILES file has {len(smiles_list)} rows but label matrix has {n_mol} rows."

    # Identify description and id columns
    desc_col = next((c for c in assay_df.columns
                     if any(k in c.lower() for k in ["description", "text", "desc"])), None)
    id_col   = next((c for c in assay_df.columns
                     if "assay" in c.lower() and "id" in c.lower()), assay_df.columns[0])
    if desc_col is None:
        raise ValueError(f"Cannot find description column. Columns: {assay_df.columns.tolist()}")
    assert len(assay_df) == n_assay, \
        f"assay.tsv has {len(assay_df)} rows but label matrix has {n_assay} columns."

    assay_ids   = assay_df[id_col].tolist()
    assay_texts = assay_df[desc_col].fillna("").tolist()

    # Assay popularity: number of active molecules per assay (column sum)
    # We use this to pick the single most-informative assay per molecule
    assay_popularity = (labels_array == 1).sum(axis=0)   # shape (n_assay,)

    rows = []
    for mol_idx in tqdm(range(n_mol), desc="Building pairs (1 per molecule)"):
        smi   = smiles_list[mol_idx]
        canon = canonicalize(smi)
        if not canon:
            continue

        # Indices of assays where this molecule is active
        active_assays = np.where(labels_array[mol_idx] == 1)[0]
        if len(active_assays) == 0:
            continue

        # Pick the most popular active assay (most data = best-described)
        best_assay = active_assays[np.argmax(assay_popularity[active_assays])]
        text = assay_texts[best_assay].strip()
        if not text:
            continue

        rows.append({
            "smiles":           smi,
            "canonical_smiles": canon,
            "assay_id":         assay_ids[best_assay],
            "text":             text,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["canonical_smiles", "text"])
    df = df.reset_index(drop=True)

    # Cap at max_pairs with reproducible sampling
    if max_pairs and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=seed).reset_index(drop=True)
        print(f"  Sampled down to {max_pairs} pairs (from {len(rows)} unique molecules)")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    raw_dir    = os.path.join(args.dataspace, "ChEMBL_data", "degree_threshold_0", "raw")
    output_dir = os.path.join(args.dataspace, "ChEMBL_data")
    os.makedirs(output_dir, exist_ok=True)

    # Verify required files
    required = ["assay.tsv", "smiles_test.csv", "smiles_train.csv",
                "labels_test.npz", "labels_train.npz"]
    for fname in required:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing: {fpath}\n\n"
                "Download with:\n"
                "  python download_ChEMBL_data.py --dataspace ../data"
            )

    # Load assay descriptions
    print("Loading assay.tsv ...")
    assay_df = pd.read_csv(os.path.join(raw_dir, "assay.tsv"), sep="\t")
    print(f"  Columns : {assay_df.columns.tolist()}")
    print(f"  Rows    : {len(assay_df)}")
    print(f"  Sample  :\n{assay_df.head(2).to_string()}\n")

    # Load label matrices
    print("Loading labels_test.npz ...")
    test_labels  = np.load(os.path.join(raw_dir, "labels_test.npz"),  allow_pickle=True)
    test_labels  = test_labels[list(test_labels.keys())[0]]
    if hasattr(test_labels, "toarray"):
        test_labels = test_labels.toarray()

    print("Loading labels_train.npz ...")
    train_labels = np.load(os.path.join(raw_dir, "labels_train.npz"), allow_pickle=True)
    train_labels = train_labels[list(train_labels.keys())[0]]
    if hasattr(train_labels, "toarray"):
        train_labels = train_labels.toarray()

    # Build test pairs
    print("\n--- Building TEST retrieval pairs ---")
    test_df = build_retrieval_pairs(
        smiles_csv    = os.path.join(raw_dir, "smiles_test.csv"),
        labels_array  = test_labels,
        assay_df      = assay_df,
        max_pairs     = args.max_pairs,
        seed          = args.seed,
    )

    # Build train pairs (optional)
    train_df = None
    if args.build_train:
        print("\n--- Building TRAIN retrieval pairs ---")
        train_df = build_retrieval_pairs(
            smiles_csv    = os.path.join(raw_dir, "smiles_train.csv"),
            labels_array  = train_labels,
            assay_df      = assay_df,
            max_pairs     = args.max_pairs,
            seed          = args.seed,
        )

    # PubChemSTM filtering (optional)
    if not args.skip_pubchem_filter:
        if not os.path.exists(args.pubchemstm_smiles_path):
            raise FileNotFoundError(
                f"PubChemSTM SMILES not found: {args.pubchemstm_smiles_path}\n"
                "Use --skip_pubchem_filter to skip, or download CID2SMILES.csv:\n"
                "  python download_ChEMBL_data.py --dataspace ../data"
            )
        pubchem_smiles = load_pubchemstm_smiles(args.pubchemstm_smiles_path)
        n_before = len(test_df)
        test_df  = test_df[~test_df["canonical_smiles"].isin(pubchem_smiles)].reset_index(drop=True)
        print(f"\nPubChemSTM filter removed {n_before - len(test_df)} test pairs.")
        if train_df is not None:
            n_before  = len(train_df)
            train_df  = train_df[~train_df["canonical_smiles"].isin(pubchem_smiles)].reset_index(drop=True)
            print(f"PubChemSTM filter removed {n_before - len(train_df)} train pairs.")

    # Save
    # "filtered" filename used when PubChemSTM filter was applied (mirrors DrugBank naming)
    suffix = "filtered" if not args.skip_pubchem_filter else "all"
    test_out = os.path.join(output_dir, f"ChEMBL_retrieval_test_{suffix}.csv")
    test_df.to_csv(test_out, index=False)
    print(f"\nSaved → {test_out}  ({len(test_df)} rows)")

    if train_df is not None:
        train_out = os.path.join(output_dir, f"ChEMBL_retrieval_train_{suffix}.csv")
        train_df.to_csv(train_out, index=False)
        print(f"Saved → {train_out}  ({len(train_df)} rows)")

    # Summary
    print("\n=== Summary ===")
    print(f"  Test  : {len(test_df):>5} pairs | "
          f"{test_df['canonical_smiles'].nunique()} molecules | "
          f"{test_df['assay_id'].nunique()} unique assays")
    if train_df is not None:
        print(f"  Train : {len(train_df):>5} pairs | "
              f"{train_df['canonical_smiles'].nunique()} molecules | "
              f"{train_df['assay_id'].nunique()} unique assays")
    print(f"\nSample rows:")
    print(test_df[["canonical_smiles", "assay_id", "text"]].head(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataspace", type=str, default="../data")
    parser.add_argument("--pubchemstm_smiles_path", type=str,
                        default="../data/PubChemSTM_data/raw/CID2SMILES.csv")
    parser.add_argument("--skip_pubchem_filter", action="store_true")
    parser.add_argument("--build_train", action="store_true")
    parser.add_argument(
        "--max_pairs", type=int, default=3000,
        help="Cap on number of pairs in each split. DrugBank sets have ~1K-3K pairs. "
             "Set to 0 or -1 for no cap (gives ~182K pairs — very slow for T-choose-one).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_pairs <= 0:
        args.max_pairs = None
    main(args)