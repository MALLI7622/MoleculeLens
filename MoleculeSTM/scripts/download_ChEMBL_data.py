"""
download_ChEMBL_data.py

Downloads only the ChEMBL files needed for retrieval from the MoleculeSTM
HuggingFace dataset repo, rather than the full 2.79 GB snapshot.

Files downloaded (~300 MB total):
    ChEMBL_data/degree_threshold_0/raw/assay.tsv          (213 kB)
    ChEMBL_data/degree_threshold_0/raw/smiles_test.csv    (8.35 MB)
    ChEMBL_data/degree_threshold_0/raw/smiles_train.csv   (12.5 MB)
    ChEMBL_data/degree_threshold_0/raw/labels_test.npz    (5.08 MB)
    ChEMBL_data/degree_threshold_0/raw/labels_train.npz   (7.62 MB)

Usage (run from scripts/ folder):
    python download_ChEMBL_data.py --dataspace ../data
"""

import os
import argparse
from huggingface_hub import hf_hub_download

REPO_ID   = "chao1224/MoleculeSTM"
REPO_TYPE = "dataset"

# Only the files we actually need — skip rdkit_molecule.pkl (180 MB) and labels.npz
CHEMBL_FILES = [
    "ChEMBL_data/degree_threshold_0/raw/assay.tsv",
    "ChEMBL_data/degree_threshold_0/raw/smiles_test.csv",
    "ChEMBL_data/degree_threshold_0/raw/smiles_train.csv",
    "ChEMBL_data/degree_threshold_0/raw/labels_test.npz",
    "ChEMBL_data/degree_threshold_0/raw/labels_train.npz",
]


def main(args):
    raw_dir = os.path.join(args.dataspace, "ChEMBL_data", "degree_threshold_0", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading ChEMBL files to: {raw_dir}\n")

    for repo_path in CHEMBL_FILES:
        filename = os.path.basename(repo_path)
        local_path = os.path.join(raw_dir, filename)

        if os.path.exists(local_path) and not args.force:
            print(f"  [skip] {filename} already exists")
            continue

        print(f"  Downloading {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=repo_path,
            local_dir=args.dataspace,
        )
        print(f"  Done → {local_path}")

    print("\nAll ChEMBL files downloaded.")
    print("Now run:")
    print("  python prepare_ChEMBL_retrieval_data.py --dataspace ../data --skip_pubchem_filter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataspace", type=str, default="../data",
                        help="Root data directory (default: ../data)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist.")
    args = parser.parse_args()
    main(args)