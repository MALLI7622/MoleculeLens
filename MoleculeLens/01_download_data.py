# -*- coding: utf-8 -*-
"""
01_download_data.py
====================
Downloads approved drug-mechanism pairs from the ChEMBL API and saves
chembl_mechanisms.csv.

Fixes applied vs original notebook:
  - BUG #1 : max_phase=3 → max_phase=4  (approved drugs only, as in the paper)
  - BUG #11: molecule_chembl_id__in now receives a list, not a comma-joined string
  - BUG #5 : `import time` moved to top-level (was missing in pref_name cell)
  - Duplicate imports removed
  - Single consistent entry-point with argparse for output path

Usage:
    pip install chembl_webresource_client tqdm pandas
    python 01_download_data.py --out chembl_mechanisms.csv
"""

import time
import argparse
import pandas as pd
from itertools import islice
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

# ---------------------------------------------------------------------------
# ChEMBL clients
# ---------------------------------------------------------------------------
molecule  = new_client.molecule
mechanism = new_client.mechanism
target    = new_client.target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chunked(iterable, size):
    """Yield successive chunks of `size` from an iterable."""
    it = iter(iterable)
    for first in it:
        yield [first, *list(islice(it, size - 1))]


def fetch_smiles_batch(ids, batch_size=200):
    """Return {chembl_id: canonical_smiles} for all ids, using batched API calls."""
    id_to_smiles = {}
    batches = list(chunked(ids, batch_size))
    for batch in tqdm(batches, desc="Fetching SMILES in batches"):
        # FIX #11: pass list directly, not ",".join(batch)
        q = molecule.filter(molecule_chembl_id__in=batch).only([
            "molecule_chembl_id",
            "molecule_structures",
        ])
        for rec in q:
            mid = rec.get("molecule_chembl_id")
            smi = (rec.get("molecule_structures") or {}).get("canonical_smiles")
            if mid and smi:
                id_to_smiles[mid] = smi
    return id_to_smiles


def get_target_name(tchembl, retries=3):
    """Fetch preferred target name with simple retry logic."""
    if not tchembl:
        return None
    for _ in range(retries):
        try:
            t = target.get(tchembl)
            if t:
                return t.get("pref_name") or t.get("target_type")
        except Exception:
            time.sleep(0.5)
    return None


def get_pref_name(chembl_id):
    """Fetch preferred drug name from molecule endpoint."""
    try:
        m = molecule.get(chembl_id)
        return (m or {}).get("pref_name")
    except Exception:
        time.sleep(0.2)
        return None


def build_text_rich(row):
    """Concatenate mechanism + target + action + drug name into text_rich."""
    parts = [row["mechanism_of_action"]]
    if isinstance(row.get("target_name"), str):
        parts.append(f"Target: {row['target_name']}.")
    if isinstance(row.get("action_type"), str):
        parts.append(f"Action: {row['action_type']}.")
    if isinstance(row.get("pref_name"), str):
        parts.append(f"Drug: {row['pref_name']}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    # ------------------------------------------------------------------
    # 1. Fetch mechanism rows — APPROVED drugs only (max_phase=4)
    # ------------------------------------------------------------------
    # FIX #1: was max_phase=3 in the original notebook (line 27 / cell 006)
    print("Fetching mechanism rows for approved drugs (max_phase=4) ...")
    mech_rows = mechanism.filter(max_phase=4, molecular_mechanism=1)
    df = pd.DataFrame(mech_rows)
    print(f"  Raw rows: {len(df)}")

    # Keep only useful columns (gracefully handle missing ones)
    keep = [c for c in [
        "molecule_chembl_id", "mechanism_of_action", "target_chembl_id",
        "action_type", "max_phase", "direct_interaction", "molecular_mechanism",
    ] if c in df.columns]
    df = df[keep]

    # ------------------------------------------------------------------
    # 2. Fetch canonical SMILES via batched API
    # ------------------------------------------------------------------
    ids = df["molecule_chembl_id"].dropna().astype(str).unique().tolist()
    print(f"  Unique molecule IDs: {len(ids)}")
    id_to_smiles = fetch_smiles_batch(ids, batch_size=200)
    df["smiles"] = df["molecule_chembl_id"].map(id_to_smiles)

    # ------------------------------------------------------------------
    # 3. Fetch target names
    # ------------------------------------------------------------------
    tqdm.pandas(desc="Fetching target names")
    df["target_name"] = df["target_chembl_id"].progress_apply(get_target_name)

    # ------------------------------------------------------------------
    # 4. Fetch preferred drug names (for text_rich)
    # ------------------------------------------------------------------
    if "molecule_chembl_id" in df.columns:
        tqdm.pandas(desc="Fetching preferred drug names")
        df["pref_name"] = df["molecule_chembl_id"].progress_apply(get_pref_name)

    # ------------------------------------------------------------------
    # 5. Clean + deduplicate
    # ------------------------------------------------------------------
    df = df.dropna(subset=["smiles", "mechanism_of_action"])
    df = df[df["mechanism_of_action"].str.len() > 15]
    df = df.drop_duplicates(
        subset=["molecule_chembl_id", "target_chembl_id"], keep="first"
    ).reset_index(drop=True)
    print(f"  After cleaning: {len(df)} rows")

    # ------------------------------------------------------------------
    # 6. Build text_rich
    # ------------------------------------------------------------------
    df["text_rich"] = df.apply(build_text_rich, axis=1)

    # ------------------------------------------------------------------
    # 7. Finalise columns and save
    # ------------------------------------------------------------------
    cols = [c for c in [
        "molecule_chembl_id", "smiles", "mechanism_of_action",
        "target_chembl_id", "target_name", "action_type",
        "max_phase", "pref_name", "text_rich",
    ] if c in df.columns]
    df = df[cols]

    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows → {args.out}")
    print(df[["molecule_chembl_id", "smiles", "text_rich"]].head(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="chembl_mechanisms.csv",
                        help="Output CSV path (default: chembl_mechanisms.csv)")
    main(parser.parse_args())