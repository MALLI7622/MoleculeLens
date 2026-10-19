"""E05b: data funnel for the all-phase ChEMBL mechanism dataset (for the paper's data section).

Re-downloads the mechanism table with the same filters as e05_download_expanded.py, but applies the
cleaning steps one at a time and records how many rows each removes. Rows dropped for lacking a SMILES
are broken down by ChEMBL molecule_type (antibody, protein, oligonucleotide, ...), so the exclusion of
biologics can be stated exactly. Also checks the result against the committed CSV.
Output: nmi_results/e05b_data_funnel.{json,md}
"""
import json
import sys
import time
from itertools import islice
from pathlib import Path

import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data"
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "nmi_results"
COMMITTED = HERE.parent / "chembl_mechanisms_allphases.csv"


def retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print("retry", k, e, flush=True)
            time.sleep(2 * (k + 1))
    return None


def get_all(resource, params, key):
    """Page through a ChEMBL REST resource."""
    out, offset = [], 0
    while True:
        p = dict(params, format="json", limit=1000, offset=offset)
        js = retry(lambda: requests.get(f"{API}/{resource}", params=p, timeout=120).json())
        if js is None:
            sys.exit(f"failed to fetch {resource} at offset {offset}")
        out.extend(js[key])
        offset += 1000
        if not js["page_meta"]["next"]:
            break
    return out


def chunked(it, n):
    it = iter(it)
    for first in it:
        yield [first, *list(islice(it, n - 1))]


t0 = time.time()
status = retry(lambda: requests.get(f"{API}/status", params={"format": "json"}, timeout=60).json()) or {}
all_rows = get_all("mechanism", {}, "mechanisms")
df = pd.DataFrame(all_rows)
print("all mechanism rows", len(df), f"{time.time()-t0:.0f}s", flush=True)

steps = [("ChEMBL drug_mechanism table, all rows", len(df), None)]


def step(label, new):
    global df
    steps.append((label, len(new), len(df) - len(new)))
    df = new.reset_index(drop=True)


step("molecular_mechanism = 1 (the e05 download filter)", df[df["molecular_mechanism"] == 1])
step("has a molecule ChEMBL id", df[df["molecule_chembl_id"].notna()])

ids = df["molecule_chembl_id"].astype(str).unique().tolist()
mol = {}
for b in chunked(ids, 100):
    for r in get_all("molecule", {"molecule_chembl_id__in": ",".join(b),
                                  "only": "molecule_chembl_id,molecule_structures,molecule_type,max_phase"},
                     "molecules"):
        mol[r["molecule_chembl_id"]] = r
print("molecules fetched", len(mol), f"{time.time()-t0:.0f}s", flush=True)

df["smiles"] = df["molecule_chembl_id"].map(lambda m: ((mol.get(m) or {}).get("molecule_structures") or {}).get("canonical_smiles"))
df["molecule_type"] = df["molecule_chembl_id"].map(lambda m: (mol.get(m) or {}).get("molecule_type")).fillna("not returned")
no_smi = df[df["smiles"].isna()]
no_smi_types = {"rows": no_smi["molecule_type"].value_counts().to_dict(),
                "molecules": no_smi.drop_duplicates("molecule_chembl_id")["molecule_type"].value_counts().to_dict()}
step("molecule has a SMILES structure", df[df["smiles"].notna()])
kept_types = df.drop_duplicates("molecule_chembl_id")["molecule_type"].value_counts().to_dict()
step("mechanism text present", df[df["mechanism_of_action"].notna()])
step("mechanism text longer than 15 characters", df[df["mechanism_of_action"].str.len() > 15])
step("unique (molecule, target) pair", df.drop_duplicates(subset=["molecule_chembl_id", "target_chembl_id"], keep="first"))

df["max_phase"] = df["molecule_chembl_id"].map(lambda m: (mol.get(m) or {}).get("max_phase")).fillna(df["max_phase"])
phase = df["max_phase"].map(lambda v: "unknown" if pd.isna(v) else str(int(float(v)))).value_counts().to_dict()

old = pd.read_csv(COMMITTED)
key = lambda d: set(zip(d["molecule_chembl_id"], d["target_chembl_id"].astype(str)))  # noqa: E731
new_k, old_k = key(df), key(old)
check = {"committed_rows": len(old), "rerun_rows": len(df), "pairs_in_both": len(new_k & old_k),
         "only_in_rerun": len(new_k - old_k), "only_in_committed": len(old_k - new_k)}

out = {"chembl_release": status.get("chembl_db_version"), "steps": [{"step": s, "rows_left": n, "removed": r} for s, n, r in steps],
       "no_smiles_by_molecule_type": no_smi_types, "kept_molecule_types": kept_types,
       "final_unique_molecules": int(df["molecule_chembl_id"].nunique()), "final_by_max_phase": phase,
       "check_vs_committed_csv": check}
RESULTS.mkdir(exist_ok=True)
(RESULTS / "e05b_data_funnel.json").write_text(json.dumps(out, indent=2))

L = [f"ChEMBL release: {out['chembl_release']}", "", "| Step | Rows left | Removed |", "|---|---|---|"]
L += [f"| {s} | {n:,} | {'-' if r is None else f'{r:,}'} |" for s, n, r in steps]
L += ["", "**Rows removed for lacking a SMILES, by molecule type** (rows / distinct molecules):", "",
      "| Molecule type | Rows | Molecules |", "|---|---|---|"]
L += [f"| {t} | {n:,} | {no_smi_types['molecules'].get(t, 0):,} |" for t, n in no_smi_types["rows"].items()]
L += ["", f"Molecule types kept: {kept_types}", f"Final: {len(df):,} pairs, {out['final_unique_molecules']:,} molecules; by max phase: {phase}",
      f"Check vs committed CSV: {check}"]
(RESULTS / "e05b_data_funnel.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
