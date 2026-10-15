"""E05a: download ChEMBL mechanism rows for ALL development phases (not only max_phase=4).

Same cleaning and text recipe as 01_download_data.py, but:
  * no max_phase filter (keeps clinical-phase / preclinical compounds with curated mechanisms);
  * names and SMILES fetched in batches, targets fetched once per unique target (faster);
  * keeps max_phase so subsets can be analysed.
Output: chembl_mechanisms_allphases.csv
"""
import sys
import time
from itertools import islice

import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data"


def get_all(resource, params, key):
    """Page through a ChEMBL REST resource."""
    out, offset = [], 0
    while True:
        p = dict(params, format="json", limit=1000, offset=offset)
        js = retry(lambda: requests.get(f"{API}/{resource}", params=p, timeout=120).json())
        if js is None:
            break
        out.extend(js[key])
        offset += 1000
        if not js["page_meta"]["next"]:
            break
    return out

OUT = sys.argv[1] if len(sys.argv) > 1 else "chembl_mechanisms_allphases.csv"


def chunked(it, n):
    it = iter(it)
    for first in it:
        yield [first, *list(islice(it, n - 1))]


def retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print("retry", k, e, flush=True)
            time.sleep(2 * (k + 1))
    return None


t0 = time.time()
rows = get_all("mechanism", {"molecular_mechanism": 1}, "mechanisms")
df = pd.DataFrame(rows)[["molecule_chembl_id", "mechanism_of_action", "target_chembl_id",
                         "action_type", "max_phase", "direct_interaction", "molecular_mechanism"]]
print("raw mechanism rows", len(df), f"{time.time()-t0:.0f}s", flush=True)

ids = df["molecule_chembl_id"].dropna().astype(str).unique().tolist()
smi, name, mphase = {}, {}, {}
for b in chunked(ids, 100):
    recs = get_all("molecule", {"molecule_chembl_id__in": ",".join(b),
                                "only": "molecule_chembl_id,molecule_structures,pref_name,max_phase"},
                   "molecules")
    for r in recs or []:
        mid = r.get("molecule_chembl_id")
        s = (r.get("molecule_structures") or {}).get("canonical_smiles")
        if s:
            smi[mid] = s
        name[mid] = r.get("pref_name")
        mphase[mid] = r.get("max_phase")
print("smiles fetched", len(smi), f"{time.time()-t0:.0f}s", flush=True)

tids = df["target_chembl_id"].dropna().astype(str).unique().tolist()
tname = {}
for b in chunked(tids, 100):
    recs = get_all("target", {"target_chembl_id__in": ",".join(b),
                              "only": "target_chembl_id,pref_name,target_type"}, "targets")
    for r in recs or []:
        tname[r["target_chembl_id"]] = r.get("pref_name") or r.get("target_type")
print("targets fetched", len(tname), f"{time.time()-t0:.0f}s", flush=True)

df["smiles"] = df["molecule_chembl_id"].map(smi)
df["pref_name"] = df["molecule_chembl_id"].map(name)
df["max_phase"] = df["molecule_chembl_id"].map(mphase).fillna(df.get("max_phase"))
df["target_name"] = df["target_chembl_id"].map(tname)

df = df.dropna(subset=["smiles", "mechanism_of_action"])
df = df[df["mechanism_of_action"].str.len() > 15]
df = df.drop_duplicates(subset=["molecule_chembl_id", "target_chembl_id"], keep="first").reset_index(drop=True)


def build_text_rich(row):
    parts = [row["mechanism_of_action"]]
    if isinstance(row.get("target_name"), str):
        parts.append(f"Target: {row['target_name']}.")
    if isinstance(row.get("action_type"), str):
        parts.append(f"Action: {row['action_type']}.")
    if isinstance(row.get("pref_name"), str):
        parts.append(f"Drug: {row['pref_name']}.")
    return " ".join(parts)


df["text_rich"] = df.apply(build_text_rich, axis=1)
cols = ["molecule_chembl_id", "smiles", "mechanism_of_action", "target_chembl_id", "target_name",
        "action_type", "max_phase", "pref_name", "text_rich"]
df[cols].to_csv(OUT, index=False)
print("saved", len(df), "rows ->", OUT, f"{time.time()-t0:.0f}s")
print(df["max_phase"].value_counts(dropna=False).to_string())
