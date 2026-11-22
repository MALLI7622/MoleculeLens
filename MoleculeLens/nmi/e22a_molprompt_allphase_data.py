"""E22a: build the all-phase training set for MolPrompt (fair comparison with the E21 final model).

Same training pool as E21 config F: all-phase ChEMBL rows whose scaffold is not in the canonical val/test
scaffolds and whose molecule is not a canonical test molecule, minus training molecules over 150 atoms
(MolPrompt runs out of memory on them; the first attempt on 2026-10-19 crashed with CUDA OOM). Validation and test are the canonical
val (245) and test (435) rows, in manifest order, so MolPrompt's stored test embeddings line up with
C.canonical_split()["test"].
Writes chembl_allphase_molprompt.csv and splits/chembl_allphase_molprompt.json; then
prepare_chembl.py featurises it into ~/MolPrompt/data_allphase.
"""
import json

import pandas as pd
from rdkit import Chem

import core as C

MAX_ATOMS = 150

dfc = C.load_df()
dfa = C.load_df(C.ROOT / "chembl_mechanisms_allphases.csv")
dfa["scaffold"] = dfa["smiles"].map(C.murcko).fillna("")
dfc["scaffold"] = dfc["smiles"].map(C.murcko).fillna("")
man = json.load(open(C.CANON_MANIFEST))["splits"]
val, test = [int(i) for i in man["val"]], [int(i) for i in man["test"]]
assert test == [int(i) for i in C.canonical_split()["test"]]

held = set(dfc.scaffold.values[val + test])
pool = dfa[~dfa.scaffold.isin(held) & ~dfa.molecule_chembl_id.isin(set(dfc.molecule_chembl_id.values[test]))]
# Graphormer memory grows with atoms^2: training molecules over MAX_ATOMS exhaust the T4 (16 GB) at batch 6.
# The same cap is applied to the MoleculeLens equal-data run (e22b); val/test are untouched.
n_atoms = pool.smiles.map(lambda s: Chem.MolFromSmiles(s).GetNumAtoms())
print("training rows dropped by the atom cap:", int((n_atoms > MAX_ATOMS).sum()))
pool = pool[n_atoms <= MAX_ATOMS]
cols = ["molecule_chembl_id", "smiles", "mechanism_of_action", "target_chembl_id", "target_name",
        "action_type", "max_phase", "pref_name", "text_rich"]
comb = pd.concat([pool[cols], dfc.iloc[val][cols], dfc.iloc[test][cols]], ignore_index=True)
n_tr, n_va = len(pool), len(val)
splits = {"train": list(range(n_tr)), "val": list(range(n_tr, n_tr + n_va)), "test": list(range(n_tr + n_va, len(comb)))}
comb.to_csv(C.ROOT / "chembl_allphase_molprompt.csv", index=False)
json.dump({"csv": str(C.ROOT / "chembl_allphase_molprompt.csv"), "seed": 0,
           "split_method": f"E21 config F: all-phase pool (<= {MAX_ATOMS} atoms) + canonical val/test",
           "split_row_counts": {k: len(v) for k, v in splits.items()}, "splits": splits},
          open(C.ROOT / "splits" / "chembl_allphase_molprompt.json", "w"), indent=1)
print({k: len(v) for k, v in splits.items()})
