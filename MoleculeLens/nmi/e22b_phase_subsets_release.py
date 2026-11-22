"""E22b: which development phases to train on, the equal-data MoleculeLens run, and release weights.

All runs use the E21 final config (parent ECFP4 8192 bits, mean-pooled text, dim 1024). The training pool
is the E21 all-phase pool (scaffold-disjoint from the held-out rows, test molecules excluded), restricted to:
  approved only (phase 4) | phase >= 3 | all phases (= E21 final) | all phases, <= 150 atoms (= MolPrompt E22 train)
plus the paper's canonical approved train for reference.
  * canonical scaffold test, 5 seeds, no-drug and rich text;
  * 10 scaffold partitions (seed 0) for the three phase subsets.
Release: final heads trained on the canonical split (reproduce the reported numbers) and on all 5,035 pairs,
saved with their config to MoleculeLens/nmi_models/ (git-ignored; mirrored to GCS).
"""
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy import stats

import core as C

RDLogger.DisableLog("rdApp.*")
MAX_ATOMS = 150
FINAL = dict(nbits=8192, parent=True, dim=1024)

dfc = C.load_df()
dfa = C.load_df(C.ROOT / "chembl_mechanisms_allphases.csv")
for d in (dfc, dfa):
    d["scaffold"] = d["smiles"].map(C.murcko).fillna("")
dfa["phase"] = pd.to_numeric(dfa["max_phase"], errors="coerce").fillna(-1)
dfa["n_atoms"] = dfa["smiles"].map(lambda s: Chem.MolFromSmiles(s).GetNumAtoms())
canon = C.canonical_split()

POOLS = {"approved only (phase 4)": dfa.phase == 4,
         "phase >= 3": dfa.phase >= 3,
         "all phases (E21 final)": dfa.phase > -np.inf,
         f"all phases, <= {MAX_ATOMS} atoms (MolPrompt E22 train)": dfa.n_atoms <= MAX_ATOMS}


def build(split_c, pool):
    """Training rows (canonical train if pool is None, else the filtered all-phase pool) + test rows of dfc."""
    te = dfc.iloc[split_c["test"]]
    if pool is None:
        tr = dfc.iloc[split_c["train"]]
    else:
        ho = list(split_c["test"]) + list(split_c.get("val", []))
        tr = dfa[pool & ~dfa.molecule_chembl_id.isin(set(te.molecule_chembl_id)) & ~dfa.scaffold.isin(set(dfc.scaffold.values[ho]))]
    comb = pd.concat([tr, te], ignore_index=True)
    return comb, {"train": list(range(len(tr))), "test": list(range(len(tr), len(comb)))}


def features(comb, text):
    Z = C.encode_texts_meanpool(C.text_variant(comb, text)).numpy()
    X = C.fingerprint_matrix(comb.smiles.tolist(), nbits=FINAL["nbits"], parent=FINAL["parent"])
    return Z, X


def run(split_c, pool, text, seed):
    comb, sp = build(split_c, pool)
    Z, X = features(comb, text)
    r = C.run_split(comb, sp, text, seed=seed, Zall=Z, Xall=X, dim=FINAL["dim"], return_model=True)
    par = comb.smiles.map(C.parent_smiles).values[np.array(sp["test"])]
    r["metrics"]["R@1_struct_parent"] = C.structure_r1(r["S"], par)
    return r


out = {"canonical_5seeds": {}, "scaffold_10_partitions": {}}
for nm, pool in [("paper approved train (2,019)", None)] + list(POOLS.items()):
    out["canonical_5seeds"][nm] = {}
    for text in ["nodrug", "rich"]:
        rs = [run(canon, pool, text, s) for s in range(5)]
        d = pd.DataFrame([r["metrics"] for r in rs])
        out["canonical_5seeds"][nm][text] = {"n_train": rs[0]["n_train"],
                                             **{k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in ["R@1", "R@1_struct_parent", "MRR"]}}
    print(nm, {t: round(out["canonical_5seeds"][nm][t]["R@1_struct_parent"][0], 3) for t in ["nodrug", "rich"]}, flush=True)

for nm in list(POOLS)[:3]:
    out["scaffold_10_partitions"][nm] = {}
    for text in ["nodrug", "rich"]:
        d = pd.DataFrame([run(C.scaffold_split(dfc, p), POOLS[nm], text, 0)["metrics"] for p in range(10)])
        out["scaffold_10_partitions"][nm][text] = {k: {"mean": float(d[k].mean()), "ci95_t": [float(x) for x in stats.t.interval(
            0.95, len(d) - 1, loc=d[k].mean(), scale=stats.sem(d[k]))]} for k in ["R@1_struct_parent", "MRR"]}
    print(nm, "10 partitions done", flush=True)

# ---------------- release weights ----------------
MODELS = C.ROOT / "nmi_models"
MODELS.mkdir(exist_ok=True)
cfg = {**FINAL, "radius": 2, "text_model": C.TEXT_MODEL, "text_pooling": "mean (masked, final layer)",
       "max_text_len": C.MAX_TXT_LEN, "train_hp": C.HP, "fingerprint": "Morgan bit vector of the salt-stripped parent (core.parent_smiles)"}
released = {}
for text in ["nodrug", "rich"]:
    r = run(canon, POOLS["all phases (E21 final)"], text, 0)
    pt, pm = r["model"]
    f = MODELS / f"moleculelens_final_{text}_canonical.pt"
    torch.save({"proj_text": pt.state_dict(), "proj_mol": pm.state_dict(), "config": cfg, "text_variant": text,
                "trained_on": "E21 all-phase pool, canonical scaffold split (seed 0)", "canonical_test_metrics": r["metrics"]}, f)
    groups = dfa["target_chembl_id"].fillna("UNK").astype(str).values
    Z, X = features(dfa, text)
    pt, pm = C.train_bridge(Z, X, groups, seed=0, dim=FINAL["dim"])
    g = MODELS / f"moleculelens_final_{text}_alldata.pt"
    torch.save({"proj_text": pt.state_dict(), "proj_mol": pm.state_dict(), "config": cfg, "text_variant": text,
                "trained_on": f"all {len(dfa)} pairs of chembl_mechanisms_allphases.csv (no held-out set)"}, g)
    released[text] = {"canonical": f.name, "alldata": g.name, "canonical_test_metrics": r["metrics"]}
out["released_models"] = released
C.save_json(out, C.RESULTS / "e22b_phase_subsets_release.json")

L = ["**Which phases to train on: canonical scaffold test (435), final config, 5 seeds; R@1 on the structure-deduplicated gallery:**", "",
     "| Training set | n_train | no-drug R@1 struct | no-drug MRR | rich R@1 struct | rich MRR |", "|---|---|---|---|---|---|"]
for nm, v in out["canonical_5seeds"].items():
    a, b = v["nodrug"], v["rich"]
    L.append(f"| {nm} | {a['n_train']:,} | {a['R@1_struct_parent'][0]:.3f} ± {a['R@1_struct_parent'][1]:.3f} | {a['MRR'][0]:.3f} | "
             f"{b['R@1_struct_parent'][0]:.3f} ± {b['R@1_struct_parent'][1]:.3f} | {b['MRR'][0]:.3f} |")
L += ["", "**10 scaffold partitions (seed 0), R@1 struct [95% t-CI]:**", "", "| Training set | no-drug | rich |", "|---|---|---|"]
for nm, v in out["scaffold_10_partitions"].items():
    c = lambda t: f"{v[t]['R@1_struct_parent']['mean']:.3f} [{v[t]['R@1_struct_parent']['ci95_t'][0]:.3f}, {v[t]['R@1_struct_parent']['ci95_t'][1]:.3f}]"  # noqa: E731
    L.append(f"| {nm} | {c('nodrug')} | {c('rich')} |")
L += ["", "Released weights (MoleculeLens/nmi_models/): " + ", ".join(f"{v['canonical']}, {v['alldata']}" for v in released.values())]
open(C.RESULTS / "e22b_phase_subsets_release.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
