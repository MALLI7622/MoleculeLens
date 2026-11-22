"""E22c: MoleculeLens vs MolPrompt at equal training data, no-drug text, canonical scaffold test.

Two data regimes, each model trained on exactly the same rows:
  paper data : canonical approved train (2,019)      MolPrompt = retrieval_chembl_sharedsplit_bs6_nodrug
  all-phase  : E21 pool, <= 150 atoms (e22a)          MolPrompt = retrieval_chembl_allphase_bs6_nodrug
MoleculeLens uses the E21 final config (seed 0 for the paired test; 5-seed means are in e22b).
Scored on the structure-deduplicated (unique parent) gallery, paired parent-cluster bootstrap, as in E21.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

import core as C

RDLogger.DisableLog("rdApp.*")
MP = "/home/cheriearjun/MolPrompt/save_model"
dfc = C.load_df()
dfa = C.load_df(C.ROOT / "chembl_mechanisms_allphases.csv")
for d in (dfc, dfa):
    d["scaffold"] = d["smiles"].map(C.murcko).fillna("")
canon = C.canonical_split()
te = dfc.iloc[canon["test"]]
held = set(dfc.scaffold.values[list(canon["test"]) + list(canon["val"])])
pool = dfa[~dfa.scaffold.isin(held) & ~dfa.molecule_chembl_id.isin(set(te.molecule_chembl_id))]
pool = pool[pool.smiles.map(lambda s: Chem.MolFromSmiles(s).GetNumAtoms()) <= 150]
TRAINS = {"paper data": dfc.iloc[canon["train"]], "all-phase, <= 150 atoms": pool}
NPZ = {"paper data": f"{MP}/retrieval_chembl_sharedsplit_bs6_nodrug/chembl_retrieval_best_metrics.npz",
       "all-phase, <= 150 atoms": f"{MP}/retrieval_chembl_allphase_bs6_nodrug/chembl_retrieval_best_metrics.npz"}
par = te.smiles.map(C.parent_smiles).values


def struct_hits(S, groups):
    g = C.group_ids(groups); G = g.max() + 1
    rep = np.array([np.where(g == k)[0][0] for k in range(G)])
    Sg = S[:, rep]; own = Sg[np.arange(len(g)), g]
    gt = np.sum(Sg > own[:, None] + 1e-7, 1); eq = np.sum(np.abs(Sg - own[:, None]) <= 1e-7, 1)
    return (gt == 0) / eq


cl = C.group_ids(par); members = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
out = {}
for nm, tr in TRAINS.items():
    comb = pd.concat([tr, te], ignore_index=True)
    sp = {"train": list(range(len(tr))), "test": list(range(len(tr), len(comb)))}
    Z = C.encode_texts_meanpool(C.text_variant(comb, "nodrug")).numpy()
    X = C.fingerprint_matrix(comb.smiles.tolist(), nbits=8192, parent=True)
    S_ml = C.run_split(comb, sp, "nodrug", seed=0, Zall=Z, Xall=X, dim=1024, return_model=True)["S"]
    npz = np.load(NPZ[nm])
    S_mp = (F.normalize(torch.tensor(npz["test_text"]), dim=1) @ F.normalize(torch.tensor(npz["test_graph"]), dim=1).T).numpy()
    h_ml, h_mp = struct_hits(S_ml, par), struct_hits(S_mp, par)
    rng = np.random.default_rng(0)
    diffs = np.array([np.mean(h_ml[i]) - np.mean(h_mp[i]) for i in
                      (np.concatenate([members[p] for p in rng.integers(0, len(members), len(members))]) for _ in range(10000))])
    out[nm] = {"n_train": len(tr), "MoleculeLens": float(h_ml.mean()), "MolPrompt": float(h_mp.mean()),
               "MolPrompt_own_scorer_R@1": float(npz["test_t2m_r1"]), "MolPrompt_best_epoch": int(npz["best_epoch"]),
               "diff": float(h_ml.mean() - h_mp.mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
               "p_le_0": float(np.mean(diffs <= 0))}
    print(nm, out[nm], flush=True)
C.save_json(out, C.RESULTS / "e22c_molprompt_allphase_compare.json")
L = ["**Equal training data, no-drug text, canonical test; R@1 on the structure-deduplicated gallery (paired parent-cluster bootstrap):**", "",
     "| Training data | n_train | MoleculeLens | MolPrompt | difference [95% CI] | P(diff <= 0) | MolPrompt best epoch |", "|---|---|---|---|---|---|---|"]
for nm, v in out.items():
    L.append(f"| {nm} | {v['n_train']:,} | {v['MoleculeLens']:.3f} | {v['MolPrompt']:.3f} | {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
             f"{v['p_le_0']:.4f} | {v['MolPrompt_best_epoch']} |")
open(C.RESULTS / "e22c_molprompt_allphase_compare.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
