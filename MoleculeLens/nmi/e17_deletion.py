"""E17: does removing the most salient substructures hurt retrieval more than removing random ones?

For each of the 435 test pairs (paper checkpoint), zero the top-k salient PRESENT bits of the
correct molecule (k = 1, 3, 5, 10) and measure (a) the drop in the exact normalised score s_ii and
(b) the share of rank-1 hits that are lost.  Controls: k random present bits (20 draws) and the
k LEAST salient present bits.  Also compares ranking by the fast proxy with ranking by the
exact leave-one-bit-out drop.  Plus an atom-level variant: delete the atoms of the top-1
environment (RDKit), recompute the fingerprint, re-score.
"""
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

import core as C

df = C.load_df()
te = np.array(C.canonical_split()["test"])
dft = df.iloc[te].reset_index(drop=True)
X = C.fingerprint_matrix(df["smiles"].tolist())[te]
pt = torch.nn.Linear(768, 256); pm = torch.nn.Linear(2048, 256)
pt.load_state_dict(torch.load(C.ROOT / "outputs/proj_text.pt", map_location="cpu"))
pm.load_state_dict(torch.load(C.ROOT / "outputs/proj_mol.pt", map_location="cpu"))
pt.eval(); pm.eval()
Zt = torch.load(C.ROOT / "outputs/Z_text_test.pt").numpy()

with torch.no_grad():
    BT = F.normalize(pt(torch.tensor(Zt)), dim=1)
    BM = F.normalize(pm(torch.tensor(X)), dim=1)
S0 = (BT @ BM.T).numpy()


@torch.no_grad()
def score_and_rank(i, x_new):
    bm = F.normalize(pm(torch.tensor(x_new, dtype=torch.float32)[None]), dim=1)[0]
    s = float(BT[i] @ bm)
    row = S0[i].copy(); row[i] = s
    return s, int(1 + np.sum(row > s))


rng = np.random.default_rng(0)
K = [1, 3, 5, 10]
res = {k: {"top": [], "rand": [], "least": [], "exact_top": [], "hit_top": [], "hit_rand": [], "hit_least": []} for k in K}
base_hit = np.array([1 + np.sum(S0[i] > S0[i, i]) == 1 for i in range(len(te))])
for i in range(len(te)):
    a = C.saliency(pt, pm, Zt[i], X[i])
    present = np.where(X[i] > 0)[0]
    by_attr = present[np.argsort(-a[present])]             # most positive first
    # exact leave-one-bit-out drops
    exact = []
    for b in present:
        x2 = X[i].copy(); x2[b] = 0
        exact.append(S0[i, i] - score_and_rank(i, x2)[0])
    by_exact = present[np.argsort(-np.array(exact))]
    for k in K:
        if len(present) <= k:
            continue
        for name, bits in [("top", by_attr[:k]), ("least", by_attr[-k:]), ("exact_top", by_exact[:k])]:
            x2 = X[i].copy(); x2[bits] = 0
            s, r = score_and_rank(i, x2)
            res[k][name].append(S0[i, i] - s)
            if name in ("top", "least"):
                res[k][f"hit_{name}"].append(r == 1)
        d, h = [], []
        for _ in range(20):
            x2 = X[i].copy(); x2[rng.choice(present, k, replace=False)] = 0
            s, r = score_and_rank(i, x2); d.append(S0[i, i] - s); h.append(r == 1)
        res[k]["rand"].append(np.mean(d)); res[k]["hit_rand"].append(np.mean(h))

summ = {}
for k in K:
    hits = base_hit[:len(res[k]["hit_top"])] if len(res[k]["hit_top"]) == len(te) else None
    summ[k] = {"mean_score_drop_top": float(np.mean(res[k]["top"])),
               "mean_score_drop_exact_top": float(np.mean(res[k]["exact_top"])),
               "mean_score_drop_random": float(np.mean(res[k]["rand"])),
               "mean_score_drop_least": float(np.mean(res[k]["least"])),
               "top_gt_random_share": float(np.mean(np.array(res[k]["top"]) > np.array(res[k]["rand"]))),
               "rank1_retained_top": float(np.mean(res[k]["hit_top"])),
               "rank1_retained_random": float(np.mean(res[k]["hit_rand"])),
               "rank1_retained_least": float(np.mean(res[k]["hit_least"])),
               "n_pairs": len(res[k]["top"])}
    print(k, summ[k])

# atom-level deletion of the top-1 environment, for pairs retrieved at rank 1
atom = {"n": 0, "lost_rank1": 0, "mean_drop": []}
for i in np.where(base_hit)[0]:
    smi = dft.smiles[i]
    m = Chem.MolFromSmiles(smi)
    a = C.saliency(pt, pm, Zt[i], X[i])
    info = C.bit_environments(smi)
    b = int(np.argmax(a))
    if b not in info:
        continue
    atoms = C.env_atoms(m, *info[b][0])
    if len(atoms) >= m.GetNumAtoms() - 1:
        continue
    rw = Chem.RWMol(m)
    for ai in sorted(atoms, reverse=True):
        rw.RemoveAtom(ai)
    try:
        m2 = rw.GetMol()
        for at in m2.GetAtoms():          # atoms that lost a neighbour keep valence via implicit H
            at.SetNoImplicit(False); at.SetNumExplicitHs(0)
        Chem.SanitizeMol(m2)
        smi2 = Chem.MolToSmiles(m2)
        x2 = C.fingerprint_matrix([smi2])[0] if Chem.MolFromSmiles(smi2) else None
    except Exception:  # noqa: BLE001
        x2 = None
    if x2 is None or x2.sum() == 0:
        continue
    s, r = score_and_rank(i, x2)
    atom["n"] += 1; atom["lost_rank1"] += int(r > 1); atom["mean_drop"].append(S0[i, i] - s)
atom["mean_drop"] = float(np.mean(atom["mean_drop"])) if atom["mean_drop"] else None
print("atom-level", atom)

C.save_json({"bit_deletion": summ, "atom_deletion_top1_env": atom, "baseline_rank1_hits": int(base_hit.sum())},
            C.RESULTS / "e17_deletion.json")
L = ["| k bits zeroed | score drop: top-k salient (proxy) | top-k (exact) | random k | least-salient k | top>random (pairs) | rank-1 kept: top / random / least |",
     "|---|---|---|---|---|---|---|"]
for k in K:
    s = summ[k]
    L.append(f"| {k} | {s['mean_score_drop_top']:.4f} | {s['mean_score_drop_exact_top']:.4f} | {s['mean_score_drop_random']:.4f} | "
             f"{s['mean_score_drop_least']:.4f} | {100*s['top_gt_random_share']:.1f}% | "
             f"{100*s['rank1_retained_top']:.1f}% / {100*s['rank1_retained_random']:.1f}% / {100*s['rank1_retained_least']:.1f}% |")
L += ["", f"Atom-level: deleting the atoms of the top-1 salient environment in {atom['n']} correctly retrieved molecules "
      f"loses the rank-1 hit in {atom['lost_rank1']} ({100*atom['lost_rank1']/max(atom['n'],1):.1f}%); mean score drop "
      f"{atom['mean_drop']:.4f}." if atom["n"] else "Atom-level deletion: no valid cases."]
open(C.RESULTS / "e17_deletion.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
