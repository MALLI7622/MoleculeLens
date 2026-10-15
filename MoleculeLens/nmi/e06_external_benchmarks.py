"""E6: generalisation to established molecule-text benchmarks (R3 W1/Q3).

ChEBI-20 (Text2Mol / MolT5 split: 26,407 / 3,301 / 3,300) and PubChem324kV2 (MolCA split:
12,000 / 1,000 / 2,000).  Names are already replaced by 'The molecule is ...' in both.
MoleculeLens trained exactly as in the paper (ECFP4-2048 + frozen S-Biomed-RoBERTa, two linear heads,
InfoNCE; no target labels exist here, so the same-target terms vanish), text max length 256 because
descriptions are long.  Variants: CLS (paper) and mean-pooled final layer (see E14).
Evaluation: text -> molecule over the full test gallery (R@1, R@10, R@20, MRR); for ChEBI-20 also
over the whole 33,010-molecule dataset (Text2Mol protocol ranks against all molecules).
Both benchmarks use random splits -> scaffold overlap reported, and metrics on the scaffold-novel
test subset (test molecules whose Murcko scaffold never occurs in training).
"""
import json

import numpy as np
import pandas as pd

import core as C

C.MAX_TXT_LEN = 256
EXT = "/home/cheriearjun/ext_data"


def load_chebi(split):
    d = pd.read_csv(f"{EXT}/chebi20_{split}.txt", sep="\t", quoting=3)
    return d.rename(columns={"SMILES": "smiles", "description": "text"})[["smiles", "text"]]


def load_pubchem(split):
    js = json.load(open(f"{EXT}/pubchem324k_{split}.json"))
    return pd.DataFrame({"smiles": js["smiles"], "text": [t.strip() for t in js["text"]]})


def evaluate(S, sm_te, novel_mask):
    ranks = 1 + np.sum(S > np.diag(S)[:, None], axis=1) if S.shape[0] == S.shape[1] else None
    m = {"R@1": float(np.mean(ranks <= 1)), "R@10": float(np.mean(ranks <= 10)), "R@20": float(np.mean(ranks <= 20)),
         "MRR": float(np.mean(1 / ranks)), "R@1_struct": C.structure_r1(S, sm_te),
         "R@1_scaffold_novel_subset": float(np.mean(ranks[novel_mask] <= 1)),
         "MRR_scaffold_novel_subset": float(np.mean(1 / ranks[novel_mask]))}
    return m


out = {}
for bench, loader, splits in [("ChEBI-20", load_chebi, ("train", "validation", "test")),
                              ("PubChem324kV2", load_pubchem, ("train", "valid", "test"))]:
    tr, va, te = [loader(s).dropna().reset_index(drop=True) for s in splits]
    allm = pd.concat([tr, va, te], ignore_index=True)
    sc_tr = set(tr.smiles.map(C.murcko).fillna(""))
    sc_te = te.smiles.map(C.murcko).fillna("")
    novel = ~sc_te.isin(sc_tr).values
    info = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "test_scaffold_seen_in_train": float(1 - novel.mean()), "n_scaffold_novel_test": int(novel.sum())}
    Xtr = C.fingerprint_matrix(tr.smiles.tolist()); Xte = C.fingerprint_matrix(te.smiles.tolist())
    res = {"info": info}
    for pool in ["cls", "mean"]:
        enc = C.encode_texts if pool == "cls" else C.encode_texts_meanpool
        Ztr = enc(tr.text.tolist()).numpy(); Zte = enc(te.text.tolist()).numpy()
        rows = []
        for s in range(3):
            pt, pm = C.train_bridge(Ztr, Xtr, np.arange(len(tr)), seed=s)
            S = C.similarity(pt, pm, Zte, Xte)
            m = evaluate(S, te.smiles.values, novel)
            if bench == "ChEBI-20" and s == 0:
                Xall = C.fingerprint_matrix(allm.smiles.tolist())
                Sall = C.similarity(pt, pm, Zte, Xall)
                own = Sall[np.arange(len(te)), len(tr) + len(va) + np.arange(len(te))]
                r = 1 + np.sum(Sall > own[:, None], axis=1)
                m["full_dataset_gallery"] = {"R@1": float(np.mean(r <= 1)), "R@10": float(np.mean(r <= 10)),
                                             "MRR": float(np.mean(1 / r)), "gallery": int(Sall.shape[1])}
            rows.append(m)
            print(bench, pool, s, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
        d = pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, float)} for r in rows])
        res[pool] = {"mean": {k: float(d[k].mean()) for k in d}, "std": {k: float(d[k].std(ddof=1)) for k in d}}
        if bench == "ChEBI-20":
            res[pool]["full_dataset_gallery_seed0"] = rows[0]["full_dataset_gallery"]
    out[bench] = res

C.save_json(out, C.RESULTS / "e06_external_benchmarks.json")
L = []
for bench, res in out.items():
    i = res["info"]
    L += [f"**{bench}** (train {i['n_train']} / test {i['n_test']}; {100*i['test_scaffold_seen_in_train']:.1f}% of test molecules "
          f"have a Murcko scaffold seen in training; {i['n_scaffold_novel_test']} scaffold-novel test molecules):", "",
          "| MoleculeLens text pooling (3 seeds) | R@1 | R@10 | R@20 | MRR | R@1 struct | R@1 scaffold-novel subset |", "|---|---|---|---|---|---|---|"]
    for pool in ["cls", "mean"]:
        m, s = res[pool]["mean"], res[pool]["std"]
        L.append(f"| {pool} | {m['R@1']:.3f} ± {s['R@1']:.3f} | {m['R@10']:.3f} | {m['R@20']:.3f} | {m['MRR']:.3f} | "
                 f"{m['R@1_struct']:.3f} | {m['R@1_scaffold_novel_subset']:.3f} |")
    if bench == "ChEBI-20":
        for pool in ["cls", "mean"]:
            f = res[pool]["full_dataset_gallery_seed0"]
            L.append(f"\nChEBI-20 {pool}, ranked against all {f['gallery']} molecules (Text2Mol protocol): "
                     f"Hits@1 {f['R@1']:.3f}, Hits@10 {f['R@10']:.3f}, MRR {f['MRR']:.3f}")
    L.append("")
open(C.RESULTS / "e06_external_benchmarks.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
