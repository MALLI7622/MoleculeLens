"""E2: MoleculeSTM on the canonical scaffold test (R3 Q1: 'no scaffold row for MoleculeSTM').

(a) zero-shot: pretrained MoleculeSTM-SMILES (MegaMolBART + SciBERT + latent heads) embeds the 435 test
    texts and molecules; full-gallery metrics, rich and no-drug text.
(b) fair bridge: MoleculeSTM's frozen 256-d molecule and text latents for all 2,699 rows, then the
    SAME two linear heads + loss as MoleculeLens trained on the canonical train split (5 seeds).
    This isolates encoder quality from training budget.
Loader and embedding code are imported from 05_compare_moleculestm_vs_thinbridges.py.
"""
import importlib.util

import numpy as np
import pandas as pd
import torch

import core as C

spec = importlib.util.spec_from_file_location("m05", C.ROOT / "05_compare_moleculestm_vs_thinbridges.py")
m05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m05)

MSTM = "/home/cheriearjun/MoleculeSTM"
comps = m05.load_moleculestm(MSTM, f"{MSTM}/data/pretrained_MoleculeSTM/SMILES", C.DEVICE)
df = C.load_df()
canon = C.canonical_split()
te = np.array(canon["test"])
smi = df["smiles"].tolist()

out = {}
emb = {}
for v in ["rich", "nodrug", "mech"]:
    texts = C.text_variant(df, v)
    path = C.CACHE / f"mstm_{v}.pt"
    if path.exists():
        M, T = torch.load(path)
    else:
        M, T = m05.moleculestm_embeddings(smi, texts, comps, C.DEVICE, batch_size=32, max_seq_len=512)
        torch.save((M, T), path)
    emb[v] = (M.numpy(), T.numpy())
    S = T.numpy()[te] @ M.numpy()[te].T
    m = C.strip(C.metrics(S)); m["R@1_struct"] = C.structure_r1(S, df["smiles"].values[te])
    out[f"zero-shot {v}"] = m
    print("zero-shot", v, m)

for v in ["rich", "nodrug", "mech"]:
    M, T = emb[v]
    d = pd.DataFrame([C.run_split(df, canon, v, seed=s, Zall=T, Xall=M)["metrics"] for s in range(5)])
    out[f"MoleculeSTM encoders + linear bridge {v}"] = {**{k: float(d[k].mean()) for k in d},
                                                         "R@1_std": float(d["R@1"].std(ddof=1))}
    print("bridge", v, out[f"MoleculeSTM encoders + linear bridge {v}"])

C.save_json(out, C.RESULTS / "e02_moleculestm_scaffold.json")
L = ["| MoleculeSTM variant (scaffold test, N=435) | R@1 | R@1 struct | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|"]
for k, m in out.items():
    L.append(f"| {k} | {m['R@1']:.3f} | {m['R@1_struct']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['MRR']:.3f} |")
open(C.RESULTS / "e02_moleculestm_scaffold.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
