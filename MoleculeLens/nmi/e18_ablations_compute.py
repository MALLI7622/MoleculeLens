"""E18-E20: ECFP radius ablation, shared-space dimension ablation, compute report.

E18: radius 1/2/3/4 (ECFP2/4/6/8), 2048 bits, 5 seeds, rich + nodrug; fingerprint time.
E19: shared dimension 32/64/128/256/512/1024, 5 seeds, rich.
E20: wall-clock of every stage on this machine (cold text encoding of all 2,699 rows, fingerprints,
     training one model, evaluation, Eq.1 attribution for all 435 pairs, exact-gradient check for all
     435 pairs, all-layer encoding for the logit lens).
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import core as C

df = C.load_df()
canon = C.canonical_split()
te = np.array(canon["test"])
Z = {v: C.encode_texts(C.text_variant(df, v)).numpy() for v in ["rich", "nodrug"]}

e18 = {}
for r in [1, 2, 3, 4]:
    t0 = time.time()
    C.CACHE.joinpath("_tmp").mkdir(exist_ok=True)
    XX = C.fingerprint_matrix(df["smiles"].tolist(), radius=r)
    fp_t = time.time() - t0
    row = {"fingerprint_seconds_(cached_if_~0)": fp_t}
    for v in ["rich", "nodrug"]:
        d = pd.DataFrame([C.run_split(df, canon, v, seed=s, Zall=Z[v], Xall=XX)["metrics"] for s in range(5)])
        row[v] = {k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in ["R@1", "R@1_struct", "R@5", "R@10", "MRR"]}
    e18[f"ECFP{2*r}"] = row
    print("radius", r, row["rich"]["R@1"], row["nodrug"]["R@1"])

X = C.fingerprint_matrix(df["smiles"].tolist())
e19 = {}
for dim in [32, 64, 128, 256, 512, 1024]:
    d = pd.DataFrame([C.run_split(df, canon, "rich", seed=s, Zall=Z["rich"], Xall=X, dim=dim)["metrics"] for s in range(5)])
    e19[dim] = {k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in ["R@1", "R@1_struct", "R@5", "R@10", "MRR"]}
    print("dim", dim, e19[dim]["R@1"])

# E20 timings -----------------------------------------------------------------------
T = {}
texts = C.text_variant(df, "rich")
tok, enc = C._encoder()
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=C.MAX_TXT_LEN, return_tensors="pt").to(C.DEVICE)
        enc(**b).last_hidden_state[:, 0, :].cpu()
torch.cuda.synchronize()
T["text_encoding_2699_rows_final_layer"] = time.time() - t0
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=C.MAX_TXT_LEN, return_tensors="pt").to(C.DEVICE)
        enc(**b, output_hidden_states=True)
torch.cuda.synchronize()
T["text_encoding_2699_rows_all_13_layers"] = time.time() - t0
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
t0 = time.time()
for s in df.smiles:
    AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
T["ecfp4_2699_molecules"] = time.time() - t0
tr = np.array(canon["train"])
groups = df["target_chembl_id"].fillna("UNK").astype(str).values
t0 = time.time()
pt, pm = C.train_bridge(Z["rich"][tr], X[tr], groups[tr], seed=0)
T["train_heads_100_epochs"] = time.time() - t0
t0 = time.time()
S = C.similarity(pt, pm, Z["rich"][te], X[te]); C.metrics(S)
T["evaluate_435x435"] = time.time() - t0
t0 = time.time()
for i in range(len(te)):
    C.saliency(pt, pm, Z["rich"][te][i], X[te][i])
T["eq1_attribution_435_pairs"] = time.time() - t0
t0 = time.time()
for i in range(len(te)):
    x = torch.tensor(X[te][i], requires_grad=True)
    bt = torch.nn.functional.normalize(pt(torch.tensor(Z["rich"][te][i]))[None], dim=1)[0]
    s = torch.nn.functional.normalize(pm(x)[None], dim=1)[0] @ bt
    s.backward()
T["exact_gradient_435_pairs"] = time.time() - t0
T["device"] = torch.cuda.get_device_name(0)
T["gpu"] = "NVIDIA Tesla T4 16GB, GCP n1-standard-4 (4 vCPU, 15 GB RAM)"
print(T)

C.save_json({"E18_radius": e18, "E19_dim": e19, "E20_timing_seconds": T}, C.RESULTS / "e18_ablations_compute.json")
L = ["**E18 - ECFP radius (2048 bits, 5 seeds):**", "", "| Fingerprint | R@1 rich | R@1 struct rich | MRR rich | R@1 no-drug | MRR no-drug |", "|---|---|---|---|---|---|"]
for k, v in e18.items():
    L.append(f"| {k} | {v['rich']['R@1'][0]:.3f} ± {v['rich']['R@1'][1]:.3f} | {v['rich']['R@1_struct'][0]:.3f} | {v['rich']['MRR'][0]:.3f} | "
             f"{v['nodrug']['R@1'][0]:.3f} ± {v['nodrug']['R@1'][1]:.3f} | {v['nodrug']['MRR'][0]:.3f} |")
L += ["", "**E19 - shared-space dimension (rich, 5 seeds):**", "", "| dim | R@1 | R@1 struct | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|"]
for k, v in e19.items():
    L.append(f"| {k} | " + " | ".join(f"{v[m][0]:.3f} ± {v[m][1]:.3f}" for m in ["R@1", "R@1_struct", "R@5", "R@10", "MRR"]) + " |")
L += ["", "**E20 - wall-clock on " + T["gpu"] + ":**", "", "| Stage | seconds |", "|---|---|"]
L += [f"| {k} | {v:.2f} |" for k, v in T.items() if isinstance(v, float)]
open(C.RESULTS / "e18_ablations_compute.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
