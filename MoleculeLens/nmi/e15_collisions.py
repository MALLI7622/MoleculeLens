"""E15: is a salient bit a unique substructure?  (R3 W2#3, Q8)

1. Collision analysis over all 2,699 molecules: unfolded Morgan identifiers (radius 2) folded to
   2048 bits; a bit 'collides' if it hosts >1 distinct unfolded identifier in the dataset.
   Reported for all active bits, for bits that appear in top-10 saliency on the test set, and the
   share of test top-10 salient (pair, bit) entries that are collided bits.
2. Training exposure: how many training molecules switch each salient bit on; share of test top-10
   salient entries on bits seen in < 5 training molecules.
3. Retrieval vs. fold size: 1024 / 2048 / 4096 / 8192 / 16384 bits (5 seeds).
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import core as C

df = C.load_df()
canon = C.canonical_split()
te, tr = np.array(canon["test"]), np.array(canon["train"])
X = C.fingerprint_matrix(df["smiles"].tolist())
Zr = C.encode_texts(C.text_variant(df, "rich")).numpy()

ids_per_bit = defaultdict(set)
for s in df.smiles:
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    for uid in AllChem.GetMorganFingerprint(m, 2).GetNonzeroElements():
        ids_per_bit[uid % 2048].add(uid)
active = [b for b in ids_per_bit]
collided = {b for b in active if len(ids_per_bit[b]) > 1}

pt = torch.nn.Linear(768, 256); pm = torch.nn.Linear(2048, 256)
pt.load_state_dict(torch.load(C.ROOT / "outputs/proj_text.pt", map_location="cpu"))
pm.load_state_dict(torch.load(C.ROOT / "outputs/proj_mol.pt", map_location="cpu"))
Zt = torch.load(C.ROOT / "outputs/Z_text_test.pt").numpy()
train_count = X[tr].sum(0)
entries, salient_bits = [], set()
for i in range(len(te)):
    a = C.saliency(pt.eval(), pm.eval(), Zt[i], X[te][i])
    order = [b for b in np.argsort(-np.abs(a)) if a[b] != 0][:10]
    salient_bits.update(order)
    entries += [(i, b) for b in order]
part = {
    "active_bits": len(active), "collided_active_bits": len(collided),
    "collision_rate_active": len(collided) / len(active),
    "mean_ids_per_active_bit": float(np.mean([len(ids_per_bit[b]) for b in active])),
    "salient_bits_distinct": len(salient_bits),
    "collision_rate_salient_bits": len(salient_bits & collided) / len(salient_bits),
    "share_top10_entries_on_collided_bits": float(np.mean([b in collided for _, b in entries])),
    "share_top10_entries_on_bits_seen_lt5_train": float(np.mean([train_count[b] < 5 for _, b in entries])),
    "share_top10_entries_on_bits_never_seen_train": float(np.mean([train_count[b] == 0 for _, b in entries])),
    "median_train_count_of_salient_entries": float(np.median([train_count[b] for _, b in entries])),
    "bits_never_on_in_train": int((train_count == 0).sum()),
}
print(part)

fold = {}
for nb in [1024, 2048, 4096, 8192, 16384]:
    XX = C.fingerprint_matrix(df["smiles"].tolist(), nbits=nb)
    rows = [C.run_split(df, canon, "rich", seed=s, Zall=Zr, Xall=XX)["metrics"] for s in range(5)]
    d = pd.DataFrame(rows)
    ids = defaultdict(set)
    for b_ids in ids_per_bit.values():
        for u in b_ids:
            ids[u % nb].add(u)
    fold[nb] = {"collision_rate": float(np.mean([len(v) > 1 for v in ids.values()])),
                **{k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in ["R@1", "R@1_struct", "R@5", "MRR"]}}
    print(nb, fold[nb])

C.save_json({"collisions_2048": part, "fold_size": fold}, C.RESULTS / "e15_collisions.json")
L = [f"2048-bit ECFP4 over the 2,699 molecules: {part['collided_active_bits']}/{part['active_bits']} active bits "
     f"({100*part['collision_rate_active']:.1f}%) host more than one distinct environment "
     f"(mean {part['mean_ids_per_active_bit']:.2f} environments per bit).",
     f"Bits appearing in a test top-10 saliency list: {part['salient_bits_distinct']} distinct, "
     f"{100*part['collision_rate_salient_bits']:.1f}% collided; {100*part['share_top10_entries_on_collided_bits']:.1f}% "
     f"of top-10 (pair, bit) entries fall on collided bits.",
     f"Training exposure: median {part['median_train_count_of_salient_entries']:.0f} training molecules per salient entry; "
     f"{100*part['share_top10_entries_on_bits_seen_lt5_train']:.1f}% of entries on bits seen in <5 training molecules, "
     f"{100*part['share_top10_entries_on_bits_never_seen_train']:.1f}% on bits never seen in training "
     f"({part['bits_never_on_in_train']} of 2048 bits never fire in training).",
     "", "| Fold size | collision rate | R@1 | R@1 struct | R@5 | MRR |", "|---|---|---|---|---|---|"]
for nb, v in fold.items():
    L.append(f"| {nb} | {100*v['collision_rate']:.1f}% | " + " | ".join(f"{v[k][0]:.3f} ± {v[k][1]:.3f}" for k in ["R@1", "R@1_struct", "R@5", "MRR"]) + " |")
open(C.RESULTS / "e15_collisions.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
