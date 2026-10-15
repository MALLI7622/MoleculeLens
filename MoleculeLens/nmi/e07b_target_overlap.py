"""E7b: is the target-split rich-text score inflated by the same drug appearing in training under
another target?  Paper config, target-disjoint partitions 0-4: (a) as in E7; (b) training rows whose
molecule_chembl_id occurs in the test set removed."""
import numpy as np
import pandas as pd

import core as C

df = C.load_df()
X = C.fingerprint_matrix(df["smiles"].tolist())
Z = {v: C.encode_texts(C.text_variant(df, v)).numpy() for v in ["rich", "nodrug"]}
rows = []
for p in range(5):
    sp = C.target_split(df, p)
    te_m = set(df.molecule_chembl_id.values[sp["test"]])
    ov = float(np.mean([m in set(df.molecule_chembl_id.values[sp["train"]]) for m in df.molecule_chembl_id.values[sp["test"]]]))
    sp2 = dict(sp, train=[i for i in sp["train"] if df.molecule_chembl_id.values[i] not in te_m])
    for v in ["rich", "nodrug"]:
        a = C.run_split(df, sp, v, seed=0, Zall=Z[v], Xall=X)["metrics"]["R@1_struct"]
        b = C.run_split(df, sp2, v, seed=0, Zall=Z[v], Xall=X)["metrics"]["R@1_struct"]
        rows.append({"partition": p, "text": v, "test_rows_whose_molecule_is_in_train": ov,
                     "R@1_struct_as_E7": a, "R@1_struct_molecule_overlap_removed": b})
d = pd.DataFrame(rows)
g = d.groupby("text").mean()
C.save_json({"rows": rows}, C.RESULTS / "e07b_target_overlap.json")
L = [f"Target-disjoint split: on average {100*d.test_rows_whose_molecule_is_in_train.mean():.1f}% of test rows have their exact molecule "
     "in training (same drug, different target).", "",
     "| Text | R@1 struct as in E7 | R@1 struct, test molecules removed from training |", "|---|---|---|"]
for v in ["rich", "nodrug"]:
    L.append(f"| {v} | {g.loc[v, 'R@1_struct_as_E7']:.3f} | {g.loc[v, 'R@1_struct_molecule_overlap_removed']:.3f} |")
open(C.RESULTS / "e07b_target_overlap.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
