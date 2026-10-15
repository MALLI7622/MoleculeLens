"""E7: text-side out-of-distribution splits (R1).

  scaffold : paper recipe, partitions 0-4 (reference)
  target   : target-disjoint, 80/10/10 by unique target_chembl_id, partitions 0-4
  joint    : scaffold- AND target-disjoint: 30% of scaffolds and 30% of targets held out;
             test = rows with held-out scaffold AND held-out target; rows touching only one
             held-out set are discarded; seeds 0-4
Each with rich / nodrug / mech text, init seed 0.  Reports R@1, R@1 struct, MRR, n_test.
"""
import numpy as np
import pandas as pd

import core as C

df = C.load_df()
X = C.fingerprint_matrix(df["smiles"].tolist())
Z = {v: C.encode_texts(C.text_variant(df, v)).numpy() for v in ["rich", "nodrug", "mech"]}
builders = {"scaffold": C.scaffold_split, "target": C.target_split, "joint": C.joint_split}

rows = []
for name, fn in builders.items():
    for p in range(5):
        sp = fn(df, p)
        for v in ["rich", "nodrug", "mech"]:
            r = C.run_split(df, sp, v, seed=0, Zall=Z[v], Xall=X)
            rows.append({"split": name, "partition": p, "text": v, "n_test": r["n_test"],
                         "n_train": r["n_train"], **r["metrics"]})
        print(name, p, [f"{x['text']} {x['R@1']:.3f}" for x in rows[-3:]], "n_test", rows[-1]["n_test"])

d = pd.DataFrame(rows)
g = d.groupby(["split", "text"]).agg(["mean", "std"])
summary = {f"{s}|{t}": {k: [float(g.loc[(s, t), (k, "mean")]), float(g.loc[(s, t), (k, "std")])]
                        for k in ["R@1", "R@1_struct", "MRR", "R@5", "n_test"]}
           for s, t in g.index}
C.save_json({"rows": rows, "summary": summary}, C.RESULTS / "e07_target_splits.json")
L = ["| Split | Text | n_test (mean) | R@1 | R@1 struct | R@5 | MRR |", "|---|---|---|---|---|---|---|"]
for s in builders:
    for t in ["rich", "nodrug", "mech"]:
        x = summary[f"{s}|{t}"]
        L.append(f"| {s} | {t} | {x['n_test'][0]:.0f} | " + " | ".join(
            f"{x[k][0]:.3f} ± {x[k][1]:.3f}" for k in ["R@1", "R@1_struct", "R@5", "MRR"]) + " |")
open(C.RESULTS / "e07_target_splits.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
