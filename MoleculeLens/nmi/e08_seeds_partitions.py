"""E8-E11: statistical rigour.

E8  : headline configuration (full loss, canonical split) across 10 initialisation seeds,
      for rich / retrained no-drug / mechanism-only text.
E9  : parent-structure cluster bootstrap CIs for all metrics (seed-0 headline model).
E10 : scaffold partitions built with the paper's own recipe (07_make_chembl_split_manifest.py);
      partition 0 == canonical manifest (checked).  Describes what each partition holds out.
E11 : name-leakage drop per partition (retrained no-drug with correct pairing, and same-model),
      mean, std and 95% t-interval across 10 partitions.
"""
import numpy as np
import pandas as pd
from scipy import stats

import core as C

df = C.load_df()
smiles = df["smiles"].values
canon = C.canonical_split()
Z = {v: C.encode_texts(C.text_variant(df, v)).numpy() for v in ["rich", "nodrug", "mech"]}
X = C.fingerprint_matrix(df["smiles"].tolist())
KEYS = ["R@1", "R@5", "R@10", "MRR", "R@1_struct"]


def summarise(rows):
    d = pd.DataFrame(rows)
    return {k: {"mean": float(d[k].mean()), "std": float(d[k].std(ddof=1)),
                "ci95_t": [float(x) for x in stats.t.interval(0.95, len(d) - 1, loc=d[k].mean(),
                                                              scale=stats.sem(d[k]))]}
            for k in KEYS if k in d}


out = {}
# ---------------- E8 ----------------
e8 = {}
for v in ["rich", "nodrug", "mech"]:
    rows = []
    for s in range(10):
        r = C.run_split(df, canon, v, seed=s, Zall=Z[v], Xall=X)
        rows.append({"seed": s, **r["metrics"]})
        if v == "rich" and s == 0:
            parents = pd.Series(smiles[canon["test"]]).map(C.parent_smiles).values
            out["E9_bootstrap_seed0_rich"] = {
                "point": r["metrics"],
                "ci_parent_cluster": C.cluster_bootstrap(np.array(r["ranks"]), parents),
                "ci_row": C.row_bootstrap(np.array(r["ranks"])),
                "n_rows": len(parents), "n_parent_clusters": int(len(set(parents)))}
    e8[v] = {"per_seed": rows, "summary": summarise(rows)}
    print("E8", v, {k: f"{e8[v]['summary'][k]['mean']:.3f}±{e8[v]['summary'][k]['std']:.3f}" for k in KEYS})
out["E8_init_seeds_canonical_split"] = e8

# ---------------- E10 / E11 ----------------
p0 = C.scaffold_split(df, 0)
out["E10_partition0_equals_canonical"] = all(sorted(p0[k]) == sorted(canon[k]) for k in ("train", "val", "test"))
print("partition 0 == canonical:", out["E10_partition0_equals_canonical"])

scaf = df["smiles"].map(C.murcko).fillna("").values
rows, desc = [], []
for p in range(10):
    sp = C.scaffold_split(df, p)
    te = np.array(sp["test"])
    rr = C.run_split(df, sp, "rich", seed=0, Zall=Z["rich"], Xall=X, return_model=True)
    rn = C.run_split(df, sp, "nodrug", seed=0, Zall=Z["nodrug"], Xall=X)
    pt, pm = rr["model"]
    S_same = C.similarity(pt, pm, Z["nodrug"][te], X[te])
    same = C.metrics(S_same)
    r1r, r1n, r1s = rr["metrics"]["R@1"], rn["metrics"]["R@1"], float(np.mean(same["ranks"] == 1))
    rows.append({"partition": p, "n_test": len(te), **rr["metrics"],
                 "R@1_nodrug_retrained": r1n, "R@1_nodrug_same_model": r1s,
                 "R@1_struct_nodrug": rn["metrics"]["R@1_struct"],
                 "leak_drop_retrained": (r1r - r1n) / r1r, "leak_drop_same_model": (r1r - r1s) / r1r})
    tsc = pd.Series(scaf[te])
    vc = tsc.value_counts()
    desc.append({"partition": p, "n_test_rows": len(te), "n_test_scaffolds": int(tsc.nunique()),
                 "acyclic_rows": int((tsc == "").sum()),
                 "largest_scaffolds": {k if k else "<acyclic>": int(c) for k, c in vc.head(3).items()},
                 "top_targets": df.iloc[te]["target_name"].value_counts().head(3).to_dict(),
                 "overlap_with_canonical_test": float(len(set(te) & set(canon["test"])) / len(te))})
    print(f"P{p} n={len(te)} R@1 rich={r1r:.3f} nodrug={r1n:.3f} same={r1s:.3f} struct={rr['metrics']['R@1_struct']:.3f}")

d = pd.DataFrame(rows)
summ = {}
for k in ["R@1", "R@5", "R@10", "MRR", "R@1_struct", "R@1_nodrug_retrained", "R@1_nodrug_same_model",
          "R@1_struct_nodrug", "leak_drop_retrained", "leak_drop_same_model"]:
    summ[k] = {"mean": float(d[k].mean()), "std": float(d[k].std(ddof=1)),
               "ci95_t": [float(x) for x in stats.t.interval(0.95, len(d) - 1, loc=d[k].mean(), scale=stats.sem(d[k]))],
               "min": float(d[k].min()), "max": float(d[k].max())}
out["E10_E11_partitions"] = {"rows": rows, "summary": summ, "description": desc,
                             "recipe": "07_make_chembl_split_manifest.py recipe, 80/10/10 by unique scaffold count, seeds 0-9"}
C.save_json(out, C.RESULTS / "e08_seeds_partitions.json")

# markdown
L = ["**E8 - headline config, canonical split, 10 init seeds (mean ± std):**", "",
     "| Text | R@1 | R@1 struct | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|"]
for v in ["rich", "nodrug", "mech"]:
    s = e8[v]["summary"]
    L.append(f"| {v} | " + " | ".join(f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}" for k in ["R@1", "R@1_struct", "R@5", "R@10", "MRR"]) + " |")
b = out["E9_bootstrap_seed0_rich"]
L += ["", f"**E9 - seed-0 rich, parent-cluster bootstrap ({b['n_parent_clusters']} clusters / {b['n_rows']} rows):** " +
      ", ".join(f"{k} {b['point'][k]:.3f} [{b['ci_parent_cluster'][k][0]:.3f}, {b['ci_parent_cluster'][k][1]:.3f}]"
                for k in ["R@1", "R@5", "R@10", "MRR"]),
      "", f"**E10 - partition 0 reproduces the canonical manifest:** {out['E10_partition0_equals_canonical']}", "",
      "| Partition | n_test | R@1 rich | R@1 struct | R@1 no-drug (retrained) | R@1 same-model no-name | drop retrained | drop same-model |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['partition']} | {r['n_test']} | {r['R@1']:.3f} | {r['R@1_struct']:.3f} | {r['R@1_nodrug_retrained']:.3f} | "
             f"{r['R@1_nodrug_same_model']:.3f} | {100*r['leak_drop_retrained']:.1f}% | {100*r['leak_drop_same_model']:.1f}% |")
L.append("| **mean [95% t-CI]** | | " + " | ".join(
    f"{summ[k]['mean']:.3f} [{summ[k]['ci95_t'][0]:.3f}, {summ[k]['ci95_t'][1]:.3f}]"
    for k in ["R@1", "R@1_struct", "R@1_nodrug_retrained", "R@1_nodrug_same_model"]) + " | " + " | ".join(
    f"{100*summ[k]['mean']:.1f}% [{100*summ[k]['ci95_t'][0]:.1f}, {100*summ[k]['ci95_t'][1]:.1f}]"
    for k in ["leak_drop_retrained", "leak_drop_same_model"]) + " |")
open(C.RESULTS / "e08_seeds_partitions.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
