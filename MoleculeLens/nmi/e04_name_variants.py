"""E4: what in the drug name drives retrieval?  (R3 Q6: INN stems)

Text variants (canonical split, 5 init seeds each, model retrained per variant):
  rich      paper text (mechanism + target + action + full drug name)
  nodrug    name removed
  stem      name replaced by its last 5 letters ('-tinib', '-azole'...): INN-stem proxy
  shuffled  names permuted across rows (a name is present but belongs to another drug)
  nameonly  'Drug: X.' only, no mechanism text
  mech      mechanism_of_action only
Same-model probes: the rich model evaluated on stem / shuffled / nodrug text.
Stem overlap: fraction of test drugs whose 5-letter stem also occurs in a training drug name.
"""
import numpy as np
import pandas as pd

import core as C

df = C.load_df()
canon = C.canonical_split()
te, tr = np.array(canon["test"]), np.array(canon["train"])
X = C.fingerprint_matrix(df["smiles"].tolist())
V = ["rich", "nodrug", "stem", "shuffled", "nameonly", "mech"]
Z = {v: C.encode_texts(C.text_variant(df, v)).numpy() for v in V}

res = {}
for v in V:
    rows = [C.run_split(df, canon, v, seed=s, Zall=Z[v], Xall=X)["metrics"] for s in range(5)]
    d = pd.DataFrame(rows)
    res[v] = {k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in d}
    print(v, {k: f"{m:.3f}±{s:.3f}" for k, (m, s) in res[v].items()})

r = C.run_split(df, canon, "rich", seed=0, Zall=Z["rich"], Xall=X, return_model=True)
pt, pm = r["model"]
same = {}
for v in ["rich", "nodrug", "stem", "shuffled"]:
    S = C.similarity(pt, pm, Z[v][te], X[te])
    m = C.strip(C.metrics(S)); m["R@1_struct"] = C.structure_r1(S, df["smiles"].values[te])
    same[v] = m

names = df["pref_name"].fillna("").astype(str).str.split().str[0].str.lower()
stem = names.str[-5:]
train_stems = set(stem.iloc[tr][names.iloc[tr].str.len() > 0])
test_has = names.iloc[te].str.len() > 0
overlap = float(np.mean([s in train_stems for s in stem.iloc[te][test_has]]))

out = {"retrained_5seeds": res, "same_model_rich_seed0": same, "test_stem_seen_in_train": overlap}
C.save_json(out, C.RESULTS / "e04_name_variants.json")
L = ["| Text variant (retrained, 5 seeds) | R@1 | R@1 struct | R@5 | MRR |", "|---|---|---|---|---|"]
for v in V:
    L.append(f"| {v} | " + " | ".join(f"{res[v][k][0]:.3f} ± {res[v][k][1]:.3f}" for k in ["R@1", "R@1_struct", "R@5", "MRR"]) + " |")
L += ["", "Same rich model (seed 0), text changed at inference: " +
      ", ".join(f"{v}: R@1 {m['R@1']:.3f} (struct {m['R@1_struct']:.3f})" for v, m in same.items()),
      "", f"Test drugs whose 5-letter name stem also appears among training drug names: {100*overlap:.1f}%"]
open(C.RESULTS / "e04_name_variants.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
