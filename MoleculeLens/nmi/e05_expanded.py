"""E5: expand beyond approved drugs (R3 Q4).

Data: chembl_mechanisms_allphases.csv (all development phases, same cleaning + text recipe).

E5a  dataset statistics by phase (rows, molecules, scaffolds, targets, share with a drug name).
E5b  more data, same test: canonical scaffold test (435 rows) fixed; train on
       (i)  canonical train (2,019 approved rows)
       (ii) all-phase rows whose scaffold is NOT in the canonical val/test scaffolds and whose
            molecule is not in the canonical test (scaffold-disjoint by construction)
       (iii) the approved-only subset of (ii) (separates "more phases" from "newer ChEMBL")
     5 seeds x rich / nodrug / mech.
E5c  new all-phase scaffold benchmark: 5 partitions (07 recipe), rich / nodrug / mech,
     metrics overall and for approved vs. non-approved test rows, plus R@1 ceilings.
"""
import numpy as np
import pandas as pd

import core as C

ALL = C.ROOT / "chembl_mechanisms_allphases.csv"
dfa = C.load_df(ALL)
dfa["phase"] = pd.to_numeric(dfa["max_phase"], errors="coerce").fillna(-1)
dfc = C.load_df()
canon = C.canonical_split()

# ---------------- E5a ----------------
dfa["scaffold"] = dfa["smiles"].map(C.murcko).fillna("")
stats = {}
for nm, sub in [("all", dfa), ("approved (4)", dfa[dfa.phase == 4]), ("phase 3", dfa[dfa.phase == 3]),
                ("phase 2", dfa[dfa.phase == 2]), ("phase 1", dfa[dfa.phase == 1]), ("unknown", dfa[dfa.phase < 1])]:
    stats[nm] = {"rows": len(sub), "molecules": int(sub.molecule_chembl_id.nunique()),
                 "scaffolds": int(sub.scaffold.nunique()), "targets": int(sub.target_chembl_id.nunique()),
                 "share_with_drug_name": float(sub.pref_name.notna().mean())}
print(stats)

# ---------------- E5b ----------------
csc = dfc["smiles"].map(C.murcko).fillna("").values
held = set(csc[canon["test"]]) | set(csc[canon["val"]])
test_mols = set(dfc.molecule_chembl_id.values[canon["test"]]) | set(dfc.molecule_chembl_id.values[canon["val"]])
pool = dfa[~dfa.scaffold.isin(held) & ~dfa.molecule_chembl_id.isin(test_mols)]
trains = {"(i) canonical approved train": dfc.iloc[canon["train"]],
          "(ii) all-phase, scaffold-disjoint": pool,
          "(iii) approved-only subset of (ii)": pool[pool.phase == 4]}
testdf = dfc.iloc[canon["test"]]
e5b = {}
for nm, trdf in trains.items():
    comb = pd.concat([trdf, testdf], ignore_index=True)
    sp = {"train": list(range(len(trdf))), "test": list(range(len(trdf), len(comb)))}
    X = C.fingerprint_matrix(comb["smiles"].tolist())
    e5b[nm] = {"n_train": len(trdf)}
    for v in ["rich", "nodrug", "mech"]:
        Z = C.encode_texts(C.text_variant(comb, v)).numpy()
        d = pd.DataFrame([C.run_split(comb, sp, v, seed=s, Zall=Z, Xall=X)["metrics"] for s in range(5)])
        e5b[nm][v] = {k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in ["R@1", "R@1_struct", "R@5", "R@10", "MRR"]}
    print(nm, len(trdf), {v: round(e5b[nm][v]["R@1"][0], 3) for v in ["rich", "nodrug", "mech"]})

# ---------------- E5c ----------------
X = C.fingerprint_matrix(dfa["smiles"].tolist())
Z = {v: C.encode_texts(C.text_variant(dfa, v)).numpy() for v in ["rich", "nodrug", "mech"]}
rows = []
for p in range(5):
    sp = C.scaffold_split(dfa, p)
    te = np.array(sp["test"])
    appr = dfa.phase.values[te] == 4
    for v in ["rich", "nodrug", "mech"]:
        r = C.run_split(dfa, sp, v, seed=0, Zall=Z[v], Xall=X)
        rk = np.array(r["ranks"])
        txt = pd.Series(C.text_variant(dfa, v)).iloc[te]
        rows.append({"partition": p, "text": v, "n_test": len(te), **r["metrics"],
                     "R@1_approved_rows": float(np.mean(rk[appr] == 1)),
                     "R@1_nonapproved_rows": float(np.mean(rk[~appr] == 1)),
                     "n_approved": int(appr.sum()), "R@1_ceiling": float(txt.nunique() / len(te))})
    print(p, [(x["text"], round(x["R@1"], 3)) for x in rows[-3:]])
d = pd.DataFrame(rows)
e5c = {v: {k: [float(d[d.text == v][k].mean()), float(d[d.text == v][k].std(ddof=1))]
           for k in ["n_test", "R@1", "R@1_struct", "R@5", "MRR", "R@1_approved_rows", "R@1_nonapproved_rows", "R@1_ceiling"]}
       for v in ["rich", "nodrug", "mech"]}

C.save_json({"E5a_stats": stats, "E5b_more_data_same_test": e5b, "E5c_allphase_benchmark": {"rows": rows, "summary": e5c}},
            C.RESULTS / "e05_expanded.json")
L = ["**E5a - all-phase ChEMBL mechanism set:**", "", "| Subset | rows | molecules | scaffolds | targets | with drug name |", "|---|---|---|---|---|---|"]
L += [f"| {k} | {v['rows']} | {v['molecules']} | {v['scaffolds']} | {v['targets']} | {100*v['share_with_drug_name']:.0f}% |" for k, v in stats.items()]
L += ["", "**E5b - same canonical test (435), more training data (5 seeds):**", "",
      "| Training set | n_train | R@1 rich | R@1 struct rich | MRR rich | R@1 no-drug | R@1 mech-only |", "|---|---|---|---|---|---|---|"]
for nm, v in e5b.items():
    L.append(f"| {nm} | {v['n_train']} | {v['rich']['R@1'][0]:.3f} ± {v['rich']['R@1'][1]:.3f} | {v['rich']['R@1_struct'][0]:.3f} | "
             f"{v['rich']['MRR'][0]:.3f} | {v['nodrug']['R@1'][0]:.3f} ± {v['nodrug']['R@1'][1]:.3f} | {v['mech']['R@1'][0]:.3f} ± {v['mech']['R@1'][1]:.3f} |")
L += ["", "**E5c - all-phase scaffold benchmark (5 partitions):**", "",
      "| Text | n_test | R@1 | R@1 struct | MRR | R@1 approved rows | R@1 non-approved rows | R@1 ceiling |", "|---|---|---|---|---|---|---|---|"]
for v, s in e5c.items():
    L.append(f"| {v} | {s['n_test'][0]:.0f} | {s['R@1'][0]:.3f} ± {s['R@1'][1]:.3f} | {s['R@1_struct'][0]:.3f} | {s['MRR'][0]:.3f} | "
             f"{s['R@1_approved_rows'][0]:.3f} | {s['R@1_nonapproved_rows'][0]:.3f} | {s['R@1_ceiling'][0]:.3f} |")
open(C.RESULTS / "e05_expanded.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
