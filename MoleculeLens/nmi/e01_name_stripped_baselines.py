"""E01: name-stripped comparison of MoleculeLens vs. baselines on the canonical scaffold test (N=435).

* MoleculeLens: saved paper checkpoints (rich / retrained-nodrug / mechanism-only), evaluated with the
  CORRECT pairing (text head and molecule head from the same model).  The paper's 0.110 "retrained
  no-drug" number paired the no-drug text head with the RICH model's molecule embeddings (see
  09b_diagonal_artifact_sync.py, which loads outputs/Bm_test.pt for every condition) - reported here
  as `paper_mismatched_pairing` for transparency.
* Same-model no-name (rich model, 'Drug: X.' removed at inference).
* MolPrompt (fine-tuned on our scaffold train split): per-example ranks recomputed from saved test
  embeddings -> exact metrics, bootstrap CIs and paired tests.
* KV-PLM (zero-shot, pretrained retrieval checkpoint): only aggregate metrics are stored -> Wilson CI.
* Ceiling: when several drugs share an identical query text (common once names are removed), at most one
  of them can be ranked first under single-label diagonal evaluation -> R@1 ceiling = #unique texts / N.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import core as C

df = C.load_df()
sp = C.canonical_split()
te = np.array(sp["test"])
dft = df.iloc[te].reset_index(drop=True)
parents = dft["smiles"].map(C.parent_smiles).values
X = C.fingerprint_matrix(df["smiles"].tolist())[te]


def load_heads(sfx, sub=""):
    pt = torch.nn.Linear(768, 256); pm = torch.nn.Linear(2048, 256)
    pt.load_state_dict(torch.load(C.ROOT / f"outputs/{sub}proj_text{sfx}.pt", map_location="cpu"))
    pm.load_state_dict(torch.load(C.ROOT / f"outputs/{sub}proj_mol{sfx}.pt", map_location="cpu"))
    return pt.eval(), pm.eval()


def saved_Z(sfx, sub=""):
    return torch.load(C.ROOT / f"outputs/{sub}Z_text_test{sfx}.pt", map_location="cpu").numpy()


rows, ranks_by, S_by = [], {}, {}


def add(name, S, note, protocol):
    m = C.metrics(S)
    r = m["ranks"]
    ranks_by[name] = r
    S_by[name] = S
    rows.append(dict(model=name, protocol=protocol, note=note, **C.strip(m),
                     ci_row=C.row_bootstrap(r), ci_parent=C.cluster_bootstrap(r, parents)))


# --- MoleculeLens ----------------------------------------------------------------
pt, pm = load_heads("")
S_rich = C.similarity(pt, pm, saved_Z(""), X)
add("MoleculeLens rich", S_rich, "paper checkpoint", "trained on scaffold train split")
add("MoleculeLens same-model no-name", C.similarity(pt, pm, saved_Z("_nodrug"), X),
    "rich heads, 'Drug: X.' removed at inference", "trained (rich)")
pt2, pm2 = load_heads("_nodrug")
add("MoleculeLens retrained no-drug", C.similarity(pt2, pm2, saved_Z("_nodrug"), X),
    "no-drug heads + no-drug text, CORRECT pairing", "trained on scaffold train split")
# paper's mismatched pairing (no-drug text head x rich molecule head)
bt = F.normalize(torch.load(C.ROOT / "outputs/Bt_test_nodrug.pt"), dim=1)
bm_rich = F.normalize(torch.load(C.ROOT / "outputs/Bm_test.pt"), dim=1)
add("[paper] retrained no-drug (mismatched)", (bt @ bm_rich.T).numpy(),
    "no-drug TEXT head x RICH molecule head, as in 09b (bug)", "-")
try:
    ptm, pmm = load_heads("", "mechanism_only/")
    add("MoleculeLens mechanism-only", C.similarity(ptm, pmm, saved_Z("", "mechanism_only/"), X),
        "paper checkpoint, mechanism_of_action only", "trained on scaffold train split")
except FileNotFoundError as e:
    print("mechanism-only checkpoint missing:", e)

# --- MolPrompt ---------------------------------------------------------------------
for sfx, nm in [("", "MolPrompt rich"), ("_nodrug", "MolPrompt no-drug")]:
    d = np.load(f"/home/cheriearjun/MolPrompt/save_model/retrieval_chembl_sharedsplit_bs6{sfx}/"
                "chembl_retrieval_best_metrics.npz")
    t = F.normalize(torch.tensor(d["test_text"]), dim=1); g = F.normalize(torch.tensor(d["test_graph"]), dim=1)
    S = (t @ g.T).numpy()
    add(nm, S, f"fine-tuned (best epoch {int(d['best_epoch'])} on val); stored R@1={float(d['test_t2m_r1']):.4f}",
        "fine-tuned on scaffold train split")


# --- KV-PLM (aggregate only) --------------------------------------------------------
def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


for f, nm in [("kv_plm_aligned.json", "KV-PLM rich"), ("kv_plm_aligned_nodrug.json", "KV-PLM no-drug")]:
    js = json.load(open(C.ROOT / "results" / f))
    rows.append(dict(model=nm, protocol="zero-shot (pretrained retrieval ckpt)", note="aggregate only",
                     **{"R@1": js["Recall@1"], "R@5": js["Recall@5"], "R@10": js["Recall@10"], "MRR": js["MRR"]},
                     ci_row={"R@1": wilson(js["Recall@1"], 435)}, ci_parent=None))

# --- ceilings ------------------------------------------------------------------------
ceil = {}
for v in ["rich", "nodrug", "mech"]:
    txt = pd.Series(C.text_variant(df, v)).iloc[te]
    ceil[v] = {"unique_texts": int(txt.nunique()), "R@1_ceiling": float(txt.nunique() / len(te))}


# --- paired tests (parent-cluster bootstrap of the R@1 difference) -------------------
def paired(a, b, n_boot=10000, seed=0):
    ra, rb = ranks_by[a], ranks_by[b]
    cl = C.group_ids(parents)
    members = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[p] for p in rng.integers(0, len(members), len(members))])
        diffs.append(np.mean(ra[idx] == 1) - np.mean(rb[idx] == 1))
    d = np.mean(ra == 1) - np.mean(rb == 1)
    return {"diff_R@1": float(d), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_le_0": float(np.mean(np.array(diffs) <= 0))}


tests = {
    "ML rich - MolPrompt rich": paired("MoleculeLens rich", "MolPrompt rich"),
    "ML retrained no-drug - MolPrompt no-drug": paired("MoleculeLens retrained no-drug", "MolPrompt no-drug"),
    "ML retrained no-drug - MolPrompt rich": paired("MoleculeLens retrained no-drug", "MolPrompt rich"),
}

# --- tie handling: optimistic (shared eval) vs strict (ties count against the model) ---------
def strict_r1(S):
    d = np.diag(S)
    return float(np.mean(np.sum(S >= d[:, None], axis=1) == 1))


def expected_r1(S):
    """Expected R@1 under uniformly random tie-breaking (row gallery)."""
    d = np.diag(S)
    gt = np.sum(S > d[:, None] + 1e-7, axis=1)
    eq = np.sum(np.abs(S - d[:, None]) <= 1e-7, axis=1)          # includes itself
    return float(np.mean((gt == 0) / eq))


smi_groups = C.group_ids(dft["smiles"].values)


def structure_r1(S):
    """Gallery deduplicated to unique SMILES (identical structures collapse to one entry);
    a query is correct if its own structure ranks first; residual ties broken at random (expected)."""
    G = smi_groups.max() + 1
    rep = np.array([np.where(smi_groups == g)[0][0] for g in range(G)])
    Sg = S[:, rep]                                       # identical structures -> identical columns
    own = Sg[np.arange(len(S)), smi_groups]
    gt = np.sum(Sg > own[:, None] + 1e-7, axis=1)
    eq = np.sum(np.abs(Sg - own[:, None]) <= 1e-7, axis=1)
    return float(np.mean((gt == 0) / eq))


Xfull = X
dup_mol = int(len(dft) - pd.Series([x.tobytes() for x in Xfull]).nunique())
ties = {k: {"R@1_optimistic": float(np.mean(ranks_by[k] == 1)), "R@1_strict": strict_r1(S_by[k]),
            "R@1_random_tiebreak": expected_r1(S_by[k]), "R@1_structure_gallery": structure_r1(S_by[k])}
        for k in S_by}
ties["_note"] = f"{dup_mol} test rows share an identical ECFP4 vector with another test row"

out = {"ties": ties, "rows": rows, "ceilings": ceil, "paired_tests": tests, "n_test": 435,
       "n_parent_clusters": int(len(set(parents)))}
C.save_json(out, C.RESULTS / "e01_name_stripped.json")

# markdown
lines = ["| Model | Protocol | R@1 [95% CI, parent-cluster] | R@5 | R@10 | MRR |", "|---|---|---|---:|---:|---:|"]
for r in rows:
    ci = (r["ci_parent"] or r["ci_row"])["R@1"]
    lines.append(f"| {r['model']} | {r['protocol']} | {C.fmt_ci(r['R@1'], ci)} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['MRR']:.3f} |")
md = "\n".join(lines)
md += "\n\nR@1 ceilings (identical query texts): " + ", ".join(
    f"{k}: {v['R@1_ceiling']:.3f} ({v['unique_texts']} unique)" for k, v in ceil.items())
md += "\n\nPaired R@1 differences (parent-cluster bootstrap):\n" + "\n".join(
    f"- {k}: {v['diff_R@1']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}], P(diff<=0)={v['p_le_0']:.4f}"
    for k, v in tests.items())
md += ("\n\nTie handling - R@1 under: optimistic (shared eval) | random tie-break | strict | "
       "structure-deduplicated gallery (unique SMILES, random tie-break):\n") + "\n".join(
    f"- {k}: {v['R@1_optimistic']:.3f} | {v['R@1_random_tiebreak']:.3f} | {v['R@1_strict']:.3f} | {v['R@1_structure_gallery']:.3f}"
    for k, v in ties.items() if not k.startswith("_"))
md += "\n- " + ties["_note"]
open(C.RESULTS / "e01_name_stripped.md", "w").write(md + "\n")
print(md)
