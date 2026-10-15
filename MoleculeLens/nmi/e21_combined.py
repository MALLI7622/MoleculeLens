"""E21: combined experiment - everything the individual experiments justified, in one model.

Step-wise ablation on the canonical scaffold test (5 seeds each), NAME-FREE text (mechanism + target +
action) as the honest main condition, rich text reported alongside:
  A  paper config        : ECFP4 2048, CLS, dim 256, approved train
  B  + 8192 bits         (E15)
  C  + parent fingerprints (E12; removes counterion / salt-name shortcut)
  D  + mean-pooled text  (E14)
  E  + dim 1024          (E19)
  F  + all-phase training data, scaffold-disjoint from the test (E5)
Final config (F) then evaluated with the full protocol:
  * canonical split: R@1, R@1 struct (unique parent), MRR, parent-cluster bootstrap CI;
    paired comparison with MolPrompt (no-drug, canonical) on the structure-deduplicated gallery;
  * 10 scaffold partitions (07 recipe): mean and 95% t-CI;
  * target-disjoint and joint splits (5 partitions);
  * interpretability re-check on the final model: E13 pharmacophore grounding (correct vs wrong text,
    chance), E17 deletion (top-k vs random), counterion share of top-1 saliency.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy import stats
from scipy.stats import wilcoxon

import core as C
from pharm import CLASSES, classify, score, top3_chance

dfc = C.load_df()
dfa = C.load_df(C.ROOT / "chembl_mechanisms_allphases.csv")
dfa["phase"] = pd.to_numeric(dfa["max_phase"], errors="coerce").fillna(-1)
dfa["scaffold"] = dfa["smiles"].map(C.murcko).fillna("")
dfc["scaffold"] = dfc["smiles"].map(C.murcko).fillna("")
canon = C.canonical_split()

CFG = {
    "A paper": dict(nbits=2048, parent=False, pool="cls", dim=256, allphase=False),
    "B +8192 bits": dict(nbits=8192, parent=False, pool="cls", dim=256, allphase=False),
    "C +parent fp": dict(nbits=8192, parent=True, pool="cls", dim=256, allphase=False),
    "D +mean-pool text": dict(nbits=8192, parent=True, pool="mean", dim=256, allphase=False),
    "E +dim 1024": dict(nbits=8192, parent=True, pool="mean", dim=1024, allphase=False),
    "F +all-phase train": dict(nbits=8192, parent=True, pool="mean", dim=1024, allphase=True),
}


def build(split_c, allphase, mode="scaffold"):
    """Combined frame = training rows (+ all-phase pool if requested) followed by test rows of dfc.
    The all-phase pool respects the split type: held-out scaffolds (scaffold), held-out targets (target),
    or both (joint); test molecules are always excluded."""
    te = dfc.iloc[split_c["test"]]
    if allphase:
        ho = list(split_c["test"]) + list(split_c.get("val", []))
        keep = ~dfa.molecule_chembl_id.isin(set(te.molecule_chembl_id))
        if mode in ("scaffold", "joint"):
            keep &= ~dfa.scaffold.isin(set(dfc.scaffold.values[ho]))
        if mode in ("target", "joint"):
            keep &= ~dfa.target_chembl_id.isin(set(dfc.target_chembl_id.values[ho]))
        tr = dfa[keep]
    else:
        tr = dfc.iloc[split_c["train"]]
    comb = pd.concat([tr, te], ignore_index=True)
    return comb, {"train": list(range(len(tr))), "test": list(range(len(tr), len(comb)))}


def run(cfg, split_c, text, seed, return_model=False, mode="scaffold"):
    comb, sp = build(split_c, cfg["allphase"], mode)
    texts = C.text_variant(comb, text)
    Z = (C.encode_texts_meanpool(texts) if cfg["pool"] == "mean" else C.encode_texts(texts)).numpy()
    X = C.fingerprint_matrix(comb.smiles.tolist(), nbits=cfg["nbits"], parent=cfg["parent"])
    r = C.run_split(comb, sp, text, seed=seed, Zall=Z, Xall=X, dim=cfg["dim"], return_model=True)
    te = np.array(sp["test"])
    par = comb.smiles.map(C.parent_smiles).values[te]
    r["metrics"]["R@1_struct_parent"] = C.structure_r1(r["S"], par)
    if return_model:
        r["Z_te"], r["X_te"], r["comb_te"] = Z[te], X[te], comb.iloc[te].reset_index(drop=True)
    return r


out = {}
# ---------------- step-wise ablation ----------------
abl = {}
for name, cfg in CFG.items():
    abl[name] = {}
    for text in ["nodrug", "rich"]:
        d = pd.DataFrame([run(cfg, canon, text, s)["metrics"] for s in range(5)])
        abl[name][text] = {k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in d}
    print(name, {t: round(abl[name][t]["R@1_struct_parent"][0], 3) for t in ["nodrug", "rich"]})
out["ablation_canonical_5seeds"] = abl

FINAL = CFG["F +all-phase train"]
# ---------------- final: canonical, bootstrap, vs MolPrompt ----------------
r = run(FINAL, canon, "nodrug", 0, return_model=True)
par = r["comb_te"].smiles.map(C.parent_smiles).values
out["final_canonical_nodrug_seed0"] = {"metrics": r["metrics"],
                                      "ci_parent_cluster": C.cluster_bootstrap(np.array(r["ranks"]), par)}
npz = np.load("/home/cheriearjun/MolPrompt/save_model/retrieval_chembl_sharedsplit_bs6_nodrug/chembl_retrieval_best_metrics.npz")
Smp = (F.normalize(torch.tensor(npz["test_text"]), dim=1) @ F.normalize(torch.tensor(npz["test_graph"]), dim=1).T).numpy()


def struct_hits(S, groups):
    g = C.group_ids(groups); G = g.max() + 1
    rep = np.array([np.where(g == k)[0][0] for k in range(G)])
    Sg = S[:, rep]; own = Sg[np.arange(len(g)), g]
    gt = np.sum(Sg > own[:, None] + 1e-7, 1); eq = np.sum(np.abs(Sg - own[:, None]) <= 1e-7, 1)
    return (gt == 0) / eq


h_ml, h_mp = struct_hits(r["S"], par), struct_hits(Smp, par)
cl = C.group_ids(par); members = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
rng = np.random.default_rng(0)
diffs = [np.mean(h_ml[i]) - np.mean(h_mp[i]) for i in
         (np.concatenate([members[p] for p in rng.integers(0, len(members), len(members))]) for _ in range(10000))]
out["final_vs_molprompt_nodrug_struct_parent"] = {
    "MoleculeLens_final": float(h_ml.mean()), "MolPrompt_nodrug": float(h_mp.mean()),
    "diff": float(h_ml.mean() - h_mp.mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    "p_le_0": float(np.mean(np.array(diffs) <= 0))}
print(out["final_vs_molprompt_nodrug_struct_parent"])

# ---------------- final: partitions + OOD splits ----------------
part = {}
for text in ["nodrug", "rich"]:
    rows = [run(FINAL, C.scaffold_split(dfc, p), text, 0)["metrics"] for p in range(10)]
    d = pd.DataFrame(rows)
    part[text] = {k: {"mean": float(d[k].mean()), "ci95_t": [float(x) for x in stats.t.interval(
        0.95, len(d) - 1, loc=d[k].mean(), scale=stats.sem(d[k]))]} for k in ["R@1", "R@1_struct_parent", "MRR"]}
out["final_10_scaffold_partitions"] = part
base_part = {}
for text in ["nodrug", "rich"]:
    d = pd.DataFrame([run(CFG["A paper"], C.scaffold_split(dfc, p), text, 0)["metrics"] for p in range(10)])
    base_part[text] = {k: float(d[k].mean()) for k in ["R@1", "R@1_struct_parent", "MRR"]}
out["paper_config_10_scaffold_partitions"] = base_part
ood = {}
for nm, fn in [("target", C.target_split), ("joint", C.joint_split)]:
    for cfgname in ["A paper", "F +all-phase train"]:
        for text in ["nodrug", "rich"]:
            d = pd.DataFrame([run(CFG[cfgname], fn(dfc, p), text, 0, mode=nm)["metrics"] for p in range(5)])
            ood[f"{nm}|{cfgname}|{text}"] = {k: float(d[k].mean()) for k in ["R@1", "R@1_struct_parent", "MRR"]}
out["final_ood_splits"] = ood
print(ood)

# ---------------- final: interpretability re-check ----------------
pt, pm = r["model"]
dft = r["comb_te"]
Zte, Xte = r["Z_te"], r["X_te"]
cls_of = classify((dft.mechanism_of_action.fillna("") + " " + dft.target_name.fillna("")).values)
recs = []
for i in range(len(dft)):
    k = cls_of[i]
    if k < 0:
        continue
    psmi = C.parent_smiles(dft.smiles[i])
    mol = Chem.MolFromSmiles(psmi)
    pharm = set(a for m in mol.GetSubstructMatches(Chem.MolFromSmarts(CLASSES[k][2])) for a in m)
    if not pharm:
        continue
    info = C.bit_environments(psmi, nbits=8192)

    def atom_sal(a):
        s = np.zeros(mol.GetNumAtoms())
        for b, envs in info.items():
            if a[b] == 0:
                continue
            for c, rr in envs:
                at = list(C.env_atoms(mol, c, rr)); s[at] += a[b] / len(at) / len(envs)
        return s
    rec = {"class": CLASSES[k][0], "chance": len(pharm) / mol.GetNumAtoms(), "top3_chance": top3_chance(mol.GetNumAtoms(), len(pharm))}
    rec["mass"], rec["top1"], rec["top3"] = score(atom_sal(C.saliency(pt, pm, Zte[i], Xte[i])), pharm)
    w = [score(atom_sal(C.saliency(pt, pm, Zte[j], Xte[i])), pharm) for j in rng.choice(np.where(cls_of != k)[0], 20, replace=False)]
    rec["mass_wrongtext"], rec["top1_wrongtext"], rec["top3_wrongtext"] = [float(np.mean(x)) for x in zip(*w)]
    recs.append(rec)
g = pd.DataFrame(recs)
out["final_pharmacophore_grounding"] = {
    "n_pairs": len(g), **{c: float(g[c].mean()) for c in ["mass", "mass_wrongtext", "chance", "top1", "top1_wrongtext", "top3", "top3_wrongtext", "top3_chance"]},
    "p_mass_gt_chance": float(wilcoxon(g["mass"], g["chance"], alternative="greater").pvalue),
    "p_mass_gt_wrongtext": float(wilcoxon(g["mass"], g["mass_wrongtext"], alternative="greater").pvalue)}
print(out["final_pharmacophore_grounding"])

with torch.no_grad():
    BT = F.normalize(pt(torch.tensor(Zte)), dim=1); BM = F.normalize(pm(torch.tensor(Xte)), dim=1)
S0 = (BT @ BM.T).numpy()
dele = {}
for k in [1, 3, 5]:
    top, rnd = [], []
    for i in range(len(dft)):
        a = C.saliency(pt, pm, Zte[i], Xte[i]); present = np.where(Xte[i] > 0)[0]
        if len(present) <= k:
            continue
        for bits, store in [(present[np.argsort(-a[present])][:k], top), (rng.choice(present, k, replace=False), rnd)]:
            x2 = Xte[i].copy(); x2[bits] = 0
            with torch.no_grad():
                s = float(BT[i] @ F.normalize(pm(torch.tensor(x2)[None]), dim=1)[0])
            store.append(S0[i, i] - s)
    dele[k] = {"drop_top": float(np.mean(top)), "drop_random": float(np.mean(rnd))}
out["final_deletion"] = dele
C.save_json(out, C.RESULTS / "e21_combined.json")

# ---------------- markdown ----------------
L = ["**Step-wise ablation, canonical scaffold test, 5 seeds; R@1 on the structure-deduplicated (unique parent) gallery:**", "",
     "| Config | no-drug R@1 struct | no-drug MRR | rich R@1 struct | rich MRR |", "|---|---|---|---|---|"]
for nm, v in abl.items():
    L.append(f"| {nm} | {v['nodrug']['R@1_struct_parent'][0]:.3f} ± {v['nodrug']['R@1_struct_parent'][1]:.3f} | {v['nodrug']['MRR'][0]:.3f} | "
             f"{v['rich']['R@1_struct_parent'][0]:.3f} ± {v['rich']['R@1_struct_parent'][1]:.3f} | {v['rich']['MRR'][0]:.3f} |")
f = out["final_canonical_nodrug_seed0"]; c = f["ci_parent_cluster"]
m = out["final_vs_molprompt_nodrug_struct_parent"]
L += ["", f"**Final model, canonical split, no-drug text (seed 0):** R@1 {f['metrics']['R@1']:.3f} [{c['R@1'][0]:.3f}, {c['R@1'][1]:.3f}], "
      f"R@1 struct {f['metrics']['R@1_struct_parent']:.3f}, MRR {f['metrics']['MRR']:.3f} [{c['MRR'][0]:.3f}, {c['MRR'][1]:.3f}].",
      f"**vs MolPrompt no-drug** (same deduplicated gallery): {m['MoleculeLens_final']:.3f} vs {m['MolPrompt_nodrug']:.3f}, "
      f"diff {m['diff']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}], P(diff<=0) = {m['p_le_0']:.4f}.", "",
      "| 10 scaffold partitions | paper config R@1 struct | final R@1 struct [95% t-CI] | final MRR |", "|---|---|---|---|"]
for t in ["nodrug", "rich"]:
    p = part[t]
    L.append(f"| {t} | {base_part[t]['R@1_struct_parent']:.3f} | {p['R@1_struct_parent']['mean']:.3f} "
             f"[{p['R@1_struct_parent']['ci95_t'][0]:.3f}, {p['R@1_struct_parent']['ci95_t'][1]:.3f}] | {p['MRR']['mean']:.3f} |")
L += ["", "| OOD split (5 partitions) | text | paper config R@1 struct | final R@1 struct |", "|---|---|---|---|"]
for nm in ["target", "joint"]:
    for t in ["nodrug", "rich"]:
        L.append(f"| {nm} | {t} | {ood[f'{nm}|A paper|{t}']['R@1_struct_parent']:.3f} | {ood[f'{nm}|F +all-phase train|{t}']['R@1_struct_parent']:.3f} |")
gg = out["final_pharmacophore_grounding"]
L += ["", f"**Final model interpretability:** pharmacophore mass {gg['mass']:.2f} (wrong text {gg['mass_wrongtext']:.2f}, chance {gg['chance']:.2f}); "
      f"top-3 hit {gg['top3']:.2f} (wrong {gg['top3_wrongtext']:.2f}, chance {gg['top3_chance']:.2f}); n={gg['n_pairs']}; "
      f"p(>chance)={gg['p_mass_gt_chance']:.1e}, p(>wrong text)={gg['p_mass_gt_wrongtext']:.1e}. Counterion saliency: 0 by construction (parent fingerprints).",
      "Deletion (score drop, top-k salient vs random k): " + ", ".join(f"k={k}: {v['drop_top']:.3f} vs {v['drop_random']:.3f}" for k, v in dele.items())]
open(C.RESULTS / "e21_combined.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
