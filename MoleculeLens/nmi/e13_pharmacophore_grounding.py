"""E13: quantitative chemical grounding of the saliency (R1, R3 W2: '8 hand-picked examples').

Automated proxy for an expert study.  16 mechanism classes are mapped to a literature pharmacophore
(SMARTS).  For every test pair (all 435 scaffold-test pairs, paper checkpoint) whose mechanism/target
matches a class AND whose molecule contains the pharmacophore, atom-level saliency is computed by
spreading each present bit's Eq.-1 attribution evenly over the atoms of its environments (as in the
paper's Fig. 3 colouring).  Metrics per pair:
  * mass   : share of positive attribution mass on pharmacophore atoms
  * top1   : is the most salient atom a pharmacophore atom?
  * top3   : is any of the 3 most salient atoms a pharmacophore atom?
Nulls:
  (a) chance        : share of heavy atoms in the pharmacophore (expected mass / top1 under random atoms)
  (b) wrong text    : same molecule, saliency under 20 mechanism texts from OTHER classes (does the
                      MECHANISM text steer the saliency to the pharmacophore, or is it a molecule prior?)
  (c) untrained     : the paper's saved initial heads (proj_*_0.pt)
Tests: one-sided Wilcoxon signed-rank (correct text > wrong text; correct > chance), per class + pooled.
"""
import re

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import wilcoxon

import core as C

CLASSES = [
    ("carbonic anhydrase inhibitor", r"carbonic anhydrase", "[SX4](=O)(=O)[NX3;H2]"),
    ("beta-lactam / PBP", r"penicillin-binding|beta-lactamase", "O=C1[#6,#7][#6][#7]1"),
    ("HMG-CoA reductase inhibitor", r"HMG-CoA", "[OX2H][CX4][CX4][CX4]([OX2H])[CX4]C(=O)[OX2H1,OX1-,OX2]"),
    ("steroid nuclear receptor", r"glucocorticoid|mineralocorticoid|androgen receptor|progesterone receptor|estrogen receptor",
     "[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~1~2"),
    ("nucleoside/nucleobase polymerase-RT inhibitor", r"polymerase inhibitor|reverse transcriptase inhibitor|thymidylate",
     "[$([#6]1~[#7]~[#6]~[#7]~[#6]2~[#6]~1~[#7]~[#6]~[#7]~2),$([#6]1(=O)~[#7]~[#6]~[#6]~[#6]~[#7]~1)]"),
    ("fluoroquinolone / gyrase", r"gyrase|topoisomerase IV", "O=C([OX2H1,OX1-])c1c[nX3]c2ccccc2c1=O"),
    ("beta-adrenergic ligand", r"beta-?\d? adrenergic|beta adrenergic|adrenergic receptor beta", "[OX2H][CX4][CX4][NX3]"),
    ("proton pump inhibitor", r"potassium-transporting ATPase", "c1nc2ccccc2[nX3]1"),
    ("benzodiazepine GABA-A modulator", r"GABA-A receptor", "[#7]1~[#6]~[#6]~[#7]~[#6]~c2ccccc~21"),
    ("COX inhibitor", r"cyclooxygenase", "[$(c[CX4]C(=O)[OX2H1,OX1-]),$(c[CX4H]([CH3])C(=O)[OX2H1,OX1-]),$(cC(=O)[OX2H1,OX1-]),$([SX4](=O)(=O)([NH2,CH3])c)]"),
    ("angiotensin II receptor antagonist", r"angiotensin II", "[$(c1nn[nH]n1),$(c1n[nH]nn1),$(c1ccc(cc1)-c1ccccc1)]"),
    ("dihydropyridine Ca-channel blocker", r"calcium channel", "[#6]1=[#6][#7H][#6]=[#6][#6]1"),
    ("DHFR inhibitor", r"dihydrofolate reductase", "[NX3H2]c1nc([NX3H2])[c,n][c,n]n1"),
    ("DHPS sulfonamide antibacterial", r"dihydropteroate", "c[SX4](=O)(=O)[NX3]"),
    ("bisphosphonate FPPS inhibitor", r"farnesyl", "P(=O)(O)(O)[CX4]P(=O)(O)O"),
    ("azole CYP51 antifungal", r"lanosterol|sterol 14|CYP51|51A1", "[$([nX3]1cncc1),$([nX3]1cncn1)]"),
]

df = C.load_df()
te = np.array(C.canonical_split()["test"])
dft = df.iloc[te].reset_index(drop=True)
X = C.fingerprint_matrix(df["smiles"].tolist())[te]
Zt = torch.load(C.ROOT / "outputs/Z_text_test.pt").numpy()


def heads(sfx):
    pt = torch.nn.Linear(768, 256); pm = torch.nn.Linear(2048, 256)
    pt.load_state_dict(torch.load(C.ROOT / f"outputs/proj_text{sfx}.pt", map_location="cpu"))
    pm.load_state_dict(torch.load(C.ROOT / f"outputs/proj_mol{sfx}.pt", map_location="cpu"))
    return pt.eval(), pm.eval()


trained, untrained = heads(""), heads("_0")


def atom_saliency(mol, info, a):
    s = np.zeros(mol.GetNumAtoms())
    for b, envs in info.items():
        if a[b] == 0:
            continue
        for c, r in envs:
            at = list(C.env_atoms(mol, c, r))
            s[at] += a[b] / len(at) / len(envs)
    return s


def score(sal, pharm):
    pos = np.clip(sal, 0, None)
    mass = pos[list(pharm)].sum() / pos.sum() if pos.sum() > 0 else 0.0
    order = np.argsort(-sal)
    return float(mass), float(order[0] in pharm), float(len(set(order[:3]) & pharm) > 0)


text = (dft.mechanism_of_action.fillna("") + " " + dft.target_name.fillna("")).values
cls_of = np.full(len(dft), -1)
for k, (_, rx, _) in enumerate(CLASSES):
    for i in range(len(dft)):
        if cls_of[i] < 0 and re.search(rx, text[i], re.I):
            cls_of[i] = k
rng = np.random.default_rng(0)
recs = []
for i in range(len(dft)):
    k = cls_of[i]
    if k < 0:
        continue
    mol = Chem.MolFromSmiles(dft.smiles[i])
    patt = Chem.MolFromSmarts(CLASSES[k][2])
    pharm = set(a for m in mol.GetSubstructMatches(patt) for a in m)
    if not pharm:
        continue
    info = C.bit_environments(dft.smiles[i])
    heavy = mol.GetNumAtoms()
    r = {"row": i, "class": CLASSES[k][0], "drug": dft.pref_name[i], "chance": len(pharm) / heavy}
    r["mass"], r["top1"], r["top3"] = score(atom_saliency(mol, info, C.saliency(*trained, Zt[i], X[i])), pharm)
    r["mass_untrained"], r["top1_untrained"], r["top3_untrained"] = score(
        atom_saliency(mol, info, C.saliency(*untrained, Zt[i], X[i])), pharm)
    others = np.where(cls_of != k)[0]
    w = [score(atom_saliency(mol, info, C.saliency(*trained, Zt[j], X[i])), pharm)
         for j in rng.choice(others, 20, replace=False)]
    r["mass_wrongtext"], r["top1_wrongtext"], r["top3_wrongtext"] = [float(np.mean(x)) for x in zip(*w)]
    r["top3_chance"] = 1 - np.prod([(heavy - len(pharm) - t) / (heavy - t) for t in range(3)])
    recs.append(r)

d = pd.DataFrame(recs)


def test(a, b):
    diff = np.asarray(a) - np.asarray(b)
    if len(diff) < 5 or np.all(diff == 0):
        return None
    return float(wilcoxon(a, b, alternative="greater").pvalue)


summ = {}
for nm, sub in list(d.groupby("class")) + [("ALL", d)]:
    summ[nm] = {"n_pairs": len(sub),
                **{c: float(sub[c].mean()) for c in ["mass", "mass_wrongtext", "mass_untrained", "chance",
                                                      "top1", "top1_wrongtext", "top1_untrained",
                                                      "top3", "top3_wrongtext", "top3_chance"]},
                "p_mass_gt_chance": test(sub["mass"], sub["chance"]),
                "p_mass_gt_wrongtext": test(sub["mass"], sub["mass_wrongtext"]),
                "p_mass_gt_untrained": test(sub["mass"], sub["mass_untrained"])}
C.save_json({"per_pair": recs, "summary": summ, "classes": [c[:2] + (c[2],) for c in CLASSES],
             "n_test_pairs_matched_to_class": int((cls_of >= 0).sum())}, C.RESULTS / "e13_pharmacophore_grounding.json")
L = [f"{int((cls_of >= 0).sum())}/435 test pairs fall in one of the {len(CLASSES)} mechanism classes; "
     f"{len(d)} of them contain the class pharmacophore and are scored.", "",
     "| Class | n | mass on pharmacophore: correct text / wrong text / untrained / chance | top-1 atom in pharm: correct / wrong / chance | top-3: correct / wrong / chance | p (mass > chance) | p (correct > wrong text) |",
     "|---|---|---|---|---|---|---|"]
for nm, s in sorted(summ.items(), key=lambda x: (x[0] == "ALL", -x[1]["n_pairs"])):
    fp = lambda p: "-" if p is None else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")  # noqa: E731
    L.append(f"| {'**ALL**' if nm == 'ALL' else nm} | {s['n_pairs']} | {s['mass']:.2f} / {s['mass_wrongtext']:.2f} / "
             f"{s['mass_untrained']:.2f} / {s['chance']:.2f} | {s['top1']:.2f} / {s['top1_wrongtext']:.2f} / {s['chance']:.2f} | "
             f"{s['top3']:.2f} / {s['top3_wrongtext']:.2f} / {s['top3_chance']:.2f} | {fp(s['p_mass_gt_chance'])} | {fp(s['p_mass_gt_wrongtext'])} |")
open(C.RESULTS / "e13_pharmacophore_grounding.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
