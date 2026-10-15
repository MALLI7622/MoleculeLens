"""Shared pharmacophore definitions for E13 (MoleculeLens) and E16 (MolPrompt)."""
import re

import numpy as np
from rdkit import Chem

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


def classify(mech_and_target_texts):
    """Index into CLASSES for each row (first matching regex), -1 if none."""
    out = np.full(len(mech_and_target_texts), -1)
    for k, (_, rx, _) in enumerate(CLASSES):
        for i, t in enumerate(mech_and_target_texts):
            if out[i] < 0 and re.search(rx, t, re.I):
                out[i] = k
    return out


def pharm_atoms(smiles, k):
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(CLASSES[k][2])
    return mol, set(a for m in mol.GetSubstructMatches(patt) for a in m)


def score(sal, pharm):
    """(positive-mass share on pharmacophore atoms, top-1 atom in pharm, any top-3 atom in pharm)."""
    sal = np.asarray(sal, dtype=float)
    pos = np.clip(sal, 0, None)
    mass = pos[list(pharm)].sum() / pos.sum() if pos.sum() > 0 else 0.0
    order = np.argsort(-sal)
    return float(mass), float(order[0] in pharm), float(len(set(order[:3]) & pharm) > 0)


def top3_chance(heavy, npharm):
    return 1 - np.prod([(heavy - npharm - t) / (heavy - t) for t in range(3)])
