"""E12: salts / counterions in the saliency (R1, Fig. 3A acetate and Fig. 3D Na+).

1. Paper checkpoint: for every test pair, are the top salient bits counterion-only environments
   (every occurrence centred on an atom outside the largest fragment)?
2. Retrain with parent-structure fingerprints (RDKit FragmentParent before ECFP4), 10 seeds,
   compare retrieval with the original fingerprints.
3. Fig. 3 case studies recomputed with the parent model: top-6 bits, their environments, and
   whether the top-3 salient atoms hit the expected pharmacophore (SMARTS).
"""
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

import core as C

df = C.load_df()
canon = C.canonical_split()
te = np.array(canon["test"])
dft = df.iloc[te].reset_index(drop=True)
X = C.fingerprint_matrix(df["smiles"].tolist())
Xp = C.fingerprint_matrix(df["smiles"].tolist(), parent=True)
Zr = C.encode_texts(C.text_variant(df, "rich")).numpy()


def counterion_bits(smi):
    """Bits whose every environment is centred outside the largest fragment."""
    m = Chem.MolFromSmiles(smi)
    frags = Chem.GetMolFrags(m)
    if len(frags) == 1:
        return set(), False
    main = set(max(frags, key=len))
    info = C.bit_environments(smi)
    return {b for b, envs in info.items() if all(c not in main for c, _ in envs)}, True


def load_paper():
    pt = torch.nn.Linear(768, 256); pm = torch.nn.Linear(2048, 256)
    pt.load_state_dict(torch.load(C.ROOT / "outputs/proj_text.pt", map_location="cpu"))
    pm.load_state_dict(torch.load(C.ROOT / "outputs/proj_mol.pt", map_location="cpu"))
    return pt.eval(), pm.eval()


# 1. counterion saliency in the paper model ---------------------------------------
pt, pm = load_paper()
Zt = torch.load(C.ROOT / "outputs/Z_text_test.pt").numpy()
n_salt = top1_ci = top6_ci = 0
mass = []
for i in range(len(dft)):
    cb, salt = counterion_bits(dft.smiles[i])
    if not salt:
        continue
    n_salt += 1
    a = C.saliency(pt, pm, Zt[i], X[te][i])
    order = np.argsort(-np.abs(a))
    order = order[np.abs(a[order]) > 0]
    top1_ci += int(order[0] in cb)
    top6_ci += int(len(set(order[:6]) & cb) > 0)
    mass.append(float(np.abs(a[list(cb)]).sum() / np.abs(a).sum()) if cb else 0.0)
part1 = {"test_rows_with_salt_or_counterion": n_salt, "of_435": len(dft),
         "top1_bit_is_counterion_only": top1_ci, "any_top6_bit_counterion_only": top6_ci,
         "mean_abs_attribution_mass_on_counterion_bits": float(np.mean(mass)) if mass else 0.0}
print(part1)

# 1b. is the counterion saliency driven by salt words in the drug name? ------------
SALT_WORDS = r"SODIUM|POTASSIUM|CALCIUM|MAGNESIUM|LITHIUM|ZINC|HYDROCHLORIDE|HYDROBROMIDE|CHLORIDE|BROMIDE|IODIDE|ACETATE|" \
             r"MESYLATE|MALEATE|FUMARATE|TARTRATE|BITARTRATE|CITRATE|SULFATE|PHOSPHATE|SUCCINATE|BESYLATE|TOSYLATE|" \
             r"LACTATE|GLUCONATE|NITRATE|OXALATE|MALATE|PAMOATE|DIPROPIONATE|HYCLATE|TROMETHAMINE|MEGLUMINE|ESYLATE|NAPSYLATE"


def counterion_top1(p1, p2, Z):
    hits, n = 0, 0
    for i in range(len(dft)):
        cb, salt = counterion_bits(dft.smiles[i])
        if not salt:
            continue
        a = C.saliency(p1, p2, Z[i], X[te][i])
        o = [b for b in np.argsort(-np.abs(a)) if a[b] != 0]
        hits += int(o[0] in cb); n += 1
    return hits / n


pt_nd = torch.nn.Linear(768, 256); pm_nd = torch.nn.Linear(2048, 256)
pt_nd.load_state_dict(torch.load(C.ROOT / "outputs/proj_text_nodrug.pt", map_location="cpu"))
pm_nd.load_state_dict(torch.load(C.ROOT / "outputs/proj_mol_nodrug.pt", map_location="cpu"))
Zt_nd = torch.load(C.ROOT / "outputs/Z_text_test_nodrug.pt").numpy()
salt_rows = [i for i in range(len(dft)) if counterion_bits(dft.smiles[i])[1]]
named = dft.pref_name.fillna("").str.upper().str.contains(SALT_WORDS, regex=True).values
part1["salt_rows_whose_drug_name_names_the_salt"] = int(named[salt_rows].sum())
part1["top1_counterion_share_rich_model"] = top1_ci / max(n_salt, 1)
part1["top1_counterion_share_nodrug_model"] = counterion_top1(pt_nd.eval(), pm_nd.eval(), Zt_nd)
part1["top1_counterion_share_rich_model_name_removed_at_inference"] = counterion_top1(pt, pm, Zt_nd)
print(part1)

# 2. retrain with parent fingerprints --------------------------------------------
cmp = {}
par_te = df["smiles"].map(C.parent_smiles).values[te]
for nm, XX in [("original", X), ("parent", Xp)]:
    rows = []
    for sd in range(10):
        r = C.run_split(df, canon, "rich", seed=sd, Zall=Zr, Xall=XX, return_model=True)
        r["metrics"]["R@1_struct_parent"] = C.structure_r1(r["S"], par_te)
        rows.append(r["metrics"])
    d = pd.DataFrame(rows)
    cmp[nm] = {k: [float(d[k].mean()), float(d[k].std(ddof=1))] for k in d}
    print(nm, {k: f"{v[0]:.3f}±{v[1]:.3f}" for k, v in cmp[nm].items()})

# 3. case studies -----------------------------------------------------------------
CASES = {"MAFENIDE": ("sulfonamide", "S(=O)(=O)N"),
         "CARBAMAZEPINE": ("dibenzazepine carboxamide", "O=C(N)N1c2ccccc2C=Cc2ccccc21"),
         "ALBUTEROL": ("catecholamine-like ethanolamine", "OCc1cc(C(O)CN)ccc1O"),
         "ACYCLOVIR": ("guanine (purine) base", "[#6]1~[#7]~[#6]2~[#6](~[#7]~1)~[#7]~[#6]~[#7]~2")}
rp = C.run_split(df, canon, "rich", seed=0, Zall=Zr, Xall=Xp, return_model=True)
ptp, pmp = rp["model"]
cases = {}
for key, (label, smarts) in CASES.items():
    hit = dft.index[dft.pref_name.fillna("").str.upper().str.contains(key)]
    if len(hit) == 0:
        continue
    i = int(hit[0]); smi = dft.smiles[i]
    patt = Chem.MolFromSmarts(smarts)
    rec = {"name": dft.pref_name[i], "expected": label}
    for nm, (p1, p2, XX, par) in {"paper_model_original_fp": (pt, pm, X, False),
                                  "parent_model_parent_fp": (ptp, pmp, Xp, True)}.items():
        mol = Chem.MolFromSmiles(C.parent_smiles(smi) if par else smi)
        pharm = set(a for match in mol.GetSubstructMatches(patt) for a in match)
        a = C.saliency(p1, p2, Zr[te][i], XX[te][i])
        info = C.bit_environments(smi, parent=par)
        order = [b for b in np.argsort(-np.abs(a)) if a[b] != 0][:6]
        bits = []
        for b in order:
            c, r = info[b][0]
            bits.append({"bit": int(b), "attr": float(a[b]), "env": C.env_smiles(mol, c, r)})
        top3_atoms = set().union(*[C.env_atoms(mol, *info[b][0]) for b in order[:3]])
        rec[nm] = {"top6": bits, "top3_atoms_overlap_pharmacophore": len(top3_atoms & pharm) > 0,
                   "pharmacophore_atoms_found": len(pharm) > 0}
    cases[key] = rec
    print(key, {k: [x["env"] for x in v["top6"]] if isinstance(v, dict) else v for k, v in rec.items()})

out = {"counterion_saliency_paper_model": part1, "retrieval_original_vs_parent_fp_10seeds": cmp,
       "fig3_cases": cases}
C.save_json(out, C.RESULTS / "e12_parent_structures.json")
L = [f"Share of salt rows whose top-1 salient bit is counterion-only: rich model {100*part1['top1_counterion_share_rich_model']:.0f}%, "
     f"same rich model with the name removed at inference {100*part1['top1_counterion_share_rich_model_name_removed_at_inference']:.0f}%, "
     f"no-drug model {100*part1['top1_counterion_share_nodrug_model']:.0f}%. "
     f"{part1['salt_rows_whose_drug_name_names_the_salt']}/{part1['test_rows_with_salt_or_counterion']} salt rows have the salt named in the drug name.", "",
     f"Paper checkpoint: {part1['test_rows_with_salt_or_counterion']}/435 test rows contain a salt or counterion; "
     f"top-1 salient bit is counterion-only in {part1['top1_bit_is_counterion_only']} of them, "
     f"a counterion-only bit is in the top 6 for {part1['any_top6_bit_counterion_only']}; "
     f"mean |attribution| mass on counterion-only bits {100*part1['mean_abs_attribution_mass_on_counterion_bits']:.1f}%.",
     "", "| Fingerprint (10 seeds) | R@1 | R@1 struct (unique SMILES) | R@1 struct (unique parent) | R@5 | MRR |", "|---|---|---|---|---|---|"]
for nm in cmp:
    L.append(f"| {nm} | " + " | ".join(f"{cmp[nm][k][0]:.3f} ± {cmp[nm][k][1]:.3f}" for k in ["R@1", "R@1_struct", "R@1_struct_parent", "R@5", "MRR"]) + " |")
L += ["", "| Fig. 3 case | expected | paper model top-3 envs | hits pharmacophore | parent model top-3 envs | hits pharmacophore |",
      "|---|---|---|---|---|---|"]
for k, r in cases.items():
    a, b = r["paper_model_original_fp"], r["parent_model_parent_fp"]
    L.append(f"| {r['name']} | {r['expected']} | {', '.join(x['env'] for x in a['top6'][:3])} | "
             f"{a['top3_atoms_overlap_pharmacophore']} | {', '.join(x['env'] for x in b['top6'][:3])} | "
             f"{b['top3_atoms_overlap_pharmacophore']} |")
open(C.RESULTS / "e12_parent_structures.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
