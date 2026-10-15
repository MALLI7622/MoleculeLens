"""E16: attribution for the strongest baseline (MolPrompt) on the same pharmacophore test (R3 Q9).

MolPrompt (fine-tuned on our scaffold train split, best checkpoint) is a Graphormer-style graph
encoder.  Atom attribution = gradient x input at the atom-embedding layer (graph_node_feature),
for the score s = cos(text_i, graph_i) using MolPrompt's own stored test text embedding; summed over
the hidden dimension.  Scored with exactly the same 16 pharmacophore classes / metrics as E13, with
the wrong-text control (20 texts from other classes) and chance.  Also reports cost per explanation.

Run from ~/MolPrompt (its imports are relative to that directory).
"""
import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

sys.path.insert(0, "/home/cheriearjun/MolPrompt")
os.chdir("/home/cheriearjun/MolPrompt")
import main_chembl_retrieval as M  # noqa: E402
from data_provider.collator_0313 import collator  # noqa: E402
from model.contrastive_GraST_gp import GraSTSimclr  # noqa: E402

sys.path.insert(0, "/home/cheriearjun/MoleculeLens/nmi")
import core as C  # noqa: E402
from pharm import CLASSES, classify, pharm_atoms, score, top3_chance  # noqa: E402

sys.argv = ["x"]
args = M.parse_args()
args.split_manifest = str(C.CANON_MANIFEST)
dataset, _, _, _, test_loader = M.prepare_loaders(args)
test_ds = test_loader.dataset
dev = torch.device("cuda")
model = GraSTSimclr(temperature=args.temperature, drop_ratio=args.drop_ratio, graph_hidden_dim=args.graph_hidden_dim,
                    bert_hidden_dim=args.bert_hidden_dim, projection_dim=args.projection_dim, lr=args.lr,
                    weight_decay=args.weight_decay).to(dev)
ck = torch.load("save_model/retrieval_chembl_sharedsplit_bs6/chembl_retrieval_best.pth", map_location=dev)
model.load_state_dict(ck.get("model_state_dict", ck))
model.eval()
npz = np.load("save_model/retrieval_chembl_sharedsplit_bs6/chembl_retrieval_best_metrics.npz")
T = F.normalize(torch.tensor(npz["test_text"]), dim=1).to(dev)

df = C.load_df()
te = np.array(C.canonical_split()["test"])
dft = df.iloc[te].reset_index(drop=True)
cls_of = classify((dft.mechanism_of_action.fillna("") + " " + dft.target_name.fillna("")).values)

store = {}
def _hook(m, i, o):
    o.retain_grad()
    store["emb"] = o          # must return None, otherwise the output is replaced


model.graph_encoder.graph_node_feature.register_forward_hook(_hook)
coll = partial(collator, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20)


def atom_attr(i, text_vec):
    batch = coll([test_ds[i]])
    model.zero_grad()
    _, g = model.encode_batch(batch)
    s = F.normalize(g, dim=1)[0] @ text_vec
    s.backward()
    e = store["emb"]
    a = (e.grad * e).sum(-1)[0].detach().cpu().numpy()
    return a[1:]                      # position 0 is the virtual graph token


rng = np.random.default_rng(0)
recs, n_mismatch, t_expl = [], 0, []
for i in range(len(dft)):
    k = cls_of[i]
    if k < 0:
        continue
    mol, pharm = pharm_atoms(dft.smiles[i], k)
    if not pharm:
        continue
    t0 = time.time()
    a = atom_attr(i, T[i])
    t_expl.append(time.time() - t0)
    n = mol.GetNumAtoms()
    if len(a) < n:
        n_mismatch += 1
        continue
    a = a[:n]
    r = {"row": i, "class": CLASSES[k][0], "drug": dft.pref_name[i], "chance": len(pharm) / n,
         "top3_chance": top3_chance(n, len(pharm))}
    r["mass"], r["top1"], r["top3"] = score(a, pharm)
    others = np.where(cls_of != k)[0]
    w = [score(atom_attr(i, T[j])[:n], pharm) for j in rng.choice(others, 20, replace=False)]
    r["mass_wrongtext"], r["top1_wrongtext"], r["top3_wrongtext"] = [float(np.mean(x)) for x in zip(*w)]
    recs.append(r)
d = pd.DataFrame(recs)


def p(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return None if len(diff) < 5 or np.all(diff == 0) else float(wilcoxon(a, b, alternative="greater").pvalue)


summ = {nm: {"n_pairs": len(s), **{c: float(s[c].mean()) for c in ["mass", "mass_wrongtext", "chance", "top1",
                                                                    "top1_wrongtext", "top3", "top3_wrongtext", "top3_chance"]},
             "p_mass_gt_chance": p(s["mass"], s["chance"]), "p_mass_gt_wrongtext": p(s["mass"], s["mass_wrongtext"])}
        for nm, s in list(d.groupby("class")) + [("ALL", d)]}
out = {"summary": summ, "per_pair": recs, "atom_count_mismatches": n_mismatch,
       "seconds_per_explanation_molprompt": float(np.mean(t_expl))}
C.save_json(out, C.RESULTS / "e16_molprompt_attribution.json")
fp = lambda v: "-" if v is None else (f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")  # noqa: E731
L = [f"MolPrompt gradient x input at atom embeddings; {len(d)} scored pairs; {fp(out['seconds_per_explanation_molprompt'])} s per explanation "
     f"(atom-order mismatches skipped: {n_mismatch}).", "",
     "| Class | n | mass: correct / wrong text / chance | top-1: correct / wrong / chance | top-3: correct / wrong / chance | p (> chance) | p (correct > wrong) |",
     "|---|---|---|---|---|---|---|"]
for nm, s in sorted(summ.items(), key=lambda x: (x[0] == "ALL", -x[1]["n_pairs"])):
    L.append(f"| {'**ALL**' if nm == 'ALL' else nm} | {s['n_pairs']} | {s['mass']:.2f} / {s['mass_wrongtext']:.2f} / {s['chance']:.2f} | "
             f"{s['top1']:.2f} / {s['top1_wrongtext']:.2f} / {s['chance']:.2f} | {s['top3']:.2f} / {s['top3_wrongtext']:.2f} / {s['top3_chance']:.2f} | "
             f"{fp(s['p_mass_gt_chance'])} | {fp(s['p_mass_gt_wrongtext'])} |")
open(C.RESULTS / "e16_molprompt_attribution.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
