"""E23: PCdes benchmark (R3 W1/Q3) - the external benchmark missing from E6.

Data: KV-PLM's own files (Ret/align_smiles.txt, Ret/align_des_filt3.txt), 15,000 PubChem
molecule-description pairs. Split exactly as KV-PLM's Ret/preprocess.py does it: sequential
slices train 0:10500, dev 10500:12000, test 12000:15000, so the 3,000-molecule test gallery
matches the published numbers.

Two evaluation protocols, both reported in the literature:
  paragraph : the whole description is one query (one row per molecule).
  sentence  : KV-PLM's Ret/calcu_test.py protocol - the description is split on '.', sentences with
              fewer than 5 words are dropped, and a molecule's score is the mean cosine over its
              sentences. Text->molecule, which is the direction that script actually evaluates.

Configs: the paper's (ECFP4 2048, CLS, dim 256) and the E21 final one (parent 8192, mean-pooled,
dim 1024). Also reports scaffold overlap between train and test, metrics on the scaffold-novel test
subset, and how often a molecule's own name/synonym appears in its description (the leakage control
that motivates our benchmark).
"""



import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import core as C

RDLogger.DisableLog("rdApp.*")
C.MAX_TXT_LEN = 256
PC = "/home/cheriearjun/ext_data/pcdes"
N, TR, VA = 15000, 10500, 12000          # KV-PLM Ret/preprocess.py
CFGS = {"paper (2048, cls, 256)": dict(nbits=2048, parent=False, pool="cls", dim=256),
        "final (parent 8192, mean, 1024)": dict(nbits=8192, parent=True, pool="mean", dim=1024)}

smiles = [l.strip() for l in open(f"{PC}/align_smiles.txt")][:N]
corpus = [l.strip() for l in open(f"{PC}/align_des_filt3.txt")][:N]
assert len(smiles) == len(corpus) == N, (len(smiles), len(corpus))
df = pd.DataFrame({"smiles": smiles, "text": corpus})
df["ok"] = df.smiles.map(lambda s: Chem.MolFromSmiles(s) is not None)
n_bad = int((~df.ok).sum())
# Keep every row so the split indices stay KV-PLM's; unparseable SMILES get an all-zero fingerprint
# (C.fingerprint_matrix already does this), exactly as a failed lookup would behave.
print(f"unparseable SMILES: {n_bad}", flush=True)

idx = {"train": np.arange(0, TR), "val": np.arange(TR, VA), "test": np.arange(VA, N)}
tr, te = df.iloc[idx["train"]].reset_index(drop=True), df.iloc[idx["test"]].reset_index(drop=True)
sc_tr = set(tr.smiles.map(C.murcko).fillna(""))
novel = ~te.smiles.map(C.murcko).fillna("").isin(sc_tr).values


def sentences(text):
    return [s.strip() for s in text.split(".") if len(s.split()) >= 5]   # KV-PLM's filter


sent_te = [sentences(t) for t in te.text]
owner = np.repeat(np.arange(len(te)), [len(s) for s in sent_te])
flat = [s for ss in sent_te for s in ss]
n_nosent = int(sum(len(s) == 0 for s in sent_te))
print(f"test sentences: {len(flat)} over {len(te)} molecules; {n_nosent} molecules have none", flush=True)


def batch64_acc(S, n_rep=20, bs=64, seed=0):
    """KV-PLM's 'Acc': top-1 accuracy within randomly sampled mini-batches of 64 pairs, NOT the full
    gallery (KV-PLM Methods: 'the accuracy of the top retrieval result in randomly sampled mini-batches
    (64 pairs in each mini-batch)'). Their Recall@20 is the full-3k-gallery number instead."""
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    accs = []
    for _ in range(n_rep):
        perm = rng.permutation(n)
        for b in range(n // bs):
            idx = perm[b * bs:(b + 1) * bs]
            sub = S[np.ix_(idx, idx)]
            accs.append(np.mean(np.argmax(sub, axis=1) == np.arange(len(idx))))
    return float(np.mean(accs))


def rank_metrics(S, novel_mask):
    """S[q, m]: query q against molecule m; the correct molecule is on the diagonal."""
    d = np.diag(S)[:, None]
    ranks = 1 + np.sum(S > d, axis=1)
    g = {f"R@{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10, 20)}
    g["MRR"] = float(np.mean(1 / ranks))
    g["acc_batch64"] = batch64_acc(S)
    g["R@1_scaffold_novel"] = float(np.mean(ranks[novel_mask] <= 1))
    g["n_scaffold_novel"] = int(novel_mask.sum())
    return g


out = {"data": {"n_pairs": N, "n_train": len(tr), "n_test": len(te), "unparseable_smiles": n_bad,
                "test_scaffold_seen_in_train": float(1 - novel.mean()), "n_test_sentences": len(flat),
                "test_molecules_without_usable_sentence": n_nosent},
       "results": {}}

for name, cfg in CFGS.items():
    enc = C.encode_texts_meanpool if cfg["pool"] == "mean" else C.encode_texts
    Ztr = enc(tr.text.tolist()).numpy()
    Zte = enc(te.text.tolist()).numpy()
    Zsent = enc(flat).numpy()
    Xtr = C.fingerprint_matrix(tr.smiles.tolist(), nbits=cfg["nbits"], parent=cfg["parent"])
    Xte = C.fingerprint_matrix(te.smiles.tolist(), nbits=cfg["nbits"], parent=cfg["parent"])
    para, sent = [], []
    for s in range(3):
        pt, pm = C.train_bridge(Ztr, Xtr, np.arange(len(tr)), seed=s, dim=cfg["dim"])
        para.append(rank_metrics(C.similarity(pt, pm, Zte, Xte), novel))
        Ssent = C.similarity(pt, pm, Zsent, Xte)                       # [n_sentences, 3000]
        agg = np.zeros((len(te), len(te)))
        np.add.at(agg, owner, Ssent)
        cnt = np.bincount(owner, minlength=len(te))[:, None]
        has = cnt[:, 0] > 0
        agg[has] /= cnt[has]
        agg[~has] = -np.inf                                            # no sentence -> never retrieved
        sent.append(rank_metrics(agg, novel))
    out["results"][name] = {p: {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))]
                                for k in rs[0]} for p, rs in [("paragraph", para), ("sentence", sent)]}
    print(name, {p: round(v["R@1"][0], 3) for p, v in out["results"][name].items()}, flush=True)

# ---------------- are PCdes descriptions name-free? ----------------
# NOTE: KV-PLM's align_smdic.pkl maps to SMILES strings, not names, so it cannot be used to detect
# name leakage. What is checkable without an external name source: PCdes descriptions mostly open with
# a pronoun ("it is a ...", "it appears as ..."), i.e. the subject name was replaced when the set was built.
pron = te.text.str.lower().str.lstrip().str.startswith(("it ", "a ", "an ", "the "))
out["name_handling"] = {"share_starting_with_pronoun_or_article": float(pron.mean()),
                        "note": "PCdes descriptions appear to be built with the molecule name removed; "
                                "not quantified against a real name list, so treat as an observation, not a measurement"}
print("name handling:", out["name_handling"], flush=True)

C.save_json(out, C.RESULTS / "e23_pcdes.json")
d = out["data"]
L = [f"**PCdes** (KV-PLM files; {d['n_pairs']:,} pairs, split {d['n_train']:,} / 1,500 / {d['n_test']:,} exactly as "
     f"`Ret/preprocess.py`). {100*d['test_scaffold_seen_in_train']:.1f}% of test molecules have a Murcko scaffold seen in "
     f"training ({d['n_test'] - out['results'][list(CFGS)[0]]['paragraph']['n_scaffold_novel'][0]:.0f} seen / "
     f"{out['results'][list(CFGS)[0]]['paragraph']['n_scaffold_novel'][0]:.0f} novel). Unparseable SMILES: {d['unparseable_smiles']}. "
     f"Sentence protocol: {d['n_test_sentences']:,} sentences over {d['n_test']:,} molecules.", "",
     "Text → molecule, 3 seeds. R@k and MRR are over the full 3,000-molecule gallery; **Acc(64)** is "
     "KV-PLM's own metric, top-1 within randomly sampled 64-pair mini-batches, which is what the published "
     "'Acc' columns report (their Recall@20 is the full-gallery number).", "",
     "| Config | Protocol | Acc(64) | R@1 | R@5 | R@10 | R@20 | MRR | R@1 scaffold-novel |",
     "|---|---|---|---|---|---|---|---|---|"]
for name, v in out["results"].items():
    for p in ["paragraph", "sentence"]:
        r = v[p]
        L.append(f"| {name} | {p} | {100*r['acc_batch64'][0]:.2f} | {r['R@1'][0]:.3f} ± {r['R@1'][1]:.3f} | {r['R@5'][0]:.3f} | "
                 f"{r['R@10'][0]:.3f} | {100*r['R@20'][0]:.2f} | {r['MRR'][0]:.3f} | {r['R@1_scaffold_novel'][0]:.3f} |")
L += ["", "Published PCdes text→molecule numbers, all **fine-tuned on the PCdes training set** "
      "(KV-PLM Table 1 / MoMu Fig. 3a; Acc = 64-pair mini-batch, Rec@20 = full 3k gallery):", "",
      "| Model | sentence Acc(64) | sentence R@20 | paragraph Acc(64) | paragraph R@20 |", "|---|---|---|---|---|",
      "| Sci-BERT | 50.12 | 68.02 | 61.75 | 60.77 |", "| KV-PLM | 54.22 | 71.80 | 64.95 | 64.27 |",
      "| KV-PLM* | 55.61 | 74.77 | 77.03 | 75.47 |", "| MoMu-S | 55.44 | 76.92 | 80.22 | 79.02 |",
      "| MoMu-K | 54.94 | 78.29 | 81.45 | 80.62 |"]
L += ["", f"**Names:** {100*out['name_handling']['share_starting_with_pronoun_or_article']:.0f}% of test descriptions open with "
      "a pronoun or article ('it is a ...'), so PCdes was built with the molecule name removed. Not verified against a real "
      "name list — an observation, not a measurement."]
open(C.RESULTS / "e23_pcdes.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
