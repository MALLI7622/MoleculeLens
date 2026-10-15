"""E00: check that nmi/core.py reproduces the paper pipeline.

1. Saved paper checkpoint (outputs/proj_*.pt) + core features -> must give R@1 0.225 on the canonical test.
2. Vectorised loss == original loop loss on a random batch.
3. Retrain with core.train_bridge (seed 0) -> should land near 0.225.
"""
import importlib.util
import numpy as np
import torch

import core as C

df = C.load_df()
sp = C.canonical_split()
te = np.array(sp["test"])
Z = C.encode_texts(C.text_variant(df, "rich"), all_layers=True)       # cache all layers once
Zf = Z[:, 12, :].numpy()
X = C.fingerprint_matrix(df["smiles"].tolist())
out = {}

# 1. saved checkpoint
pt = torch.nn.Linear(768, 256); pm = torch.nn.Linear(2048, 256)
pt.load_state_dict(torch.load(C.ROOT / "outputs/proj_text.pt", map_location="cpu"))
pm.load_state_dict(torch.load(C.ROOT / "outputs/proj_mol.pt", map_location="cpu"))
m = C.strip(C.metrics(C.similarity(pt.eval(), pm.eval(), Zf[te], X[te])))
print("saved checkpoint:", m)
out["saved_checkpoint"] = m

# also the saved no-drug checkpoint with no-drug text
Znd = C.encode_texts(C.text_variant(df, "nodrug"), all_layers=True)[:, 12, :].numpy()
pt.load_state_dict(torch.load(C.ROOT / "outputs/proj_text_nodrug.pt", map_location="cpu"))
pm.load_state_dict(torch.load(C.ROOT / "outputs/proj_mol_nodrug.pt", map_location="cpu"))
m = C.strip(C.metrics(C.similarity(pt.eval(), pm.eval(), Znd[te], X[te])))
print("saved nodrug checkpoint:", m)
out["saved_checkpoint_nodrug"] = m

# 2. loss equivalence
spec = importlib.util.spec_from_file_location("orig", C.ROOT / "03_train_scaffold_split.py")
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
torch.manual_seed(1)
bt, bm = torch.randn(64, 256), torch.randn(64, 256)
grp = [str(g) for g in np.random.default_rng(0).integers(0, 10, 64)]
l_orig, _ = orig.contrastive_loss_with_weights(bt, bm, grp, "cpu", T=0.07, margin=0.1)
l_new = C.full_loss(bt, bm, C.group_ids(grp), 0.07, 0.1)
print("loss orig vs vectorised:", float(l_orig), float(l_new))
out["loss_equivalence"] = {"orig": float(l_orig), "vectorised": float(l_new),
                           "abs_diff": abs(float(l_orig) - float(l_new))}

# 3. retrain
t = C.Timer()
with t("train_seed0"):
    r = C.run_split(df, sp, "rich", seed=0, Zall=Zf, Xall=X)
print("retrained seed0:", r["metrics"], f"{t.t['train_seed0']:.1f}s")
out["retrained_seed0"] = r["metrics"]
out["train_seconds"] = t.t["train_seed0"]
out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
C.save_json(out, C.RESULTS / "e00_sanity.json")
