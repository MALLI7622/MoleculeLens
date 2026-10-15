"""E14: text-side logit lens done properly (R3 W2#4, Q10).

The paper applied the layer-12 head unchanged to earlier layers.  Here a SEPARATE linear head
(and molecule head) is trained on each layer's CLS embedding, canonical split, 3 init seeds,
rich and no-drug text.  Also reported: the paper's reused-head numbers for comparison, and a
mean-pooled variant (CLS of early layers is not a sentence summary).
"""
import numpy as np
import pandas as pd
import torch

import core as C

df = C.load_df()
canon = C.canonical_split()
X = C.fingerprint_matrix(df["smiles"].tolist())
te = np.array(canon["test"])
rows = []
for v in ["rich", "nodrug"]:
    H = C.encode_texts(C.text_variant(df, v), all_layers=True).numpy()       # [N,13,768]
    # paper-style reused head (layer-12 head applied to every layer)
    r12 = C.run_split(df, canon, v, seed=0, Zall=H[:, 12, :], Xall=X, return_model=True)
    pt12, pm12 = r12["model"]
    for L in range(13):
        reused = C.metrics(C.similarity(pt12, pm12, H[te, L, :], X[te]))
        per = [C.run_split(df, canon, v, seed=s, Zall=H[:, L, :], Xall=X)["metrics"] for s in range(3)]
        rows.append({"text": v, "layer": L,
                     "R@1_reused_L12_head": float(np.mean(reused["ranks"] == 1)),
                     "R@1_own_head_mean": float(np.mean([p["R@1"] for p in per])),
                     "R@1_own_head_std": float(np.std([p["R@1"] for p in per], ddof=1)),
                     "R@1_struct_own_head": float(np.mean([p["R@1_struct"] for p in per])),
                     "MRR_own_head": float(np.mean([p["MRR"] for p in per]))})
        print(v, L, f"reused {rows[-1]['R@1_reused_L12_head']:.3f}  own {rows[-1]['R@1_own_head_mean']:.3f}")

# mean-pooled token states at each layer (rich only), own head
from transformers import AutoModel, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
enc = AutoModel.from_pretrained(C.TEXT_MODEL).to(C.DEVICE).eval()
texts = C.text_variant(df, "rich")
P = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=C.MAX_TXT_LEN, return_tensors="pt").to(C.DEVICE)
        hs = enc(**b, output_hidden_states=True).hidden_states
        m = b["attention_mask"].unsqueeze(-1).float()
        P.append(torch.stack([(h * m).sum(1) / m.sum(1) for h in hs], 1).cpu())
P = torch.cat(P).numpy()
mp = []
for L in range(13):
    per = [C.run_split(df, canon, "rich", seed=s, Zall=P[:, L, :], Xall=X)["metrics"]["R@1"] for s in range(3)]
    mp.append({"layer": L, "R@1_meanpool_own_head": float(np.mean(per))})
C.save_json({"cls": rows, "meanpool_rich": mp}, C.RESULTS / "e14_per_layer_heads.json")
d = pd.DataFrame(rows)
Ls = ["| Layer | rich: reused L12 head (paper) | rich: own head | rich: own head, mean-pool | no-drug: reused | no-drug: own head |",
      "|---|---|---|---|---|---|"]
for L in range(13):
    a = d[(d.text == "rich") & (d.layer == L)].iloc[0]; b = d[(d.text == "nodrug") & (d.layer == L)].iloc[0]
    Ls.append(f"| {L} | {a['R@1_reused_L12_head']:.3f} | {a['R@1_own_head_mean']:.3f} ± {a['R@1_own_head_std']:.3f} | "
              f"{mp[L]['R@1_meanpool_own_head']:.3f} | {b['R@1_reused_L12_head']:.3f} | "
              f"{b['R@1_own_head_mean']:.3f} ± {b['R@1_own_head_std']:.3f} |")
open(C.RESULTS / "e14_per_layer_heads.md", "w").write("\n".join(Ls) + "\n")
print("\n".join(Ls))
