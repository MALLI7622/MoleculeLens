"""
Track 3 raw analysis: exact-replay contrastive logit lens.

This script re-encodes the test texts through the frozen encoder, applies the
saved projection heads at every RoBERTa depth, and writes raw Track 3 outputs
under outputs/track3/. Paper-facing figures are generated separately by
10b_logit_lens_figures.py so each publication artifact has a single writer.
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

from paper_eval_utils import DEFAULT_TEXT_MAX_LENGTH, diagonal_metric_summary

os.makedirs("outputs/track3", exist_ok=True)
# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── load saved tensors ───────────────────────────────────────────────────
print("Loading saved tensors...")
Wt_state = torch.load("outputs/proj_text.pt",        map_location="cpu")
Wt_nd_state = torch.load("outputs/proj_text_nodrug.pt", map_location="cpu")
Bm       = torch.load("outputs/Bm_test.pt",          map_location="cpu")   # [435, 256]

W_t = Wt_state["weight"]   # [256, 768]
b_t = Wt_state["bias"]     # [256]
W_t_nd = Wt_nd_state["weight"]
b_t_nd = Wt_nd_state["bias"]

df       = pd.read_csv("outputs/test_df.csv")
df_nd    = pd.read_csv("outputs/test_df_nodrug.csv")
N        = len(df)

texts_rich   = df["text_rich"].tolist()
texts_nodrug = df_nd["text_rich"].tolist()

print(f"  W_t: {W_t.shape}  W_t_nodrug: {W_t_nd.shape}  Bm: {Bm.shape}  N={N}")

# ── load RoBERTa ─────────────────────────────────────────────────────────
print("Loading frozen S-Biomed-RoBERTa...")
MODEL_NAME = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
MAX_TXT_LEN = DEFAULT_TEXT_MAX_LENGTH
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
roberta    = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
roberta.eval()
N_LAYERS = 13   # embedding(0) + 12 transformer layers

# ── encoder ──────────────────────────────────────────────────────────────
def encode_all_layers(texts, batch_size=64, desc=""):
    """
    Returns CLS hidden states at all 13 positions.
    Shape: [N, 13, 768]
    """
    all_cls = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        enc = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_TXT_LEN
        )
        with torch.no_grad():
            out = roberta(**enc, output_hidden_states=True)
        # hidden_states: tuple of 13 tensors, each [B, seq_len, 768]
        # CLS = index 0
        cls_stack = torch.stack(
            [h[:, 0, :] for h in out.hidden_states], dim=1
        )  # [B, 13, 768]
        all_cls.append(cls_stack)
        if desc:
            done = min(start + batch_size, len(texts))
            print(f"  {desc}: {done}/{len(texts)}", end="\r")
    if desc:
        print()
    return torch.cat(all_cls, dim=0)   # [N, 13, 768]


def project_layer(H_layer, w_t, b_t_vec):
    """Project one layer's CLS states via W_t, L2-normalize. → [N, 256]"""
    # H_layer: [N, 768]
    proj = H_layer @ w_t.T + b_t_vec   # [N, 256]
    return F.normalize(proj, dim=-1)


def retrieval_metrics(Bt_proj, Bm_gallery):
    sims = (Bt_proj @ Bm_gallery.T).cpu().numpy()
    metrics = diagonal_metric_summary(sims, ks=(1, 5, 10))
    hits_r1 = torch.from_numpy((metrics["ranks"] == 1).astype(np.int64))
    return metrics, hits_r1


# ── Step 1: encode all texts at all layers ───────────────────────────────
print("\nEncoding text_rich at all layers...")
H_rich   = encode_all_layers(texts_rich,   desc="rich")    # [435, 13, 768]
print("Encoding text_nodrug at all layers...")
H_nodrug = encode_all_layers(texts_nodrug, desc="nodrug")  # [435, 13, 768]

torch.save(H_rich,   "outputs/track3/H_rich_all_layers.pt")
torch.save(H_nodrug, "outputs/track3/H_nodrug_all_layers.pt")
print("Saved layer-wise CLS representations.")

bt_saved_rich = torch.load("outputs/Bt_test.pt", map_location="cpu")
bt_saved_nodrug = torch.load("outputs/Bt_test_nodrug.pt", map_location="cpu")
layer12_rich = project_layer(H_rich[:, 12, :], W_t, b_t)
layer12_nodrug = project_layer(H_nodrug[:, 12, :], W_t_nd, b_t_nd)
print(
    "Layer-12 embedding max abs diff vs saved outputs:",
    f"rich={float((layer12_rich - bt_saved_rich).abs().max()):.6g}",
    f"nodrug={float((layer12_nodrug - bt_saved_nodrug).abs().max()):.6g}",
)

# ── Step 2: recall curve for both conditions ─────────────────────────────
print("\nComputing Recall@1 / MRR per layer...")

results = {cond: {"r1": [], "r5": [], "r10": [], "mrr": [], "hits_r1": []}
           for cond in ["rich", "nodrug"]}

for layer in range(N_LAYERS):
    for cond, H, w_t_mat, b_t_vec, gallery in [
        ("rich", H_rich, W_t, b_t, Bm),
        # Match the canonical paper leakage ablation by holding the rich
        # molecule gallery fixed and changing only the text condition.
        ("nodrug", H_nodrug, W_t_nd, b_t_nd, Bm),
    ]:
        Bt_l = project_layer(H[:, layer, :], w_t_mat, b_t_vec)
        metrics, hits = retrieval_metrics(Bt_l, gallery)
        results[cond]["r1"].append(metrics["Recall@1"])
        results[cond]["r5"].append(metrics["Recall@5"])
        results[cond]["r10"].append(metrics["Recall@10"])
        results[cond]["mrr"].append(metrics["MRR"])
        results[cond]["hits_r1"].append(hits)
    print(f"  Layer {layer:2d}  R@1 rich={results['rich']['r1'][-1]:.3f}  "
          f"nodrug={results['nodrug']['r1'][-1]:.3f}  "
          f"gap={results['rich']['r1'][-1]-results['nodrug']['r1'][-1]:.3f}")

# save numeric results
rows = []
for layer in range(N_LAYERS):
    for cond in ["rich", "nodrug"]:
        rows.append({
            "layer": layer, "condition": cond,
            "recall_1":  results[cond]["r1"][layer],
            "recall_5":  results[cond]["r5"][layer],
            "recall_10": results[cond]["r10"][layer],
            "mrr":       results[cond]["mrr"][layer],
        })
pd.DataFrame(rows).to_csv("outputs/track3/recall_by_layer.csv", index=False)
print("\nSaved: outputs/track3/recall_by_layer.csv")

# find emergence and divergence layers
gap = [results["rich"]["r1"][l] - results["nodrug"]["r1"][l] for l in range(N_LAYERS)]
emergence_layer = int(np.argmax(results["rich"]["r1"]))        # peak R@1 layer
divergence_layer = int(np.argmax(gap))                          # max gap layer
first_sig = next((l for l in range(1, N_LAYERS) if gap[l] > 0.03), divergence_layer)

print(f"\nKey layers:")
print(f"  Peak R@1 (text_rich):  layer {emergence_layer}  "
      f"R@1={results['rich']['r1'][emergence_layer]:.3f}")
print(f"  Peak R@1 (text_nodrug): layer {int(np.argmax(results['nodrug']['r1']))}  "
      f"R@1={results['nodrug']['r1'][int(np.argmax(results['nodrug']['r1']))]:.3f}")
print(f"  Max leakage gap:       layer {divergence_layer}  "
      f"gap={gap[divergence_layer]:.3f}")
print(f"  First significant gap (>0.03): layer {first_sig}")

# ── Step 3: per-target-family analysis ──────────────────────────────────
print("\nAnalysing per-target-family recall curves...")

def assign_family(row):
    moa = str(row["mechanism_of_action"]).lower()
    tgt = str(row["target_name"]).lower()
    if "kinase" in moa or "kinase" in tgt:               return "Kinase"
    if any(x in moa for x in ["adrenergic", "opioid", "dopamine",
                               "serotonin", "muscarinic", "gpcr",
                               "receptor agonist", "receptor antagonist"]):
        if "nuclear" not in moa and "glucocorticoid" not in moa \
           and "androgen" not in moa and "retinoid" not in moa:
            return "GPCR"
    if any(x in moa for x in ["channel", "blocker"]):   return "Ion Channel"
    if any(x in moa for x in ["glucocorticoid", "androgen", "retinoid",
                               "nuclear receptor", "estrogen"]):
        return "Nuclear Receptor"
    if any(x in moa for x in ["ribosome", "penicillin", "dihydropteroate",
                               "gyrase", "topoisomerase", "bacterial"]):
        return "Antibacterial"
    if any(x in moa for x in ["herpesvirus", "viral", "hiv", "polymerase inh",
                               "reverse transcriptase"]):
        return "Antiviral"
    if "cyclooxygenase" in moa or "cox" in moa:         return "COX Inhibitor"
    if "diagnostic" in moa:                              return "Diagnostic"
    return "Other"

df["family"] = df.apply(assign_family, axis=1)
family_counts = df["family"].value_counts()
print("  Target family distribution:")
for fam, cnt in family_counts.items():
    print(f"    {fam}: {cnt}")

# per-family recall curves (rich only, top families)
top_families = family_counts[family_counts >= 10].index.tolist()
family_curves = {}
for fam in top_families:
    idx = df[df["family"] == fam].index.tolist()
    curve_r1 = []
    for layer in range(N_LAYERS):
        Bt_l = project_layer(H_rich[idx, layer, :], W_t, b_t)
        Bm_fam = Bm[idx]
        sims = Bt_l @ Bm_fam.T
        curve_r1.append(diagonal_metric_summary(sims, ks=(1,))["Recall@1"])
    family_curves[fam] = curve_r1
    peak_l = int(np.argmax(curve_r1))
    print(f"  {fam:20s}  n={len(idx):3d}  "
          f"peak layer={peak_l}  peak R@1={curve_r1[peak_l]:.3f}")

# save
pd.DataFrame(family_curves, index=range(N_LAYERS)).to_csv(
    "outputs/track3/recall_by_layer_family.csv")
print("Saved: outputs/track3/recall_by_layer_family.csv")

# ── Step 4: correct vs incorrect lock-in layer ───────────────────────────
print("\nAnalysing lock-in layer for correct vs incorrect retrievals...")

# For each drug, find the first layer where it's correctly retrieved
# and compare between final-rank-1 (correct) vs final-rank≥2 (incorrect)
final_hits = results["rich"]["hits_r1"][-1]   # [N] bool at layer 12

lockein_correct   = []
lockein_incorrect = []

for i in range(N):
    first_hit = None
    for layer in range(N_LAYERS):
        if results["rich"]["hits_r1"][layer][i].item():
            first_hit = layer
            break
    if final_hits[i].item():
        if first_hit is not None:
            lockein_correct.append(first_hit)
    else:
        if first_hit is not None:
            lockein_incorrect.append(first_hit)
        else:
            lockein_incorrect.append(N_LAYERS)   # never correct

print(f"  Correct at R@1 (layer 12): {final_hits.sum().item()}/{N}")
if lockein_correct:
    print(f"  Mean lock-in layer (correct):   {np.mean(lockein_correct):.1f}")
if lockein_incorrect:
    print(f"  Mean first-hit layer (incorrect): {np.mean([x for x in lockein_incorrect if x < N_LAYERS]):.1f}")

# ── Step 5: attention analysis at divergence layer ───────────────────────
print(f"\nToken-level attention at divergence layer {divergence_layer}...")

# Re-run with output_attentions=True for a subset of texts
# We look at which tokens CLS attends to in the divergence layer
roberta_attn = AutoModel.from_pretrained(MODEL_NAME,
                                         output_hidden_states=True,
                                         output_attentions=True)
roberta_attn.eval()

# Sample: first 8 correctly retrieved text_rich texts
correct_r1_idx = [i for i in range(N) if final_hits[i].item()][:8]
sample_texts_rich   = [texts_rich[i]   for i in correct_r1_idx]
sample_texts_nodrug = [texts_nodrug[i] for i in correct_r1_idx]

def get_cls_attention_at_layer(texts, layer_idx):
    """Returns (tokens, attn_weights) for each text in batch.
    attn_weights: mean over heads, CLS row → shape [seq_len] per sample."""
    enc = tokenizer(texts, return_tensors="pt",
                    padding=True, truncation=True, max_length=MAX_TXT_LEN)
    with torch.no_grad():
        out = roberta_attn(**enc, output_attentions=True)
    # attentions: tuple of N_layers × [B, n_heads, seq, seq]
    attn = out.attentions[layer_idx - 1]   # layer 0 is embedding, attn starts at 1
    # mean over heads, CLS row (index 0)
    cls_attn = attn.mean(dim=1)[:, 0, :]  # [B, seq_len]
    input_ids = enc["input_ids"]
    tokens_batch = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in input_ids]
    return tokens_batch, cls_attn, enc["attention_mask"]

# only do attention at layers 1..12
attn_layer = max(1, divergence_layer)
tokens_r,  attn_r,  mask_r  = get_cls_attention_at_layer(sample_texts_rich,   attn_layer)
tokens_nd, attn_nd, mask_nd = get_cls_attention_at_layer(sample_texts_nodrug, attn_layer)

# For each example, find top-5 tokens by CLS attention
attn_results = []
for k, idx in enumerate(correct_r1_idx[:4]):   # first 4 for display
    for cond, toks, attn_w, mask in [
        ("rich",   tokens_r[k],  attn_r[k],  mask_r[k]),
        ("nodrug", tokens_nd[k], attn_nd[k], mask_nd[k]),
    ]:
        seq_len = mask.sum().item()
        toks_valid = toks[:seq_len]
        attn_valid = attn_w[:seq_len].numpy()
        top5_idx   = np.argsort(attn_valid)[-5:][::-1]
        top5_toks  = [(toks_valid[j], float(attn_valid[j])) for j in top5_idx]
        attn_results.append({
            "drug_idx": idx,
            "drug_name": df.iloc[idx]["pref_name"],
            "condition": cond,
            "layer": attn_layer,
            "top5_tokens": str(top5_toks),
        })
        print(f"  [{idx}] {df.iloc[idx]['pref_name'][:20]:20s} {cond:6s} "
              f"layer {attn_layer}  top-5: "
              + "  ".join(f"{t}({w:.3f})" for t, w in top5_toks[:5]))

pd.DataFrame(attn_results).to_csv("outputs/track3/attention_analysis.csv", index=False)
print("Saved: outputs/track3/attention_analysis.csv")

with open("outputs/track3/run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_name": MODEL_NAME,
            "max_text_length": MAX_TXT_LEN,
            "evaluation": "diagonal_optimistic",
            "gallery_rich": "Bm_test.pt",
            "gallery_nodrug": "Bm_test.pt",
            "leakage_semantics": "fixed_rich_gallery_text_ablation",
            "layer12_max_abs_diff_rich": float((layer12_rich - bt_saved_rich).abs().max()),
            "layer12_max_abs_diff_nodrug": float((layer12_nodrug - bt_saved_nodrug).abs().max()),
        },
        f,
        indent=2,
    )
print("Saved: outputs/track3/run_metadata.json")


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════

layers = list(range(N_LAYERS))
layer_labels = ["embed"] + [str(l) for l in range(1, N_LAYERS)]

RICH_COL   = "#2166ac"
NODRUG_COL = "#d6604d"
FAM_COLS   = plt.cm.Set2(np.linspace(0, 1, 8))

# ── Figure A: main layer-emergence panel ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel 1: R@1 curve
ax = axes[0]
ax.plot(layers, results["rich"]["r1"],   "o-", color=RICH_COL,   lw=2,
        ms=5, label="text\_rich", zorder=3)
ax.plot(layers, results["nodrug"]["r1"], "s--", color=NODRUG_COL, lw=2,
        ms=5, label="text\_nodrug", zorder=3)
ax.fill_between(layers,
                results["nodrug"]["r1"], results["rich"]["r1"],
                alpha=0.15, color="#888888", label="leakage gap")
ax.axvline(first_sig, color="#555555", lw=1, ls=":", alpha=0.8,
           label=f"first gap > 0.03\n(layer {first_sig})")
ax.axvline(emergence_layer, color=RICH_COL, lw=1, ls="--", alpha=0.6,
           label=f"peak rich\n(layer {emergence_layer})")
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels, rotation=45)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel("Recall@1")
ax.set_title("(A) Layer-emergence curve\nRecall@1 vs RoBERTa depth")
ax.legend(fontsize=6.5, frameon=False, loc="upper left")

# Panel 2: MRR curve
ax = axes[1]
ax.plot(layers, results["rich"]["mrr"],   "o-", color=RICH_COL,   lw=2, ms=5,
        label="text\_rich")
ax.plot(layers, results["nodrug"]["mrr"], "s--", color=NODRUG_COL, lw=2, ms=5,
        label="text\_nodrug")
ax.fill_between(layers,
                results["nodrug"]["mrr"], results["rich"]["mrr"],
                alpha=0.15, color="#888888")
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels, rotation=45)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel("MRR")
ax.set_title("(B) MRR vs RoBERTa depth")
ax.legend(fontsize=6.5, frameon=False)

# Panel 3: per-family R@1 at each layer (top families only)
ax = axes[2]
family_subset = [f for f in top_families if family_counts[f] >= 15][:6]
for fi, fam in enumerate(family_subset):
    ax.plot(layers, family_curves[fam], "o-", color=FAM_COLS[fi],
            lw=1.5, ms=4, label=f"{fam} (n={family_counts[fam]})")
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels, rotation=45)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel("Recall@1")
ax.set_title("(C) Per-mechanism-family\nlayer-emergence (text\_rich)")
ax.legend(fontsize=6, frameon=False, ncol=1)

plt.suptitle(
    "Track 3 — Contrastive Logit Lens: how drug-retrievable information builds across RoBERTa layers",
    fontsize=10, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("outputs/track3/layer_emergence_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/track3/layer_emergence_curves.png")


# ── Figure B: raw summary figure ─────────────────────────────────────────
fig2 = plt.figure(figsize=(13, 4.5))
gs = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.38)

# ---- Panel A: R@1 + gap annotation ----
ax = fig2.add_subplot(gs[0])
ax.plot(layers, results["rich"]["r1"],   "o-", color=RICH_COL,   lw=2.2,
        ms=5.5, label="text\\_rich", zorder=4)
ax.plot(layers, results["nodrug"]["r1"], "s--", color=NODRUG_COL, lw=2.2,
        ms=5.5, label="text\\_nodrug", zorder=4)
ax.fill_between(layers,
                results["nodrug"]["r1"], results["rich"]["r1"],
                alpha=0.18, color="#aaaaaa", label="leakage gap")
# annotate divergence
ax.annotate(
    f"gap first\n>0.03\nlayer {first_sig}",
    xy=(first_sig, results["rich"]["r1"][first_sig]),
    xytext=(first_sig + 1.5, results["rich"]["r1"][first_sig] + 0.025),
    fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
    color="#333333"
)
# annotate peak
ax.annotate(
    f"peak\nlayer {emergence_layer}\nR@1={results['rich']['r1'][emergence_layer]:.3f}",
    xy=(emergence_layer, results["rich"]["r1"][emergence_layer]),
    xytext=(emergence_layer - 3.5, results["rich"]["r1"][emergence_layer] + 0.02),
    fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8, color=RICH_COL),
    color=RICH_COL
)
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel("Recall@1")
ax.set_title("(A) Layer-emergence curve", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

# ---- Panel B: leakage gap per layer ----
ax = fig2.add_subplot(gs[1])
gap_arr = np.array(gap)
colours = ["#e45756" if g > 0.03 else "#aec7e8" for g in gap_arr]
ax.bar(layers, gap_arr, color=colours, edgecolor="none", width=0.7)
ax.axhline(0.03, color="#555555", lw=1, ls="--", label="threshold = 0.03")
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel("R@1 gap (rich − nodrug)")
ax.set_title("(B) Leakage gap per layer\n(red = significant gap > 0.03)", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

# ---- Panel C: correct lock-in vs. incorrect ----
ax = fig2.add_subplot(gs[2])
bins = range(0, N_LAYERS + 1)
if lockein_correct:
    ax.hist(lockein_correct, bins=bins, alpha=0.75, color=RICH_COL,
            label=f"correct at R@1 (n={len(lockein_correct)})",
            density=True, edgecolor="white", lw=0.4)
inc_vals = [x for x in lockein_incorrect if x < N_LAYERS]
if inc_vals:
    ax.hist(inc_vals, bins=bins, alpha=0.65, color=NODRUG_COL,
            label=f"never correct (first hit, n={len(inc_vals)})",
            density=True, edgecolor="white", lw=0.4)
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel("Density")
ax.set_title("(C) First correct retrieval layer\n(correct vs. incorrect pairs)", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

fig2.suptitle(
    "Track 3 raw summary — Contrastive Logit Lens",
    fontsize=9.5, fontweight="bold", y=1.01
)
fig2.savefig("outputs/track3/fig_logit_lens_raw.pdf",
             bbox_inches="tight", format="pdf")
fig2.savefig("outputs/track3/fig_logit_lens_raw.png",
             bbox_inches="tight", dpi=200)
plt.close(fig2)
print("Saved: outputs/track3/fig_logit_lens_raw.pdf/.png")


# ── Figure C: raw family heatmap ─────────────────────────────────────────
if len(family_subset) >= 3:
    fam_matrix = np.array([family_curves[f] for f in family_subset])
    fig3, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(fam_matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=max(fam_matrix.max(), 0.3))
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=7)
    ax.set_yticks(range(len(family_subset)))
    ax.set_yticklabels(
        [f"{f} (n={family_counts[f]})" for f in family_subset], fontsize=8
    )
    ax.set_xlabel("RoBERTa layer")
    plt.colorbar(im, ax=ax, label="Recall@1", shrink=0.8)
    ax.set_title(
        "Per-mechanism-family Recall@1 across RoBERTa layers (text\_rich)\n"
        "Earlier peak = mechanism information encoded in shallower layers",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig("outputs/track3/family_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig("outputs/track3/fig_logit_lens_family_raw.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print("Saved: outputs/track3/family_heatmap.png + fig_logit_lens_family_raw.png")


# ── Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRACK 3 COMPLETE — Contrastive Logit Lens")
print("=" * 60)
print(f"  N test pairs:     {N}")
print(f"  Layers analysed:  {N_LAYERS}  (embed + 12 transformer)")
print()
print(f"  text_rich peak:   layer {emergence_layer}  "
      f"R@1={results['rich']['r1'][emergence_layer]:.3f}  "
      f"MRR={results['rich']['mrr'][emergence_layer]:.3f}")
ndpeak = int(np.argmax(results['nodrug']['r1']))
print(f"  text_nodrug peak: layer {ndpeak}  "
      f"R@1={results['nodrug']['r1'][ndpeak]:.3f}  "
      f"MRR={results['nodrug']['mrr'][ndpeak]:.3f}")
print(f"  Max leakage gap:  layer {divergence_layer}  "
      f"gap={gap[divergence_layer]:.3f}")
print(f"  First sig. gap:   layer {first_sig}  (>0.03 threshold)")
print()
for fam in family_subset:
    pk = int(np.argmax(family_curves[fam]))
    print(f"  {fam:20s}  peak layer={pk}  R@1={family_curves[fam][pk]:.3f}")
print()
print("Outputs in: outputs/track3/")
print("Paper figures are refreshed by: 10b_logit_lens_figures.py")
