"""
Paper-facing figures for Track 3 — Contrastive Logit Lens.

This script is the sole writer for the Track 3 publication figures. It reads
the raw outputs from outputs/track3/ and rewrites the paper figures under
MoleculeLens-paper/figures/.
"""

import os, io, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from transformers import AutoTokenizer, AutoModel

from paper_eval_utils import DEFAULT_TEXT_MAX_LENGTH, diagonal_metric_summary

os.makedirs("MoleculeLens-paper/figures", exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

RICH_COL   = "#2166ac"
NODRUG_COL = "#d6604d"

# ── load ──────────────────────────────────────────────────────────────────
print("Loading saved data...")
df         = pd.read_csv("outputs/test_df.csv")
df_nd      = pd.read_csv("outputs/test_df_nodrug.csv")
layer_df   = pd.read_csv("outputs/track3/recall_by_layer.csv")
family_df  = pd.read_csv("outputs/track3/recall_by_layer_family.csv", index_col=0)
attn_df    = pd.read_csv("outputs/track3/attention_analysis.csv")
Bm         = torch.load("outputs/Bm_test.pt", map_location="cpu")
H_rich     = torch.load("outputs/track3/H_rich_all_layers.pt",   map_location="cpu")
H_nodrug   = torch.load("outputs/track3/H_nodrug_all_layers.pt", map_location="cpu")
Wt_state   = torch.load("outputs/proj_text.pt", map_location="cpu")
W_t        = Wt_state["weight"]
b_t        = Wt_state["bias"]

N_LAYERS = 13
layers   = list(range(N_LAYERS))
layer_labels = ["emb"] + [str(l) for l in range(1, N_LAYERS)]

rich_r1   = layer_df[layer_df["condition"] == "rich"]["recall_1"].values
nodrug_r1 = layer_df[layer_df["condition"] == "nodrug"]["recall_1"].values
rich_mrr  = layer_df[layer_df["condition"] == "rich"]["mrr"].values
nodrug_mrr= layer_df[layer_df["condition"] == "nodrug"]["mrr"].values
gap       = rich_r1 - nodrug_r1


def project_layer(H_layer):
    proj = H_layer @ W_t.T + b_t
    return F.normalize(proj, dim=-1)


# ── target family assignment (must mirror 10_contrastive_logit_lens.py) ───
def assign_family(row):
    moa = str(row["mechanism_of_action"]).lower()
    tgt = str(row.get("target_name", "")).lower()
    if "kinase" in moa or "kinase" in tgt:                        return "Kinase"
    if any(x in moa for x in ["adrenergic", "opioid", "dopamine",
                               "serotonin", "muscarinic", "gpcr",
                               "receptor agonist", "receptor antagonist"]):
        if not any(x in moa for x in ["glucocorticoid", "androgen", "retinoid",
                                      "nuclear receptor"]):
            return "GPCR"
    if any(x in moa for x in ["channel", "blocker"]):            return "Ion Channel"
    if any(x in moa for x in ["glucocorticoid", "androgen",
                               "retinoid", "nuclear receptor", "estrogen"]):
        return "Nuclear Receptor"
    if any(x in moa for x in ["ribosome", "penicillin",
                               "dihydropteroate", "gyrase",
                               "topoisomerase", "bacterial"]):    return "Antibacterial"
    if any(x in moa for x in ["herpesvirus", "viral",
                               "hiv", "polymerase inh",
                               "reverse transcriptase"]):         return "Antiviral"
    if "cyclooxygenase" in moa or "cox" in moa:                   return "COX Inhibitor"
    if "diagnostic" in moa:                                       return "Diagnostic"
    return "Other"

df["family"] = df.apply(assign_family, axis=1)
family_counts = df["family"].value_counts()
top_families = family_counts[family_counts >= 10].index.tolist()

# ══════════════════════════════════════════════════════════════════════════
# RE-GENERATE FIGURE 4 (main logit lens) — save as PDF (vector)
# ══════════════════════════════════════════════════════════════════════════

print("Regenerating Figure 4 (main logit lens) as PDF...")

first_sig = next((l for l in range(1, N_LAYERS) if gap[l] > 0.03), 12)
emergence_layer = int(np.argmax(rich_r1))

fig4 = plt.figure(figsize=(13, 4.5))
gs4 = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.38)

# Panel A: R@1 curve
ax = fig4.add_subplot(gs4[0])
ax.plot(layers, rich_r1,   "o-", color=RICH_COL,   lw=2.2, ms=5.5,
        label="text_rich",   zorder=4)
ax.plot(layers, nodrug_r1, "s--", color=NODRUG_COL, lw=2.2, ms=5.5,
        label="text_nodrug", zorder=4)
ax.fill_between(layers, nodrug_r1, rich_r1, alpha=0.18, color="#aaaaaa",
                label="leakage gap")
ax.annotate(
    f"first gap > 0.03\nlayer {first_sig}",
    xy=(first_sig, rich_r1[first_sig]),
    xytext=(first_sig + 0.7, rich_r1[first_sig] + 0.028),
    fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
    color="#333333"
)
ax.annotate(
    f"peak R@1={rich_r1[emergence_layer]:.3f}\nlayer {emergence_layer}",
    xy=(emergence_layer, rich_r1[emergence_layer]),
    xytext=(emergence_layer - 4.5, rich_r1[emergence_layer] + 0.02),
    fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8, color=RICH_COL),
    color=RICH_COL
)
ax.set_xticks(layers); ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer"); ax.set_ylabel("Recall@1")
ax.set_title("(A) Layer-emergence curve", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

# Panel B: leakage gap bars
ax = fig4.add_subplot(gs4[1])
colours = ["#e45756" if g > 0.03 else "#aec7e8" for g in gap]
ax.bar(layers, gap, color=colours, edgecolor="none", width=0.7)
ax.axhline(0.03, color="#555555", lw=1, ls="--", label="threshold = 0.03")
ax.set_xticks(layers); ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel(r"R@1 gap (rich $-$ nodrug)")
ax.set_title("(B) Leakage gap per layer\n(red = gap > 0.03)", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

# Panel C: MRR curve
ax = fig4.add_subplot(gs4[2])
ax.plot(layers, rich_mrr,   "o-", color=RICH_COL,   lw=2.2, ms=5.5,
        label="text_rich")
ax.plot(layers, nodrug_mrr, "s--", color=NODRUG_COL, lw=2.2, ms=5.5,
        label="text_nodrug")
ax.fill_between(layers, nodrug_mrr, rich_mrr, alpha=0.18, color="#aaaaaa")
ax.set_xticks(layers); ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer"); ax.set_ylabel("MRR")
ax.set_title("(C) MRR vs RoBERTa depth", fontweight="bold")
ax.legend(fontsize=7, frameon=False)

fig4.savefig("MoleculeLens-paper/figures/fig_logit_lens.pdf",
             bbox_inches="tight", format="pdf")
fig4.savefig("MoleculeLens-paper/figures/fig_logit_lens.png",
             bbox_inches="tight", dpi=200)
plt.close(fig4)
print("  Saved: fig_logit_lens.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Logit Lens Detail: family heatmap + attention bars
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 5 (logit lens detail)...")

# ── get attention at layer 12 for 4 case pairs ────────────────────────────
MODEL_NAME = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
roberta_attn = AutoModel.from_pretrained(MODEL_NAME,
                                         output_hidden_states=True,
                                         output_attentions=True)
roberta_attn.eval()

# Pick 4 varied cases: one per major family, all correctly retrieved at R@1
sim_r = project_layer(H_rich[:, 12, :]) @ Bm.T
correct_r1 = (diagonal_metric_summary(sim_r, ks=(1,))["ranks"] == 1).tolist()

cases = []
seen_fams = set()
for i, row in df.iterrows():
    if correct_r1[i] and row["family"] not in seen_fams and len(cases) < 4:
        cases.append(i)
        seen_fams.add(row["family"])
print(f"  Selected {len(cases)} case studies for attention panel: {[df.iloc[i]['pref_name'] for i in cases]}")

ATTN_LAYER = 11   # 0-indexed transformer block = layer 12 in our 0..12 scheme

def get_attn(texts, layer_idx):
    """Returns (token_lists, cls_attn_mean_heads [B, seq_len], mask)."""
    enc = tokenizer(texts, return_tensors="pt",
                    padding=True, truncation=True, max_length=DEFAULT_TEXT_MAX_LENGTH)
    with torch.no_grad():
        out = roberta_attn(**enc, output_attentions=True)
    attn = out.attentions[layer_idx]           # [B, heads, seq, seq]
    cls_attn = attn.mean(dim=1)[:, 0, :]      # [B, seq_len]
    toks_batch = [tokenizer.convert_ids_to_tokens(ids.tolist())
                  for ids in enc["input_ids"]]
    return toks_batch, cls_attn, enc["attention_mask"]

batch_rich   = [df["text_rich"].iloc[i]    for i in cases]
batch_nodrug = [df_nd["text_rich"].iloc[i] for i in cases]
toks_r,  attn_r,  mask_r  = get_attn(batch_rich,   ATTN_LAYER)
toks_nd, attn_nd, mask_nd = get_attn(batch_nodrug, ATTN_LAYER)

# clean up token display: strip Ġ prefix, collapse subwords
def clean_tok(t):
    t = t.replace("Ġ", " ").replace("Ċ", "\n").strip()
    return t if t else "_"

# ── build Figure 5 ────────────────────────────────────────────────────────
fig5 = plt.figure(figsize=(14, 9))
outer = gridspec.GridSpec(2, 1, figure=fig5, hspace=0.55,
                          height_ratios=[1.1, 1.9])

# ── Row 1: per-family heatmap ─────────────────────────────────────────────
ax_heat = fig5.add_subplot(outer[0])

ordered_fams = sorted(top_families,
                      key=lambda f: family_df[f].iloc[12] if f in family_df.columns else 0,
                      reverse=True)
fam_cols = [f for f in ordered_fams if f in family_df.columns]
fam_matrix = np.array([family_df[f].values for f in fam_cols])   # [n_fam, 13]

im = ax_heat.imshow(fam_matrix, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=max(fam_matrix.max(), 0.3),
                    interpolation="nearest")
ax_heat.set_xticks(layers)
ax_heat.set_xticklabels(layer_labels, fontsize=7.5)
ax_heat.set_yticks(range(len(fam_cols)))
ax_heat.set_yticklabels(
    [f"{f}  (n={family_counts[f]})" for f in fam_cols], fontsize=8
)
ax_heat.set_xlabel("RoBERTa layer", fontsize=8)

# annotate each cell with value at layer 12
for fi, fam in enumerate(fam_cols):
    v = fam_matrix[fi, 12]
    ax_heat.text(12, fi, f"{v:.2f}", ha="center", va="center",
                 fontsize=7, fontweight="bold",
                 color="white" if v > 0.25 else "#333333")

cb = plt.colorbar(im, ax=ax_heat, orientation="vertical",
                  fraction=0.025, pad=0.01, label="Recall@1")
cb.ax.tick_params(labelsize=7)
ax_heat.set_title("(A) Per-mechanism-family Recall@1 across RoBERTa layers (text\\_rich)\n"
                  "All families peak at the final layer; value annotated at layer 12.",
                  fontweight="bold", fontsize=8.5)

# ── Row 2: 4×2 attention panels (rich | nodrug) ───────────────────────────
inner = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=outer[1], hspace=0.55, wspace=0.35
)

for col_idx, i in enumerate(cases):
    drug_name = df.iloc[i]["pref_name"].title()
    family    = df.iloc[i]["family"]
    sim_score = (project_layer(H_rich[[i], 12, :]) @ Bm[[i]].T)[0, 0].item()

    for row_idx, (toks, attn_w, mask, cond, col) in enumerate([
        (toks_r[col_idx],  attn_r[col_idx],  mask_r[col_idx],
         "text\\_rich",    RICH_COL),
        (toks_nd[col_idx], attn_nd[col_idx], mask_nd[col_idx],
         "text\\_nodrug",  NODRUG_COL),
    ]):
        ax = fig5.add_subplot(inner[row_idx, col_idx])

        seq_len    = int(mask.sum().item())
        toks_valid = [clean_tok(t) for t in toks[:seq_len]]
        attn_valid = attn_w[:seq_len].numpy()

        # remove <s> and </s> for cleaner display, keep top-8
        filtered = [(t, a) for t, a in zip(toks_valid, attn_valid)
                    if t not in {"<s>", "</s>", ""}][:20]
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)[:8]
        tok_labels = [t for t, _ in filtered_sorted]
        attn_vals  = [a for _, a in filtered_sorted]

        bar_colours = [col] * len(attn_vals)
        ax.barh(range(len(tok_labels)), attn_vals[::-1],
                color=bar_colours, edgecolor="none", height=0.65, alpha=0.85)
        ax.set_yticks(range(len(tok_labels)))
        ax.set_yticklabels(tok_labels[::-1], fontsize=6.5)
        ax.set_xlabel("CLS attn weight", fontsize=6.5)

        if row_idx == 0:
            ax.set_title(
                f"({chr(66 + col_idx)}) {drug_name}\n({family})\nsim={sim_score:.3f}",
                fontsize=7.5, fontweight="bold", pad=3
            )
        if col_idx == 0:
            ax.text(-0.35, 0.5, cond, transform=ax.transAxes,
                    fontsize=7, rotation=90, va="center", ha="center",
                    color=col, fontweight="bold")

fig5.savefig("MoleculeLens-paper/figures/fig_logit_lens_detail.pdf",
             bbox_inches="tight", format="pdf")
fig5.savefig("MoleculeLens-paper/figures/fig_logit_lens_detail.png",
             bbox_inches="tight", dpi=200)
plt.close(fig5)
print("  Saved: fig_logit_lens_detail.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════
# RE-SAVE TRACK 1 FIGURES as clean PDF (already exist but confirm format)
# ══════════════════════════════════════════════════════════════════════════
print("Verifying Track 1 PDFs exist...")
for fname in ["fig_molecular_lens.pdf", "fig_leakage_substructure.pdf",
              "fig_wrong_close.pdf"]:
    path = f"MoleculeLens-paper/figures/{fname}"
    exists = os.path.exists(path)
    size   = os.path.getsize(path) // 1024 if exists else 0
    print(f"  {fname}: {'OK' if exists else 'MISSING'}  ({size} KB)")

print("\nAll Track 3 paper figures done.")
print("outputs:")
for f in sorted(os.listdir("MoleculeLens-paper/figures/")):
    size = os.path.getsize(f"MoleculeLens-paper/figures/{f}") // 1024
    print(f"  {f:45s} {size:5d} KB")
