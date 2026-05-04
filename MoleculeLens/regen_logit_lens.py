"""Standalone regenerator for fig_logit_lens — no model loading needed."""
import os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("MoleculeLens-paper/figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

RICH_COL   = "#2166ac"
NODRUG_COL = "#d6604d"

layer_df = pd.read_csv("outputs/track3/recall_by_layer.csv")
N_LAYERS = 13
layers   = list(range(N_LAYERS))
layer_labels = ["emb"] + [str(l) for l in range(1, N_LAYERS)]

rich_r1    = layer_df[layer_df["condition"] == "rich"]["recall_1"].values
nodrug_r1  = layer_df[layer_df["condition"] == "nodrug"]["recall_1"].values
rich_mrr   = layer_df[layer_df["condition"] == "rich"]["mrr"].values
nodrug_mrr = layer_df[layer_df["condition"] == "nodrug"]["mrr"].values
gap        = rich_r1 - nodrug_r1

first_sig = next((l for l in range(1, N_LAYERS) if gap[l] > 0.03), 12)
emergence_layer = int(np.argmax(rich_r1))

fig4 = plt.figure(figsize=(13, 4.2))
gs4  = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.38)

# Panel A: R@1 curve
ax = fig4.add_subplot(gs4[0])
ax.plot(layers, rich_r1,   "o-",  color=RICH_COL,   lw=2.2, ms=5.5, label="text_rich",   zorder=4)
ax.plot(layers, nodrug_r1, "s--", color=NODRUG_COL, lw=2.2, ms=5.5, label="text_nodrug", zorder=4)
ax.fill_between(layers, nodrug_r1, rich_r1, alpha=0.18, color="#aaaaaa", label="leakage gap")
ax.annotate(
    f"first gap > 0.03\nlayer {first_sig}",
    xy=(first_sig, rich_r1[first_sig]),
    xytext=(first_sig + 0.7, rich_r1[first_sig] + 0.028),
    fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"), color="#333333"
)
ax.annotate(
    f"peak R@1={rich_r1[emergence_layer]:.3f}\nlayer {emergence_layer}",
    xy=(emergence_layer, rich_r1[emergence_layer]),
    xytext=(emergence_layer - 4.5, rich_r1[emergence_layer] + 0.02),
    fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8, color=RICH_COL), color=RICH_COL
)
ax.set_xticks(layers); ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer"); ax.set_ylabel("Recall@1")
ax.set_title("(A) Layer-emergence curve", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)

# Panel B: leakage gap bars
ax = fig4.add_subplot(gs4[1])
colours = ["#e45756" if g > 0.03 else "#aec7e8" for g in gap]
ax.bar(layers, gap, color=colours, edgecolor="none", width=0.7)
ax.axhline(0.03, color="#555555", lw=1, ls="--", label="threshold = 0.03")
ax.set_xticks(layers); ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel(r"R@1 gap (rich $-$ nodrug)")
ax.set_title("(B) Leakage gap per layer\n(red = gap > 0.03)", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)

# Panel C: MRR curve
ax = fig4.add_subplot(gs4[2])
ax.plot(layers, rich_mrr,   "o-",  color=RICH_COL,   lw=2.2, ms=5.5, label="text_rich")
ax.plot(layers, nodrug_mrr, "s--", color=NODRUG_COL, lw=2.2, ms=5.5, label="text_nodrug")
ax.fill_between(layers, nodrug_mrr, rich_mrr, alpha=0.18, color="#aaaaaa")
ax.set_xticks(layers); ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
ax.set_xlabel("RoBERTa layer"); ax.set_ylabel("MRR")
ax.set_title("(C) MRR vs RoBERTa depth", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)

fig4.tight_layout()
fig4.savefig("MoleculeLens-paper/figures/fig_logit_lens.pdf", bbox_inches="tight", format="pdf")
fig4.savefig("MoleculeLens-paper/figures/fig_logit_lens.png", bbox_inches="tight", dpi=200)
plt.close(fig4)

shutil.copy("MoleculeLens-paper/figures/fig_logit_lens.pdf", "/home/cheriearjun/figures/fig_logit_lens.pdf")
shutil.copy("MoleculeLens-paper/figures/fig_logit_lens.png", "/home/cheriearjun/figures/fig_logit_lens.png")
print("Saved: fig_logit_lens.pdf + .png")
