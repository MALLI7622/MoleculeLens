"""Standalone regenerator for fig_logit_lens — no model loading needed."""
import os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "MoleculeLens-paper", "figures")
ROOT_FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(ROOT_FIG_DIR, exist_ok=True)


def mirror_to_root(stem):
    for ext in ("pdf", "png"):
        shutil.copy(
            os.path.join(PAPER_FIG_DIR, f"{stem}.{ext}"),
            os.path.join(ROOT_FIG_DIR, f"{stem}.{ext}"),
        )

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15.5,
    "axes.labelsize": 17.0, "axes.titlesize": 17.0,
    "xtick.labelsize": 15.0, "ytick.labelsize": 15.0,
    "legend.fontsize": 15.0,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

RICH_COL   = "#2166ac"
NODRUG_COL = "#d6604d"

layer_df = pd.read_csv("outputs/track3/recall_by_layer.csv")
N_LAYERS = 13
layers   = list(range(N_LAYERS))
layer_labels = ["emb"] + [str(l) for l in range(1, N_LAYERS)]
shown_layers = [0, 3, 6, 9, 12]
shown_layer_labels = [layer_labels[l] for l in shown_layers]

rich_r1    = layer_df[layer_df["condition"] == "rich"]["recall_1"].values
nodrug_r1  = layer_df[layer_df["condition"] == "nodrug"]["recall_1"].values
rich_mrr   = layer_df[layer_df["condition"] == "rich"]["mrr"].values
nodrug_mrr = layer_df[layer_df["condition"] == "nodrug"]["mrr"].values
gap        = rich_r1 - nodrug_r1

first_sig = next((l for l in range(1, N_LAYERS) if gap[l] > 0.03), 12)
emergence_layer = int(np.argmax(rich_r1))

fig4 = plt.figure(figsize=(10.6, 3.8), constrained_layout=True)
gs4  = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.42)

# Panel A: R@1 curve
ax = fig4.add_subplot(gs4[0])
ax.plot(layers, rich_r1,   "o-",  color=RICH_COL,   lw=2.2, ms=5.5, label="text_rich",   zorder=4)
ax.plot(layers, nodrug_r1, "s--", color=NODRUG_COL, lw=2.2, ms=5.5, label="text_nodrug", zorder=4)
ax.fill_between(layers, nodrug_r1, rich_r1, alpha=0.18, color="#aaaaaa", label="leakage gap")
ax.set_xticks(shown_layers); ax.set_xticklabels(shown_layer_labels, rotation=0)
ax.set_xlim(-0.4, 12.4)
ax.set_xlabel("RoBERTa layer"); ax.set_ylabel("Recall@1")
ax.set_title("(A) Recall@1", fontweight="bold")
ax.legend(frameon=False)

# Panel B: leakage gap bars
ax = fig4.add_subplot(gs4[1])
colours = ["#e45756" if g > 0.03 else "#aec7e8" for g in gap]
ax.bar(layers, gap, color=colours, edgecolor="none", width=0.7)
ax.axhline(0.03, color="#555555", lw=1, ls="--", label="threshold = 0.03")
ax.set_xticks(shown_layers); ax.set_xticklabels(shown_layer_labels, rotation=0)
ax.set_xlim(-0.4, 12.4)
ax.set_xlabel("RoBERTa layer")
ax.set_ylabel(r"R@1 gap (rich $-$ nodrug)")
ax.set_title("(B) Gap", fontweight="bold")
ax.legend(frameon=False)

# Panel C: MRR curve
ax = fig4.add_subplot(gs4[2])
ax.plot(layers, rich_mrr,   "o-",  color=RICH_COL,   lw=2.2, ms=5.5, label="text_rich")
ax.plot(layers, nodrug_mrr, "s--", color=NODRUG_COL, lw=2.2, ms=5.5, label="text_nodrug")
ax.fill_between(layers, nodrug_mrr, rich_mrr, alpha=0.18, color="#aaaaaa")
ax.set_xticks(shown_layers); ax.set_xticklabels(shown_layer_labels, rotation=0)
ax.set_xlim(-0.4, 12.4)
ax.set_xlabel("RoBERTa layer"); ax.set_ylabel("MRR")
ax.set_title("(C) MRR", fontweight="bold")
ax.legend(frameon=False)

fig4.savefig(os.path.join(PAPER_FIG_DIR, "fig_logit_lens.pdf"),
             bbox_inches="tight", pad_inches=0.02, format="pdf")
fig4.savefig(os.path.join(PAPER_FIG_DIR, "fig_logit_lens.png"),
             bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.close(fig4)

mirror_to_root("fig_logit_lens")
print("Saved: fig_logit_lens.pdf + .png")
