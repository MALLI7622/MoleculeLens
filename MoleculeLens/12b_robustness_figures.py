"""Generate robustness figures from saved CSVs."""
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

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 11.5, "axes.titlesize": 12.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

ci_df = pd.read_csv("outputs/robustness/bootstrap_ci.csv")
ms_df = pd.read_csv("outputs/robustness/multiseed_results.csv")

SEEDS = ms_df["seed"].tolist()
RICH_COL   = "#2166ac"
NODRUG_COL = "#d6604d"

# ── forest-plot data: pull rows cleanly ───────────────────────────────────
forest_rows = [
    {"label": "ML (text_rich)",    "col": RICH_COL},
    {"label": "ML (text_nodrug)",  "col": NODRUG_COL},
    {"label": "MolPrompt",         "col": "#4dac26"},
    {"label": "KV-PLM",            "col": "#f1a340"},
    {"label": "Graphormer (0-shot)","col":"#888888"},
]
# Map labels to ci_df rows
ci_map = {
    "ML (text_rich)":     "MoleculeLens (text_rich)",
    "ML (text_nodrug)":   "MoleculeLens (text_nodrug)",
    "MolPrompt":          "MolPrompt",
    "KV-PLM":             "KV-PLM",
    "Graphormer (0-shot)":"Graphormer (0-shot)",
}


def get_metric(row, primary, fallback):
    if primary in row.index:
        return float(row[primary])
    return float(row[fallback])


for row in forest_rows:
    match = ci_df[ci_df["model"] == ci_map[row["label"]]]
    if len(match) == 0:
        # fallback: partial match
        match = ci_df[ci_df["model"].str.contains(
            ci_map[row["label"]].split()[0], case=False)]
    r = match.iloc[0]
    row["mean"] = get_metric(r, "R@1", "R1")
    row["lo"]   = get_metric(r, "R@1_lo", "R1_lo")
    row["hi"]   = get_metric(r, "R@1_hi", "R1_hi")

# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.6, 4.55))
gs  = gridspec.GridSpec(
    1, 3, figure=fig, wspace=0.48,
    left=0.085, right=0.985, bottom=0.25, top=0.82
)

# ── Panel A: Forest plot (bootstrap CI) ───────────────────────────────────
ax = fig.add_subplot(gs[0])
for i, row in enumerate(forest_rows):
    ax.plot([row["lo"], row["hi"]], [i, i],
            color=row["col"], lw=3.5, solid_capstyle="round", alpha=0.7)
    ax.plot(row["mean"], i, "o", color=row["col"], ms=8, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(row["hi"] + 0.005, i, f" {row['mean']:.3f}",
            va="center", fontsize=15.0, color=row["col"], fontweight="bold")

ax.set_yticks(range(len(forest_rows)))
ax.set_yticklabels([r["label"] for r in forest_rows], fontsize=15.0)
ax.tick_params(axis="x", labelsize=15.0)
ax.set_xlabel("Recall@1  (↑ better)", fontsize=17.0)
ax.set_xlim(-0.01, 0.28)
ax.set_title("(A) Bootstrap CI", fontsize=17.0, fontweight="bold")
ax.axvline(0, color="#dddddd", lw=0.7)
ax.text(0.97, 0.03, "point = mean\nbar = 95% CI",
        ha="right", va="bottom", transform=ax.transAxes,
        fontsize=15.0, color="#888888")

# ── Panel B: Multi-seed R@1 ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1])
rich_v   = ms_df["R@1_rich"].values
nodrug_v = ms_df["R@1_nodrug"].values
mu_r, sd_r = rich_v.mean(),   rich_v.std()
mu_n, sd_n = nodrug_v.mean(), nodrug_v.std()

ax.plot(SEEDS, rich_v,   "o-", color=RICH_COL,   lw=2.2, ms=7,
        markeredgecolor="white", markeredgewidth=0.8,
        label=f"text_rich  {mu_r:.3f}±{sd_r:.3f}")
ax.plot(SEEDS, nodrug_v, "s--", color=NODRUG_COL, lw=2.2, ms=7,
        markeredgecolor="white", markeredgewidth=0.8,
        label=f"text_nodrug {mu_n:.3f}±{sd_n:.3f}")

ax.axhline(mu_r, color=RICH_COL,   lw=1, ls=":", alpha=0.6)
ax.axhline(mu_n, color=NODRUG_COL, lw=1, ls=":", alpha=0.6)
ax.fill_between(SEEDS, mu_r - sd_r, mu_r + sd_r,
                alpha=0.12, color=RICH_COL)
ax.fill_between(SEEDS, mu_n - sd_n, mu_n + sd_n,
                alpha=0.12, color=NODRUG_COL)

ax.set_xlabel("Scaffold split seed", fontsize=17.0)
ax.set_ylabel("Recall@1", fontsize=17.0)
ax.set_xticks(SEEDS)
ax.tick_params(axis="both", labelsize=15.0)
ax.set_title("(B) Seed stability", fontsize=17.0, fontweight="bold")
ax.set_xlim(min(SEEDS) - 0.15, max(SEEDS) + 0.55)
ax.text(max(SEEDS) + 0.15, rich_v[-1],
        "rich",
        color=RICH_COL, fontsize=13.2, va="center", ha="left",
        fontweight="bold")
ax.text(max(SEEDS) + 0.15, nodrug_v[-1],
        "no-drug",
        color=NODRUG_COL, fontsize=13.2, va="center", ha="left",
        fontweight="bold")

# ── Panel C: Leakage-drop variance ────────────────────────────────────────
ax = fig.add_subplot(gs[2])
drops = ms_df["leakage_drop_pct"].values
mu_d, sd_d = drops.mean(), drops.std()

bars = ax.bar(SEEDS, drops, color="#9467bd", edgecolor="white",
              width=0.6, alpha=0.85)
ax.axhline(mu_d, color="#333333", lw=1.5, ls="--",
           zorder=3)
ax.fill_between([-0.5, max(SEEDS) + 0.5],
                mu_d - sd_d, mu_d + sd_d,
                alpha=0.15, color="#9467bd")

# value labels on bars
for bar, v in zip(bars, drops):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=14.2,
            color="#9467bd", fontweight="bold")

ax.set_xlabel("Scaffold split seed", fontsize=17.0)
ax.set_ylabel("Leakage drop (%)", fontsize=17.0)
ax.set_xticks(SEEDS)
ax.tick_params(axis="both", labelsize=15.0)
ax.set_title("(C) Leakage drop", fontsize=17.0, fontweight="bold")
ax.set_ylim(0, max(drops) * 1.18)

pdf_path = os.path.join(PAPER_FIG_DIR, "fig_robustness.pdf")
png_path = os.path.join(PAPER_FIG_DIR, "fig_robustness.png")
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, format="pdf")
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
shutil.copy(pdf_path, os.path.join(ROOT_FIG_DIR, "fig_robustness.pdf"))
shutil.copy(png_path, os.path.join(ROOT_FIG_DIR, "fig_robustness.png"))
plt.close(fig)
print("Saved: fig_robustness.pdf/.png")

# ── Print paper-ready table ───────────────────────────────────────────────
print("\nPaper-ready multi-seed table:")
print(f"{'Seed':>5} {'n_test':>7} {'R@1':>7} {'R@5':>7} "
      f"{'R@10':>7} {'MRR':>7} {'nodrug':>8} {'drop%':>7}")
for _, r in ms_df.iterrows():
    print(f"  {int(r.seed):3d}   {int(r.n_test):5d}  "
          f"{r['R@1_rich']:.3f}  {r['R@5_rich']:.3f}  "
          f"{r['R@10_rich']:.3f}  {r['MRR_rich']:.3f}  "
          f"{r['R@1_nodrug']:.3f}  {r['leakage_drop_pct']:.1f}%")
print(f"  {'Mean':>3}          "
      f"{ms_df['R@1_rich'].mean():.3f}  {ms_df['R@5_rich'].mean():.3f}  "
      f"{ms_df['R@10_rich'].mean():.3f}  {ms_df['MRR_rich'].mean():.3f}  "
      f"{ms_df['R@1_nodrug'].mean():.3f}  "
      f"{ms_df['leakage_drop_pct'].mean():.1f}%")
print(f"  {'Std':>3}          "
      f"{ms_df['R@1_rich'].std():.3f}  {ms_df['R@5_rich'].std():.3f}  "
      f"{ms_df['R@10_rich'].std():.3f}  {ms_df['MRR_rich'].std():.3f}  "
      f"{ms_df['R@1_nodrug'].std():.3f}  "
      f"{ms_df['leakage_drop_pct'].std():.1f}%")
