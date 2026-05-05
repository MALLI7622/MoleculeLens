"""Generate robustness figures from saved CSVs."""
import os
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
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
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
fig = plt.figure(figsize=(10.8, 3.45))
gs  = gridspec.GridSpec(
    1, 3, figure=fig, wspace=0.36,
    left=0.075, right=0.985, bottom=0.19, top=0.90
)

# ── Panel A: Forest plot (bootstrap CI) ───────────────────────────────────
ax = fig.add_subplot(gs[0])
for i, row in enumerate(forest_rows):
    ax.plot([row["lo"], row["hi"]], [i, i],
            color=row["col"], lw=3.5, solid_capstyle="round", alpha=0.7)
    ax.plot(row["mean"], i, "o", color=row["col"], ms=8, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(row["hi"] + 0.005, i, f" {row['mean']:.3f}",
            va="center", fontsize=7.5, color=row["col"], fontweight="bold")

ax.set_yticks(range(len(forest_rows)))
ax.set_yticklabels([r["label"] for r in forest_rows], fontsize=8)
ax.set_xlabel("Recall@1  (↑ better)")
ax.set_xlim(-0.01, 0.28)
ax.set_title("(A) Bootstrap 95% CI\nRecall@1 — N = 435 test pairs", fontweight="bold")
ax.axvline(0, color="#dddddd", lw=0.7)
ax.text(0.97, 0.03, "point = mean\nbar = 95% CI",
        ha="right", va="bottom", transform=ax.transAxes,
        fontsize=6.5, color="#888888")

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

ax.set_xlabel("Scaffold split seed")
ax.set_ylabel("Recall@1")
ax.set_xticks(SEEDS)
ax.set_title("(B) Multi-seed stability\n5 independent scaffold splits", fontweight="bold")
ax.set_xlim(min(SEEDS) - 0.15, max(SEEDS) + 0.62)
ax.text(max(SEEDS) + 0.10, rich_v[-1], f"rich\n{mu_r:.3f}±{sd_r:.3f}",
        color=RICH_COL, fontsize=6.6, va="center")
ax.text(max(SEEDS) + 0.10, nodrug_v[-1], f"nodrug\n{mu_n:.3f}±{sd_n:.3f}",
        color=NODRUG_COL, fontsize=6.6, va="center")

# annotate test-set sizes
for s, n in zip(SEEDS, ms_df["n_test"].values):
    ax.text(s, ax.get_ylim()[0] + 0.005 if ax.get_ylim()[0] > 0 else 0.02,
            f"n={n}", ha="center", va="bottom", fontsize=6, color="#888888")

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
            f"{v:.1f}%", ha="center", va="bottom", fontsize=7,
            color="#9467bd", fontweight="bold")

ax.set_xlabel("Scaffold split seed")
ax.set_ylabel("Leakage drop (R@1 rich − nodrug) / R@1 rich × 100")
ax.set_xticks(SEEDS)
ax.set_title("(C) Drug-name leakage drop\nacross scaffold splits", fontweight="bold")
ax.text(0.03, 0.96, f"mean = {mu_d:.1f}%\n±1 std = {sd_d:.1f} pp",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

fig.savefig("MoleculeLens-paper/figures/fig_robustness.pdf",
            bbox_inches="tight", pad_inches=0.02, format="pdf")
fig.savefig("MoleculeLens-paper/figures/fig_robustness.png",
            bbox_inches="tight", pad_inches=0.02, dpi=200)
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
