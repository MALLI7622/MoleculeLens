# -*- coding: utf-8 -*-
"""
06_plot_results.py
==================
Standalone results plotter for the CLOOB ablation suite.

Run this at ANY point — it reads whatever results.json files exist in
outputs_cloob/ and plots only the variants that have finished.
Safe to run after 1, 2, 3, or all 4 variants complete.

Usage:
    python 06_plot_results.py --outdir outputs_cloob
    python 06_plot_results.py --outdir outputs_cloob --show   # also opens plots
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Consistent colors and labels per variant
VARIANT_COLORS = {
    "A": "#4878CF",   # blue
    "B": "#6ACC65",   # green
    "C": "#D65F5F",   # red
    "D": "#B47CC7",   # purple
}
VARIANT_ABBR = {
    "A": "A: InfoNCE+NoHop",
    "B": "B: InfoLOOB+NoHop",
    "C": "C: InfoNCE+Hop",
    "D": "D: InfoLOOB+Hop (CLOOB)",
}

METRICS = ["recall_at_1", "mrr", "recall_at_5", "recall_at_10",
           "T4_S2T", "T4_T2S", "T10_S2T", "T10_T2S", "T20_S2T", "T20_T2S"]
METRIC_LABELS = ["R@1", "MRR", "R@5", "R@10",
                 "T4 S→T", "T4 T→S", "T10 S→T", "T10 T→S", "T20 S→T", "T20 T→S"]


# =============================================================================
# LOAD RESULTS
# =============================================================================
def load_all_results(outdir: str) -> list:
    """Load all results.json files found under outdir."""
    pattern = os.path.join(outdir, "variant_*/results.json")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No results.json files found in {outdir}/variant_*/")
        print("Make sure at least one variant has finished running.")
        return []

    results = []
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        results.append(r)
        cond = "nodrug" if r["remove_drug_name"] else "withdrug"
        print(f"  Loaded: Variant {r['variant']} ({cond})  "
              f"R@1={r['recall_at_1']:.3f}  MRR={r['mrr']:.3f}  "
              f"T4_S2T={r['T4_S2T']:.3f}")
    return results


# =============================================================================
# TERMINAL TABLE
# =============================================================================
def print_table(results: list):
    rows_w = [r for r in results if not r["remove_drug_name"]]
    rows_n = [r for r in results if     r["remove_drug_name"]]

    col_keys = ["recall_at_1", "mrr", "recall_at_5", "recall_at_10",
                "T4_S2T", "T4_T2S", "T10_S2T", "T10_T2S", "T20_S2T", "T20_T2S"]
    col_hdrs = ["R@1", "MRR", "R@5", "R@10",
                "T4 S→T", "T4 T→S", "T10 S→T", "T10 T→S", "T20 S→T", "T20 T→S"]

    def print_section(rows, title):
        if not rows:
            return
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}")
        hdr = f"{'Variant':<28}" + "".join(f"{h:>9}" for h in col_hdrs)
        print(hdr)
        print("-" * 100)
        for r in sorted(rows, key=lambda x: x["variant"]):
            label = f"  {r['variant']} ({VARIANT_ABBR.get(r['variant'], r['variant'])})"
            row   = f"{label:<28}" + "".join(f"{r[c]:>9.3f}" for c in col_keys)
            print(row)

    print_section(rows_w, "WITH DRUG NAME")
    print_section(rows_n, "NO DRUG NAME  (Leakage Ablation)")

    # Leakage drop table
    if rows_w and rows_n:
        print(f"\n{'='*100}")
        print("  % PERFORMANCE DROP  (with drug → no drug name)")
        print(f"{'='*100}")
        hdr = f"{'Variant':<28}" + "".join(f"{h:>9}" for h in col_hdrs)
        print(hdr)
        print("-" * 100)
        for rw in sorted(rows_w, key=lambda x: x["variant"]):
            rn_match = [r for r in rows_n if r["variant"] == rw["variant"]]
            if not rn_match:
                continue
            rn    = rn_match[0]
            label = f"  {rw['variant']} ({VARIANT_ABBR.get(rw['variant'], rw['variant'])})"
            row   = f"{label:<28}"
            for c in col_keys:
                drop = (rw[c] - rn[c]) / rw[c] * 100 if rw[c] > 0 else 0
                row += f"{drop:>8.1f}%"
            print(row)
        print("=" * 100)


# =============================================================================
# PLOT 1: Main ablation bar chart (with drug name)
# =============================================================================
def plot_main_ablation(rows_w: list, outdir: str, show: bool = False):
    if not rows_w:
        print("  Skipping main ablation plot — no 'with drug name' results yet.")
        return

    rows_w = sorted(rows_w, key=lambda x: x["variant"])
    n_vars = len(rows_w)
    x      = np.arange(len(METRICS))
    width  = 0.8 / n_vars   # auto-scale bar width based on how many variants done

    fig, ax = plt.subplots(figsize=(15, 5))
    for k, r in enumerate(rows_w):
        var  = r["variant"]
        vals = [r[m] for m in METRICS]
        offset = (k - n_vars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=VARIANT_ABBR.get(var, var),
                      color=VARIANT_COLORS.get(var, "gray"),
                      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Thin Bridges CLOOB Ablation — With Drug Name\n"
                 f"({n_vars}/4 variants complete)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    plt.tight_layout()
    out = os.path.join(outdir, "summary_withdrug.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Saved → {out}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# PLOT 2: T-choose-one focused comparison
# =============================================================================
def plot_t_choose_one(rows_w: list, outdir: str, show: bool = False):
    if not rows_w:
        print("  Skipping T-choose-one plot — no results yet.")
        return

    rows_w   = sorted(rows_w, key=lambda x: x["variant"])
    T_keys   = ["T4_S2T", "T4_T2S", "T10_S2T", "T10_T2S", "T20_S2T", "T20_T2S"]
    T_labels = ["T=4 S→T", "T=4 T→S", "T=10 S→T", "T=10 T→S", "T=20 S→T", "T=20 T→S"]
    random_vals = [1/4, 1/4, 1/10, 1/10, 1/20, 1/20]

    n_vars = len(rows_w)
    x      = np.arange(len(T_keys))
    width  = 0.7 / (n_vars + 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot each variant
    for k, r in enumerate(rows_w):
        var  = r["variant"]
        vals = [r[m] for m in T_keys]
        offset = (k - n_vars / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=VARIANT_ABBR.get(var, var),
               color=VARIANT_COLORS.get(var, "gray"),
               alpha=0.85)

    # Random baseline
    offset = (n_vars - n_vars / 2 + 0.5) * width
    ax.bar(x + offset, random_vals, width,
           label="Random", color="gray", alpha=0.4)

    # MoleculeSTM reference lines (from your ablation_analysis.docx)
    moleculestm_vals = [0.926, 0.933, 0.830, 0.862, 0.739, 0.763]
    for i, (xi, mv) in enumerate(zip(x, moleculestm_vals)):
        ax.hlines(mv, xi - 0.4, xi + 0.4,
                  colors="red", linestyles="--", linewidth=1.5,
                  label="MoleculeSTM (zero-shot)" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(T_labels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("T-choose-one Accuracy — All Variants vs MoleculeSTM\n"
                 "(red dashed = MoleculeSTM zero-shot reference)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(outdir, "t_choose_one_comparison.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Saved → {out}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# PLOT 3: Recall@1 + MRR focused comparison
# =============================================================================
def plot_recall_mrr(rows_w: list, outdir: str, show: bool = False):
    if not rows_w:
        print("  Skipping Recall/MRR plot — no results yet.")
        return

    rows_w  = sorted(rows_w, key=lambda x: x["variant"])
    metrics = ["recall_at_1", "mrr", "recall_at_5", "recall_at_10"]
    labels  = ["Recall@1", "MRR", "Recall@5", "Recall@10"]
    # MoleculeSTM references from your paper
    mstm    = [0.089, 0.171, 0.243, 0.337]

    n_vars = len(rows_w)
    x      = np.arange(len(metrics))
    width  = 0.65 / (n_vars + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for k, r in enumerate(rows_w):
        var  = r["variant"]
        vals = [r[m] for m in metrics]
        offset = (k - n_vars / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=VARIANT_ABBR.get(var, var),
               color=VARIANT_COLORS.get(var, "gray"),
               alpha=0.85)

    # MoleculeSTM reference
    offset = (n_vars - n_vars / 2 + 0.5) * width
    ax.bar(x + offset, mstm, width,
           label="MoleculeSTM (zero-shot)", color="red", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall Metrics — All Variants vs MoleculeSTM", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(outdir, "recall_mrr_comparison.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Saved → {out}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# PLOT 4: Leakage ablation (with vs no drug name per variant)
# =============================================================================
def plot_leakage(rows_w: list, rows_n: list, outdir: str, show: bool = False):
    if not rows_w:
        print("  Skipping leakage plot — no 'with drug name' results yet.")
        return
    if not rows_n:
        print("  Skipping leakage plot — no 'no drug name' results yet.")
        print("  Run with --remove_drug_name first, then re-run this script.")
        return

    # Only plot variants that have BOTH conditions
    variants_both = [r["variant"] for r in rows_w
                     if any(rn["variant"] == r["variant"] for rn in rows_n)]
    if not variants_both:
        print("  Skipping leakage plot — no variants have both conditions yet.")
        return

    leak_metrics = ["recall_at_1", "T4_S2T", "T10_S2T", "T20_S2T"]
    leak_labels  = ["Recall@1", "T=4 S→T", "T=10 S→T", "T=20 S→T"]

    n = len(variants_both)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, sorted(variants_both)):
        rw = next(r for r in rows_w if r["variant"] == var)
        rn = next(r for r in rows_n if r["variant"] == var)

        vals_w   = [rw[m] for m in leak_metrics]
        vals_n   = [rn[m] for m in leak_metrics]
        drop_pct = [(w - nd) / w * 100 if w > 0 else 0
                    for w, nd in zip(vals_w, vals_n)]

        xi = np.arange(len(leak_metrics))
        col = VARIANT_COLORS.get(var, "gray")

        ax.bar(xi - 0.2, vals_w, 0.35,
               label="With drug name", color=col, alpha=0.85)
        ax.bar(xi + 0.2, vals_n, 0.35,
               label="No drug name",   color=col, alpha=0.35)

        # Annotate drop %
        for i, (w, nd, dp) in enumerate(zip(vals_w, vals_n, drop_pct)):
            ax.annotate(f"↓{dp:.0f}%",
                        xy=(xi[i], max(w, nd) + 0.02),
                        ha="center", fontsize=9, color="black")

        ax.set_xticks(xi)
        ax.set_xticklabels(leak_labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(f"{VARIANT_ABBR.get(var, var)}\n"
                     f"Avg drop: {np.mean(drop_pct):.1f}%", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Score", fontsize=11)

    plt.suptitle("Drug Name Leakage Ablation — Performance Drop When Name Removed",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, "leakage_ablation.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Saved → {out}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# PLOT 5: Loss curves (if epoch_losses.npy files exist)
# =============================================================================
def plot_loss_curves(outdir: str, show: bool = False):
    import glob
    files = sorted(glob.glob(os.path.join(outdir, "variant_*/epoch_losses.npy")))
    if not files:
        print("  Skipping loss curves — no epoch_losses.npy files found yet.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for f in files:
        # Parse variant name from directory
        parts   = f.split(os.sep)
        dirname = parts[-2]                        # e.g. "variant_D" or "variant_D_nodrug"
        nodrug  = "nodrug" in dirname
        var     = dirname.replace("variant_", "").replace("_nodrug", "")
        label   = f"Variant {var}" + (" (nodrug)" if nodrug else "")
        losses  = np.load(f)
        style   = "--" if nodrug else "-"
        ax.plot(range(1, len(losses) + 1), losses,
                style, linewidth=1.5,
                color=VARIANT_COLORS.get(var, "gray"),
                label=label, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title("Training Loss Curves — All Variants", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(outdir, "all_loss_curves.png")
    plt.savefig(out, dpi=130)
    print(f"  Saved → {out}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main(args):
    print(f"\nLoading results from: {args.outdir}/")
    print("-" * 60)
    results = load_all_results(args.outdir)

    if not results:
        return

    rows_w = [r for r in results if not r["remove_drug_name"]]
    rows_n = [r for r in results if     r["remove_drug_name"]]

    print(f"\nFound: {len(rows_w)} with-drug-name variants, "
          f"{len(rows_n)} no-drug-name variants\n")

    # Save combined CSV
    df = pd.DataFrame(results)
    csv_out = os.path.join(args.outdir, "summary_table.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved → {csv_out}\n")

    # Terminal table
    print_table(results)

    # Plots
    print("\nGenerating plots ...")
    plot_main_ablation(rows_w, args.outdir, show=args.show)
    plot_t_choose_one(rows_w, args.outdir,  show=args.show)
    plot_recall_mrr(rows_w, args.outdir,    show=args.show)
    plot_leakage(rows_w, rows_n, args.outdir, show=args.show)
    plot_loss_curves(args.outdir, show=args.show)

    print(f"\nAll plots saved to {args.outdir}/")
    print("\nFiles generated:")
    for fname in ["summary_table.csv", "summary_withdrug.png",
                  "t_choose_one_comparison.png", "recall_mrr_comparison.png",
                  "leakage_ablation.png", "all_loss_curves.png"]:
        fpath = os.path.join(args.outdir, fname)
        exists = "✓" if os.path.exists(fpath) else "✗ (not yet)"
        print(f"  {exists}  {fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot CLOOB ablation results — safe to run after any variant finishes")
    parser.add_argument("--outdir", type=str, default="outputs_cloob",
                        help="Directory containing variant_X/ subdirectories")
    parser.add_argument("--show", action="store_true",
                        help="Also open plots interactively (requires display)")
    main(parser.parse_args())