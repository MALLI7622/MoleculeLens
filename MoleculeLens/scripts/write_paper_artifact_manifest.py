#!/usr/bin/env python3
"""Write a machine-readable summary of the current paper-facing artifacts."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
        ).strip()
        return {
            "commit": commit,
            "dirty": bool(status),
        }
    except Exception:
        return {
            "commit": None,
            "dirty": None,
        }


def main() -> None:
    comparison = json.loads((ROOT / "results/model_comparison_manifest.json").read_text())
    attribution_validation = json.loads((ROOT / "results/attribution_validation.json").read_text())
    incorrect_attribution = json.loads(
        (ROOT / "results/incorrect_attribution_summary.json").read_text()
    )
    full_loss_multiseed = json.loads((ROOT / "results/full_loss_multiseed_summary.json").read_text())
    mechanism_only = json.loads((ROOT / "results/mechanism_only_ablation.json").read_text())
    track3_meta = json.loads((ROOT / "outputs/track3/run_metadata.json").read_text())
    track3_df = pd.read_csv(ROOT / "outputs/track3/recall_by_layer.csv")
    leakage_df = pd.read_csv(ROOT / "outputs/track1/leakage_per_pair.csv")
    wrong_close_df = pd.read_csv(ROOT / "outputs/track1/wrong_close_analysis.csv")
    bootstrap_df = pd.read_csv(ROOT / "outputs/robustness/bootstrap_ci.csv")
    multiseed_df = pd.read_csv(ROOT / "outputs/robustness/multiseed_results.csv")

    scaffold_row = next(
        row for row in comparison["primary_rows"] if row["model"] == "MoleculeLens (ours)"
    )
    track3_layer12 = track3_df[track3_df["layer"] == 12]
    track3_rich = track3_layer12[track3_layer12["condition"] == "rich"].iloc[0]
    track3_nodrug = track3_layer12[track3_layer12["condition"] == "nodrug"].iloc[0]

    wrong_corr = wrong_close_df["attr_correlation"].dropna()
    rich_ci = bootstrap_df[bootstrap_df["model"] == "MoleculeLens (text_rich)"].iloc[0]
    nodrug_ci = bootstrap_df[bootstrap_df["model"] == "MoleculeLens (text_nodrug)"].iloc[0]
    nodrug_same_model_ci = bootstrap_df[
        bootstrap_df["model"] == "MoleculeLens (text_nodrug, same_model)"
    ].iloc[0]

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_refresh_command": "bash scripts/run_paper_artifact_refresh.sh",
        "git": git_info(),
        "artifact_writers": {
            "comparison": "scripts/build_paper_comparison_table.py",
            "track1_csvs": "09b_diagonal_artifact_sync.py",
            "track1_figures": "08c_paper_figures.py",
            "track3_raw": "10_contrastive_logit_lens.py",
            "track3_figures": "10b_logit_lens_figures.py",
            "robustness_raw": "12_robustness.py",
            "robustness_figure": "12b_robustness_figures.py",
        },
        "paper_outputs": {
            "comparison_table": "results/model_comparison_table.md",
            "comparison_manifest": "results/model_comparison_manifest.json",
            "track1_leakage_csv": "outputs/track1/leakage_per_pair.csv",
            "track1_wrong_close_csv": "outputs/track1/wrong_close_analysis.csv",
            "track1_incorrect_attribution_csv": "outputs/track1/incorrect_attribution_summary.csv",
            "track3_recall_csv": "outputs/track3/recall_by_layer.csv",
            "track3_run_metadata": "outputs/track3/run_metadata.json",
            "bootstrap_ci_csv": "outputs/robustness/bootstrap_ci.csv",
            "full_loss_multiseed_csv": "outputs/robustness/full_loss_multiseed_results.csv",
            "mechanism_only_outdir": "outputs/mechanism_only",
            "attribution_validation_json": "results/attribution_validation.json",
            "incorrect_attribution_json": "results/incorrect_attribution_summary.json",
            "full_loss_multiseed_json": "results/full_loss_multiseed_summary.json",
            "mechanism_only_json": "results/mechanism_only_ablation.json",
            "paper_pdf": "MoleculeLens-paper/neurips_2026.pdf",
        },
        "comparison": {
            "evaluation": comparison.get("evaluation", {}),
            "moleculelens_scaffold": scaffold_row,
            "moleculelens_mechanism_only": mechanism_only,
        },
        "track1": {
            "leakage_semantics": "same_model_rich_head_text_ablation",
            "jaccard_mean": float(leakage_df["attr_jaccard_overlap"].mean()),
            "jaccard_std": float(leakage_df["attr_jaccard_overlap"].std(ddof=0)),
            "jaccard_lt_0_2": int((leakage_df["attr_jaccard_overlap"] < 0.2).sum()),
            "leakage_candidates": int(
                (
                    (leakage_df["hit_rich"] == 1)
                    & (leakage_df["hit_nodrug"] == 0)
                    & (leakage_df["attr_jaccard_overlap"] < 0.2)
                ).sum()
            ),
            "wrong_close_pairs": int(len(wrong_close_df)),
            "wrong_close_mean_rho": float(wrong_corr.mean()),
            "wrong_close_std_rho": float(wrong_corr.std(ddof=0)),
            "wrong_close_rho_gt_0_5": int((wrong_corr > 0.5).sum()),
            "wrong_close_same_moa": int(wrong_close_df["moa_correct_match"].sum()),
            "attribution_validation": attribution_validation,
            "incorrect_attribution": incorrect_attribution,
        },
        "track3": {
            "exact_replay": track3_meta,
            "layer12": {
                "text_rich": {
                    "Recall@1": float(track3_rich["recall_1"]),
                    "MRR": float(track3_rich["mrr"]),
                },
                "text_nodrug": {
                    "Recall@1": float(track3_nodrug["recall_1"]),
                    "MRR": float(track3_nodrug["mrr"]),
                },
                "gap": float(track3_rich["recall_1"] - track3_nodrug["recall_1"]),
            },
        },
        "robustness": {
            "bootstrap_ci": {
                "text_rich": {
                    "Recall@1": float(rich_ci["R1"]),
                    "ci_lo": float(rich_ci["R1_lo"]),
                    "ci_hi": float(rich_ci["R1_hi"]),
                },
                "text_nodrug": {
                    "Recall@1": float(nodrug_ci["R1"]),
                    "ci_lo": float(nodrug_ci["R1_lo"]),
                    "ci_hi": float(nodrug_ci["R1_hi"]),
                    "semantics": "retrained_no_drug_head",
                },
                "text_nodrug_same_model": {
                    "Recall@1": float(nodrug_same_model_ci["R1"]),
                    "ci_lo": float(nodrug_same_model_ci["R1_lo"]),
                    "ci_hi": float(nodrug_same_model_ci["R1_hi"]),
                    "semantics": "same_model_rich_head_text_ablation",
                },
            },
            "multiseed_mean_r1_rich": float(multiseed_df["R@1_rich"].mean()),
            "multiseed_std_r1_rich": float(multiseed_df["R@1_rich"].std()),
            "multiseed_mean_leakage_drop_pct": float(multiseed_df["leakage_drop_pct"].mean()),
            "full_loss_multiseed": full_loss_multiseed,
        },
    }

    out_path = ROOT / "results/paper_artifact_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
