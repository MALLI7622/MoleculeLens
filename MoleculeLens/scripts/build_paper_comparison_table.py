#!/usr/bin/env python3
"""Build the paper-style Thin Bridges comparison table.

This script restores the earlier comparison framing:
  - primary ranking by Recall@1 / MRR / Recall@5 / Recall@10
  - leakage analysis by Recall@1
  - T-choose-one shown only as a secondary reference

Inputs:
  1. A with-drug comparison_results.csv from the earlier head-to-head script.
  2. A no-drug comparison_results.csv from the earlier head-to-head script.
  3. A scaffold output directory containing Bt_test*.pt / Bm_test*.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_eval_utils import diagonal_metric_summary

PRIMARY_METRICS = ("Recall@1", "MRR", "Recall@5", "Recall@10")
T_METRICS = (
    "T=4 S->T",
    "T=4 T->S",
    "T=10 S->T",
    "T=10 T->S",
    "T=20 S->T",
    "T=20 T->S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-csv", required=True, help="With-drug comparison_results.csv")
    parser.add_argument("--nodrug-csv", required=True, help="No-drug comparison_results.csv")
    parser.add_argument(
        "--scaffold-outdir",
        required=True,
        help="Output directory from 03_train_scaffold_split.py",
    )
    parser.add_argument("--molprompt-with-npz", default=None, help="Optional MolPrompt with-drug NPZ")
    parser.add_argument("--molprompt-nodrug-npz", default=None, help="Optional MolPrompt no-drug NPZ")
    parser.add_argument(
        "--molprompt-global-with-npz",
        default=None,
        help="Optional MolPrompt full-gallery/global with-drug NPZ",
    )
    parser.add_argument(
        "--molprompt-global-nodrug-npz",
        default=None,
        help="Optional MolPrompt full-gallery/global no-drug NPZ",
    )
    parser.add_argument("--kvplm-with-json", default=None, help="Optional KV-PLM with-drug metrics JSON")
    parser.add_argument("--kvplm-nodrug-json", default=None, help="Optional KV-PLM no-drug metrics JSON")
    parser.add_argument(
        "--kvplm-global-with-json",
        default=None,
        help="Optional KV-PLM full-gallery/global with-drug metrics JSON",
    )
    parser.add_argument(
        "--kvplm-global-nodrug-json",
        default=None,
        help="Optional KV-PLM full-gallery/global no-drug metrics JSON",
    )
    parser.add_argument(
        "--graphormer-global-with-csv",
        default=None,
        help="Optional Graphormer full-gallery/global with-drug results CSV",
    )
    parser.add_argument(
        "--graphormer-global-nodrug-csv",
        default=None,
        help="Optional Graphormer full-gallery/global no-drug results CSV",
    )
    parser.add_argument("--out", required=True, help="Output markdown path")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional JSON manifest path with resolved metrics and sources",
    )
    return parser.parse_args()


def load_head_to_head_csv(path: Path) -> dict[str, dict[str, float]]:
    by_model: dict[str, dict[str, float]] = {
        "MoleculeSTM": {},
        "Thin Bridges (Global)": {},
        "Random": {},
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row["Metric"].strip()
            if row.get("MoleculeSTM"):
                by_model["MoleculeSTM"][metric] = float(row["MoleculeSTM"])
            if row.get("ThinBridges"):
                by_model["Thin Bridges (Global)"][metric] = float(row["ThinBridges"])
            if row.get("Random"):
                by_model["Random"][metric] = float(row["Random"])
    return by_model


def t_choose_one(
    sim: np.ndarray,
    t_list: tuple[int, ...] = (4, 10, 20),
    n_trials: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = sim.shape[0]
    trials = min(n_trials, n)
    results: dict[str, float] = {}
    all_indices = np.arange(n)

    for t in t_list:
        s2t_hits = 0
        t2s_hits = 0
        for i in range(trials):
            negatives = rng.choice(all_indices[all_indices != i], size=t - 1, replace=False)
            candidates = np.concatenate([[i], negatives])
            if candidates[np.argmax(sim[i, candidates])] == i:
                s2t_hits += 1
            if candidates[np.argmax(sim[candidates, i])] == i:
                t2s_hits += 1
        results[f"T={t} S->T"] = s2t_hits / trials
        results[f"T={t} T->S"] = t2s_hits / trials
    return results


def load_scaffold_metrics(scaffold_outdir: Path, nodrug: bool) -> dict[str, float]:
    suffix = "_nodrug" if nodrug else ""
    bt = torch.load(scaffold_outdir / f"Bt_test{suffix}.pt", map_location="cpu")
    # The canonical paper leakage ablation holds the rich molecule gallery fixed
    # and swaps only the text condition.
    bm = torch.load(scaffold_outdir / "Bm_test.pt", map_location="cpu")
    sim = (bt @ bm.T).numpy()
    metrics = diagonal_metric_summary(sim, ks=(1, 5, 10))
    t_metrics = t_choose_one(sim)
    return {
        "Recall@1": metrics["Recall@1"],
        "MRR": metrics["MRR"],
        "Recall@5": metrics["Recall@5"],
        "Recall@10": metrics["Recall@10"],
        **t_metrics,
        "gallery_size": float(sim.shape[0]),
    }


def load_molprompt_metrics(path: Path) -> dict[str, float]:
    archive = np.load(path)
    test_graph = archive["test_graph"]
    test_text = archive["test_text"]
    sim = test_text @ test_graph.T
    t_metrics = t_choose_one(sim)
    return {
        "Recall@1": float(archive["test_t2m_r1"]),
        "MRR": float(archive["test_t2m_mrr"]),
        "Recall@5": float(archive["test_t2m_r5"]),
        "Recall@10": float(archive["test_t2m_r10"]),
        **t_metrics,
        "gallery_size": float(test_graph.shape[0]),
    }


def load_json_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in data.items()}


def load_graphormer_csv(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row["Metric"].strip()
            value = row.get("ZeroShot")
            if value is None or value == "":
                continue
            metrics[metric] = float(value)
    return metrics


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct_drop(with_value: float, without_value: float) -> str:
    if with_value == 0:
        return "0.0%"
    return f"{(with_value - without_value) / with_value * 100.0:.1f}%"


def render_primary_table(rows: list[dict[str, object]]) -> str:
    header = [
        "| Model | Eval Set | Recall@1 | MRR | Recall@5 | Recall@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    body = []
    for row in rows:
        metrics = row["metrics"]
        body.append(
            "| " + " | ".join(
                [
                    str(row["model"]),
                    str(row["eval_set"]),
                    fmt(metrics["Recall@1"]),
                    fmt(metrics["MRR"]),
                    fmt(metrics["Recall@5"]),
                    fmt(metrics["Recall@10"]),
                ]
            ) + " |"
        )
    return "\n".join(header + body)


def render_leakage_table(rows: list[dict[str, object]]) -> str:
    header = [
        "| Model | Eval Set | With Drug Name | No Drug Name | Abs Drop | % Drop |",
        "|---|---|---:|---:|---:|---:|",
    ]
    body = []
    for row in rows:
        with_value = row["with"]["Recall@1"]
        without_value = row["without"]["Recall@1"]
        body.append(
            "| " + " | ".join(
                [
                    str(row["model"]),
                    str(row["eval_set"]),
                    fmt(with_value),
                    fmt(without_value),
                    fmt(with_value - without_value),
                    fmt_pct_drop(with_value, without_value),
                ]
            ) + " |"
        )
    return "\n".join(header + body)


def render_t_table(rows: list[dict[str, object]]) -> str:
    header = [
        "| Model | Eval Set | T=4 S->T | T=4 T->S | T=10 S->T | T=10 T->S | T=20 S->T | T=20 T->S |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    body = []
    for row in rows:
        metrics = row["metrics"]
        body.append(
            "| " + " | ".join(
                [
                    str(row["model"]),
                    str(row["eval_set"]),
                    fmt(metrics["T=4 S->T"]),
                    fmt(metrics["T=4 T->S"]),
                    fmt(metrics["T=10 S->T"]),
                    fmt(metrics["T=10 T->S"]),
                    fmt(metrics["T=20 S->T"]),
                    fmt(metrics["T=20 T->S"]),
                ]
            ) + " |"
        )
    return "\n".join(header + body)


def render_sources(rows: list[dict[str, object]]) -> str:
    header = [
        "| Model | With Drug Source | No Drug Source |",
        "|---|---|---|",
    ]
    body = []
    for row in rows:
        body.append(
            "| " + " | ".join(
                [
                    str(row["model"]),
                    f"`{row['with_source']}`",
                    f"`{row['without_source']}`",
                ]
            ) + " |"
        )
    return "\n".join(header + body)


def main() -> None:
    args = parse_args()
    with_csv = Path(args.with_csv).expanduser()
    nodrug_csv = Path(args.nodrug_csv).expanduser()
    scaffold_outdir = Path(args.scaffold_outdir).expanduser()

    with_data = load_head_to_head_csv(with_csv)
    nodrug_data = load_head_to_head_csv(nodrug_csv)
    scaffold_with = load_scaffold_metrics(scaffold_outdir, nodrug=False)
    scaffold_without = load_scaffold_metrics(scaffold_outdir, nodrug=True)
    molprompt_with = None
    molprompt_without = None
    if args.molprompt_with_npz and args.molprompt_nodrug_npz:
        molprompt_with = load_molprompt_metrics(Path(args.molprompt_with_npz).expanduser())
        molprompt_without = load_molprompt_metrics(Path(args.molprompt_nodrug_npz).expanduser())
    molprompt_global_with = None
    molprompt_global_without = None
    if args.molprompt_global_with_npz and args.molprompt_global_nodrug_npz:
        molprompt_global_with = load_molprompt_metrics(
            Path(args.molprompt_global_with_npz).expanduser()
        )
        molprompt_global_without = load_molprompt_metrics(
            Path(args.molprompt_global_nodrug_npz).expanduser()
        )
    kvplm_with = None
    kvplm_without = None
    if args.kvplm_with_json and args.kvplm_nodrug_json:
        kvplm_with = load_json_metrics(Path(args.kvplm_with_json).expanduser())
        kvplm_without = load_json_metrics(Path(args.kvplm_nodrug_json).expanduser())
    kvplm_global_with = None
    kvplm_global_without = None
    if args.kvplm_global_with_json and args.kvplm_global_nodrug_json:
        kvplm_global_with = load_json_metrics(Path(args.kvplm_global_with_json).expanduser())
        kvplm_global_without = load_json_metrics(Path(args.kvplm_global_nodrug_json).expanduser())
    graphormer_global_with = None
    graphormer_global_without = None
    if args.graphormer_global_with_csv and args.graphormer_global_nodrug_csv:
        graphormer_global_with = load_graphormer_csv(
            Path(args.graphormer_global_with_csv).expanduser()
        )
        graphormer_global_without = load_graphormer_csv(
            Path(args.graphormer_global_nodrug_csv).expanduser()
        )

    primary_rows = [
        {
            "model": "MoleculeSTM (zero-shot)",
            "eval_set": "Full gallery from earlier head-to-head run (N=2699)",
            "metrics": with_data["MoleculeSTM"],
        },
        {
            "model": "Thin Bridges (Global)",
            "eval_set": "Full gallery from earlier head-to-head run (N=2699; train=test)",
            "metrics": with_data["Thin Bridges (Global)"],
        },
    ]
    if graphormer_global_with is not None:
        primary_rows.append(
            {
                "model": "Graphormer (zero-shot, Global)",
                "eval_set": "Full gallery global zero-shot run (N=2699)",
                "metrics": graphormer_global_with,
            }
        )
    if molprompt_global_with is not None:
        primary_rows.append(
            {
                "model": "MolPrompt (Global)",
                "eval_set": f"Full gallery train=test run from saved NPZ (N={int(molprompt_global_with['gallery_size'])})",
                "metrics": molprompt_global_with,
            }
        )
    if kvplm_global_with is not None:
        primary_rows.append(
            {
                "model": "KV-PLM (Global)",
                "eval_set": f"Full gallery zero-shot run (N={int(kvplm_global_with['gallery_size'])})",
                "metrics": kvplm_global_with,
            }
        )
    primary_rows.append(
            {
                "model": "MoleculeLens (ours)",
                "eval_set": f"Held-out scaffold test from saved outputs (N={int(scaffold_with['gallery_size'])})",
                "metrics": scaffold_with,
            }
    )
    if molprompt_with is not None:
        primary_rows.append(
            {
                "model": "MolPrompt (Scaffold)",
                "eval_set": f"Held-out shared/scaffold split from saved NPZ (N={int(molprompt_with['gallery_size'])})",
                "metrics": molprompt_with,
            }
        )
    if kvplm_with is not None:
        primary_rows.append(
            {
                "model": "KV-PLM (Scaffold)",
                "eval_set": f"Held-out aligned scaffold test (N={int(kvplm_with['gallery_size'])})",
                "metrics": kvplm_with,
            }
        )

    leakage_rows = [
        {
            "model": "MoleculeSTM (zero-shot)",
            "eval_set": "Full gallery from earlier head-to-head run (N=2699)",
            "with": with_data["MoleculeSTM"],
            "without": nodrug_data["MoleculeSTM"],
            "with_source": with_csv,
            "without_source": nodrug_csv,
        },
        {
            "model": "Thin Bridges (Global)",
            "eval_set": "Full gallery from earlier head-to-head run (N=2699; train=test)",
            "with": with_data["Thin Bridges (Global)"],
            "without": nodrug_data["Thin Bridges (Global)"],
            "with_source": with_csv,
            "without_source": nodrug_csv,
        },
    ]
    if graphormer_global_with is not None and graphormer_global_without is not None:
        leakage_rows.append(
            {
                "model": "Graphormer (zero-shot, Global)",
                "eval_set": "Full gallery global zero-shot run (N=2699)",
                "with": graphormer_global_with,
                "without": graphormer_global_without,
                "with_source": Path(args.graphormer_global_with_csv).expanduser(),
                "without_source": Path(args.graphormer_global_nodrug_csv).expanduser(),
            }
        )
    if molprompt_global_with is not None and molprompt_global_without is not None:
        leakage_rows.append(
            {
                "model": "MolPrompt (Global)",
                "eval_set": f"Full gallery train=test run from saved NPZ (N={int(molprompt_global_with['gallery_size'])})",
                "with": molprompt_global_with,
                "without": molprompt_global_without,
                "with_source": Path(args.molprompt_global_with_npz).expanduser(),
                "without_source": Path(args.molprompt_global_nodrug_npz).expanduser(),
            }
        )
    if kvplm_global_with is not None and kvplm_global_without is not None:
        leakage_rows.append(
            {
                "model": "KV-PLM (Global)",
                "eval_set": f"Full gallery zero-shot run (N={int(kvplm_global_with['gallery_size'])})",
                "with": kvplm_global_with,
                "without": kvplm_global_without,
                "with_source": Path(args.kvplm_global_with_json).expanduser(),
                "without_source": Path(args.kvplm_global_nodrug_json).expanduser(),
            }
        )
    leakage_rows.extend(
        [
        {
            "model": "MoleculeLens (ours)",
            "eval_set": f"Held-out scaffold test from saved outputs (N={int(scaffold_with['gallery_size'])})",
            "with": scaffold_with,
            "without": scaffold_without,
            "with_source": scaffold_outdir / "Bt_test.pt",
            "without_source": scaffold_outdir / "Bt_test_nodrug.pt",
        },
        ]
    )
    if molprompt_with is not None and molprompt_without is not None:
        leakage_rows.append(
            {
                "model": "MolPrompt (Scaffold)",
                "eval_set": f"Held-out shared/scaffold split from saved NPZ (N={int(molprompt_with['gallery_size'])})",
                "with": molprompt_with,
                "without": molprompt_without,
                "with_source": Path(args.molprompt_with_npz).expanduser(),
                "without_source": Path(args.molprompt_nodrug_npz).expanduser(),
            }
        )
    if kvplm_with is not None and kvplm_without is not None:
        leakage_rows.append(
            {
                "model": "KV-PLM (Scaffold)",
                "eval_set": f"Held-out aligned scaffold test (N={int(kvplm_with['gallery_size'])})",
                "with": kvplm_with,
                "without": kvplm_without,
                "with_source": Path(args.kvplm_with_json).expanduser(),
                "without_source": Path(args.kvplm_nodrug_json).expanduser(),
            }
        )

    markdown = "\n".join(
        [
            "# MoleculeLens Camera-Ready Comparison",
            "",
            "Primary retrieval comparison should be based on full-gallery retrieval metrics",
            "(`Recall@1`, `MRR`, `Recall@5`, `Recall@10`).",
            "",
            "## Primary Retrieval Comparison",
            "",
            render_primary_table(primary_rows),
            "",
            "## Drug-Name Leakage Ablation (Recall@1)",
            "",
            render_leakage_table(leakage_rows),
            "",
            "## Secondary T-Choose-One Reference",
            "",
            "These are useful as a secondary diagnostic, but should not replace",
            "full-gallery retrieval metrics as the primary ranking criterion.",
            "",
            render_t_table(primary_rows),
            "",
            "## Notes",
            "",
            "- `Thin Bridges (Global)` is included only as a memorization reference because it uses train=test.",
            "- `MolPrompt (Global)` uses the same full-gallery train=val=test protocol to provide a like-for-like memorization-heavy reference.",
            "- `KV-PLM (Global)` and `Graphormer (zero-shot, Global)` are full-gallery zero-shot evaluations over the same 2,699-pair gallery.",
            "- `MoleculeLens (ours)` is the scientifically valid held-out result from the saved scaffold outputs.",
            "- `MoleculeSTM (zero-shot)` and `Thin Bridges (Global)` come from the earlier head-to-head CSVs.",
            "- Random Recall@1 differs by gallery size: about `1/2699 = 0.0004` for the full-gallery runs and "
            f"`1/{int(scaffold_with['gallery_size'])} = {1.0 / scaffold_with['gallery_size']:.4f}` for the scaffold test set.",
            "",
            "## Sources",
            "",
            render_sources(leakage_rows),
            "",
        ]
    )

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    if args.manifest_out:
        manifest = {
            "primary_rows": [
                {
                    "model": row["model"],
                    "eval_set": row["eval_set"],
                    "metrics": {k: float(v) for k, v in row["metrics"].items() if k != "gallery_size"},
                    "gallery_size": int(row["metrics"].get("gallery_size", 2699.0)),
                }
                for row in primary_rows
            ],
            "leakage_rows": [
                {
                    "model": row["model"],
                    "eval_set": row["eval_set"],
                    "with": {k: float(v) for k, v in row["with"].items() if k != "gallery_size"},
                    "without": {k: float(v) for k, v in row["without"].items() if k != "gallery_size"},
                    "with_source": str(row["with_source"]),
                    "without_source": str(row["without_source"]),
                }
                for row in leakage_rows
            ],
            "evaluation": {
                "moleculelens_scaffold": "diagonal_optimistic",
                "definition": "rank(i) = 1 + count_j[S[i,j] > S[i,i]]",
                "moleculelens_leakage": "fixed_rich_gallery_text_ablation",
            },
            "inputs": {
                "with_csv": str(with_csv),
                "nodrug_csv": str(nodrug_csv),
                "scaffold_outdir": str(scaffold_outdir),
                "molprompt_with_npz": args.molprompt_with_npz,
                "molprompt_nodrug_npz": args.molprompt_nodrug_npz,
                "molprompt_global_with_npz": args.molprompt_global_with_npz,
                "molprompt_global_nodrug_npz": args.molprompt_global_nodrug_npz,
                "kvplm_with_json": args.kvplm_with_json,
                "kvplm_nodrug_json": args.kvplm_nodrug_json,
                "kvplm_global_with_json": args.kvplm_global_with_json,
                "kvplm_global_nodrug_json": args.kvplm_global_nodrug_json,
                "graphormer_global_with_csv": args.graphormer_global_with_csv,
                "graphormer_global_nodrug_csv": args.graphormer_global_nodrug_csv,
            },
            }
        manifest_path = Path(args.manifest_out).expanduser()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
