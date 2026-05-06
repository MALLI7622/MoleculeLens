#!/usr/bin/env python3
"""Summarize the mechanism-only MoleculeLens ablation."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from paper_eval_utils import diagonal_metric_summary


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs/mechanism_only"


def main() -> None:
    bt = torch.load(OUTDIR / "Bt_test.pt", map_location="cpu")
    bm = torch.load(OUTDIR / "Bm_test.pt", map_location="cpu")
    sim = (bt @ bm.T).numpy()
    metrics = diagonal_metric_summary(sim, ks=(1, 5, 10))
    summary = {
        "semantics": (
            "MoleculeLens trained and evaluated on mechanism_of_action only; "
            "target name, action type, and explicit drug field are removed."
        ),
        "outdir": str(OUTDIR.relative_to(ROOT)),
        "N": int(sim.shape[0]),
        "Recall@1": float(metrics["Recall@1"]),
        "MRR": float(metrics["MRR"]),
        "Recall@5": float(metrics["Recall@5"]),
        "Recall@10": float(metrics["Recall@10"]),
    }
    out_path = ROOT / "results/mechanism_only_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
