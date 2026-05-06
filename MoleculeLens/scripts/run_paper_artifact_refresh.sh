#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cheriearjun
MOLLENS="$ROOT/MoleculeLens"
MOLPROMPT="$ROOT/MolPrompt"
CONDA_SH="$ROOT/miniconda3/etc/profile.d/conda.sh"

source "$CONDA_SH"
conda activate MoleculeSTM

cd "$MOLLENS"

python scripts/build_paper_comparison_table.py \
    --with-csv "$MOLLENS/outputs/comparison_withdrug/comparison_results.csv" \
    --nodrug-csv "$MOLLENS/outputs/comparison_nodrug/comparison_results.csv" \
    --scaffold-outdir "$MOLLENS/outputs" \
    --molprompt-with-npz "$MOLPROMPT/save_model/retrieval_chembl_sharedsplit_bs6/chembl_retrieval_best_metrics.npz" \
    --molprompt-nodrug-npz "$MOLPROMPT/save_model/retrieval_chembl_sharedsplit_bs6_nodrug/chembl_retrieval_best_metrics.npz" \
    --kvplm-with-json "$MOLLENS/results/kv_plm_aligned.json" \
    --kvplm-nodrug-json "$MOLLENS/results/kv_plm_aligned_nodrug.json" \
    --out "$MOLLENS/results/model_comparison_table.md" \
    --manifest-out "$MOLLENS/results/model_comparison_manifest.json"

python 09b_diagonal_artifact_sync.py
python 09c_attribution_validation.py
python 09d_incorrect_attribution_diagnostic.py
python 12c_full_loss_multiseed.py
python 03_train_scaffold_split.py \
    --csv "$MOLLENS/chembl_mechanisms.csv" \
    --outdir "$MOLLENS/outputs/mechanism_only" \
    --text_field mechanism_of_action
python 13_mechanism_only_summary.py
python 10_contrastive_logit_lens.py
python 08c_paper_figures.py
python 10b_logit_lens_figures.py
python 12b_robustness_figures.py
python scripts/write_paper_artifact_manifest.py

cd "$MOLLENS/MoleculeLens-paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error neurips_2026.tex

echo "Paper artifact refresh complete."
