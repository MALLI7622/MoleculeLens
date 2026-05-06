#!/usr/bin/env bash
set -euo pipefail

# Legacy aligned-baseline refresh.
# This script updates aligned baseline artifacts only. Camera-ready paper
# tables/figures are regenerated exclusively by scripts/run_paper_artifact_refresh.sh.

ROOT=/home/cheriearjun
MOLLENS="$ROOT/MoleculeLens"
MOLPROMPT="$ROOT/MolPrompt"
MSTM="$ROOT/MoleculeSTM"
CONDA_SH="$ROOT/miniconda3/etc/profile.d/conda.sh"
WAIT_PID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait-pid)
            WAIT_PID="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

source "$CONDA_SH"
conda activate MoleculeSTM
export LD_LIBRARY_PATH="$ROOT/miniconda3/envs/MoleculeSTM/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism:$ROOT/apex:$ROOT/apex/build/lib.linux-x86_64-cpython-37:${PYTHONPATH:-}"

mkdir -p "$MSTM/logs" "$MSTM/results_aligned" "$MOLLENS/results"

if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "Waiting for PID $WAIT_PID to release the GPU..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
fi

python "$MOLLENS/scripts/extract_molprompt_metrics.py" \
    --npz "$MOLPROMPT/save_model/retrieval_chembl_sharedsplit_bs6/chembl_retrieval_best_metrics.npz" \
    --out "$MOLLENS/results/molprompt.json"

python "$MOLLENS/scripts/extract_molprompt_metrics.py" \
    --npz "$MOLPROMPT/save_model/retrieval_chembl_sharedsplit_bs6_nodrug/chembl_retrieval_best_metrics.npz" \
    --out "$MOLLENS/results/molprompt_nodrug.json"

python "$MOLLENS/scripts/evaluate_kvplm_aligned.py" \
    --source-csv "$MOLLENS/chembl_mechanisms.csv" \
    --split-manifest "$MOLLENS/splits/chembl_scaffold_seed0_train80_val10_test10.json" \
    --with-out "$MOLLENS/results/kv_plm_aligned.json" \
    --nodrug-out "$MOLLENS/results/kv_plm_aligned_nodrug.json"

cd "$MSTM"

python -u scripts/downstream_01_retrieval_ChEMBL.py \
    --molecule_type=SMILES \
    --input_model_dir=data/pretrained_MoleculeSTM/SMILES \
    --vocab_path=MoleculeSTM/bart_vocab.txt \
    --source_csv "$MOLLENS/chembl_mechanisms.csv" \
    --split_manifest "$MOLLENS/splits/chembl_scaffold_seed0_train80_val10_test10.json" \
    --text_column text_rich \
    --smiles_column smiles \
    --output_dir "$MSTM/results_aligned" \
    > "$MSTM/logs/moleculestm_aligned_with_names.log" 2>&1

python "$MOLLENS/scripts/extract_moleculestm_metrics.py" \
    --csv "$MSTM/results_aligned/results_aligned_mechanism_sharedsplit_SMILES.csv" \
    --out "$MOLLENS/results/moleculestm_aligned.json"

python -u scripts/downstream_01_retrieval_ChEMBL.py \
    --molecule_type=SMILES \
    --input_model_dir=data/pretrained_MoleculeSTM/SMILES \
    --vocab_path=MoleculeSTM/bart_vocab.txt \
    --source_csv "$MOLLENS/chembl_mechanisms.csv" \
    --split_manifest "$MOLLENS/splits/chembl_scaffold_seed0_train80_val10_test10.json" \
    --text_column text_rich \
    --smiles_column smiles \
    --remove_drug_name \
    --output_dir "$MSTM/results_aligned" \
    > "$MSTM/logs/moleculestm_aligned_no_names.log" 2>&1

python "$MOLLENS/scripts/extract_moleculestm_metrics.py" \
    --csv "$MSTM/results_aligned/results_aligned_mechanism_sharedsplit_SMILES_nodrug.csv" \
    --out "$MOLLENS/results/moleculestm_aligned_nodrug.json"

python "$MOLLENS/scripts/build_comparison_table.py" \
    --input "MoleculeLens=$MOLLENS/results/moleculelens.json" \
    --input "MoleculeSTM=$MOLLENS/results/moleculestm_aligned.json" \
    --input "MolPrompt=$MOLLENS/results/molprompt.json" \
    --input "KV-PLM=$MOLLENS/results/kv_plm.json" \
    --leakage-input "MoleculeLens=$MOLLENS/results/moleculelens.json|$MOLLENS/results/moleculelens_nodrug.json" \
    --leakage-input "MoleculeSTM=$MOLLENS/results/moleculestm_aligned.json|$MOLLENS/results/moleculestm_aligned_nodrug.json" \
    --leakage-input "MolPrompt=$MOLLENS/results/molprompt.json|$MOLLENS/results/molprompt_nodrug.json" \
    --leakage-input "KV-PLM=$MOLLENS/results/kv_plm.json|$MOLLENS/results/kv_plm_nodrug.json" \
    --out "$MOLLENS/results/aligned_model_comparison_table.md" \
    --manifest-out "$MOLLENS/results/aligned_model_comparison_manifest.json"

echo "Aligned comparison refresh complete."
echo "For camera-ready paper artifacts, run: bash $MOLLENS/scripts/run_paper_artifact_refresh.sh"
