#!/bin/bash
# E22: MoleculeLens phase subsets + release weights (e22b), then MolPrompt (no-drug text) retrained on the
# E21 all-phase training pool capped at 150 atoms (e22a), then the paired comparison (e22c).
# Shuts the VM down at the end, whatever happened (see NEXT_STEPS.md E22: a crash once left it idle 23 days).
source ~/miniconda3/bin/activate MoleculeSTM
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
LOG=~/MoleculeLens/nmi_logs/e22_molprompt_allphase.log
trap 'echo "shutdown $(date)" >> $LOG; sudo shutdown -h now' EXIT
cd ~/MoleculeLens/nmi
python e22b_phase_subsets_release.py > ../nmi_logs/e22b_phase_subsets_release.log 2>&1
echo "e22b exit $? $(date)" > $LOG
python e22a_molprompt_allphase_data.py >> $LOG 2>&1 || exit 1
cd ~/MolPrompt
P=data_allphase/MoleculeNet_data/chembl/processed
rm -rf $P   # prepare_chembl.py silently reuses existing files
python prepare_chembl.py --source ~/MoleculeLens/chembl_allphase_molprompt.csv --dataspace_path data_allphase >> $LOG 2>&1 || exit 1
# 30 epochs: earlier runs peaked at epoch 23-27 with half the data (see NEXT_STEPS.md E22)
python -u main_chembl_retrieval.py --device 0 --batch_size 6 --epochs 30 --remove_drug_name \
  --source_csv ~/MoleculeLens/chembl_allphase_molprompt.csv \
  --data_processed_path $P/data_processed.pt --smiles_csv $P/smiles.csv \
  --split_manifest ~/MoleculeLens/splits/chembl_allphase_molprompt.json \
  --output_dir save_model/retrieval_chembl_allphase_bs6_nodrug >> $LOG 2>&1
echo "molprompt exit $? $(date)" >> $LOG
cd ~/MoleculeLens/nmi
python e22c_molprompt_allphase_compare.py > ../nmi_logs/e22c_molprompt_allphase_compare.log 2>&1
echo "e22c exit $? $(date)" >> $LOG
