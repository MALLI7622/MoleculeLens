#!/bin/bash
# Run experiment scripts sequentially; one log per script.
source ~/miniconda3/bin/activate MoleculeSTM
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/MoleculeLens/nmi
for s in "$@"; do
  echo "=== $s start $(date)" >> ../nmi_logs/batch.log
  python $s > ../nmi_logs/${s%.py}.log 2>&1
  echo "=== $s exit $? $(date)" >> ../nmi_logs/batch.log
done
