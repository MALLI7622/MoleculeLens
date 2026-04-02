#!/bin/bash
# MoleculeSTM Environment Setup Script
# Usage: bash setup_env.sh
# Assumes: MolBART and apex are sibling directories of this repo
# i.e.  ~/MoleculeSTM/  ~/MolBART/  ~/apex/

set -e

# --- Locate and initialise conda ---
CONDA_BASE=""
for candidate in \
    "$HOME/miniconda3" "$HOME/anaconda3" \
    "/opt/conda" "/usr/local/miniconda3" "/usr/local/anaconda3"
do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "Using conda at: $CONDA_BASE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"  # parent of MoleculeSTM
MOLBART_DIR="$REPO_ROOT/MolBART"
APEX_DIR="$REPO_ROOT/apex"

ENV_NAME="MoleculeSTM"
PIP_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

echo "=== [1/7] Creating conda environment: $ENV_NAME (Python 3.7) ==="
# Accept Anaconda TOS if required (non-interactive installs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda create -n $ENV_NAME python=3.7 -y

echo "=== [2/7] Installing conda packages (rdkit, cudatoolkit, spacy) ==="
conda install -n $ENV_NAME -y -c rdkit rdkit=2020.09.1.0
conda install -n $ENV_NAME -y -c conda-forge cudatoolkit=11.1
conda install -n $ENV_NAME -y -c conda-forge spacy
conda install -n $ENV_NAME -y boto3

echo "=== [3/7] Installing PyTorch 1.9.1+cu111 ==="
$PIP_BIN install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "=== [4/7] Installing PyTorch Geometric (PyG 2.0.3) ==="
$PIP_BIN install \
    torch-scatter==2.0.9 \
    torch-sparse==0.6.12 \
    torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 \
    torch-geometric==2.0.3 \
    -f https://data.pyg.org/whl/torch-1.9.1+cu111.html

echo "=== [5/7] Installing pip packages from requirements.txt ==="
$PIP_BIN install -r "$SCRIPT_DIR/requirements.txt"
$PIP_BIN install -e "$SCRIPT_DIR"

echo "=== [6/7] Installing Megatron-LM from MolBART ==="
if [ -d "$MOLBART_DIR/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism" ]; then
    $PIP_BIN install "$MOLBART_DIR/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism"
else
    echo "WARNING: MolBART not found at $MOLBART_DIR — cloning..."
    git clone https://github.com/MolecularAI/MolBART.git --branch megatron-molbart-with-zinc "$MOLBART_DIR"
    $PIP_BIN install "$MOLBART_DIR/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism"
fi

echo "=== [7/7] Installing apex (CUDA C++ compilation) ==="
if [ -d "$APEX_DIR" ]; then
    pushd "$APEX_DIR"
    $PIP_BIN install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --global-option="--cpp_ext" --global-option="--cuda_ext" .
    popd
else
    echo "WARNING: apex not found at $APEX_DIR — cloning..."
    git clone https://github.com/chao1224/apex.git "$APEX_DIR"
    pushd "$APEX_DIR"
    $PIP_BIN install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --global-option="--cpp_ext" --global-option="--cuda_ext" .
    popd
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate with:"
echo "  conda activate $ENV_NAME"
echo "  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH\""
echo ""
echo "Add the LD_LIBRARY_PATH line to your ~/.bashrc or conda activate hook to make it permanent."
