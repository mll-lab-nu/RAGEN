#!/bin/bash

# Exit on error
set -e

echo "Starting setup for B200 (verl v0.6.1)..."

# Create and activate conda environment if not exists
if ! conda env list | grep -q "ragen"; then
    echo "[Step] Creating conda environment 'ragen' with Python 3.12..."
    conda create -n ragen python=3.12 -y
else
    echo "[Step] Conda environment 'ragen' already exists"
fi

# Need to source conda for script environment
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate ragen

echo "[Step] Initializing and updating submodules..."
git submodule update --init --recursive

echo "[Step] Installing RAGEN package (no deps)..."
pip install -e . --no-deps

echo "[Step] Installing verl and its dependencies..."
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..

echo "[Step] Installing additional dependencies for environments..."
pip install IPython matplotlib gym gym_sokoban gymnasium "gymnasium[toy-text]" debugpy together anthropic faiss-cpu==1.11.0 numpy==1.26.4

# echo "[Step] Downloading datasets..."
# python scripts/download_data.py

# echo "[Step] Installing spatial environment..."
# pip install -e ragen/env/spatial/Base

# echo "[Step] Setup WebShop environment + data..."
# bash scripts/setup_webshop.sh

# echo "[Step] Installing Lean environment dependencies..."
# pip install -e ".[lean]"

# echo "[Step] Setup Alfworld environment + data..."
# pip install alfworld
# Note: alfworld-download might be interactive or take a long time
# alfworld-download --extra

echo "------------------------------------------------"
echo "Setup completed successfully!"
echo "To activate the environment, run: conda activate ragen"
