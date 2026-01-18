This setup targets verl v0.6.1 and has been validated on B200 with bandit, sokoban, and frozen_lake environments.

## Clone and Setup

```bash
# Clone the repository
git clone https://github.com/CHIGUI0/RAGEN.git
cd RAGEN

# Create and activate conda environment
conda create -n ragen python=3.12 -y
conda activate ragen

# Switch to the update_verl branch and init submodules
git switch update_verl
git submodule update --init --recursive
```

## Install RAGEN and verl

```bash
# Install RAGEN package (no deps to avoid conflicts)
pip install -e . --no-deps

# Install verl and its dependencies
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..

# Install additional dependencies for environments
pip install IPython matplotlib gym gym_sokoban gymnasium "gymnasium[toy-text]" debugpy together anthropic faiss-cpu==1.11.0 numpy==1.26.4
```

## Optional

```bash
# Download datasets
python scripts/download_data.py

# Install spatial environment
pip install -e ragen/env/spatial/Base

# Setup WebShop environment + data (includes JDK and spaCy models)
bash scripts/setup_webshop.sh

# Install Lean environment dependencies
pip install -e ".[lean]"
```
