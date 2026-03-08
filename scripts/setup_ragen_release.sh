#!/usr/bin/env bash

set -euo pipefail

# Environment setup script for the RAGEN environment.
#
# Validation:
# - Verified on NVIDIA H100, H200, and B200.
#
# Environment coverage:
# - Supports bandit, sokoban, frozenlake, metamathqa, countdown

ENV_NAME="ragen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

print_step() {
    echo
    echo "[setup_ragen_release] $1"
}

ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required but was not found in PATH." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
}

validate_repo_root() {
    if [[ ! -d "${PROJECT_ROOT}/verl" ]]; then
        echo "Could not find the RAGEN repository root from ${SCRIPT_DIR}." >&2
        exit 1
    fi
}

ensure_env() {
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        print_step "Creating conda environment ${ENV_NAME} with Python 3.12"
        conda create -n "${ENV_NAME}" python=3.12 -y
    else
        print_step "Using existing conda environment ${ENV_NAME}"
    fi

    print_step "Activating conda environment ${ENV_NAME}"
    conda activate "${ENV_NAME}"
}

main() {
    ensure_conda
    validate_repo_root
    cd "${PROJECT_ROOT}"

    ensure_env

    print_step "Initializing git submodules"
    git submodule update --init --recursive

    print_step "Installing RAGEN in editable mode"
    pip install -e . --no-deps

    print_step "Installing verl dependencies for v0.6.1"
    pushd verl >/dev/null
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
    pip install --no-deps -e .
    popd >/dev/null

    print_step "Installing release environment dependencies"
    pip install \
        IPython \
        matplotlib \
        gym \
        gym_sokoban \
        gymnasium \
        "gymnasium[toy-text]" \
        debugpy \
        together \
        anthropic \
        faiss-cpu==1.11.0 \
        numpy==1.26.4

    print_step "Downloading project data"
    python scripts/download_data.py

    print_step "Setup complete"
    echo "Activate with: conda activate ${ENV_NAME}"
}

main "$@"
