#!/usr/bin/env bash

set -euo pipefail

# Environment setup script for the RAGEN environment.
#
# Validation:
# - Verified on NVIDIA H100, H200, and B200.
#
# Environment coverage:
# - Supports bandit, sokoban, frozenlake, metamathqa, countdown
#
# Optional environments (install with flags):
#   --with-search    Search (HotpotQA) environment (~87 GB data download)
#
# Examples:
#   bash scripts/setup_ragen_release.sh                   # base only
#   bash scripts/setup_ragen_release.sh --with-search     # base + search

ENV_NAME="ragen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse optional environment flags
WITH_SEARCH=0
for arg in "$@"; do
    case "$arg" in
        --with-search)  WITH_SEARCH=1 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

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

setup_search() {
    print_step "Installing search environment dependencies..."
    pip install sentence-transformers flask

    local DATA_DIR="./search_data"
    local INDICES_DIR="${DATA_DIR}/prebuilt_indices"
    local WIKI_DIR="${DATA_DIR}/wikipedia"

    print_step "Downloading search index data (wiki corpus + FAISS index shards, ~87 GB)..."
    python scripts/download_search_index.py --data_dir "$DATA_DIR"

    # Merge FAISS index shards
    local INDEX_FILE="${INDICES_DIR}/e5_Flat.index"
    if [ -f "$INDEX_FILE" ]; then
        echo "e5_Flat.index already exists ($(du -h "$INDEX_FILE" | cut -f1))"
    else
        print_step "Merging index shards -> e5_Flat.index..."
        if [ -f "${INDICES_DIR}/part_aa" ] && [ -f "${INDICES_DIR}/part_ab" ]; then
            cat "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab" > "$INDEX_FILE"
            rm -f "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab"
            echo "Created e5_Flat.index ($(du -h "$INDEX_FILE" | cut -f1))"
        else
            echo "ERROR: Index shards not found in ${INDICES_DIR}" >&2
            exit 1
        fi
    fi

    # Convert wiki-18.jsonl -> corpus.json
    local CORPUS_FILE="${INDICES_DIR}/corpus.json"
    local WIKI_JSONL="${WIKI_DIR}/wiki-18.jsonl"

    if [ -f "$CORPUS_FILE" ]; then
        echo "corpus.json already exists ($(du -h "$CORPUS_FILE" | cut -f1))"
    else
        print_step "Converting wiki-18.jsonl -> corpus.json..."
        if [ ! -f "$WIKI_JSONL" ]; then
            echo "ERROR: ${WIKI_JSONL} not found" >&2
            exit 1
        fi
        python3 -c "
import json
from tqdm import tqdm

input_path = '${WIKI_JSONL}'
output_path = '${CORPUS_FILE}'

print(f'Reading {input_path}...')
corpus = []
with open(input_path, 'r') as f:
    for line in tqdm(f, desc='Loading wiki-18.jsonl'):
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        text = doc.get('text', doc.get('contents', doc.get('content', '')))
        title = doc.get('title', '')
        if title and text:
            corpus.append(f'{title} {text}')
        elif text:
            corpus.append(text)

print(f'Writing {len(corpus)} documents to {output_path}...')
with open(output_path, 'w') as f:
    json.dump(corpus, f)
print(f'Done! corpus.json = {len(corpus)} docs')
"
    fi

    # Prepare HotpotQA parquet data
    print_step "Preparing HotpotQA parquet data..."
    python scripts/prepare_search_data.py --output_dir data/search

    print_step "Search environment setup complete"
    echo "To start the retrieval server:"
    echo "  CUDA_VISIBLE_DEVICES='' python scripts/retrieval/server.py --port 8001"
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

    # Reinstall setuptools<70 (vllm may upgrade it, breaking pkg_resources for gym_sokoban)
    pip install "setuptools<70.0.0"

    print_step "Downloading project data"
    python scripts/download_data.py

    # Optional: search environment
    if [ "$WITH_SEARCH" -eq 1 ]; then
        setup_search
    fi

    print_step "Setup complete"
    echo "Activate with: conda activate ${ENV_NAME}"
}

main "$@"
