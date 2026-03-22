#!/bin/bash

# Exit on error
set -e

echo "Setting up webshop..."
echo "NOTE: please run scripts/setup_ragen_old.sh before running this script"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

build_constraints_file() {
    local constraints_file
    constraints_file=$(mktemp -t ragen_webshop_constraints.XXXXXX)
    python - "$constraints_file" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

protected = [
    "torch",
    "transformers",
    "vllm",
    "flash-attn",
    "tokenizers",
]

lines = []
for package in protected:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    lines.append(f"{package}=={version}")

Path(sys.argv[1]).write_text("\n".join(lines) + ("\n" if lines else ""))
PY
    echo "$constraints_file"
}

ensure_conda_env() {
    if ! command -v conda &> /dev/null; then
        echo "Conda is not installed. Please install Conda first."
        exit 1
    fi
    if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "ragen" ]; then
        CONDA_PATH=$(conda info --base)
        source "$CONDA_PATH/etc/profile.d/conda.sh"
        conda activate ragen
    fi
}

ensure_conda_env

# `pkg_resources` is provided by setuptools. Reinstall the packaging toolchain
# here because this script must work after either base setup path.
print_step "Ensuring packaging toolchain is available..."
pip install -U pip "setuptools<70.0.0" wheel

CONSTRAINTS_FILE=$(build_constraints_file)
cleanup() {
    rm -f "${CONSTRAINTS_FILE}"
}
trap cleanup EXIT

# WebShop-specific system dependencies.
print_step "Installing WebShop system dependencies..."
sudo apt update
sudo apt install default-jdk -y
conda install -c conda-forge openjdk=21 maven -y

# WebShop-only Python extras. Avoid reinstalling the full base requirements
# because the B200 setup path intentionally manages those separately.
# Use constraints so pip cannot silently upgrade fragile core packages.
print_step "Installing WebShop Python dependencies..."
pip install -c "${CONSTRAINTS_FILE}" beautifulsoup4 cleantext flask html2text rank_bm25 pyserini thefuzz gdown spacy rich

# WebShop package and models.
print_step "Installing WebShop package..."
pip install -e external/webshop-minimal/ --no-dependencies
print_step "Downloading spaCy models..."
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

print_step "Downloading data..."
python scripts/download_data.py

# Optional: download full data set. Google Drive links are brittle and should
# not fail the whole setup if they are rate-limited or permissions change.
print_step "Downloading full data set (best effort)..."
conda install -c conda-forge gdown -y
FULL_DATA_DIR="external/webshop-minimal/webshop_minimal/data/full"
mkdir -p "${FULL_DATA_DIR}"
pushd "${FULL_DATA_DIR}" >/dev/null
if ! gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; then
    echo "Warning: failed to download WebShop full dataset file 'items_shuffle'. Continuing without it."
fi
if ! gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; then
    echo "Warning: failed to download WebShop full dataset file 'items_ins_v2'. Continuing without it."
fi
popd >/dev/null

echo -e "${GREEN}Installation completed successfully!${NC}"
echo "To activate the environment, run: conda activate ragen"
