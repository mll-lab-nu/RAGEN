# Scripts README

## Release Environment Setup

Use the setup flow below for the validated RAGEN environment.

This setup has been validated on `H100`, `H200`, and `B200`, and supports:
- `bandit`
- `sokoban`
- `frozenlake`
- `metamathqa`
- `countdown`

## Requirements

- `CUDA >= 12.8`

### 1. Clone the repository

```bash
git clone https://github.com/CHIGUI0/RAGEN.git
cd RAGEN
```

### 2. Create and activate the conda environment

```bash
conda create -n ragen python=3.12 -y
conda activate ragen
```

### 3. Run the environment setup script

```bash
bash scripts/setup_ragen_release.sh
```

### requirements for deepcoder env
1. pip install setuptools==68.2.2