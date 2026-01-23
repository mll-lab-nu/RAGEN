### B200 / sm_100 Compatibility Fix

If you are on an NVIDIA B200 machine and see a `UserWarning` about `sm_100` compatibility, run the following commands to upgrade your environment:

```bash
# Upgrade PyTorch to nightly with cu128 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install compatible support for B200
pip install vllm==0.11.0
python -m pip uninstall -y flashinfer-python flashinfer
python -m pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

---

# Manual Scripts to Setup Environment
```bash
conda create -n ragen python=3.12 -y
conda activate ragen

git clone git@github.com:ZihanWang314/ragen.git
cd ragen

# Install RAGEN
pip install -e .

# [B200 Only] Install PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# [Non-B200] Install PyTorch stable
# pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Setup flash-attention
# For B200:
python -m pip uninstall -y flashinfer-python flashinfer
python -m pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
# For non-B200:
# pip3 install flash-attn==2.7.4.post1 --no-build-isolation

pip install -r requirements.txt

git submodule init
git submodule update
cd verl
pip install -e .
cd ..
```
