#!/usr/bin/env bash
# One-time setup on a fresh Prime Intellect pod (ubuntu_22_cuda_12 image).
#   scp code.tgz pod:~ && ssh pod 'bash -s' < scripts/setup_pod.sh
# (scripts/launch.sh does this for you.)
set -euo pipefail
cd ~
mkdir -p icl && tar -xzf code.tgz -C icl
python3 -m pip install -q --upgrade pip
python3 -m pip install -q "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install -q numpy "huggingface_hub[hf_xet]" zstandard requests wandb matplotlib
python3 - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF
nproc; free -g | head -2; df -h ~ | tail -1
