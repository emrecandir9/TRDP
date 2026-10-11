#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-review}
case "$mode" in review|cuda) ;; *) echo 'Usage: ./scripts/setup.sh [review|cuda]' >&2; exit 2;; esac
python=${PYTHON:-python3}
"$python" -c 'import sys; assert (3, 12) <= sys.version_info[:2] <= (3, 13), "Use Python 3.12 or 3.13"'
if [[ "$mode" == cuda ]]; then
  [[ "$(uname -s)" == Linux ]] || { echo 'CUDA generation requires Linux.' >&2; exit 1; }
  nvidia-smi >/dev/null
fi
"$python" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
if [[ "$mode" == cuda ]]; then
  .venv/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install diffusers==0.41.0 transformers==5.19.0 accelerate==1.15.0 safetensors==0.8.0 huggingface-hub==1.33.0
fi
echo 'Ready. Activate with: source .venv/bin/activate'
