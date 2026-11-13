#!/usr/bin/env bash
# Offline install on the Raspberry Pi (64-bit OS, Python 3.13). No internet needed.
#   cd namma-ai-ps2 && bash bundle/install_pi.sh
set -e
cd "$(dirname "$0")/.."

echo "arch:   $(uname -m)   (need aarch64)"
echo "python: $(python3 --version)   (wheels are built for 3.13)"
echo "cpu:    $(grep -m1 -o 'Cortex-A[0-9]*' /proc/cpuinfo || true) $(grep -m1 'Model' /proc/cpuinfo || true)"
if grep -qw asimddp /proc/cpuinfo; then
  echo "INT8 dot-product: yes (Pi 5 class) -> try the INT8 model first"
else
  echo "INT8 dot-product: NO (Pi 4 class) -> INT8 may be slow or fail; use the FP32 model"
fi
[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: not a 64-bit ARM OS"; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)' || {
  echo "ERROR: Python is not 3.13, these wheels will not install. See bundle/README.md"; exit 1; }

( cd bundle/wheels && sha256sum -c --quiet ../SHA256SUMS ) && echo "wheel checksums OK"
python3 -m venv .venv-pi
. .venv-pi/bin/activate
# --no-deps: the wheel folder is the complete, tested set (executorch's metadata asks for coremltools)
pip install --no-index --no-deps bundle/wheels/*.whl
python -c "import torch, cv2, numpy; from executorch.runtime import Runtime; print('imports OK, XNNPACK:', Runtime.get().backend_registry.is_available('XnnpackBackend'))"
echo "Done. Run:  . .venv-pi/bin/activate && python main.py --source <clip-or-folder>"
