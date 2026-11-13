#!/usr/bin/env bash
# Install on the Raspberry Pi (64-bit OS). Offline for Python 3.11 / 3.12 / 3.13; any other
# version needs internet once (uv fetches Python 3.13).
#   cd namma-ai-ps2 && bash bundle/install_pi.sh
set -e
cd "$(dirname "$0")/.."

PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "arch:   $(uname -m)   (need aarch64)"
echo "python: $PYV   (offline wheels: 3.11, 3.12, 3.13)"
echo "board:  $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "ram:    $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
if grep -qw asimddp /proc/cpuinfo; then
  echo "INT8 dot-product: yes (Pi 5 class) -> INT8 model should be the fast one"
else
  echo "INT8 dot-product: NO (Pi 4 class) -> INT8 may be slow or fail; run_all.sh falls back to FP32"
fi
[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: not a 64-bit ARM OS (32-bit Pi OS cannot run these wheels)"; exit 1; }
[ "$(getconf LONG_BIT)" = "64" ] || { echo "ERROR: 64-bit kernel but 32-bit userland; needs a 64-bit OS image"; exit 1; }

rm -rf .venv-pi
case "$PYV" in
  3.11|3.12|3.13)
    python3 -m venv .venv-pi 2>/dev/null || {
      echo "python3-venv is missing. Fix:  sudo apt install -y python3-venv   (or: pip install --user virtualenv)"; exit 1; } ;;
  *)
    echo "Python $PYV has no offline wheels here -> fetching Python 3.13 with uv (needs internet)"
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    uv venv --python 3.13 --seed .venv-pi ;;
esac
. .venv-pi/bin/activate
if ls bundle/wheels/*.whl >/dev/null 2>&1; then
  ( cd bundle/wheels && sha256sum -c --quiet ../SHA256SUMS ) && echo "wheel checksums OK"
  # --no-deps: the pinned list is the complete, tested set (executorch's metadata asks for coremltools)
  pip install --no-index --find-links bundle/wheels --no-deps -r bundle/requirements-pi.txt
else
  echo "bundle/wheels/ is empty (it is not in git) -> installing the same pinned versions from the internet"
  pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r bundle/requirements-pi.txt
fi
python -c "import torch, cv2, numpy; from executorch.runtime import Runtime; print('imports OK, XNNPACK:', Runtime.get().backend_registry.is_available('XnnpackBackend'))"
echo "Done. Next:  bash run_all.sh <clip-or-folder> [gate|junction] [--every N]"
