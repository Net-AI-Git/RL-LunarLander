#!/usr/bin/env bash
# One-shot dev environment: system packages (Linux/apt) + Python venv + pip deps.
# Usage:
#   chmod +x setup_env.sh && ./setup_env.sh
# Skip apt (no sudo / non-Debian):  SKIP_SYSTEM=1 ./setup_env.sh

set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="${ROOT}/.venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

echo "==> RL-LunarLander: create env in ${VENV}"

if [[ "${SKIP_SYSTEM:-0}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system packages (needs sudo for apt)..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    swig cmake python3-opengl ffmpeg xvfb \
    python3-venv
else
  echo "==> Skipping apt (SKIP_SYSTEM=1 or no apt-get). On Linux install manually:"
  echo "    sudo apt-get update && sudo apt-get install -y swig cmake python3-opengl ffmpeg xvfb python3-venv"
fi

if [[ ! -x "${PY}" ]]; then
  python3 -m venv "${VENV}"
fi

"${PIP}" install --upgrade pip
# Python deps (Box2D / RL stack / Hub / notebooks)
"${PIP}" install \
  stable-baselines3 \
  "gymnasium[box2d]" \
  huggingface_sb3 \
  huggingface_hub \
  pyvirtualdisplay \
  optuna \
  ipywidgets \
  opencv-python-headless \
  matplotlib

echo ""
echo "Done. Activate:"
echo "  source ${VENV}/bin/activate"
echo "Or pick interpreter: ${PY}"
