#!/usr/bin/env bash
# One-shot dev environment: matches the course-style cells below, plus this repo's extras.
#
# Original notebook-style (equivalent — no need to run pip/pyvirtualdisplay twice):
#   !pip install stable-baselines3 "gymnasium[box2d]" huggingface_sb3 pyvirtualdisplay optuna ipywidgets
#   !sudo apt-get update
#   !sudo apt-get install -y python3-opengl
#   !apt install -y ffmpeg
#   !apt install -y xvfb
#   !pip install pyvirtualdisplay   # duplicate of line 1 — omitted here
#   !pip3 install pyvirtualdisplay  # duplicate — omitted here
#
# Usage:
#   chmod +x /workspace/RL-LunarLander/setup_env.sh && /workspace/RL-LunarLander/setup_env.sh
# Skip apt: SKIP_SYSTEM=1 ./setup_env.sh

set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="${ROOT}/.venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

echo "==> RL-LunarLander: create env in ${VENV}"

if [[ "${SKIP_SYSTEM:-0}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "==> System packages (same as apt-get update + opengl + ffmpeg + xvfb)..."
  sudo apt-get update -qq
  # One line = update + python3-opengl + ffmpeg + xvfb; also swig/cmake for Box2D, python3-venv for venv
  sudo apt-get install -y -qq \
    python3-opengl \
    ffmpeg \
    xvfb \
    swig \
    cmake \
    python3-venv
else
  echo "==> Skipping apt (SKIP_SYSTEM=1 or no apt-get). On Linux run manually:"
  echo "    sudo apt-get update && sudo apt-get install -y python3-opengl ffmpeg xvfb swig cmake python3-venv"
fi

if [[ ! -x "${PY}" ]]; then
  python3 -m venv "${VENV}"
fi

"${PIP}" install --upgrade pip

echo "==> Pip: course baseline (your first !pip install line, exact set)"
"${PIP}" install \
  stable-baselines3 \
  "gymnasium[box2d]" \
  huggingface_sb3 \
  pyvirtualdisplay \
  optuna \
  ipywidgets \
  opencv-python-headless

echo "==> Pip: this repo + notebooks (Hub upload, plots, Jupyter kernel — not in the short course line)"
"${PIP}" install \
  huggingface_hub \
  matplotlib \
  ipykernel

# Register Jupyter / Cursor kernel (needs ipykernel above)
"${PY}" -m ipykernel install --user \
  --name=rl-lunarlander \
  --display-name="Python (RL-LunarLander .venv)"

echo ""
echo "Done. Activate:"
echo "  source ${VENV}/bin/activate"
echo "Or: ${PY}"
echo "Jupyter kernel: Python (RL-LunarLander .venv)"
