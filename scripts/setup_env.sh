#!/usr/bin/env bash
set -euo pipefail

# Environment setup script for MemSkillJittor.
#
# Usage:
#   bash scripts/setup_env.sh
#   bash scripts/setup_env.sh --env .venv-memskill-jittor --device cu124
#   bash scripts/setup_env.sh --device cpu
#
# Notes:
# - This script does not install API keys.
# - If Jittor reports a CUDA/GCC mismatch, install a compatible compiler and
#   export CC/CXX/cc_path before running training scripts.

ENV_DIR=".venv-memskill-jittor"
PYTHON_BIN="python3"
DEVICE="cu124"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[1/6] Creating virtual environment: ${ENV_DIR}"
"${PYTHON_BIN}" -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

echo "[2/6] Upgrading pip tools"
python -m pip install --upgrade pip setuptools wheel

echo "[3/6] Installing PyTorch backend: ${DEVICE}"
case "${DEVICE}" in
  cu124)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ;;
  cu121)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ;;
  cpu)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ;;
  skip)
    echo "Skipping PyTorch installation"
    ;;
  *)
    echo "Unsupported --device value: ${DEVICE}" >&2
    echo "Supported values: cu124, cu121, cpu, skip" >&2
    exit 1
    ;;
esac

echo "[4/6] Installing Jittor"
pip install jittor

echo "[5/6] Installing common Python dependencies"
pip install \
  numpy \
  scipy \
  pandas \
  matplotlib \
  tqdm \
  pyyaml \
  python-dotenv \
  scikit-learn \
  sentence-transformers \
  transformers \
  openai \
  pytest

if [[ -f requirements.txt ]]; then
  echo "[5/6] Installing requirements.txt"
  pip install -r requirements.txt
fi

echo "[6/6] Writing .env.example"
cat > .env.example <<'EOF'
# OpenAI-compatible API settings.
# Copy this file to .env and fill in real values locally.
MEMSKILL_MODEL=
MEMSKILL_DESIGNER_MODEL=
MEMSKILL_API_BASE=
MEMSKILL_API_KEY=

# Optional offline cache settings.
# HF_HUB_OFFLINE=1
# TRANSFORMERS_OFFLINE=1
# WANDB_MODE=offline
EOF

echo
echo "Environment setup complete."
echo "Activate it with:"
echo "  source ${ENV_DIR}/bin/activate"
echo
echo "Then copy .env.example to .env and fill in API settings before online training."
