#!/usr/bin/env bash
set -euo pipefail

# Run exactly one additional outer epoch for the PyTorch baseline run.
# Usage:
#   ./scripts/run_torch_locomo_next_outer_epoch.sh 1
#   ./scripts/run_torch_locomo_next_outer_epoch.sh 2
#   ./scripts/run_torch_locomo_next_outer_epoch.sh 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-}"
if [[ -z "$TARGET" || ! "$TARGET" =~ ^[0-9]+$ || "$TARGET" -lt 1 ]]; then
  echo "Usage: $0 <target_outer_epoch_number>" >&2
  exit 1
fi

export RUN_DIR="${RUN_DIR:-./jittor_controller_repro/runs/locomo_torch_full_small_designer_epochwise}"
export RUN_NAME="${RUN_NAME:-locomo-torch-full-small-designer-epochwise}"
export TARGET_OUTER_EPOCH="$TARGET"

if [[ "$TARGET" -gt 1 ]]; then
  PREV=$((TARGET - 1))
  PREV_CKPT="$RUN_DIR/checkpoints/${RUN_NAME}_epoch_${PREV}.pt"
  if [[ ! -f "$PREV_CKPT" ]]; then
    echo "[ERROR] Previous checkpoint not found: $PREV_CKPT" >&2
    echo "        Run target epoch $PREV first, or set RUN_DIR/RUN_NAME to the existing run." >&2
    exit 1
  fi
  export LOAD_CHECKPOINT="$PREV_CKPT"
else
  export LOAD_CHECKPOINT=""
fi

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train_epoch_${TARGET}.log"

echo "[INFO] target outer epoch: $TARGET"
echo "[INFO] run_dir: $RUN_DIR"
echo "[INFO] log: $LOG"

./scripts/run_torch_locomo_full_small_designer.sh 2>&1 | tee "$LOG"
