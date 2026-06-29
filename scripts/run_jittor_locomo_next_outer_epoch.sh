#!/usr/bin/env bash
set -euo pipefail

# Run exactly one additional outer epoch for the full small-scale Jittor LoCoMo
# experiment. The target epoch is passed as the first argument:
#   ./scripts/run_jittor_locomo_next_outer_epoch.sh 1
#   ./scripts/run_jittor_locomo_next_outer_epoch.sh 2
#   ./scripts/run_jittor_locomo_next_outer_epoch.sh 3
#
# For target epoch N > 1, this script loads the checkpoint produced by epoch N-1.
# In MemSkill, --outer-epochs is the total target outer epoch count; resume logic
# skips already completed outer epochs based on completed_outer_epoch in the
# checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-}"
if [[ -z "$TARGET" || ! "$TARGET" =~ ^[0-9]+$ || "$TARGET" -lt 1 ]]; then
  echo "Usage: $0 <target_outer_epoch_number>" >&2
  exit 1
fi

export RUN_DIR="${RUN_DIR:-./jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise}"
export RUN_NAME="${RUN_NAME:-locomo-jittor-full-small-designer-epochwise}"
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

# Keep a per-epoch log while the base script also writes the latest train.log.
./scripts/run_jittor_locomo_full_small_designer.sh 2>&1 | tee "$LOG"
