#!/usr/bin/env bash
set -euo pipefail

# Full small-scale LoCoMo online training run with the Jittor controller and
# Designer enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
else
  echo "[ERROR] .env not found in $REPO_ROOT" >&2
  exit 1
fi

PY="${PY:-python}"
RUN_DIR="${RUN_DIR:-./jittor_controller_repro/runs/locomo_jittor_full_small_designer_3x5}"
ENCODER="${ENCODER:-Qwen/Qwen3-Embedding-0.6B}"
TARGET_OUTER_EPOCH="${TARGET_OUTER_EPOCH:-3}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-locomo-jittor-full-small-designer-3x5}"

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python env not found or not executable: $PY" >&2
  exit 1
fi

: "${MEMSKILL_MODEL:?MEMSKILL_MODEL is not set in .env}"
: "${MEMSKILL_DESIGNER_MODEL:?MEMSKILL_DESIGNER_MODEL is not set in .env}"
: "${MEMSKILL_API_BASE:?MEMSKILL_API_BASE is not set in .env}"
: "${MEMSKILL_API_KEY:?MEMSKILL_API_KEY is not set in .env}"

# Optional Jittor compiler overrides.
# If Jittor reports an nvcc/gcc mismatch, run this script with compatible
# compiler paths, e.g. CC=/path/to/gcc CXX=/path/to/g++ cc_path=/path/to/g++.
if [[ -n "${CC:-}" ]]; then
  export CC
fi
if [[ -n "${CXX:-}" ]]; then
  export CXX
fi
if [[ -n "${cc_path:-}" ]]; then
  export cc_path
fi

export DISABLE_MULTIPROCESSING="${DISABLE_MULTIPROCESSING:-1}"
export cache_name="${cache_name:-memskill_jittor}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$RUN_DIR" "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/api_cache"

echo "[INFO] repo: $REPO_ROOT"
echo "[INFO] python: $PY"
echo "[INFO] run_dir: $RUN_DIR"
echo "[INFO] controller_backend: jittor"
echo "[INFO] designer: enabled"
echo "[INFO] target_outer_epoch: $TARGET_OUTER_EPOCH"
if [[ -n "$LOAD_CHECKPOINT" ]]; then
  echo "[INFO] load_checkpoint: $LOAD_CHECKPOINT"
fi
echo "[INFO] encoder: $ENCODER"
if [[ -n "${cc_path:-}" ]]; then
  echo "[INFO] cc_path: $cc_path"
  echo "[INFO] g++: $("$cc_path" --version | head -n 1)"
else
  echo "[INFO] cc_path: <system default>"
fi
echo "[INFO] cache_name: $cache_name"
echo "[INFO] WANDB_MODE: $WANDB_MODE"
echo "[INFO] log: $RUN_DIR/train.log"

EXTRA_ARGS=()
if [[ -n "$LOAD_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--load-checkpoint "$LOAD_CHECKPOINT" --resume-new-wandb-run)
fi

PYTHONUNBUFFERED=1 "$PY" main.py \
  --dataset locomo \
  --data-file ./data/locomo10.json \
  --model "$MEMSKILL_MODEL" \
  --designer-model "$MEMSKILL_DESIGNER_MODEL" \
  --api \
  --api-base "$MEMSKILL_API_BASE" \
  --api-key "$MEMSKILL_API_KEY" \
  --retriever contriever \
  --state-encoder "$ENCODER" \
  --op-encoder "$ENCODER" \
  --disable-flash-attn \
  --device cuda \
  --controller-backend jittor \
  --batch-size 4 \
  --inner-epochs 5 \
  --outer-epochs "$TARGET_OUTER_EPOCH" \
  --ppo-epochs 2 \
  --minibatch-size 0 \
  --action-top-k 3 \
  --session-mode full-session \
  --mem-top-k 20 \
  --mem-top-k-eval 20 \
  --reward-metric f1 \
  --locomo-train-query-sampling-ratio 0.2 \
  --enable-designer \
  --designer-freq 1 \
  --designer-max-changes 3 \
  --designer-new-skill-hint \
  --designer-reflection-cycles 1 \
  --designer-num-clusters 3 \
  --designer-samples-per-cluster 2 \
  --op-evolution-trials 1 \
  --max-designer-evolves 3 \
  --designer-early-stop-patience 2 \
  --api-cache-mode live \
  --api-cache-dir "$RUN_DIR/api_cache" \
  --save-dir "$RUN_DIR/checkpoints" \
  --log-dir "$RUN_DIR/logs" \
  --controller-trace-records "$RUN_DIR/controller_trace_records.jsonl" \
  --run-dir "$RUN_DIR" \
  --wandb-run-name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"
