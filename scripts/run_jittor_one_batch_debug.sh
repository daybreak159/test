#!/usr/bin/env bash
set -euo pipefail

# Run from the MemSkill repository root no matter where the script is invoked.
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
RUN_DIR="${RUN_DIR:-./jittor_controller_repro/runs/locomo_jittor_one_batch_debug}"
ENCODER="${ENCODER:-Qwen/Qwen3-Embedding-0.6B}"

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
export cache_name="${cache_name:-memskill_jittor_debug}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Keep the run fully local; this avoids W&B API-key prompts in non-interactive
# terminals while still writing local wandb artifacts.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$RUN_DIR"

echo "[INFO] repo: $REPO_ROOT"
echo "[INFO] python: $PY"
echo "[INFO] run_dir: $RUN_DIR"
echo "[INFO] controller_backend: jittor"
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
  --inner-epochs 1 \
  --outer-epochs 1 \
  --ppo-epochs 1 \
  --minibatch-size 0 \
  --action-top-k 3 \
  --session-mode full-session \
  --mem-top-k 20 \
  --mem-top-k-eval 20 \
  --reward-metric f1 \
  --locomo-train-query-sampling-ratio 1.0 \
  --controller-trace-records "$RUN_DIR/controller_trace_records.jsonl" \
  --run-dir "$RUN_DIR" \
  --wandb-run-name locomo-jittor-one-batch-debug \
  2>&1 | tee "$RUN_DIR/train.log"
