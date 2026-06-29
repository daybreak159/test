"""Collect or convert API-backed MemSkill traces.

This script provides the Mode B entrypoint.  In the first pass it supports two
safe workflows:

1. Convert JSON/JSONL step dumps emitted from an instrumented MemSkill run.
2. Fall back to synthetic trace generation for dry-run validation.

The original online MemSkill run should be used to produce step records with
the fields documented in ``README.md``.  Keeping conversion separate from
training ensures PyTorch/Jittor comparisons use a fixed trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .synthetic_generator import generate_synthetic_trace
from .trace_schema import ControllerTrace, compute_returns_and_advantages, save_trace


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "steps" in data:
        return list(data["steps"])
    raise ValueError("expected a list of step records or a dict with 'steps'")


def _array(record: dict[str, Any], key: str, dtype=np.float32):
    if key not in record:
        raise ValueError(f"missing required step field: {key}")
    return np.asarray(record[key], dtype=dtype)


def convert_records_to_trace(
    records: list[dict[str, Any]],
    output: Path,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    use_record_targets: bool = False,
    source_name: str = "api-cache",
) -> ControllerTrace:
    states = []
    op_embs = []
    new_masks = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    cached_returns = []
    cached_advantages = []
    steps = []
    episodes = []

    for i, record in enumerate(records):
        states.append(_array(record, "state_embedding"))
        op_embs.append(_array(record, "op_embeddings"))
        new_masks.append(
            None if record.get("new_op_mask") is None else np.asarray(record["new_op_mask"], dtype=np.float32)
        )
        actions.append(np.asarray(record.get("selected_action", record.get("action_idx")), dtype=np.int64))
        log_probs.append(float(record.get("old_log_prob", record["log_prob"])))
        values.append(float(record["value"]))
        rewards.append(float(record.get("reward", record.get("process_reward", 0.0))))
        dones.append(float(record.get("done", 0.0)))
        if "return" in record and "advantage" in record:
            cached_returns.append(float(record["return"]))
            cached_advantages.append(float(record["advantage"]))
        steps.append(
            {
                "source": record.get("source", source_name),
                "episode_id": record.get("episode_id", record.get("trace_id")),
                "step": record.get("step", i),
                "session_idx": record.get("session_idx"),
                "candidate_ops": record.get("candidate_ops", []),
                "selected_skills": record.get("selected_skills", record.get("selected_op", [])),
                "selected_action": np.asarray(actions[-1]).tolist(),
                "process_reward": rewards[-1],
                "qa_reward": record.get("qa_reward"),
                "executor_prompt": record.get("executor_prompt", ""),
                "executor_response": record.get("executor_response", ""),
                "exec_results": record.get("exec_results", []),
            }
        )
        if "episode" in record:
            episodes.append(record["episode"])

    actions_arr = np.asarray(actions, dtype=np.int64)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.float32)
    if use_record_targets and len(cached_returns) == len(records) and len(cached_advantages) == len(records):
        returns = np.asarray(cached_returns, dtype=np.float32)
        advantages = np.asarray(cached_advantages, dtype=np.float32)
    else:
        returns, advantages = compute_returns_and_advantages(
            rewards_arr, values_arr, dones_arr, gamma=gamma, gae_lambda=gae_lambda
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    trace = ControllerTrace(
        states=np.asarray(states, dtype=np.float32),
        op_embs=op_embs,
        new_op_masks=new_masks,
        actions=actions_arr,
        log_probs=np.asarray(log_probs, dtype=np.float32),
        values=values_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        returns=returns,
        advantages=advantages.astype(np.float32),
        metadata={
            "source": source_name,
            "converted_from_records": True,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "use_record_targets": use_record_targets,
        },
        steps=steps,
        episodes=episodes,
    )
    save_trace(trace, output)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an API-cached controller trace.")
    parser.add_argument("--input-records", help="JSON/JSONL records from a MemSkill API run")
    parser.add_argument("--output", default="jittor_controller_repro/runs/api_cached_trace.npz")
    parser.add_argument("--dry-run-synthetic", action="store_true")
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use-record-targets", action="store_true",
                        help="Use cached return/advantage fields when present instead of recomputing GAE.")
    parser.add_argument("--source-name", default="api-cache")
    args = parser.parse_args()

    output = Path(args.output)
    if args.input_records:
        trace = convert_records_to_trace(
            read_records(Path(args.input_records)),
            output,
            use_record_targets=args.use_record_targets,
            source_name=args.source_name,
        )
        print(f"converted {trace.n_steps} API-cached steps to {output}")
        return
    if args.dry_run_synthetic:
        trace = generate_synthetic_trace(n_steps=args.n_steps, seed=args.seed)
        trace.metadata["source"] = "api-cache-dry-run-synthetic"
        save_trace(trace, output)
        print(f"saved dry-run trace with {trace.n_steps} steps to {output}")
        return
    raise SystemExit("Provide --input-records or use --dry-run-synthetic.")


if __name__ == "__main__":
    main()
