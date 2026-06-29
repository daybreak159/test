"""Export compact online MemSkill training logs from a checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from jittor_controller_repro.common import write_jsonl
from jittor_controller_repro.online_artifacts import write_csv


METRIC_KEYS = [
    "reward",
    "raw_performance",
    "policy_loss",
    "value_loss",
    "entropy",
    "topk_entropy",
    "topk_mass",
    "topk_bin_entropy",
    "approx_kl",
    "clip_frac",
    "explained_variance",
    "value_mean",
    "return_mean",
    "advantage_mean",
    "n_updates",
]


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _compact_step(log: dict[str, Any], index: int, backend: str, run_name: str) -> dict[str, Any]:
    row = {
        "run_name": run_name,
        "backend": backend,
        "step": index,
        "inner_epoch": int(log.get("inner_epoch", index)),
        "num_steps": len(log.get("steps", []) or []),
    }
    for key in METRIC_KEYS:
        value = log.get(key)
        if value is None:
            row[key] = 0.0
        elif isinstance(value, (int, float)):
            row[key] = float(value)
        else:
            row[key] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact online training logs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    ckpt = _load_checkpoint(args.checkpoint)
    backend = str(ckpt.get("controller_backend", "unknown"))
    run_name = args.run_name or str(ckpt.get("wandb_run_name") or Path(args.checkpoint).stem)
    logs = ckpt.get("training_logs", []) or []
    rows = [_compact_step(log, i, backend, run_name) for i, log in enumerate(logs)]
    write_jsonl(args.jsonl, rows)
    if args.csv:
        write_csv(args.csv, rows)
    print(
        f"exported {len(rows)} rows from {args.checkpoint} "
        f"(backend={backend}) to {args.jsonl}"
    )


if __name__ == "__main__":
    main()
