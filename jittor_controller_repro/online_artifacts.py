"""Utilities for online MemSkill/Jittor run artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jittor_controller_repro.common import load_jsonl, write_jsonl


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compact_metrics(training_logs: list[dict[str, Any]], backend: str = "unknown") -> list[dict[str, Any]]:
    rows = []
    for idx, log in enumerate(training_logs or []):
        row = {
            "step": idx,
            "backend": backend,
            "outer_epoch": int(log.get("outer_epoch", 0)),
            "inner_epoch": int(log.get("inner_epoch", idx)),
            "num_steps": len(log.get("steps", []) or []),
        }
        for key in METRIC_KEYS:
            row[key] = safe_float(log.get(key, 0.0))
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, Any]], metric: str, output: str | Path, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    xs = [row.get("step", i) for i, row in enumerate(rows)]
    ys = [safe_float(row.get(metric, 0.0)) for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", color="#1f5fbf")
    plt.xlabel("training update")
    plt.ylabel(metric)
    plt.title(title or f"Online Jittor {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_counter(counter: Counter, output: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    items = counter.most_common()
    labels = [name for name, _ in items] or ["none"]
    values = [count for _, count in items] or [0]
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, values, color="#1f5fbf")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def step_record_counters(step_records_path: str | Path | None) -> tuple[Counter, Counter]:
    skill_counter: Counter = Counter()
    action_counter: Counter = Counter()
    if not step_records_path:
        return skill_counter, action_counter
    for row in load_jsonl(step_records_path):
        selected = row.get("selected_skills", row.get("selected_op", []))
        if isinstance(selected, str):
            selected = [selected]
        for name in selected or []:
            skill_counter[str(name)] += 1
        for action in row.get("executor_actions", []) or []:
            action_counter[str(action)] += 1
    return skill_counter, action_counter


def write_run_config(path: str | Path, args: Any, config: Any, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args) if hasattr(args, "__dict__") else dict(args or {}),
        "config": vars(config) if hasattr(config, "__dict__") else dict(config or {}),
    }
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def write_training_summary(
    path: str | Path,
    metrics: list[dict[str, Any]],
    selected_skill_counts: Counter,
    memory_action_counts: Counter,
    backend: str,
    step_records_path: str | Path | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_reward = metrics[-1]["reward"] if metrics else 0.0
    lines = [
        "# Online MemSkill Jittor Run Summary",
        "",
        f"- backend: `{backend}`",
        f"- metric rows: `{len(metrics)}`",
        f"- step records: `{step_records_path or ''}`",
        f"- final reward: `{final_reward:.6f}`",
        "",
        "## Selected Skill Counts",
        "",
    ]
    if selected_skill_counts:
        for name, count in selected_skill_counts.most_common():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- no selected skill records")
    lines.extend(["", "## Memory Action Counts", ""])
    if memory_action_counts:
        for name, count in memory_action_counts.most_common():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- no memory action records")
    lines.extend([
        "",
        "## Generated Files",
        "",
        "- `run_config.json`",
        "- `metrics.csv`",
        "- `metrics.jsonl`",
        "- `reward_curve.png`",
        "- `policy_loss_curve.png`",
        "- `value_loss_curve.png`",
        "- `entropy_curve.png`",
        "- `selected_skill_stats.png`",
        "- `memory_action_stats.png`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_online_run_artifacts(
    run_dir: str | Path,
    training_logs: list[dict[str, Any]],
    args: Any,
    config: Any,
    backend: str,
    step_records_path: str | Path | None = None,
) -> dict[str, str]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = compact_metrics(training_logs, backend=backend)

    write_run_config(run_dir / "run_config.json", args, config, extra={"backend": backend})
    write_jsonl(run_dir / "metrics.jsonl", metrics)
    write_csv(run_dir / "metrics.csv", metrics)

    metric_outputs = {
        "reward": "reward_curve.png",
        "policy_loss": "policy_loss_curve.png",
        "value_loss": "value_loss_curve.png",
        "entropy": "entropy_curve.png",
    }
    for metric, filename in metric_outputs.items():
        plot_metric(metrics, metric, run_dir / filename)

    selected_counts, action_counts = step_record_counters(step_records_path)
    plot_counter(selected_counts, run_dir / "selected_skill_stats.png", "Selected MemSkill Counts")
    plot_counter(action_counts, run_dir / "memory_action_stats.png", "Executor Memory Action Counts")

    write_training_summary(
        run_dir / "training_summary.md",
        metrics,
        selected_counts,
        action_counts,
        backend,
        step_records_path,
    )
    return {
        "run_dir": str(run_dir),
        "metrics_jsonl": str(run_dir / "metrics.jsonl"),
        "metrics_csv": str(run_dir / "metrics.csv"),
        "summary": str(run_dir / "training_summary.md"),
    }
