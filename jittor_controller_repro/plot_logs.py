"""Plot PyTorch/Jittor training logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from jittor_controller_repro.common import load_jsonl


def _x_values(rows: list[dict]) -> list[int | float]:
    values: list[int | float] = []
    for idx, row in enumerate(rows):
        if "epoch" in row:
            values.append(row["epoch"])
        elif "step" in row:
            values.append(row["step"])
        elif "inner_epoch" in row:
            values.append(row["inner_epoch"])
        else:
            values.append(idx)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot controller reproduction logs.")
    parser.add_argument("--torch-log", default="jittor_controller_repro/runs/torch_train.jsonl")
    parser.add_argument("--jittor-log", default="jittor_controller_repro/runs/jittor_train.jsonl")
    parser.add_argument("--metric", default="total_loss")
    parser.add_argument("--output", default="jittor_controller_repro/runs/loss_curve.png")
    args = parser.parse_args()

    torch_rows = load_jsonl(args.torch_log)
    jittor_rows = load_jsonl(args.jittor_log)
    if not torch_rows and not jittor_rows:
        raise SystemExit("No logs found to plot.")

    plt.figure(figsize=(8, 4.5))
    if torch_rows:
        plt.plot(_x_values(torch_rows), [r.get(args.metric, 0.0) for r in torch_rows], marker="o", label="PyTorch original")
    if jittor_rows:
        plt.plot(_x_values(jittor_rows), [r.get(args.metric, 0.0) for r in jittor_rows], marker="o", label="Jittor")
    plt.xlabel("Epoch / Step")
    plt.ylabel(args.metric)
    plt.title(f"MemSkill Controller {args.metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(f"saved plot to {output}")


if __name__ == "__main__":
    main()
