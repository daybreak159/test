#!/usr/bin/env python3
"""Generate README figures from saved MemSkill reproduction logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


BLUE = "#2F69BF"
ORANGE = "#E28A3B"
GREEN = "#4E9F70"
GRAY = "#6B7280"
GRID = "#D8DEE9"


def configure_matplotlib() -> None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 220


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_two_backend_line(
    x_values,
    jittor_values,
    torch_values,
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x_values, jittor_values, marker="o", color=BLUE, label="Jittor")
    ax.plot(x_values, torch_values, marker="s", color=ORANGE, label="PyTorch")
    if logy:
        ax.set_yscale("log")
        ylabel += " (log scale)"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save(fig, out)


def plot_online_loss(runs_root: Path, out_dir: Path) -> None:
    base = runs_root / "compare_jittor_torch_full_small_epoch1_20260624" / "raw_metrics"
    jittor = pd.read_csv(base / "jittor_metrics.csv")
    torch = pd.read_csv(base / "torch_metrics.csv")
    metrics = [
        ("reward", "Reward", "在线流程 Reward 曲线", "online_reward_curve.png", False),
        ("value_loss", "Value Loss", "在线流程 Value Loss 曲线", "online_value_loss_curve.png", False),
        ("policy_loss", "Policy Loss", "在线流程 Policy Loss 曲线", "online_policy_loss_curve.png", False),
    ]
    for col, ylabel, title, filename, logy in metrics:
        plot_two_backend_line(
            jittor["inner_epoch"] + 1,
            jittor[col],
            torch[col],
            title,
            "Inner Epoch",
            ylabel,
            out_dir / filename,
            logy=logy,
        )


def plot_offline_loss(runs_root: Path, out_dir: Path) -> None:
    df = pd.read_csv(runs_root / "offline_paper_style_loss_20260625" / "offline_loss_comparison.csv")
    metrics = [
        ("value_loss", "Value Loss", "离线缓存 Value Loss 曲线", "offline_value_loss_curve.png", True),
        ("policy_loss", "Policy Loss", "离线缓存 Policy Loss 曲线", "offline_policy_loss_curve.png", False),
        ("total_loss", "PPO Objective Loss", "离线缓存 PPO 总目标曲线", "offline_ppo_objective_loss_curve.png", False),
    ]
    for col, ylabel, title, filename, logy in metrics:
        plot_two_backend_line(
            df["epoch"],
            df[f"jittor_{col}"],
            df[f"torch_{col}"],
            title,
            "Epoch",
            ylabel,
            out_dir / filename,
            logy=logy,
        )


def normalize_op(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "/" in text:
        text = text.split("/", 1)[0]
    return text


def collect_distribution(trace_path: Path) -> tuple[Counter, Counter]:
    skills: Counter = Counter()
    actions: Counter = Counter()
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            selected = item.get("selected_skills") or item.get("selected_op") or item.get("selected_ops") or []
            if isinstance(selected, str):
                selected = [selected]
            for value in selected:
                op = normalize_op(value)
                if op:
                    skills[op] += 1

            executor_actions = item.get("executor_actions") or []
            for value in executor_actions:
                op = normalize_op(value)
                if op:
                    actions[op] += 1
            for result in item.get("exec_results") or item.get("executor_results") or []:
                if not isinstance(result, dict):
                    continue
                value = result.get("action_type") or result.get("update_type") or result.get("operation")
                op = normalize_op(value)
                if op:
                    actions[op] += 1
    return skills, actions


def plot_distribution_compare(
    jittor_counter: Counter,
    torch_counter: Counter,
    title: str,
    out: Path,
    preferred_order: list[str],
) -> None:
    labels = [label for label in preferred_order if jittor_counter.get(label, 0) or torch_counter.get(label, 0)]
    rest = sorted((set(jittor_counter) | set(torch_counter)) - set(labels))
    labels.extend(rest)
    labels = labels[::-1]
    j_total = sum(jittor_counter.values()) or 1
    t_total = sum(torch_counter.values()) or 1
    j_values = [jittor_counter.get(label, 0) / j_total * 100 for label in labels]
    t_values = [torch_counter.get(label, 0) / t_total * 100 for label in labels]

    fig, ax = plt.subplots(figsize=(9.8, max(3.4, 0.5 * len(labels) + 1.2)))
    y = list(range(len(labels)))
    height = 0.36
    ax.barh([i + height / 2 for i in y], j_values, height=height, color=BLUE, label=f"Jittor (n={j_total})")
    ax.barh([i - height / 2 for i in y], t_values, height=height, color=ORANGE, label=f"PyTorch (n={t_total})")
    ax.set_yticks(y, labels)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("占比 (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out)


def plot_offline_distributions(runs_root: Path, out_dir: Path) -> None:
    j_trace = runs_root / "full-flow-jittor-evolve2" / "step_records.jsonl"
    t_trace = runs_root / "full_compare" / "torch_fixed_skill" / "controller_trace_records.jsonl"
    j_skills, j_actions = collect_distribution(j_trace)
    t_skills, t_actions = collect_distribution(t_trace)
    plot_distribution_compare(
        j_skills,
        t_skills,
        "缓存 trace 中的 Controller 技能选择分布对比",
        out_dir / "offline_selected_skill_distribution.png",
        ["insert", "update", "delete", "noop"],
    )
    plot_distribution_compare(
        j_actions,
        t_actions,
        "缓存 trace 中的 Executor memory action 分布对比",
        out_dir / "offline_memory_action_distribution.png",
        ["insert", "update", "delete", "noop"],
    )


def plot_benchmark(runs_root: Path, out_dir: Path) -> None:
    bench = runs_root / "controller_benchmark_locomo_real_gpu"
    j = read_json(bench / "jittor_summary.json")
    t = read_json(bench / "torch_summary.json")
    metrics = [
        ("forward_sec_mean", "forward"),
        ("evaluate_sec_mean", "evaluate"),
        ("loss_sec_mean", "loss"),
        ("train_step_sec_mean", "train step"),
        ("epoch_sec_mean", "epoch"),
    ]
    labels = [m[1] for m in metrics]
    torch_ms = [t[m[0]] * 1000 for m in metrics]
    jittor_ms = [j[m[0]] * 1000 for m in metrics]
    speedup = [t[m[0]] / j[m[0]] for m in metrics]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], torch_ms, width=width, color=ORANGE, label="PyTorch")
    ax.bar([i + width / 2 for i in x], jittor_ms, width=width, color=BLUE, label="Jittor")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("耗时 (ms, 越低越好)")
    ax.set_title("Controller-only 分阶段耗时", fontsize=15, fontweight="bold", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir / "controller_benchmark_timing.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [GREEN if v >= 1 else GRAY for v in speedup]
    bars = ax.barh(labels[::-1], speedup[::-1], color=colors[::-1])
    ax.axvline(1.0, color="#334155", linewidth=1.0)
    ax.set_xlabel("Speedup = PyTorch / Jittor")
    ax.set_title("Controller-only Jittor 相对速度", fontsize=15, fontweight="bold", pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, speedup[::-1]):
        ax.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height() / 2, f"{value:.2f}x", va="center", fontsize=10)
    save(fig, out_dir / "controller_benchmark_speedup.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("jittor_controller_repro/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("assets/figures"))
    args = parser.parse_args()

    configure_matplotlib()
    plot_online_loss(args.runs_root, args.out_dir)
    plot_offline_loss(args.runs_root, args.out_dir)
    plot_offline_distributions(args.runs_root, args.out_dir)
    plot_benchmark(args.runs_root, args.out_dir)


if __name__ == "__main__":
    main()
