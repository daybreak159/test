"""Controller-only benchmark for PyTorch and Jittor PPO controllers.

This benchmark intentionally consumes a fixed offline ControllerTrace.  It does
not call the LLM API, retriever, executor, or evaluator, so the timings focus on
the trainable PPOController path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from jittor_controller_repro.common import append_jsonl
from jittor_controller_repro.data.trace_schema import load_trace, pad_op_embeddings


def iter_minibatches(n_samples: int, minibatch_size: int, rng: np.random.Generator):
    indices = rng.permutation(n_samples)
    size = minibatch_size if minibatch_size > 0 else n_samples
    for start in range(0, n_samples, size):
        yield indices[start : min(start + size, n_samples)]


def slice_batch(batch: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    return {
        "states": [batch["states"][i] for i in indices],
        "op_embs": [batch["op_embs"][i] for i in indices],
        "new_op_masks": [batch["new_op_masks"][i] for i in indices],
        "actions": [batch["actions"][i] for i in indices],
        "log_probs": [batch["log_probs"][i] for i in indices],
        "values": [batch["values"][i] for i in indices],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {
        "backend": rows[0]["backend"],
        "trace": rows[0]["trace"],
        "n_steps": rows[0]["n_steps"],
        "state_dim": rows[0]["state_dim"],
        "op_dim": rows[0]["op_dim"],
        "action_top_k": rows[0]["action_top_k"],
        "epochs": len(rows),
    }
    numeric_keys = [
        "forward_sec",
        "evaluate_sec",
        "loss_sec",
        "train_step_sec",
        "epoch_sec",
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    if "updates" in rows[0]:
        out["updates_per_epoch"] = rows[0]["updates"]
    return out


def run_torch(args, trace, batch) -> list[dict[str, Any]]:
    from jittor_controller_repro.baselines.original_torch_runner import build_original_controller

    import torch
    import torch.optim as optim

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    controller = build_original_controller(
        state_dim=trace.state_dim,
        op_dim=trace.op_dim,
        hidden_dim=args.hidden_dim,
        action_top_k=trace.action_top_k,
        device=args.device,
    )
    optimizer = optim.Adam(controller.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    sample_state = torch.FloatTensor(trace.states[0]).to(args.device)
    sample_ops = torch.FloatTensor(trace.op_embs[0]).to(args.device)
    sample_action = trace.actions[0]

    for epoch in range(args.epochs + args.warmup_epochs):
        epoch_t = time.perf_counter()

        t0 = time.perf_counter()
        for _ in range(args.forward_repeats):
            controller.forward(sample_state, sample_ops, deterministic=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_sec = time.perf_counter() - t0

        op_padded, op_masks, new_masks = pad_op_embeddings(trace.op_embs, trace.new_op_masks)
        states_t = torch.FloatTensor(trace.states).to(args.device)
        ops_t = torch.FloatTensor(op_padded).to(args.device)
        actions_t = torch.LongTensor(trace.actions).to(args.device)
        masks_t = torch.FloatTensor(op_masks).to(args.device)
        new_masks_t = None if new_masks is None else torch.FloatTensor(new_masks).to(args.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.evaluate_repeats):
                controller.evaluate_actions(states_t, ops_t, actions_t, masks_t, new_masks_t)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        evaluate_sec = time.perf_counter() - t0

        totals: dict[str, float] = {}
        updates = 0
        loss_sec = 0.0
        train_sec = 0.0
        for _ in range(args.ppo_epochs):
            for mb_idx in iter_minibatches(trace.n_steps, args.minibatch_size, rng):
                mb = slice_batch(batch, mb_idx)
                t0 = time.perf_counter()
                loss, info = controller.compute_ppo_loss(mb, trace.returns[mb_idx], trace.advantages[mb_idx])
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                loss_sec += time.perf_counter() - t0

                t0 = time.perf_counter()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                train_sec += time.perf_counter() - t0

                totals["total_loss"] = totals.get("total_loss", 0.0) + float(loss.item())
                for key, value in info.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                updates += 1

        if epoch >= args.warmup_epochs:
            row = {
                "backend": "torch",
                "trace": args.trace,
                "epoch": epoch - args.warmup_epochs,
                "n_steps": trace.n_steps,
                "state_dim": trace.state_dim,
                "op_dim": trace.op_dim,
                "action_top_k": trace.action_top_k,
                "updates": updates,
                "forward_repeats": args.forward_repeats,
                "evaluate_repeats": args.evaluate_repeats,
                "forward_sec": forward_sec,
                "evaluate_sec": evaluate_sec,
                "loss_sec": loss_sec,
                "train_step_sec": train_sec,
                "epoch_sec": time.perf_counter() - epoch_t,
                "sample_action": np.asarray(sample_action).tolist(),
            }
            row.update({key: value / max(updates, 1) for key, value in totals.items()})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    return rows


def run_jittor(args, trace, batch) -> list[dict[str, Any]]:
    import jittor as jt
    from jittor import optim

    from jittor_controller_repro.models.jittor_controller import JittorPPOController

    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    if args.jittor_cuda >= 0:
        jt.flags.use_cuda = int(args.jittor_cuda)

    controller = JittorPPOController(
        state_dim=trace.state_dim,
        op_dim=trace.op_dim,
        hidden_dim=args.hidden_dim,
        action_top_k=trace.action_top_k,
    )
    optimizer = optim.Adam(controller.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    sample_state = trace.states[0]
    sample_ops = trace.op_embs[0]
    sample_action = trace.actions[0]

    for epoch in range(args.epochs + args.warmup_epochs):
        epoch_t = time.perf_counter()

        t0 = time.perf_counter()
        for _ in range(args.forward_repeats):
            controller.select_action(sample_state, sample_ops, deterministic=True)
        jt.sync_all()
        forward_sec = time.perf_counter() - t0

        op_padded, op_masks, new_masks = pad_op_embeddings(trace.op_embs, trace.new_op_masks)
        states_t = jt.array(trace.states)
        ops_t = jt.array(op_padded)
        actions_t = jt.array(trace.actions.astype(np.int64))
        masks_t = jt.array(op_masks)
        new_masks_t = None if new_masks is None else jt.array(new_masks)

        t0 = time.perf_counter()
        for _ in range(args.evaluate_repeats):
            controller.evaluate_actions(states_t, ops_t, actions_t, masks_t, new_masks_t)
        jt.sync_all()
        evaluate_sec = time.perf_counter() - t0

        totals: dict[str, float] = {}
        updates = 0
        loss_sec = 0.0
        train_sec = 0.0
        for _ in range(args.ppo_epochs):
            for mb_idx in iter_minibatches(trace.n_steps, args.minibatch_size, rng):
                mb = slice_batch(batch, mb_idx)
                t0 = time.perf_counter()
                loss, info = controller.compute_ppo_loss(mb, trace.returns[mb_idx], trace.advantages[mb_idx])
                jt.sync_all()
                loss_sec += time.perf_counter() - t0

                t0 = time.perf_counter()
                optimizer.step(loss)
                jt.sync_all()
                train_sec += time.perf_counter() - t0

                totals["total_loss"] = totals.get("total_loss", 0.0) + float(loss.item())
                for key, value in info.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                updates += 1

        if epoch >= args.warmup_epochs:
            row = {
                "backend": "jittor",
                "trace": args.trace,
                "epoch": epoch - args.warmup_epochs,
                "n_steps": trace.n_steps,
                "state_dim": trace.state_dim,
                "op_dim": trace.op_dim,
                "action_top_k": trace.action_top_k,
                "updates": updates,
                "forward_repeats": args.forward_repeats,
                "evaluate_repeats": args.evaluate_repeats,
                "forward_sec": forward_sec,
                "evaluate_sec": evaluate_sec,
                "loss_sec": loss_sec,
                "train_step_sec": train_sec,
                "epoch_sec": time.perf_counter() - epoch_t,
                "sample_action": np.asarray(sample_action).tolist(),
                "jittor_use_cuda": int(jt.flags.use_cuda),
            }
            row.update({key: value / max(updates, 1) for key, value in totals.items()})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark controller-only PPO paths.")
    parser.add_argument("--backend", choices=["torch", "jittor"], required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="jittor_controller_repro/runs/controller_benchmark")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--forward-repeats", type=int, default=100)
    parser.add_argument("--evaluate-repeats", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    parser.add_argument("--jittor-cuda", type=int, default=-1, help="-1 keeps default, 0 CPU, 1 CUDA")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    trace = load_trace(args.trace)
    batch = trace.to_batch()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "torch":
        rows = run_torch(args, trace, batch)
    else:
        rows = run_jittor(args, trace, batch)

    jsonl_path = out_dir / f"{args.backend}_benchmark.jsonl"
    csv_path = out_dir / f"{args.backend}_benchmark.csv"
    summary_path = out_dir / f"{args.backend}_summary.json"
    if jsonl_path.exists():
        jsonl_path.unlink()
    for row in rows:
        append_jsonl(jsonl_path, row)
    write_csv(csv_path, rows)
    summary = summarize(rows)
    summary.update({"python": os.sys.executable})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} rows to {jsonl_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
