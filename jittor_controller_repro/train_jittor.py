"""Train the Jittor PPOController on a cached trace."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from jittor_controller_repro.common import append_jsonl, require_package
from jittor_controller_repro.data.trace_schema import load_trace
from jittor_controller_repro.models.jittor_controller import JittorPPOController


def iter_minibatches(n_samples: int, minibatch_size: int, rng: np.random.Generator):
    indices = rng.permutation(n_samples)
    size = minibatch_size if minibatch_size > 0 else n_samples
    for start in range(0, n_samples, size):
        yield indices[start : min(start + size, n_samples)]


def slice_batch(batch: dict, indices: np.ndarray) -> dict:
    return {
        "states": [batch["states"][i] for i in indices],
        "op_embs": [batch["op_embs"][i] for i in indices],
        "new_op_masks": [batch["new_op_masks"][i] for i in indices],
        "actions": [batch["actions"][i] for i in indices],
        "log_probs": [batch["log_probs"][i] for i in indices],
        "values": [batch["values"][i] for i in indices],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Jittor PPOController.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--log", default="jittor_controller_repro/runs/jittor_train.jsonl")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    require_package("jittor", "Install jittor to run the Jittor reproduction.")
    import jittor as jt
    from jittor import optim

    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    trace = load_trace(args.trace)
    controller = JittorPPOController(
        state_dim=trace.state_dim,
        op_dim=trace.op_dim,
        hidden_dim=args.hidden_dim,
        action_top_k=trace.action_top_k,
    )
    optimizer = optim.Adam(controller.parameters(), lr=args.lr)
    batch = trace.to_batch()
    rng = np.random.default_rng(args.seed)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.log).exists():
        Path(args.log).unlink()

    for epoch in range(args.epochs):
        start_t = time.perf_counter()
        totals: dict[str, float] = {}
        updates = 0
        for _ in range(args.ppo_epochs):
            for mb_idx in iter_minibatches(trace.n_steps, args.minibatch_size, rng):
                mb = slice_batch(batch, mb_idx)
                loss, info = controller.compute_ppo_loss(
                    mb, trace.returns[mb_idx], trace.advantages[mb_idx]
                )
                optimizer.step(loss)
                totals["total_loss"] = totals.get("total_loss", 0.0) + float(loss.item())
                for key, value in info.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                updates += 1
        row = {
            "backend": "jittor",
            "epoch": epoch,
            "updates": updates,
            "elapsed_sec": time.perf_counter() - start_t,
        }
        row.update({key: value / max(updates, 1) for key, value in totals.items()})
        append_jsonl(args.log, row)
        print(row)


if __name__ == "__main__":
    main()
