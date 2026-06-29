"""Offline adapter smoke test for MemSkill-style controller batches."""

from __future__ import annotations

import argparse

from jittor_controller_repro.common import require_package
from jittor_controller_repro.data.trace_schema import load_trace
from jittor_controller_repro.models.jittor_controller import JittorPPOController


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Jittor controller on cached MemSkill-style inputs.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    require_package("jittor", "Install jittor to run the offline adapter.")
    trace = load_trace(args.trace)
    idx = max(0, min(args.step, trace.n_steps - 1))
    controller = JittorPPOController(
        state_dim=trace.state_dim,
        op_dim=trace.op_dim,
        hidden_dim=args.hidden_dim,
        action_top_k=trace.action_top_k,
    )
    action, log_prob, value = controller.select_action(
        trace.states[idx], trace.op_embs[idx], deterministic=True, new_op_mask=trace.new_op_masks[idx]
    )
    selected_names = []
    if trace.steps:
        candidate_ops = trace.steps[idx].get("candidate_ops", [])
        selected = action if isinstance(action, list) else [action]
        selected_names = [candidate_ops[i] for i in selected if i < len(candidate_ops)]
    print(
        {
            "step": idx,
            "action": action,
            "selected_ops": selected_names,
            "log_prob": log_prob,
            "value": value,
            "message": "Jittor controller consumed MemSkill-style state/op embeddings.",
        }
    )


if __name__ == "__main__":
    main()

