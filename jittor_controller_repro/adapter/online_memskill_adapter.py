"""Small online API smoke test for a Jittor-selected MemSkill operation.

This is Mode C scaffolding.  It validates the boundary from Jittor controller
selection to an OpenAI-compatible executor call without claiming full online
training parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from jittor_controller_repro.common import require_package
from jittor_controller_repro.data.trace_schema import load_trace
from jittor_controller_repro.models.jittor_controller import JittorPPOController


def ensure_memskill_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Online API smoke test for Jittor controller adapter.")
    parser.add_argument("--trace", required=True, help="Cached trace providing state/op embeddings")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--retriever", default="contriever")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    require_package("jittor", "Install jittor before running the online adapter.")
    ensure_memskill_on_path()
    from src.executor import Executor
    from src.operation_bank import OperationBank

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

    bank = OperationBank(encoder=None)
    candidate_ops = bank.get_candidate_operations()
    selected_indices = action if isinstance(action, list) else [action]
    selected_ops = [candidate_ops[i % len(candidate_ops)] for i in selected_indices]
    executor_args = SimpleNamespace(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        retriever=args.retriever,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    executor = Executor(executor_args)
    session_text = "Alice will visit Shanghai next month and asked the agent to remember it."
    retrieved_memories = ["Alice likes travel planning."]
    results = executor.execute_operation(selected_ops, session_text, retrieved_memories)
    print(
        json.dumps(
            {
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "selected_ops": [op.name for op in selected_ops],
                "exec_results": [str(result) for result in results],
                "message": "Online API smoke completed. This is not a full training run.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
