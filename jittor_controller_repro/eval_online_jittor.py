"""Evaluate a trained Jittor controller on a fixed demo trace.

The script is designed for PPT/video demos.  It restores a Jittor controller
checkpoint, replays fixed controller states span by span, records Top-K skills,
and optionally calls the original Executor.  The default mock executor keeps the
demo stable without API access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from jittor_controller_repro.common import write_jsonl
from jittor_controller_repro.data.trace_schema import load_trace
from jittor_controller_repro.models.jittor_controller import JittorPPOController


def ensure_memskill_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def load_jittor_controller(ckpt: dict[str, Any], trace) -> Any:
    import jittor as jt

    config = ckpt.get("config", {}) or {}
    controller = JittorPPOController(
        state_dim=int(config.get("state_dim") or trace.state_dim),
        op_dim=int(config.get("op_embedding_dim") or trace.op_dim),
        hidden_dim=int(config.get("controller_hidden_dim", 256)),
        gamma=float(config.get("gamma", 0.99)),
        gae_lambda=float(config.get("gae_lambda", 0.95)),
        clip_epsilon=float(config.get("clip_epsilon", 0.2)),
        entropy_coef=float(config.get("entropy_coef", 0.01)),
        value_coef=float(config.get("value_coef", 0.5)),
        vf_clip=float(config.get("vf_clip", 0.0)),
        new_action_p_min=float(config.get("new_action_p_min", 0.0)),
        new_action_delta_max=float(config.get("new_action_delta_max", 0.0)),
        action_top_k=int(config.get("action_top_k", trace.action_top_k)),
    )
    state = {}
    for key, value in (ckpt.get("controller_state_dict") or {}).items():
        state[key] = jt.array(value) if isinstance(value, np.ndarray) else value
    controller.load_state_dict(state)
    return controller


def operation_bank_from_checkpoint(ckpt: dict[str, Any]):
    ensure_memskill_on_path()
    from src.operation_bank import OperationBank

    op_bank_data = ckpt.get("operation_bank") or {}
    bank = OperationBank.from_dict(op_bank_data, encoder=None)
    bank.set_new_operation_names(ckpt.get("operation_bank_new_operation_names", []))
    return bank


class MockExecutor:
    """Deterministic executor used for stable demos and tests."""

    def execute_operation(self, operation, session_text, retrieved_memories):
        ensure_memskill_on_path()
        from src.executor import ExecutionResult

        ops = operation if isinstance(operation, (list, tuple)) else [operation]
        results = []
        for op in ops:
            update_type = str(getattr(op, "update_type", "insert")).upper()
            name = str(getattr(op, "name", "skill"))
            if update_type == "NOOP":
                results.append(ExecutionResult("NOOP", True, reasoning=f"mock noop via {name}"))
            elif update_type == "UPDATE" and retrieved_memories:
                results.append(ExecutionResult(
                    "UPDATE",
                    True,
                    memory_index=0,
                    memory_content=f"{retrieved_memories[0]} | mock update from {name}",
                    reasoning="mock update",
                ))
            elif update_type == "DELETE" and retrieved_memories:
                results.append(ExecutionResult("DELETE", True, memory_index=0, reasoning="mock delete"))
            else:
                snippet = " ".join(str(session_text).split())[:160]
                results.append(ExecutionResult(
                    "INSERT",
                    True,
                    memory_content=f"{snippet} [mock memory via {name}]",
                    reasoning="mock insert",
                ))
        return results


def select_candidate_names(trace_step: dict[str, Any], operation_bank, n_ops: int) -> list[str]:
    names = trace_step.get("candidate_ops") if trace_step else None
    if names:
        return [str(name) for name in names]
    ops = operation_bank.get_candidate_operations()
    return [str(getattr(op, "name", f"op_{i}")) for i, op in enumerate(ops[:n_ops])]


def result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "action_type": getattr(result, "action_type", None),
        "success": bool(getattr(result, "success", False)),
        "memory_index": int(getattr(result, "memory_index", -1)),
        "memory_content": getattr(result, "memory_content", ""),
        "reasoning": getattr(result, "reasoning", ""),
    }


def apply_mock_memory(memory_bank: list[str], results: list[Any]) -> None:
    for result in results:
        if not bool(getattr(result, "success", False)):
            continue
        action = str(getattr(result, "action_type", "")).upper()
        if action == "INSERT":
            memory_bank.append(str(getattr(result, "memory_content", "")))
        elif action == "UPDATE" and memory_bank:
            idx = max(0, min(int(getattr(result, "memory_index", 0)), len(memory_bank) - 1))
            memory_bank[idx] = str(getattr(result, "memory_content", memory_bank[idx]))
        elif action == "DELETE" and memory_bank:
            idx = max(0, min(int(getattr(result, "memory_index", 0)), len(memory_bank) - 1))
            memory_bank.pop(idx)


def build_live_executor(args):
    ensure_memskill_on_path()
    from src.executor import Executor

    executor_args = SimpleNamespace(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        retriever=args.retriever,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    return Executor(executor_args)


def write_report(path: Path, selected_rows: list[dict], action_rows: list[dict], memory_bank: list[str]) -> None:
    lines = [
        "# Jittor Controller Online Demo Trace Report",
        "",
        f"- spans: `{len(selected_rows)}`",
        f"- memory actions: `{sum(len(r.get('executor_actions', [])) for r in action_rows)}`",
        f"- final memory count: `{len(memory_bank)}`",
        "",
        "## Selected Skills",
        "",
    ]
    for row in selected_rows[:20]:
        lines.append(
            f"- span `{row['span_id']}`: {', '.join(row.get('selected_skills', []))} "
            f"(log_prob={row.get('log_prob', 0.0):.4f}, value={row.get('value', 0.0):.4f})"
        )
    lines.extend(["", "## Final Memory Bank", ""])
    for idx, memory in enumerate(memory_bank[:30]):
        lines.append(f"{idx + 1}. {memory}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Jittor controller on a fixed online demo trace.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace", required=True, help="Controller trace .npz used as fixed demo spans")
    parser.add_argument("--output-dir", default="jittor_controller_repro/runs/online_eval_demo")
    parser.add_argument("--executor-mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--api-cache-mode", choices=["off", "live", "replay"], default="off")
    parser.add_argument("--api-cache-dir", default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--retriever", default="contriever")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-spans", type=int, default=0)
    args = parser.parse_args()

    if args.api_cache_mode != "off":
        os.environ["MEMSKILL_API_CACHE_MODE"] = args.api_cache_mode
        cache_dir = args.api_cache_dir or str(Path(args.output_dir) / "api_cache")
        os.environ["MEMSKILL_API_CACHE_DIR"] = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = load_trace(args.trace)
    ckpt = load_checkpoint(args.checkpoint)
    controller = load_jittor_controller(ckpt, trace)
    operation_bank = operation_bank_from_checkpoint(ckpt)
    executor = MockExecutor() if args.executor_mode == "mock" else build_live_executor(args)

    selected_rows = []
    action_rows = []
    memory_bank: list[str] = []
    n_spans = trace.n_steps if args.max_spans <= 0 else min(args.max_spans, trace.n_steps)

    for span_id in range(n_spans):
        action, log_prob, value = controller.select_action(
            trace.states[span_id],
            trace.op_embs[span_id],
            deterministic=True,
            new_op_mask=trace.new_op_masks[span_id],
        )
        indices = action if isinstance(action, list) else [action]
        step = trace.steps[span_id] if span_id < len(trace.steps) else {}
        candidate_names = select_candidate_names(step, operation_bank, trace.op_embs[span_id].shape[0])
        selected_skills = [
            candidate_names[idx] if idx < len(candidate_names) else f"op_{idx}"
            for idx in indices
        ]
        selected_rows.append({
            "span_id": span_id,
            "action_indices": indices,
            "selected_skills": selected_skills,
            "log_prob": float(log_prob),
            "value": float(value),
        })

        ops = operation_bank.get_candidate_operations()
        selected_ops = [ops[idx % len(ops)] for idx in indices] if ops else []
        session_text = step.get("session_text", f"demo span {span_id}")
        retrieved = step.get("retrieved_memories", memory_bank[-3:])
        results = executor.execute_operation(selected_ops, session_text, retrieved)
        result_dicts = [result_to_dict(r) for r in results]
        apply_mock_memory(memory_bank, results)
        action_rows.append({
            "span_id": span_id,
            "selected_skills": selected_skills,
            "executor_actions": [r.get("action_type") for r in result_dicts],
            "results": result_dicts,
        })

    write_jsonl(output_dir / "selected_skills.jsonl", selected_rows)
    write_jsonl(output_dir / "memory_actions.jsonl", action_rows)
    with (output_dir / "final_memory_bank.json").open("w", encoding="utf-8") as f:
        json.dump({"memories": memory_bank}, f, ensure_ascii=False, indent=2)
    write_report(output_dir / "demo_trace_report.md", selected_rows, action_rows, memory_bank)
    print(f"saved demo report to {output_dir / 'demo_trace_report.md'}")


if __name__ == "__main__":
    main()
