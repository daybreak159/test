#!/usr/bin/env python3
"""
Observable one-batch MemSkill flow using the real PyTorch PPOController.

This script is intentionally small and deterministic around the trainer-side
components. It avoids LLM/API calls, but uses the real controller.py network,
PPOBuffer, return/advantage calculation, compute_ppo_loss, and optimizer step.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
MEMSKILL_ROOT = THIS_DIR.parent
if str(MEMSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMSKILL_ROOT))

from src.controller import PPOBuffer, PPOController  # noqa: E402

from debug_one_batch_flow import (  # noqa: E402
    DebugMemoryBank,
    DebugSkillBank,
    Operation,
    build_state_embedding,
    compute_process_reward,
    desired_update_type,
    evaluate_final_reward,
    execute_ops,
    stable_embedding,
    write_jsonl,
)


def tensor_head(x: torch.Tensor, n: int = 6) -> List[float]:
    return x.detach().float().cpu().reshape(-1)[:n].numpy().round(6).tolist()


def parameter_norms(controller: PPOController) -> Dict[str, float]:
    groups = {
        "state_net": controller.state_net,
        "op_net": controller.op_net,
        "actor_head": controller.actor_head,
        "critic_head": controller.critic_head,
    }
    out = {}
    with torch.no_grad():
        for name, module in groups.items():
            total = 0.0
            for p in module.parameters():
                total += float((p.detach() ** 2).sum().item())
            out[name] = round(total ** 0.5, 6)
    return out


def inspect_policy(
    controller: PPOController,
    state_np: np.ndarray,
    op_embs_np: np.ndarray,
    new_op_mask_np: np.ndarray,
    device: str,
) -> Dict:
    state = torch.tensor(state_np, dtype=torch.float32, device=device)
    ops = torch.tensor(op_embs_np, dtype=torch.float32, device=device)
    new_mask = torch.tensor(new_op_mask_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        state_h = controller.encode_state(state.unsqueeze(0))
        op_h = controller.encode_ops(ops.unsqueeze(0))
        raw_logits = controller.get_action_logits(state_h, op_h)[0]
        biased_logits = controller._apply_new_action_bias(raw_logits, new_mask)
        probs = torch.softmax(biased_logits, dim=-1)
        value = controller.get_value(state_h)[0]

    return {
        "state_h_shape": list(state_h.shape),
        "op_h_shape": list(op_h.shape),
        "state_h_first6": tensor_head(state_h),
        "raw_logits": raw_logits.detach().cpu().numpy().round(6).tolist(),
        "biased_logits": biased_logits.detach().cpu().numpy().round(6).tolist(),
        "probs": probs.detach().cpu().numpy().round(6).tolist(),
        "value_before_forward": round(float(value.item()), 6),
    }


def write_markdown(path: Path, summary: Dict, records: List[Dict]):
    lines = [
        "# Torch One Batch Step Flow",
        "",
        "这个日志使用真实 `src.controller.PPOController` 跑一条可观察训练链路；外部 LLM/executor/evaluator 用确定性 mock 代替。",
        "",
        "## 1. Trace / Sessions",
        "",
        f"- sample_id: `{summary['sample']['sample_id']}`",
    ]
    for i, session in enumerate(summary["sessions"]):
        lines.append(f"- span {i}: {session}")

    lines.extend(["", "## 2. Span Breakpoints", ""])
    for r in records:
        if r["event"] == "designer_added_skill":
            lines.append(f"### Designer Update After Span {r['after_span']}")
            lines.append(f"- added skill: `{r['new_skill']}`")
            lines.append(f"- note: {r['note']}")
            lines.append("")
            continue
        if r["event"] != "span_processed":
            continue
        lines.append(f"### Span {r['session_idx']}")
        lines.append(f"- text: {r['session_text']}")
        lines.append(f"- memory size: `{r['memory_size_before']} -> {r['memory_size_after']}`")
        lines.append(f"- retrieved indices: `{r['retrieved_indices']}`")
        lines.append(f"- state shape: `{r['state_info']['state_embedding_shape']}`, first6: `{r['state_info']['state_embedding_first6']}`")
        lines.append(f"- candidate ops: `{r['candidate_ops']}`")
        lines.append(f"- op embedding shape: `{r['op_embedding_shape']}`, new mask: `{r['new_op_mask']}`")
        lines.append(f"- state_h/op_h: `{r['policy_inspect']['state_h_shape']}` / `{r['policy_inspect']['op_h_shape']}`")
        lines.append(f"- logits: `{r['policy_inspect']['biased_logits']}`")
        lines.append(f"- probs: `{r['policy_inspect']['probs']}`")
        lines.append(f"- selected actions: `{r['actions']}` -> `{r['selected_ops']}`")
        lines.append(f"- old_log_prob: `{r['log_prob']}`, old_value: `{r['value']}`")
        lines.append(f"- process reward: `{r['process_reward_meta']}`")
        lines.append("")

    lines.extend(["## 3. Episode / Buffer", "", "```json"])
    lines.append(json.dumps(summary["episode"], ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## 4. PPO Loss And Update", "", "```json"])
    lines.append(json.dumps(summary["ppo_update"], ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## 5. Final Memory / Skill Bank", "", "```json"])
    lines.append(json.dumps(summary["final_state"], ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(out_dir: Path, seed: int = 7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_dim = 8
    state_dim = emb_dim * 2
    op_dim = emb_dim
    sessions = [
        "Alice likes tea and keeps notes about project Orion.",
        "Alice moved to Paris, so her location memory should be updated.",
        "Bob joined project Orion and Alice now coordinates with Bob.",
    ]
    sample = {"sample_id": "torch_debug_trace_001", "raw_trace": " | ".join(sessions)}

    memory_bank = DebugMemoryBank(dim=emb_dim, top_k=2)
    skill_bank = DebugSkillBank(op_dim=op_dim)
    controller = PPOController(
        state_dim=state_dim,
        op_dim=op_dim,
        hidden_dim=16,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        vf_clip=0.0,
        new_action_p_min=0.35,
        new_action_delta_max=1.0,
        action_top_k=2,
    )
    controller.set_new_action_bias_scale(0.8)
    optimizer = torch.optim.Adam(controller.parameters(), lr=3e-3)
    ppo_buffer = PPOBuffer()

    records: List[Dict] = [{"event": "sample_selected", "sample": sample, "sessions": sessions}]
    process_budget = 0.3

    for session_idx, session_text in enumerate(sessions):
        if session_idx == 1 and "capture_location" not in skill_bank.operations:
            skill_bank.add(
                "capture_location",
                "Capture durable location changes for people or entities.",
                "update",
                is_new=True,
            )
            records.append(
                {
                    "event": "designer_added_skill",
                    "after_span": 0,
                    "new_skill": "capture_location",
                    "note": "模拟 Designer 在第一个 span 后新增 skill，所以后续 step 的候选 skill 数从 4 变为 5。",
                }
            )

        query_emb = stable_embedding(session_text, emb_dim)
        memory_before = memory_bank.contents()
        retrieved_texts, retrieved_indices, retrieved_embs = memory_bank.retrieve(query_emb)
        state_np, state_info = build_state_embedding(session_text, retrieved_embs, emb_dim)
        candidate_ops = skill_bank.candidates()
        op_embs_np = np.array([op.embedding for op in candidate_ops], dtype=np.float32)
        new_op_mask_np = np.array([1.0 if op.is_new else 0.0 for op in candidate_ops], dtype=np.float32)

        policy_inspect = inspect_policy(controller, state_np, op_embs_np, new_op_mask_np, device)
        state_t = torch.tensor(state_np, dtype=torch.float32, device=device)
        op_t = torch.tensor(op_embs_np, dtype=torch.float32, device=device)
        new_mask_t = torch.tensor(new_op_mask_np, dtype=torch.float32, device=device)
        actions, log_prob, value = controller.forward(
            state_t,
            op_t,
            deterministic=True,
            new_op_mask=new_mask_t,
        )
        if isinstance(actions, int):
            action_list = [actions]
        else:
            action_list = list(actions)

        selected_ops: List[Operation] = [candidate_ops[i] for i in action_list]
        exec_results = execute_ops(session_text, selected_ops, memory_bank)
        desired = desired_update_type(session_text, memory_bank)
        process_reward, process_meta = compute_process_reward(
            selected_ops,
            desired,
            len(sessions),
            process_budget,
        )
        skill_bank.update_stats([op.name for op in selected_ops], process_reward)
        ppo_buffer.push(
            state_np,
            op_embs_np,
            action_list,
            log_prob,
            value,
            reward=process_reward,
            new_op_mask=new_op_mask_np,
        )

        records.append(
            {
                "event": "span_processed",
                "session_idx": session_idx,
                "session_text": session_text,
                "memory_size_before": len(memory_before),
                "memory_size_after": len(memory_bank.memories),
                "retrieved_texts": retrieved_texts,
                "retrieved_indices": retrieved_indices,
                "state_info": state_info,
                "candidate_ops": [f"{op.name}/{op.update_type}" for op in candidate_ops],
                "op_embedding_shape": list(op_embs_np.shape),
                "new_op_mask": new_op_mask_np.astype(int).tolist(),
                "policy_inspect": policy_inspect,
                "actions": action_list,
                "selected_ops": [f"{op.name}/{op.update_type}" for op in selected_ops],
                "log_prob": round(float(log_prob), 6),
                "value": round(float(value), 6),
                "executor_results": exec_results,
                "process_reward_meta": process_meta,
                "memory_contents_after": memory_bank.contents(),
            }
        )
        skill_bank.mark_all_old()

    final_reward, final_reward_info = evaluate_final_reward(memory_bank)
    ppo_buffer.finish_episode(
        final_reward=final_reward,
        redistribute=True,
        redistribution_decay=0.9,
        final_reward_last_ratio=0.4,
    )
    returns, advantages = ppo_buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    adv_mean = float(advantages.mean())
    adv_std = float(advantages.std() + 1e-8)
    advantages_norm = (advantages - adv_mean) / adv_std

    batch = ppo_buffer.get_batch()
    norms_before = parameter_norms(controller)
    loss, loss_info = controller.compute_ppo_loss(batch, returns, advantages_norm)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
    optimizer.step()
    norms_after = parameter_norms(controller)
    with torch.no_grad():
        loss_after, loss_info_after = controller.compute_ppo_loss(batch, returns, advantages_norm)

    summary = {
        "sample": sample,
        "device": device,
        "sessions": sessions,
        "episode": {
            "final_reward_info": final_reward_info,
            "buffer_actions": ppo_buffer.actions,
            "buffer_old_log_probs": [round(float(x), 6) for x in ppo_buffer.log_probs],
            "buffer_old_values": [round(float(x), 6) for x in ppo_buffer.values],
            "buffer_rewards_after_finish": [round(float(x), 6) for x in ppo_buffer.rewards],
            "buffer_dones": ppo_buffer.dones,
            "returns": np.round(returns, 6).tolist(),
            "advantages_raw": np.round(advantages, 6).tolist(),
            "advantages_normalized": np.round(advantages_norm, 6).tolist(),
        },
        "ppo_update": {
            "param_norms_before": norms_before,
            "loss_before": round(float(loss.item()), 6),
            "loss_info_before": loss_info,
            "grad_norm_clipped": round(float(grad_norm), 6),
            "param_norms_after": norms_after,
            "loss_after": round(float(loss_after.item()), 6),
            "loss_info_after": loss_info_after,
        },
        "final_state": {
            "memory_bank": [asdict(m) for m in memory_bank.memories],
            "skill_bank": [
                {
                    "name": op.name,
                    "update_type": op.update_type,
                    "usage_count": op.usage_count,
                    "avg_reward": round(op.avg_reward, 6),
                }
                for op in skill_bank.candidates()
            ],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "torch_debug_records.jsonl", records)
    (out_dir / "torch_debug_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(out_dir / "torch_debug_trace.md", summary, records)
    return summary


def main():
    out_dir = THIS_DIR / "runs" / "debug_torch_one_batch_flow"
    summary = run(out_dir)
    print(f"device: {summary['device']}")
    print(f"trace: {out_dir / 'torch_debug_trace.md'}")
    print(f"records: {out_dir / 'torch_debug_records.jsonl'}")
    print(f"summary: {out_dir / 'torch_debug_summary.json'}")
    print("buffer_rewards:", summary["episode"]["buffer_rewards_after_finish"])
    print("returns:", summary["episode"]["returns"])
    print("advantages_normalized:", summary["episode"]["advantages_normalized"])
    print("loss_before:", summary["ppo_update"]["loss_before"])
    print("loss_after:", summary["ppo_update"]["loss_after"])
    print("param_norms_before:", summary["ppo_update"]["param_norms_before"])
    print("param_norms_after:", summary["ppo_update"]["param_norms_after"])


if __name__ == "__main__":
    main()
