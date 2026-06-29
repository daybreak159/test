#!/usr/bin/env python3
"""
Run one observable MemSkill-style PPO batch without external LLM/API dependencies.

This is a teaching/debug runner. It keeps the same data flow as the real trainer:
sample -> sessions/spans -> MemoryBank retrieval -> state embedding -> SkillBank
candidate ops -> controller action/top-k -> executor memory update -> process reward
-> final reward -> PPOBuffer returns/advantages -> PPO loss -> parameter update.

The local shell used for this note may not have torch/jittor installed, so this file
uses a tiny NumPy controller that mirrors the formulas and tensor shapes. The logs
are meant to be read like breakpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def stable_embedding(text: str, dim: int) -> np.ndarray:
    """Deterministic text embedding used only for this observable debug run."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    counter = 0
    while len(values) < dim:
        block = hashlib.sha256(digest + str(counter).encode("ascii")).digest()
        values.extend([(b / 127.5) - 1.0 for b in block])
        counter += 1
    arr = np.array(values[:dim], dtype=np.float32)
    norm = np.linalg.norm(arr) + 1e-8
    return arr / norm


def softmax(logits: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    x = logits.astype(np.float64).copy()
    if mask is not None:
        x = np.where(mask > 0, x, -1e9)
    x = x - np.max(x)
    exp_x = np.exp(x)
    if mask is not None:
        exp_x = exp_x * (mask > 0)
    return exp_x / (np.sum(exp_x) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


@dataclass
class MemoryItem:
    content: str
    embedding: List[float]
    operation_history: List[str]


class DebugMemoryBank:
    def __init__(self, dim: int, top_k: int = 2):
        self.dim = dim
        self.top_k = top_k
        self.memories: List[MemoryItem] = []

    def retrieve(self, query_embedding: np.ndarray) -> Tuple[List[str], List[int], np.ndarray]:
        if not self.memories:
            return [], [], np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.array([m.embedding for m in self.memories], dtype=np.float32)
        sims = np.array([cosine(query_embedding, row) for row in matrix])
        indices = np.argsort(sims)[::-1][: self.top_k].tolist()
        return [self.memories[i].content for i in indices], indices, matrix[indices]

    def add(self, content: str, operation_name: str):
        self.memories.append(
            MemoryItem(
                content=content,
                embedding=stable_embedding(content, self.dim).tolist(),
                operation_history=[operation_name],
            )
        )

    def update(self, index: int, content: str, operation_name: str):
        old = self.memories[index]
        old.content = content
        old.embedding = stable_embedding(content, self.dim).tolist()
        old.operation_history.append(operation_name)

    def delete(self, index: int):
        if 0 <= index < len(self.memories):
            del self.memories[index]

    def contents(self) -> List[str]:
        return [m.content for m in self.memories]


@dataclass
class Operation:
    name: str
    description: str
    update_type: str
    embedding: List[float]
    is_new: bool = False
    usage_count: int = 0
    avg_reward: float = 0.0


class DebugSkillBank:
    def __init__(self, op_dim: int):
        self.op_dim = op_dim
        self.operations: Dict[str, Operation] = {}
        for name, desc, update_type in [
            ("delete", "Delete stale or contradicted memories.", "delete"),
            ("insert", "Insert new useful information into memory.", "insert"),
            ("noop", "Do nothing when the span has no durable information.", "noop"),
            ("update", "Update an existing memory with corrected details.", "update"),
        ]:
            self.add(name, desc, update_type, is_new=False)

    def add(self, name: str, description: str, update_type: str, is_new: bool):
        self.operations[name] = Operation(
            name=name,
            description=description,
            update_type=update_type,
            embedding=stable_embedding(description, self.op_dim).tolist(),
            is_new=is_new,
        )

    def candidates(self) -> List[Operation]:
        return [self.operations[name] for name in sorted(self.operations.keys())]

    def mark_all_old(self):
        for op in self.operations.values():
            op.is_new = False

    def update_stats(self, selected_names: Sequence[str], reward: float):
        selected = set(selected_names)
        for op in self.operations.values():
            if op.name in selected:
                op.usage_count += 1
                op.avg_reward += (reward - op.avg_reward) / max(op.usage_count, 1)


class DebugPPOBuffer:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.op_embs: List[np.ndarray] = []
        self.new_op_masks: List[np.ndarray] = []
        self.actions: List[List[int]] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def push(self, state, op_embs, new_op_mask, actions, log_prob, value, reward):
        self.states.append(state.astype(np.float32))
        self.op_embs.append(op_embs.astype(np.float32))
        self.new_op_masks.append(new_op_mask.astype(np.float32))
        self.actions.append(list(actions))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(False)

    def finish_episode(
        self,
        final_reward: float,
        redistribute: bool = True,
        redistribution_decay: float = 0.9,
        final_reward_last_ratio: float = 0.4,
    ) -> Dict:
        episode_start = 0
        for i in range(len(self.dones) - 1, -1, -1):
            if self.dones[i]:
                episode_start = i + 1
                break
        episode_length = len(self.rewards) - episode_start
        weights = None
        if redistribute and episode_length > 1:
            last_reward = final_reward * final_reward_last_ratio
            remaining = final_reward - last_reward
            weights = np.array(
                [redistribution_decay ** (episode_length - 1 - i) for i in range(episode_length)],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            for i, w in enumerate(weights):
                self.rewards[episode_start + i] += float(remaining * w)
            self.rewards[-1] += float(last_reward)
        else:
            self.rewards[-1] += float(final_reward)
        self.dones[-1] = True
        return {
            "episode_start": episode_start,
            "episode_length": episode_length,
            "redistribution_weights": None if weights is None else weights.round(6).tolist(),
            "final_reward_last_ratio": final_reward_last_ratio,
            "rewards_after_finish": [round(x, 6) for x in self.rewards],
            "dones": self.dones,
        }

    def compute_returns_and_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        n = len(self.rewards)
        rewards = np.array(self.rewards, dtype=np.float64)
        values = np.array(self.values, dtype=np.float64)
        dones = np.array(self.dones, dtype=np.float64)
        advantages = np.zeros(n, dtype=np.float64)
        last_gae = 0.0
        rows = []
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            rows.append(
                {
                    "t": t,
                    "reward": round(float(rewards[t]), 6),
                    "value": round(float(values[t]), 6),
                    "next_value": round(float(next_value), 6),
                    "done": bool(dones[t]),
                    "delta": round(float(delta), 6),
                    "advantage": round(float(last_gae), 6),
                }
            )
        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32), list(reversed(rows))

    def batch(self) -> Dict:
        return {
            "states": self.states,
            "op_embs": self.op_embs,
            "new_op_masks": self.new_op_masks,
            "actions": np.array(self.actions, dtype=np.int64),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=bool),
        }


class NumpyPPOController:
    def __init__(
        self,
        state_dim: int,
        op_dim: int,
        action_top_k: int = 2,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        new_action_p_min: float = 0.35,
        new_action_delta_max: float = 1.0,
        new_action_bias_scale: float = 0.8,
        seed: int = 7,
    ):
        self.state_dim = state_dim
        self.op_dim = op_dim
        self.feature_dim = state_dim + op_dim
        self.action_top_k = action_top_k
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.new_action_p_min = new_action_p_min
        self.new_action_delta_max = new_action_delta_max
        self.new_action_bias_scale = new_action_bias_scale
        self.rng = np.random.default_rng(seed)
        self.actor_w = self.rng.normal(0, 0.12, size=self.feature_dim).astype(np.float64)
        self.actor_b = 0.0
        self.critic_w = self.rng.normal(0, 0.12, size=state_dim).astype(np.float64)
        self.critic_b = 0.0

    def param_vector(self) -> np.ndarray:
        return np.concatenate([self.actor_w, [self.actor_b], self.critic_w, [self.critic_b]])

    def set_param_vector(self, vec: np.ndarray):
        a_end = self.feature_dim
        self.actor_w = vec[:a_end].copy()
        self.actor_b = float(vec[a_end])
        c_start = a_end + 1
        c_end = c_start + self.state_dim
        self.critic_w = vec[c_start:c_end].copy()
        self.critic_b = float(vec[c_end])

    def parameter_norms(self) -> Dict[str, float]:
        return {
            "actor_norm": float(np.linalg.norm(self.actor_w)),
            "critic_norm": float(np.linalg.norm(self.critic_w)),
        }

    def logits(self, state: np.ndarray, op_embs: np.ndarray) -> np.ndarray:
        state_expanded = np.repeat(state.reshape(1, -1), op_embs.shape[0], axis=0)
        features = np.concatenate([state_expanded, op_embs], axis=-1)
        return features @ self.actor_w + self.actor_b

    def value(self, state: np.ndarray) -> float:
        return float(state @ self.critic_w + self.critic_b)

    def apply_new_action_bias(
        self, logits: np.ndarray, new_op_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        if new_op_mask.sum() <= 0:
            return logits, {"applied": False, "reason": "no new skill"}
        probs = softmax(logits)
        p_new = float(np.sum(probs * new_op_mask))
        need_bias = p_new < self.new_action_p_min
        if not need_bias:
            return logits, {"applied": False, "p_new": p_new, "reason": "already enough mass"}
        delta = math.log(self.new_action_p_min / (p_new + 1e-8))
        delta = min(max(delta, 0.0), self.new_action_delta_max) * self.new_action_bias_scale
        return logits + delta * new_op_mask, {
            "applied": True,
            "p_new_before": p_new,
            "target": self.new_action_p_min,
            "delta": round(delta, 6),
        }

    def topk_log_prob(self, logits: np.ndarray, actions: Sequence[int]) -> float:
        probs = softmax(logits)
        remaining = 1.0
        log_prob = 0.0
        eps = 1e-8
        for action in actions:
            p_i = max(float(probs[action]), eps)
            denom = max(remaining, eps)
            log_prob += math.log(p_i) - math.log(denom)
            remaining = max(remaining - p_i, eps)
        return float(log_prob)

    def forward(
        self, state: np.ndarray, op_embs: np.ndarray, new_op_mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[List[int], float, float, Dict]:
        raw_logits = self.logits(state, op_embs)
        biased_logits, bias_info = self.apply_new_action_bias(raw_logits, new_op_mask)
        probs = softmax(biased_logits)
        k = min(self.action_top_k, len(probs))
        if deterministic:
            actions = np.argsort(biased_logits)[::-1][:k].tolist()
        else:
            remaining_indices = list(range(len(probs)))
            remaining_probs = probs.copy()
            actions = []
            for _ in range(k):
                local_probs = remaining_probs[remaining_indices]
                local_probs = local_probs / local_probs.sum()
                chosen = int(self.rng.choice(remaining_indices, p=local_probs))
                actions.append(chosen)
                remaining_indices.remove(chosen)
        log_prob = self.topk_log_prob(biased_logits, actions)
        value = self.value(state)
        return actions, log_prob, value, {
            "raw_logits": raw_logits.round(6).tolist(),
            "biased_logits": biased_logits.round(6).tolist(),
            "probs": probs.round(6).tolist(),
            "new_action_bias": bias_info,
        }

    def compute_loss(self, batch: Dict, returns: np.ndarray, advantages: np.ndarray) -> Tuple[float, Dict]:
        n = len(batch["states"])
        max_ops = max(op.shape[0] for op in batch["op_embs"])
        op_dim = batch["op_embs"][0].shape[1]
        op_padded = np.zeros((n, max_ops, op_dim), dtype=np.float64)
        op_masks = np.zeros((n, max_ops), dtype=np.float64)
        new_masks = np.zeros((n, max_ops), dtype=np.float64)
        padding_rows = []
        for i, op in enumerate(batch["op_embs"]):
            n_ops = op.shape[0]
            op_padded[i, :n_ops] = op
            op_masks[i, :n_ops] = 1.0
            new_masks[i, :n_ops] = batch["new_op_masks"][i]
            padding_rows.append({"row": i, "real_ops": n_ops, "padded_to": max_ops})

        old_log_probs = batch["log_probs"].astype(np.float64)
        old_values = batch["values"].astype(np.float64)
        actions = batch["actions"]
        new_log_probs = []
        values = []
        entropies = []
        topk_mass = []
        logits_rows = []
        for i in range(n):
            logits = self.logits(batch["states"][i], op_padded[i])
            logits = np.where(op_masks[i] > 0, logits, -1e9)
            logits, _ = self.apply_new_action_bias(logits, new_masks[i])
            probs = softmax(logits, op_masks[i])
            new_log_probs.append(self.topk_log_prob(logits, actions[i]))
            values.append(self.value(batch["states"][i]))
            entropies.append(float(-np.sum(probs[op_masks[i] > 0] * np.log(probs[op_masks[i] > 0] + 1e-8))))
            topk_mass.append(float(np.sum(probs[actions[i]])))
            logits_rows.append(logits[: int(op_masks[i].sum())].round(6).tolist())

        new_log_probs = np.array(new_log_probs, dtype=np.float64)
        values = np.array(values, dtype=np.float64)
        entropies = np.array(entropies, dtype=np.float64)
        ratio = np.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -float(np.mean(np.minimum(surr1, surr2)))
        value_loss = 0.5 * float(np.mean((values - returns) ** 2))
        entropy_loss = -float(np.mean(entropies))
        total = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        approx_kl = float(np.mean(old_log_probs - new_log_probs))
        clip_frac = float(np.mean(np.abs(ratio - 1.0) > self.clip_epsilon))
        return float(total), {
            "padding": padding_rows,
            "new_log_probs": new_log_probs.round(6).tolist(),
            "old_log_probs": old_log_probs.round(6).tolist(),
            "ratio": ratio.round(6).tolist(),
            "values_new": values.round(6).tolist(),
            "values_old": old_values.round(6).tolist(),
            "returns": np.array(returns).round(6).tolist(),
            "advantages": np.array(advantages).round(6).tolist(),
            "policy_loss": round(policy_loss, 6),
            "value_loss": round(value_loss, 6),
            "entropy": round(float(np.mean(entropies)), 6),
            "entropy_loss": round(entropy_loss, 6),
            "total_loss": round(float(total), 6),
            "approx_kl": round(approx_kl, 6),
            "clip_frac": round(clip_frac, 6),
            "topk_mass": np.array(topk_mass).round(6).tolist(),
            "logits_rows": logits_rows,
        }

    def update(self, batch: Dict, returns: np.ndarray, advantages: np.ndarray, lr: float = 0.08):
        before = self.parameter_norms()
        base_vec = self.param_vector()

        def loss_at(vec):
            old_vec = self.param_vector()
            self.set_param_vector(vec)
            loss, _ = self.compute_loss(batch, returns, advantages)
            self.set_param_vector(old_vec)
            return loss

        eps = 1e-4
        grad = np.zeros_like(base_vec)
        for i in range(len(base_vec)):
            plus = base_vec.copy()
            minus = base_vec.copy()
            plus[i] += eps
            minus[i] -= eps
            grad[i] = (loss_at(plus) - loss_at(minus)) / (2 * eps)
        self.set_param_vector(base_vec - lr * grad)
        after = self.parameter_norms()
        loss_after, info_after = self.compute_loss(batch, returns, advantages)
        return {
            "param_norms_before": {k: round(v, 6) for k, v in before.items()},
            "grad_norm": round(float(np.linalg.norm(grad)), 6),
            "learning_rate": lr,
            "param_norms_after": {k: round(v, 6) for k, v in after.items()},
            "loss_after_update": round(loss_after, 6),
            "post_update_ratio": info_after["ratio"],
            "post_update_new_log_probs": info_after["new_log_probs"],
        }


def build_state_embedding(
    session_text: str, retrieved_embeddings: Optional[np.ndarray], emb_dim: int
) -> Tuple[np.ndarray, Dict]:
    session_emb = stable_embedding(session_text, emb_dim)
    if retrieved_embeddings is None or len(retrieved_embeddings) == 0:
        memory_emb = np.zeros(emb_dim, dtype=np.float32)
    else:
        memory_emb = retrieved_embeddings.mean(axis=0).astype(np.float32)
    state = np.concatenate([session_emb, memory_emb]).astype(np.float32)
    return state, {
        "session_embedding_shape": list(session_emb.shape),
        "memory_embedding_shape": list(memory_emb.shape),
        "state_embedding_shape": list(state.shape),
        "state_embedding_first6": state[:6].round(4).tolist(),
    }


def desired_update_type(session_text: str, memory_bank: DebugMemoryBank) -> str:
    lowered = session_text.lower()
    if "ignore" in lowered or "weather" in lowered:
        return "noop"
    if "moved" in lowered or "now" in lowered or "instead" in lowered:
        return "update" if len(memory_bank.memories) > 0 else "insert"
    return "insert" if len(memory_bank.memories) == 0 else "update"


def execute_ops(
    session_text: str,
    selected_ops: Sequence[Operation],
    memory_bank: DebugMemoryBank,
) -> List[Dict]:
    results = []
    for op in selected_ops:
        before = memory_bank.contents()
        if op.update_type == "insert":
            memory_bank.add(f"Memory from span: {session_text}", op.name)
            status = "inserted"
        elif op.update_type == "update" and len(memory_bank.memories) > 0:
            memory_bank.update(0, f"{memory_bank.memories[0].content} | update: {session_text}", op.name)
            status = "updated index 0"
        elif op.update_type == "delete" and "contradiction" in session_text.lower() and len(memory_bank.memories) > 0:
            memory_bank.delete(0)
            status = "deleted index 0"
        else:
            status = "no change"
        results.append(
            {
                "operation": op.name,
                "update_type": op.update_type,
                "status": status,
                "memory_before": before,
                "memory_after": memory_bank.contents(),
            }
        )
    return results


def compute_process_reward(
    selected_ops: Sequence[Operation],
    desired: str,
    episode_length: int,
    process_budget: float,
) -> Tuple[float, Dict]:
    quota = process_budget / max(episode_length, 1)
    match_count = sum(1 for op in selected_ops if op.update_type == desired)
    match_ratio = match_count / max(len(selected_ops), 1)
    reward = quota * match_ratio
    return float(reward), {
        "process_budget_for_trace": process_budget,
        "episode_length": episode_length,
        "quota_per_span": round(quota, 6),
        "desired_update_type": desired,
        "selected_update_types": [op.update_type for op in selected_ops],
        "match_ratio": round(match_ratio, 6),
        "process_reward": round(reward, 6),
    }


def evaluate_final_reward(memory_bank: DebugMemoryBank) -> Tuple[float, Dict]:
    text = " ".join(memory_bank.contents()).lower()
    checks = {
        "alice_seen": "alice" in text,
        "tea_seen": "tea" in text,
        "paris_seen": "paris" in text,
        "bob_seen": "bob" in text,
    }
    reward = sum(checks.values()) / len(checks)
    return float(reward), {"checks": checks, "final_reward": round(float(reward), 6)}


def write_jsonl(path: Path, records: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown(path: Path, summary: Dict, records: List[Dict]):
    lines = []
    lines.append("# Debug One Batch Flow\n")
    lines.append("这个文件是一轮可观察的 PPO 训练流日志：一条 trace 被切成多个 span，逐 span 选择 skill、更新 memory，最后把 reward 写回 buffer 并更新 controller 参数。\n")
    lines.append("## 1. Sample Selected\n")
    lines.append(f"- sample_id: `{summary['sample']['sample_id']}`")
    lines.append(f"- sessions: `{len(summary['sessions'])}`")
    for i, s in enumerate(summary["sessions"]):
        lines.append(f"- span {i}: {s}")
    lines.append("\n## 2. Per Span Breakpoints\n")
    for record in records:
        if record["event"] != "span_processed":
            continue
        lines.append(f"### Span {record['session_idx']}\n")
        lines.append(f"- text: {record['session_text']}")
        lines.append(f"- memory before: {record['memory_size_before']}, after: {record['memory_size_after']}")
        lines.append(f"- retrieved indices: `{record['retrieved_indices']}`")
        lines.append(f"- state shape: `{record['state_info']['state_embedding_shape']}` first6: `{record['state_info']['state_embedding_first6']}`")
        lines.append(f"- candidate ops: `{record['candidate_ops']}`")
        lines.append(f"- op embedding shape: `{record['op_embedding_shape']}`, new mask: `{record['new_op_mask']}`")
        lines.append(f"- raw logits: `{record['controller']['raw_logits']}`")
        lines.append(f"- biased logits: `{record['controller']['biased_logits']}`")
        lines.append(f"- probs: `{record['controller']['probs']}`")
        lines.append(f"- selected actions: `{record['actions']}` -> `{record['selected_ops']}`")
        lines.append(f"- old_log_prob: `{record['log_prob']}`, value: `{record['value']}`")
        lines.append(f"- process reward: `{record['process_reward_meta']}`")
        lines.append("")
    lines.append("## 3. Episode Reward And Buffer\n")
    lines.append("```json")
    lines.append(json.dumps(summary["episode"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("\n## 4. GAE Returns / Advantages\n")
    lines.append("```json")
    lines.append(json.dumps(summary["gae"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("\n## 5. PPO Update\n")
    lines.append("```json")
    lines.append(json.dumps(summary["ppo_update"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("\n## 6. Final Memory And Skill Bank\n")
    lines.append("```json")
    lines.append(json.dumps(summary["final_state"], ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_debug_flow(out_dir: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    np.set_printoptions(suppress=True, precision=6)
    emb_dim = 8
    state_dim = emb_dim * 2
    op_dim = emb_dim
    sessions = [
        "Alice likes tea and keeps notes about project Orion.",
        "Alice moved to Paris, so her location memory should be updated.",
        "Bob joined project Orion and Alice now coordinates with Bob.",
    ]
    sample = {"sample_id": "debug_trace_001", "raw_trace": " | ".join(sessions)}
    memory_bank = DebugMemoryBank(dim=emb_dim, top_k=2)
    skill_bank = DebugSkillBank(op_dim=op_dim)
    controller = NumpyPPOController(
        state_dim=state_dim,
        op_dim=op_dim,
        action_top_k=2,
        seed=seed,
    )
    buffer = DebugPPOBuffer()
    records: List[Dict] = []
    process_budget = 0.3

    records.append({"event": "sample_selected", "sample": sample, "sessions": sessions})

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
                    "note": "模拟 Designer 在处理完第一个 span 后新增 skill；后续 step 的 num_ops 从 4 变成 5。",
                }
            )

        memory_before = memory_bank.contents()
        query_emb = stable_embedding(session_text, emb_dim)
        retrieved_texts, retrieved_indices, retrieved_embs = memory_bank.retrieve(query_emb)
        state, state_info = build_state_embedding(session_text, retrieved_embs, emb_dim)
        candidate_ops = skill_bank.candidates()
        op_embs = np.array([op.embedding for op in candidate_ops], dtype=np.float32)
        new_op_mask = np.array([1.0 if op.is_new else 0.0 for op in candidate_ops], dtype=np.float32)

        actions, log_prob, value, controller_info = controller.forward(
            state, op_embs, new_op_mask, deterministic=False
        )
        selected_ops = [candidate_ops[i] for i in actions]
        exec_results = execute_ops(session_text, selected_ops, memory_bank)
        desired = desired_update_type(session_text, memory_bank)
        process_reward, process_meta = compute_process_reward(
            selected_ops, desired, len(sessions), process_budget
        )
        skill_bank.update_stats([op.name for op in selected_ops], process_reward)
        buffer.push(state, op_embs, new_op_mask, actions, log_prob, value, process_reward)

        record = {
            "event": "span_processed",
            "session_idx": session_idx,
            "session_text": session_text,
            "memory_size_before": len(memory_before),
            "memory_size_after": len(memory_bank.memories),
            "retrieved_texts": retrieved_texts,
            "retrieved_indices": retrieved_indices,
            "state_info": state_info,
            "candidate_ops": [f"{op.name}/{op.update_type}" for op in candidate_ops],
            "op_embedding_shape": list(op_embs.shape),
            "new_op_mask": new_op_mask.astype(int).tolist(),
            "controller": controller_info,
            "actions": actions,
            "selected_ops": [f"{op.name}/{op.update_type}" for op in selected_ops],
            "log_prob": round(float(log_prob), 6),
            "value": round(float(value), 6),
            "executor_results": exec_results,
            "process_reward_meta": process_meta,
            "memory_contents_after": memory_bank.contents(),
        }
        records.append(record)
        skill_bank.mark_all_old()

    final_reward, final_reward_info = evaluate_final_reward(memory_bank)
    finish_info = buffer.finish_episode(
        final_reward=final_reward,
        redistribute=True,
        redistribution_decay=0.9,
        final_reward_last_ratio=0.4,
    )
    returns, advantages, gae_rows = buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    adv_mean = float(advantages.mean())
    adv_std = float(advantages.std() + 1e-8)
    advantages_norm = (advantages - adv_mean) / adv_std
    batch = buffer.batch()
    loss_before, loss_info_before = controller.compute_loss(batch, returns, advantages_norm)
    update_info = controller.update(batch, returns, advantages_norm, lr=0.08)

    summary = {
        "sample": sample,
        "sessions": sessions,
        "episode": {
            "final_reward_info": final_reward_info,
            "finish_episode": finish_info,
            "buffer_actions": buffer.actions,
            "buffer_old_log_probs": [round(x, 6) for x in buffer.log_probs],
            "buffer_old_values": [round(x, 6) for x in buffer.values],
            "buffer_rewards": [round(x, 6) for x in buffer.rewards],
            "buffer_dones": buffer.dones,
        },
        "gae": {
            "rows": gae_rows,
            "returns": returns.round(6).tolist(),
            "advantages_raw": advantages.round(6).tolist(),
            "advantages_normalized": advantages_norm.round(6).tolist(),
        },
        "ppo_update": {
            "loss_before_update": round(float(loss_before), 6),
            "loss_info_before": loss_info_before,
            "update_info": update_info,
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
    write_jsonl(out_dir / "debug_records.jsonl", records)
    (out_dir / "debug_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "debug_memory_bank.json").write_text(
        json.dumps(summary["final_state"]["memory_bank"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "debug_ppo_update.json").write_text(
        json.dumps(summary["ppo_update"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(out_dir / "debug_trace.md", summary, records)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="MemSkill/jittor_controller_repro/runs/debug_one_batch_flow",
        help="Directory for debug_trace.md and JSON artifacts.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    summary = run_debug_flow(out_dir=out_dir, seed=args.seed)
    print(f"debug_trace: {out_dir / 'debug_trace.md'}")
    print(f"debug_records: {out_dir / 'debug_records.jsonl'}")
    print(f"debug_summary: {out_dir / 'debug_summary.json'}")
    print("loss_before:", summary["ppo_update"]["loss_before_update"])
    print("loss_after:", summary["ppo_update"]["update_info"]["loss_after_update"])
    print("buffer_rewards:", summary["episode"]["buffer_rewards"])
    print("returns:", summary["gae"]["returns"])
    print("advantages_normalized:", summary["gae"]["advantages_normalized"])


if __name__ == "__main__":
    main()
