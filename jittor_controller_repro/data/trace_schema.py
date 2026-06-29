"""Shared trace schema for MemSkill controller reproduction.

The original MemSkill trainer stores transitions in ``PPOBuffer`` as Python
lists of NumPy arrays.  This module keeps that shape because it is the most
direct bridge between the original PyTorch controller and the Jittor port.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


@dataclass
class ControllerTrace:
    """A framework-neutral PPO controller trace.

    Attributes mirror ``PPOBuffer.get_batch()`` plus precomputed returns and
    advantages so PyTorch and Jittor can train on exactly the same evidence.
    """

    states: np.ndarray
    op_embs: list[np.ndarray]
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    new_op_masks: list[np.ndarray | None] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.log_probs = np.asarray(self.log_probs, dtype=np.float32)
        self.values = np.asarray(self.values, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.dones = np.asarray(self.dones, dtype=np.float32)
        self.returns = np.asarray(self.returns, dtype=np.float32)
        self.advantages = np.asarray(self.advantages, dtype=np.float32)
        self.op_embs = [np.asarray(op, dtype=np.float32) for op in self.op_embs]
        if not self.new_op_masks:
            self.new_op_masks = [None for _ in self.op_embs]
        else:
            self.new_op_masks = [
                None if mask is None else np.asarray(mask, dtype=np.float32)
                for mask in self.new_op_masks
            ]
        self.validate()

    @property
    def n_steps(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def op_dim(self) -> int:
        return int(self.op_embs[0].shape[1])

    @property
    def action_top_k(self) -> int:
        if self.actions.ndim == 1:
            return 1
        return int(self.actions.shape[1])

    def validate(self) -> None:
        n = int(self.states.shape[0])
        if n == 0:
            raise ValueError("ControllerTrace must contain at least one step")
        arrays = {
            "op_embs": self.op_embs,
            "log_probs": self.log_probs,
            "values": self.values,
            "rewards": self.rewards,
            "dones": self.dones,
            "returns": self.returns,
            "advantages": self.advantages,
            "new_op_masks": self.new_op_masks,
        }
        for name, value in arrays.items():
            if len(value) != n:
                raise ValueError(f"{name} length {len(value)} does not match steps {n}")
        if self.actions.shape[0] != n:
            raise ValueError("actions length does not match states")
        op_dim = self.op_embs[0].shape[1]
        for i, op in enumerate(self.op_embs):
            if op.ndim != 2:
                raise ValueError(f"op_embs[{i}] must be 2D")
            if op.shape[1] != op_dim:
                raise ValueError("all op embeddings must share op_dim")
            action = self.actions[i]
            max_action = int(np.max(action)) if np.ndim(action) > 0 else int(action)
            if max_action >= op.shape[0]:
                raise ValueError(f"action index out of range at step {i}")
            mask = self.new_op_masks[i]
            if mask is not None and mask.shape[0] != op.shape[0]:
                raise ValueError(f"new_op_mask length mismatch at step {i}")

    def to_batch(self) -> dict[str, Any]:
        """Return the batch shape consumed by original ``PPOController``."""

        return {
            "states": [self.states[i] for i in range(self.n_steps)],
            "op_embs": self.op_embs,
            "new_op_masks": self.new_op_masks,
            "actions": self.actions.tolist(),
            "log_probs": self.log_probs.tolist(),
            "values": self.values.tolist(),
        }


def object_array(items: list[Any]) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def save_trace(trace: ControllerTrace, path: str | Path) -> None:
    """Save a trace as a compressed NPZ file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(trace.metadata)
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dim": trace.state_dim,
            "op_dim": trace.op_dim,
            "action_top_k": trace.action_top_k,
            "n_steps": trace.n_steps,
        }
    )
    np.savez_compressed(
        path,
        metadata=json.dumps(metadata, ensure_ascii=False),
        steps=json.dumps(trace.steps, ensure_ascii=False),
        episodes=json.dumps(trace.episodes, ensure_ascii=False),
        states=trace.states,
        op_embs=object_array(trace.op_embs),
        new_op_masks=object_array(trace.new_op_masks),
        actions=trace.actions,
        log_probs=trace.log_probs,
        values=trace.values,
        rewards=trace.rewards,
        dones=trace.dones,
        returns=trace.returns,
        advantages=trace.advantages,
    )


def load_trace(path: str | Path) -> ControllerTrace:
    """Load a controller trace saved by :func:`save_trace`."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata"]))
        steps = json.loads(str(data["steps"])) if "steps" in data else []
        episodes = json.loads(str(data["episodes"])) if "episodes" in data else []
        return ControllerTrace(
            states=data["states"],
            op_embs=list(data["op_embs"]),
            new_op_masks=list(data["new_op_masks"]),
            actions=data["actions"],
            log_probs=data["log_probs"],
            values=data["values"],
            rewards=data["rewards"],
            dones=data["dones"],
            returns=data["returns"],
            advantages=data["advantages"],
            metadata=metadata,
            steps=steps,
            episodes=episodes,
        )


def compute_returns_and_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy implementation of MemSkill PPOBuffer GAE."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns.astype(np.float32), advantages.astype(np.float32)


def pad_op_embeddings(
    op_embs: list[np.ndarray],
    new_op_masks: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Pad variable candidate-operation embeddings for batched controllers."""

    n = len(op_embs)
    op_dim = op_embs[0].shape[1]
    max_ops = max(op.shape[0] for op in op_embs)
    padded = np.zeros((n, max_ops, op_dim), dtype=np.float32)
    masks = np.zeros((n, max_ops), dtype=np.float32)
    padded_new = None
    if new_op_masks is not None:
        padded_new = np.zeros((n, max_ops), dtype=np.float32)
    for i, op in enumerate(op_embs):
        n_ops = op.shape[0]
        padded[i, :n_ops] = op
        masks[i, :n_ops] = 1.0
        if padded_new is not None and new_op_masks[i] is not None:
            padded_new[i, :n_ops] = new_op_masks[i]
    return padded, masks, padded_new

