"""Generate deterministic synthetic controller traces.

Mode A intentionally avoids LLMs, transformers, PyTorch, and Jittor.  It gives
us a stable dataset for numerical parity tests before using API-cached traces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .trace_schema import ControllerTrace, compute_returns_and_advantages, save_trace


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def _topk_log_prob(logits: np.ndarray, actions: np.ndarray) -> float:
    probs = _softmax(logits)
    remaining = 1.0
    out = 0.0
    eps = 1e-8
    for action in actions:
        p_i = max(float(probs[int(action)]), eps)
        denom = max(remaining, eps)
        out += np.log(p_i) - np.log(denom)
        remaining = max(remaining - p_i, eps)
    return float(out)


def generate_synthetic_trace(
    n_steps: int = 128,
    state_dim: int = 128,
    op_dim: int = 128,
    action_top_k: int = 3,
    min_ops: int = 4,
    max_ops: int = 12,
    episode_len: int = 8,
    seed: int = 42,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> ControllerTrace:
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(state_dim), size=(state_dim, op_dim)).astype(np.float32)

    states: list[np.ndarray] = []
    op_embs: list[np.ndarray] = []
    new_op_masks: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    steps: list[dict] = []
    episodes: list[dict] = []

    for step in range(n_steps):
        num_ops = int(rng.integers(min_ops, max_ops + 1))
        state = rng.normal(size=(state_dim,)).astype(np.float32)
        ops = rng.normal(size=(num_ops, op_dim)).astype(np.float32)
        state_query = state @ projection
        true_scores = ops @ state_query / np.sqrt(op_dim)
        behavior_logits = true_scores + rng.normal(0.0, 0.35, size=(num_ops,))

        k = min(action_top_k, num_ops)
        chosen = np.argsort(-behavior_logits)[:k].astype(np.int64)
        reward = float(np.mean(true_scores[chosen]) - 0.05 * k)
        reward = float(np.tanh(reward))
        value = float(np.mean(true_scores) + rng.normal(0.0, 0.1))
        done = 1.0 if ((step + 1) % episode_len == 0 or step == n_steps - 1) else 0.0

        mask = np.zeros(num_ops, dtype=np.float32)
        if step % 5 == 0:
            mask[-1] = 1.0

        states.append(state)
        op_embs.append(ops)
        new_op_masks.append(mask)
        actions.append(chosen)
        log_probs.append(_topk_log_prob(behavior_logits, chosen))
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        steps.append(
            {
                "source": "synthetic",
                "step": step,
                "candidate_ops": [f"skill_{i}" for i in range(num_ops)],
                "selected_action": chosen.tolist(),
                "process_reward": reward,
                "executor_prompt": "",
                "executor_response": "",
                "exec_results": [],
            }
        )
        if done:
            episodes.append(
                {
                    "source": "synthetic",
                    "last_step": step,
                    "final_reward": reward,
                    "qa_prompts": [],
                    "qa_responses": [],
                    "f1_score": reward,
                    "llm_judge_score": 0.0,
                }
            )

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.float32)
    returns, advantages = compute_returns_and_advantages(
        rewards_arr, values_arr, dones_arr, gamma=gamma, gae_lambda=gae_lambda
    )
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return ControllerTrace(
        states=np.asarray(states, dtype=np.float32),
        op_embs=op_embs,
        new_op_masks=new_op_masks,
        actions=np.asarray(actions, dtype=np.int64),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        values=values_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        returns=returns,
        advantages=advantages.astype(np.float32),
        metadata={
            "source": "synthetic",
            "seed": seed,
            "state_dim": state_dim,
            "op_dim": op_dim,
            "action_top_k": action_top_k,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
        },
        steps=steps,
        episodes=episodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic MemSkill controller trace.")
    parser.add_argument("--output", default="jittor_controller_repro/runs/synthetic_trace.npz")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--op-dim", type=int, default=128)
    parser.add_argument("--action-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trace = generate_synthetic_trace(
        n_steps=args.n_steps,
        state_dim=args.state_dim,
        op_dim=args.op_dim,
        action_top_k=args.action_top_k,
        seed=args.seed,
    )
    save_trace(trace, Path(args.output))
    print(f"saved {trace.n_steps} synthetic steps to {args.output}")


if __name__ == "__main__":
    main()

