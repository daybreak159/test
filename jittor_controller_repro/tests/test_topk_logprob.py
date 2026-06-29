from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("nvcc_path", "")

JittorPPOController = pytest.importorskip(
    "jittor_controller_repro.models.jittor_controller"
).JittorPPOController


def _loop_joint_log_prob(selected_probs: np.ndarray) -> np.ndarray:
    joint = np.zeros(selected_probs.shape[0], dtype=np.float64)
    remaining = np.ones(selected_probs.shape[0], dtype=np.float64)
    eps = 1e-8
    for i in range(selected_probs.shape[1]):
        p_i = np.clip(selected_probs[:, i], eps, None)
        denom = np.clip(remaining, eps, None)
        joint += np.log(p_i) - np.log(denom)
        remaining = np.clip(remaining - p_i, eps, None)
    return joint


def _vectorized_joint_log_prob(selected_probs: np.ndarray) -> np.ndarray:
    eps = 1e-8
    prefix_selected = np.cumsum(selected_probs, axis=-1) - selected_probs
    remaining = np.clip(1.0 - prefix_selected, eps, None)
    safe_selected = np.clip(selected_probs, eps, None)
    return (np.log(safe_selected) - np.log(remaining)).sum(axis=-1)


def test_vectorized_joint_log_prob_matches_loop_formula():
    selected_probs = np.asarray(
        [
            [0.20, 0.30, 0.10],
            [0.05, 0.70, 0.15],
            [0.40, 0.20, 0.30],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        _vectorized_joint_log_prob(selected_probs),
        _loop_joint_log_prob(selected_probs),
        rtol=1e-7,
        atol=1e-7,
    )


def test_topk_select_action_has_no_repeated_indices():
    controller = JittorPPOController(state_dim=8, op_dim=8, hidden_dim=16, action_top_k=3)
    state = np.random.default_rng(0).normal(size=(8,)).astype("float32")
    ops = np.random.default_rng(1).normal(size=(5, 8)).astype("float32")

    action, log_prob, value = controller.select_action(state, ops, deterministic=True)

    assert isinstance(action, list)
    assert len(action) == 3
    assert len(set(action)) == 3
    assert math.isfinite(log_prob)
    assert math.isfinite(value)


def test_topk_k1_degenerates_to_single_action():
    controller = JittorPPOController(state_dim=8, op_dim=8, hidden_dim=16, action_top_k=1)
    state = np.random.default_rng(2).normal(size=(8,)).astype("float32")
    ops = np.random.default_rng(3).normal(size=(4, 8)).astype("float32")

    action, log_prob, _ = controller.select_action(state, ops, deterministic=True)

    assert isinstance(action, int)
    assert 0 <= action < 4
    assert math.isfinite(log_prob)


def test_joint_log_prob_is_not_nan_for_fixed_topk_actions():
    controller = JittorPPOController(state_dim=8, op_dim=8, hidden_dim=16, action_top_k=2)
    batch = {
        "states": [np.zeros(8, dtype="float32")],
        "op_embs": [np.eye(4, 8, dtype="float32")],
        "new_op_masks": [np.zeros(4, dtype="float32")],
        "actions": [[0, 1]],
        "log_probs": [0.0],
        "values": [0.0],
    }

    loss, info = controller.compute_ppo_loss(
        batch,
        returns=np.asarray([1.0], dtype="float32"),
        advantages=np.asarray([1.0], dtype="float32"),
    )

    assert math.isfinite(float(np.asarray(loss.numpy()).reshape(-1)[0]))
    assert math.isfinite(info["policy_loss"])
    assert math.isfinite(info["topk_mass"])
