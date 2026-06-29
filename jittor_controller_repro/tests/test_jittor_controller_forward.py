from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("nvcc_path", "")

from jittor_controller_repro.models.jittor_controller import JittorPPOController


def test_select_action_supports_variable_skill_counts():
    controller = JittorPPOController(state_dim=6, op_dim=6, hidden_dim=12, action_top_k=3)
    state = np.ones(6, dtype="float32")

    for n_ops in (3, 5, 7):
        ops = np.random.default_rng(n_ops).normal(size=(n_ops, 6)).astype("float32")
        action, log_prob, value = controller.select_action(state, ops, deterministic=True)
        assert len(action) == min(3, n_ops)
        assert all(0 <= idx < n_ops for idx in action)
        assert math.isfinite(log_prob)
        assert math.isfinite(value)


def test_evaluate_actions_returns_batch_shapes():
    import jittor as jt

    controller = JittorPPOController(state_dim=6, op_dim=6, hidden_dim=12, action_top_k=2)
    state_embs = jt.array(np.zeros((2, 6), dtype="float32"))
    op_embs = jt.array(np.zeros((2, 4, 6), dtype="float32"))
    actions = jt.array(np.asarray([[0, 1], [1, 2]], dtype="int64"))
    masks = jt.array(np.ones((2, 4), dtype="float32"))

    log_probs, values, entropy, stats = controller.evaluate_actions(state_embs, op_embs, actions, masks)

    assert tuple(log_probs.shape) == (2,)
    assert tuple(values.shape) == (2,)
    assert tuple(entropy.shape) == (2,)
    assert tuple(stats["topk_mass"].shape) == (2,)
