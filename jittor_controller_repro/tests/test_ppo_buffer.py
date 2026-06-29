from __future__ import annotations

import numpy as np

from jittor_controller_repro.data.trace_schema import ControllerTrace, compute_returns_and_advantages


def test_controller_trace_validates_consistent_field_lengths():
    trace = ControllerTrace(
        states=np.zeros((2, 4), dtype="float32"),
        op_embs=[np.zeros((3, 4), dtype="float32"), np.zeros((3, 4), dtype="float32")],
        actions=np.asarray([[0, 1], [1, 2]], dtype="int64"),
        log_probs=np.asarray([0.1, 0.2], dtype="float32"),
        values=np.asarray([0.0, 0.1], dtype="float32"),
        rewards=np.asarray([0.0, 1.0], dtype="float32"),
        dones=np.asarray([0.0, 1.0], dtype="float32"),
        returns=np.asarray([0.9, 1.0], dtype="float32"),
        advantages=np.asarray([0.9, 0.9], dtype="float32"),
    )

    assert trace.n_steps == 2
    assert trace.action_top_k == 2
    assert len(trace.to_batch()["states"]) == 2


def test_returns_and_advantages_are_finite_and_boundary_aware():
    returns, advantages = compute_returns_and_advantages(
        rewards=np.asarray([0.0, 1.0, 0.5], dtype="float32"),
        values=np.asarray([0.1, 0.2, 0.3], dtype="float32"),
        dones=np.asarray([0.0, 1.0, 1.0], dtype="float32"),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert returns.shape == (3,)
    assert advantages.shape == (3,)
    assert np.isfinite(returns).all()
    assert np.isfinite(advantages).all()
