from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("nvcc_path", "")

from jittor_controller_repro.models.jittor_controller import JittorPPOController


def test_jittor_checkpoint_state_roundtrip_allows_inference(tmp_path):
    import jittor as jt

    controller = JittorPPOController(state_dim=5, op_dim=5, hidden_dim=10, action_top_k=2)
    state_dict = {
        key: value.numpy() if hasattr(value, "numpy") else value
        for key, value in controller.state_dict().items()
    }
    checkpoint = tmp_path / "controller_state.pkl"
    jt.save(state_dict, str(checkpoint))

    loaded = jt.load(str(checkpoint))
    restored = JittorPPOController(state_dim=5, op_dim=5, hidden_dim=10, action_top_k=2)
    restored.load_state_dict({
        key: jt.array(value) if isinstance(value, np.ndarray) else value
        for key, value in loaded.items()
    })

    action, log_prob, value = restored.select_action(
        np.zeros(5, dtype="float32"),
        np.zeros((4, 5), dtype="float32"),
        deterministic=True,
    )

    assert len(action) == 2
    assert np.isfinite(log_prob)
    assert np.isfinite(value)
