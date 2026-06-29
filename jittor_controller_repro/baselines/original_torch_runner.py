"""Wrapper around the original MemSkill PyTorch PPOController.

The baseline intentionally imports the original implementation instead of a
rewritten copy.  This keeps the PyTorch side faithful to the paper repository.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Any


def memskill_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_memskill_on_path() -> None:
    root = memskill_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def import_original_controller():
    """Load the paper repository's controller file without importing src.__init__.

    Importing ``src.controller`` normally executes ``src/__init__.py``, which
    pulls in executor/designer dependencies unrelated to the controller parity
    test.  Loading the file directly keeps the baseline faithful to the original
    PPOController implementation while avoiding those extra imports.
    """

    ensure_memskill_on_path()
    controller_path = memskill_root() / "src" / "controller.py"
    spec = importlib.util.spec_from_file_location("memskill_original_controller", controller_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load original controller from {controller_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise SystemExit(
            "Unable to import original MemSkill PPOController. "
            "Install the original controller dependencies first."
        ) from exc
    PPOController = getattr(module, "PPOController", None)
    if PPOController is None:
        raise SystemExit(f"PPOController not found in {controller_path}")
    return PPOController


def build_original_controller(
    state_dim: int,
    op_dim: int,
    hidden_dim: int,
    action_top_k: int,
    device: str = "cpu",
    **kwargs: Any,
):
    """Construct the original PyTorch PPOController with reproduction defaults."""

    PPOController = import_original_controller()
    return PPOController(
        state_dim=state_dim,
        op_dim=op_dim,
        hidden_dim=hidden_dim,
        device=device,
        action_top_k=action_top_k,
        **kwargs,
    )
