"""Numerical parity checks between original PyTorch and Jittor controllers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jittor_controller_repro.baselines.original_torch_runner import build_original_controller
from jittor_controller_repro.common import require_package
from jittor_controller_repro.data.trace_schema import load_trace, pad_op_embeddings
from jittor_controller_repro.models.jittor_controller import JittorPPOController


def copy_linear_state(torch_controller, jittor_controller) -> None:
    """Copy matching linear weights from PyTorch to Jittor for parity tests."""

    import jittor as jt

    pairs = [
        (torch_controller.state_net, jittor_controller.state_net),
        (torch_controller.op_net, jittor_controller.op_net),
        (torch_controller.actor_head, jittor_controller.actor_head),
        (torch_controller.critic_head, jittor_controller.critic_head),
    ]
    for torch_seq, jt_seq in pairs:
        torch_linears = [m for m in torch_seq if m.__class__.__name__ == "Linear"]
        jt_linears = [m for m in jt_seq if m.__class__.__name__ == "Linear"]
        if len(torch_linears) != len(jt_linears):
            raise RuntimeError("linear layer count mismatch")
        for t_layer, j_layer in zip(torch_linears, jt_linears):
            j_layer.weight.assign(jt.array(t_layer.weight.detach().cpu().numpy()))
            j_layer.bias.assign(jt.array(t_layer.bias.detach().cpu().numpy()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original PyTorch and Jittor controller outputs.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    require_package("torch", "Install torch to run parity checks.")
    require_package("jittor", "Install jittor to run parity checks.")
    import torch
    import jittor as jt

    torch.manual_seed(123)
    jt.set_global_seed(123)
    trace = load_trace(args.trace)
    n = min(args.steps, trace.n_steps)
    torch_controller = build_original_controller(
        trace.state_dim, trace.op_dim, args.hidden_dim, trace.action_top_k, device=args.device
    )
    jittor_controller = JittorPPOController(
        state_dim=trace.state_dim,
        op_dim=trace.op_dim,
        hidden_dim=args.hidden_dim,
        action_top_k=trace.action_top_k,
    )
    copy_linear_state(torch_controller, jittor_controller)

    batch = trace.to_batch()
    sub_batch = {
        "states": batch["states"][:n],
        "op_embs": batch["op_embs"][:n],
        "new_op_masks": batch["new_op_masks"][:n],
        "actions": batch["actions"][:n],
        "log_probs": batch["log_probs"][:n],
        "values": batch["values"][:n],
    }
    op_padded, op_masks, new_masks = pad_op_embeddings(sub_batch["op_embs"], sub_batch["new_op_masks"])
    with torch.no_grad():
        state_t = torch.FloatTensor(np.asarray(sub_batch["states"])).to(args.device)
        op_t = torch.FloatTensor(op_padded).to(args.device)
        mask_t = torch.FloatTensor(op_masks).to(args.device)
        actions_t = torch.LongTensor(sub_batch["actions"]).to(args.device)
        torch_lp, torch_values, torch_entropy, _ = torch_controller.evaluate_actions(
            state_t, op_t, actions_t, mask_t
        )
        torch_loss, torch_info = torch_controller.compute_ppo_loss(
            sub_batch, trace.returns[:n], trace.advantages[:n]
        )

    jt_state = jt.array(np.asarray(sub_batch["states"], dtype=np.float32))
    jt_op = jt.array(op_padded)
    jt_mask = jt.array(op_masks)
    jt_actions = jt.array(np.asarray(sub_batch["actions"], dtype=np.int64))
    jt_lp, jt_values, jt_entropy, _ = jittor_controller.evaluate_actions(
        jt_state, jt_op, jt_actions, jt_mask
    )
    jt_loss, jt_info = jittor_controller.compute_ppo_loss(
        sub_batch, trace.returns[:n], trace.advantages[:n]
    )

    metrics = {
        "log_prob_max_abs_diff": float(np.max(np.abs(torch_lp.cpu().numpy() - jt_lp.numpy()))),
        "value_max_abs_diff": float(np.max(np.abs(torch_values.cpu().numpy() - jt_values.numpy()))),
        "entropy_max_abs_diff": float(np.max(np.abs(torch_entropy.cpu().numpy() - jt_entropy.numpy()))),
        "loss_abs_diff": float(abs(float(torch_loss.item()) - float(jt_loss.item()))),
    }
    for key in ["policy_loss", "value_loss", "entropy", "topk_mass", "approx_kl", "clip_frac"]:
        metrics[f"{key}_abs_diff"] = float(abs(torch_info[key] - jt_info[key]))
    print(metrics)
    failed = {k: v for k, v in metrics.items() if v > args.tolerance}
    if failed:
        raise SystemExit(f"parity check failed: {failed}")


if __name__ == "__main__":
    main()
