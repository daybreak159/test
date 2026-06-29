"""Jittor implementation of MemSkill's PPOController.

这个文件只复现 Controller 的可训练 PPO 网络部分：
- 输入的 state_embedding / op_embeddings 已经由 trainer.py 中的 encoder 提前算好；
- 本文件负责把 numpy/list 形式的轨迹数据转换成 Jittor jt.Var；
- 用 Jittor 的 nn.Module/nn.Sequential 搭建 actor、critic；
- 在 rollout 阶段选择 Top-K skill，并记录 old_log_prob / old_value；
- 在 PPO 更新阶段重新计算 new_log_prob / value / entropy，返回可反传的 total_loss。

因此它不是 MemoryBank / OperationBank 的实现，也不负责调用 LLM executor。
它对应 PyTorch 原版 src/controller.py 中 PPOController 的 Jittor 迁移版本。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.trace_schema import pad_op_embeddings


def _require_jittor():
    # Jittor 是可选依赖。这里采用延迟导入：
    # 只有真正构造 JittorPPOController 时才 import jittor，
    # 这样普通 PyTorch 训练、数据处理、脚本导入不会因为没有 Jittor 而直接失败。
    try:
        import jittor as jt
        from jittor import nn
    except ImportError as exc:
        raise SystemExit(
            "Jittor is required for this command. Install jittor before running "
            "the Jittor reproduction scripts."
        ) from exc
    return jt, nn


class JittorPPOController:
    """Factory wrapper that delays importing Jittor until construction time.

    这个外层类本身不是 Jittor 网络。
    __new__ 里动态创建内部 _Controller(nn.Module)，然后直接返回 _Controller 实例。
    这样做的好处是：外部 trainer 仍然可以像使用 PPOController 一样使用它，
    但真正的 Jittor import / nn.Module 初始化只发生在选择 Jittor backend 时。
    """

    def __new__(
        cls,
        state_dim: int = 768,
        op_dim: int = 768,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        vf_clip: float = 0.0,
        new_action_p_min: float = 0.0,
        new_action_delta_max: float = 0.0,
        action_top_k: int = 1,
    ):
        jt, nn = _require_jittor()

        class _Controller(nn.Module):
            def __init__(self):
                super().__init__()
                # 下面这些超参数与 PyTorch PPOController 保持同名同义。
                # 迁移时要保证这些字段名不变，因为 trainer.py 会统一构造 controller_kwargs。
                self.state_dim = state_dim
                self.op_dim = op_dim
                self.hidden_dim = hidden_dim
                self.gamma = gamma
                self.gae_lambda = gae_lambda
                self.clip_epsilon = clip_epsilon
                self.entropy_coef = entropy_coef
                self.value_coef = value_coef
                self.vf_clip = vf_clip
                self.new_action_p_min = new_action_p_min
                self.new_action_delta_max = new_action_delta_max
                self.new_action_bias_scale = 0.0
                self.action_top_k = action_top_k

                # Jittor 版状态编码网络。
                # 输入 state_embedding: [B, state_dim]
                # 输出 state_h:        [B, hidden_dim]
                # 这里对应 PyTorch 原版中的 nn.Sequential + nn.Linear + ReLU。
                self.state_net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )

                # Jittor 版 skill/op 编码网络。
                # 输入 op_embeddings: [B, num_ops, op_dim]
                # 输出 op_h:         [B, num_ops, hidden_dim]
                # Jittor 的 Linear 可以作用在最后一维，因此可以直接处理三维张量。
                self.op_net = nn.Sequential(
                    nn.Linear(op_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )

                # actor_head 对每个 (state, skill) pair 打一个 logit 分数。
                # 输入是 concat([state_h, op_h_i])，维度为 hidden_dim * 2。
                # 输出最后 squeeze 成 [B, num_ops]，再通过 softmax 变成动作概率。
                self.actor_head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

                # critic_head 只看 state_h，输出 V(s_t)。
                # 它预测当前状态后续能获得的累计回报，用于 GAE 和 value loss。
                self.critic_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def encode_state(self, state_embedding):
                # state_embedding 必须已经是 Jittor jt.Var。
                # 如果从 trainer/Buffer 传来的是 numpy，会在 select_action 或 compute_ppo_loss 边界处转换。
                return self.state_net(state_embedding)

            def encode_ops(self, op_embeddings):
                # op_embeddings 形状可以是 [B, num_ops, op_dim]。
                # 这里不产生 skill embedding，只把已有 embedding 投影到 controller 的 hidden_dim。
                return self.op_net(op_embeddings)

            def get_action_logits(self, state_h, op_h, mask=None):
                # 这里是 actor 前向打分的核心。
                # state_h 与 op_h 都已经是 Jittor tensor，所以后续 concat、where、softmax
                # 都保留在 Jittor 计算图里，方便 PPO loss 对 actor 参数反传。
                batch_size = state_h.shape[0]
                num_ops = op_h.shape[1]

                # state_h: [B, H]，一个 batch 中每个样本只有一个当前状态。
                # op_h: [B, num_ops, H]，每个样本有多个候选 MemSkill。
                # 这里把 state_h 扩展到 [B, num_ops, H]，再与每个 op_h_i 拼接，
                # 等价于对每个 (state, skill) pair 单独打分。
                state_expanded = state_h.unsqueeze(1).broadcast((batch_size, num_ops, self.hidden_dim))
                combined = jt.concat([state_expanded, op_h], dim=-1)

                # logits: [B, num_ops]，每个候选 skill 一个分数。
                # 分数越高，该 skill 在当前 state 下被选中的概率越大。
                logits = self.actor_head(combined).squeeze(-1)
                if mask is not None:
                    # mask=0 的位置是 padding 出来的假 skill。
                    # 设成很小的值后，softmax 概率会接近 0，不参与动作选择和 PPO 概率计算。
                    # PyTorch 原版使用 masked_fill(mask == 0, -inf)；
                    # Jittor 这里用 jt.where(mask == 0, neg_inf, logits) 完成同样的掩码逻辑。
                    # 这不是简单隐藏显示，而是从概率路径上让 padding skill 的概率约等于 0。
                    neg_inf = jt.full(logits.shape, -1.0e9)
                    logits = jt.where(mask == 0, neg_inf, logits)
                return logits

            def get_value(self, state_h):
                # critic 只基于 state_h 预测 V(s_t)，不依赖具体选了哪个 skill。
                return self.critic_head(state_h).squeeze(-1)

            def set_new_action_bias_scale(self, bias_scale: float):
                # 这个 scale 由 trainer 控制，用于逐步打开/关闭新 skill 探索偏置。
                # 它不是网络参数，不通过梯度学习。
                self.new_action_bias_scale = float(bias_scale)

            @staticmethod
            def _scalar(value):
                # Jittor 的计算结果是 jt.Var；日志、buffer 记录、Python 控制流需要普通 float。
                # 这里通过 value.numpy() 把单元素 tensor 拿回 CPU，再转成 Python float。
                # 注意：这个函数只应该用于“退出计算图”的地方，例如 loss_info 或 rollout 记录；
                # compute_ppo_loss 返回的 total_loss 不能用 _scalar，否则 optimizer.step(loss) 就无法反传。
                return float(np.asarray(value.numpy()).reshape(-1)[0])

            def _assert_state_dim(self, state_embedding):
                # 迁移时最容易踩坑的是 embedding 维度不一致：
                # 例如 Qwen embedding 可能不是 768，必须和 controller 初始化的 state_dim 对齐。
                # 这里提前断言，避免错误在 Jittor Linear 内部才以更难读的形状错误爆出来。
                actual = int(state_embedding.shape[-1])
                assert actual == self.state_dim, (
                    f"state embedding dim mismatch: expected {self.state_dim}, got {actual}"
                )

            def _assert_op_dim(self, op_embeddings):
                # op_dim 同理，OperationBank 中 skill embedding 的最后一维必须等于 controller.op_dim。
                actual = int(op_embeddings.shape[-1])
                assert actual == self.op_dim, (
                    f"operation embedding dim mismatch: expected {self.op_dim}, got {actual}"
                )

            def _apply_new_action_bias(self, logits, new_op_mask):
                # 新 skill 探索偏置：当 Designer 新增/更新 skill 后，源码希望这些 skill 至少有机会被采样验证。
                # new_op_mask 与 logits 对齐，1 表示新 skill，0 表示旧 skill 或 padding。
                # 这个函数只调整 logits 的采样倾向，不把 bias 当成可学习参数。
                if new_op_mask is None:
                    return logits
                if self.new_action_p_min <= 0.0 or self.new_action_delta_max <= 0.0:
                    return logits
                if self.new_action_bias_scale <= 0.0:
                    return logits
                single = len(logits.shape) == 1
                logits_in = logits.unsqueeze(0) if single else logits
                mask_in = new_op_mask.unsqueeze(0) if single else new_op_mask
                mask_in = mask_in.float32()

                # 用当前 logits 先算一次概率，估计“所有新 skill 的概率质量” p_new。
                # 如果 p_new 小于 new_action_p_min，就给新 skill 的 logit 加一个正偏置。
                probs = nn.softmax(logits_in, dim=-1)
                p_new = (probs * mask_in).sum(dim=-1)
                target = float(self.new_action_p_min)
                has_new = mask_in.sum(dim=-1) > 0
                need_bias = (p_new < target) & has_new

                # jt.where 是 Jittor 中做张量条件选择的方式。
                # need_bias=False 的样本用 1 / 0 等安全值占位，避免无意义的 log 或除法影响数值。
                safe_p = jt.where(need_bias, p_new, jt.ones_like(p_new))
                delta = jt.log(target / (safe_p + 1e-8))
                delta = jt.where(need_bias, delta, jt.zeros_like(delta))
                delta = jt.clamp(delta, min_v=0.0, max_v=float(self.new_action_delta_max))
                delta = delta * float(self.new_action_bias_scale)

                # stop_grad 对应 PyTorch 里的 no_grad / detach 语义：
                # 这个探索偏置用于改变 action preference，但不希望 PPO 反向传播去“学习 delta”。
                # 真正被训练的仍然是 state_net/op_net/actor_head/critic_head 的参数。
                delta = delta.stop_grad()
                out = logits_in + delta.unsqueeze(-1) * mask_in
                return out[0] if single else out

            def _compute_topk_log_prob(self, logits, actions):
                # 计算 Top-K actions 的联合 log probability。
                # PPO 需要比较同一组旧动作在新旧策略下的概率，因此这里不是只算单个 action。
                # logits:  [B, num_ops] 或 [num_ops]
                # actions: [B, K]       或 [K]
                # 返回值:  [B]          或标量 jt.Var
                single_batch = len(logits.shape) == 1
                if single_batch:
                    logits = logits.unsqueeze(0)
                    actions = actions.unsqueeze(0)
                batch_size = logits.shape[0]
                k = actions.shape[1]
                probs = nn.softmax(logits, dim=-1)

                # gather 按 actions 中记录的 skill 下标取出对应概率。
                # 这是 Jittor 里复现 PyTorch Categorical.log_prob / tensor.gather 的关键步骤。
                # selected_probs[b, i] 表示第 b 个样本中第 i 个被选中 skill 的概率。
                selected_probs = probs.gather(dim=-1, index=actions)
                eps = 1e-8

                # Vectorized without-replacement probability:
                # remaining_i = 1 - sum_{j<i} p(a_j).  The previous implementation
                # updated this value in a Python loop over K; prefix sums keep the
                # whole joint-log-probability calculation in Jittor tensor ops.
                # 这里的含义是“无放回 Top-K”联合概率：
                # 第 1 个 skill 按原始概率选；
                # 第 2 个 skill 要在去掉第 1 个之后重新归一化；
                # 第 i 个 skill 的分母是前面已选概率质量之外的 remaining_i。
                # jt.cumsum 把这个递推写成批量张量计算，减少 Python for-loop，
                # 也让 Jittor JIT 更容易把计算融合到图里。
                prefix_selected = jt.cumsum(selected_probs, dim=-1) - selected_probs
                remaining = jt.clamp(1.0 - prefix_selected, min_v=eps)
                safe_selected = jt.clamp(selected_probs, min_v=eps)
                joint = (jt.log(safe_selected) - jt.log(remaining)).sum(dim=-1)
                return joint[0] if single_batch else joint

            def _compute_topk_stats(self, logits, actions):
                # 这些 top-k 指标只用于日志诊断，不直接参与 loss：
                # topk_mass 表示被选中 Top-K skill 的总概率质量；
                # topk_entropy 表示 Top-K 内部概率是否集中；
                # topk_bin_entropy 表示“选中集合 vs 未选中集合”的二元熵。
                if len(logits.shape) == 1:
                    logits = logits.unsqueeze(0)
                if len(actions.shape) == 1:
                    actions = actions.unsqueeze(1)
                probs = nn.softmax(logits, dim=-1)
                selected_probs = probs.gather(dim=-1, index=actions)
                mass = selected_probs.sum(dim=-1)
                eps = 1e-8
                safe_mass = jt.clamp(mass, min_v=eps)
                normalized = jt.clamp(selected_probs / safe_mass.unsqueeze(-1), min_v=eps)
                topk_entropy = -(normalized * jt.log(normalized)).sum(dim=-1)
                m = jt.clamp(mass, min_v=eps, max_v=1.0 - eps)
                topk_bin_entropy = -(m * jt.log(m) + (1 - m) * jt.log(1 - m))
                return topk_entropy, mass, topk_bin_entropy

            def execute(self, state_embedding, op_embeddings, deterministic=False, new_op_mask=None):
                # 兼容旧接口：有些 trainer 代码可能调用 execute，
                # 这里直接转到 select_action，行为保持一致。
                return self.select_action(
                    state_embedding, op_embeddings, deterministic=deterministic, new_op_mask=new_op_mask
                )

            def _sample_action_indices(self, logits, k: int, deterministic: bool):
                # rollout 阶段的动作采样。
                # 注意这里返回的是 action index；真正用于 PPO ratio 的 log_prob
                # 会在 _compute_topk_log_prob 中根据 logits 重新计算。
                if k == 1:
                    if deterministic:
                        return jt.argmax(logits, dim=0)[0].reshape((1,)).int64()
                    probs = nn.softmax(logits, dim=-1)
                    return jt.multinomial(probs, 1).reshape((1,)).int64()
                if deterministic:
                    # deterministic=True 常用于评估：直接取 logit 最大的 K 个 skill。
                    _, indices = jt.topk(logits, k)
                    return indices.int64()
                # Gumbel-Top-K in Jittor.  The sampling operation is used only for
                # rollout action construction; log_prob/loss remain Jittor tensor ops.
                # Gumbel 噪声可以把“按概率随机采样 Top-K”转成 logits + noise 后取 topk。
                # 这比手写循环采样更适合 Jittor 张量化执行。
                u = jt.random(logits.shape)
                gumbel = -jt.log(-jt.log(jt.clamp(u, min_v=1e-8, max_v=1.0 - 1e-8)))
                _, indices = jt.topk(logits + gumbel, k)
                return indices.int64()

            def select_action(self, state_embedding, op_embeddings, deterministic=False, new_op_mask=None):
                # select_action 是 rollout 阶段入口。
                # 它接收 trainer 生成的 numpy state_embedding/op_embeddings，
                # 在这里转换为 Jittor jt.Var 后完成 actor/critic 前向计算。
                # 这正是 PyTorch -> Jittor 迁移的输入边界：
                # 上游 StateEncoder / OperationBank 可能返回 numpy.ndarray，
                # 但 Jittor 网络只能对 jt.Var 建立计算图和执行前向。
                if not hasattr(state_embedding, "shape") or not hasattr(state_embedding, "numpy"):
                    # np.asarray(..., dtype=np.float32) 先统一 dtype，避免 float64 输入导致额外转换。
                    # jt.array(...) 再把 numpy/list 包装成 Jittor tensor。
                    state_embedding = jt.array(np.asarray(state_embedding, dtype=np.float32))
                if not hasattr(op_embeddings, "shape") or not hasattr(op_embeddings, "numpy"):
                    op_embeddings = jt.array(np.asarray(op_embeddings, dtype=np.float32))
                self._assert_state_dim(state_embedding)
                self._assert_op_dim(op_embeddings)

                # 添加 batch 维度:
                #   state_emb: [1, state_dim]
                #   op_embs:   [1, num_ops, op_dim]
                state_emb = state_embedding.unsqueeze(0)
                op_embs = op_embeddings.unsqueeze(0)

                # state_h/op_h 是 Jittor 版 state_net/op_net 的输出。
                state_h = self.encode_state(state_emb)
                op_h = self.encode_ops(op_embs)
                logits = self.get_action_logits(state_h, op_h)[0]
                value = self.get_value(state_h)[0]

                # 如果 new_op_mask 标记了新 skill，按配置给这些 skill 的 logits 加探索 bias。
                if new_op_mask is not None:
                    if not hasattr(new_op_mask, "shape") or not hasattr(new_op_mask, "numpy"):
                        new_op_mask = jt.array(np.asarray(new_op_mask, dtype=np.float32))
                    logits = self._apply_new_action_bias(logits, new_op_mask)
                num_ops = logits.shape[0]
                k = min(self.action_top_k, num_ops)

                # actions_t 仍是 Jittor tensor，log_prob 也在 Jittor 中计算。
                # 随后为了写入 PPOBuffer，需要把 action/log_prob/value 转成 Python int/float。
                # 这些 old_log_prob / old_value 是 rollout 时旧策略留下的“冻结记录”，
                # 后面 PPO 更新会和新策略重新计算的 new_log_prob 做 ratio。
                actions_t = self._sample_action_indices(logits, k, deterministic)
                log_prob = self._compute_topk_log_prob(logits, actions_t)
                actions = [int(x) for x in np.asarray(actions_t.numpy()).reshape(-1).tolist()]
                if k == 1:
                    return actions[0], self._scalar(log_prob), self._scalar(value)
                return actions, self._scalar(log_prob), self._scalar(value)

            def evaluate_actions(self, state_embs, op_embs, actions, op_masks=None, new_op_masks=None):
                # evaluate_actions 是 PPO 更新阶段入口。
                # 它不重新采样动作，而是用当前新策略重新评估 buffer 中旧 actions 的 log_prob。
                # 这一步对应 PPO 中的 pi_theta(a_old | s_old)：
                # action 来自旧 rollout，但概率由“当前参数”的 Jittor Controller 重新计算。
                self._assert_state_dim(state_embs)
                self._assert_op_dim(op_embs)
                state_h = self.encode_state(state_embs)
                op_h = self.encode_ops(op_embs)

                # op_masks 来自 batch padding。
                # 因为一个 batch 内每条 span 的候选 skill 数不同，训练前会补齐到 max_ops。
                # get_action_logits 用 mask 把补出来的位置设成 -1e9，避免进入 softmax 概率。
                logits = self.get_action_logits(state_h, op_h, mask=op_masks)
                values = self.get_value(state_h)
                if new_op_masks is not None:
                    logits = self._apply_new_action_bias(logits, new_op_masks)
                probs = nn.softmax(logits, dim=-1)
                if op_masks is not None:
                    # entropy 是探索程度的诊断/正则项。
                    # padding skill 的概率虽然已经接近 0，但这里再次乘 mask 并重新归一化，
                    # 是为了让 entropy 只统计真实候选 skill，不把 padding 位置算进去。
                    probs_for_entropy = probs * op_masks
                    norm = jt.clamp(probs_for_entropy.sum(dim=-1, keepdims=True), min_v=1e-8)
                    probs_for_entropy = probs_for_entropy / norm
                else:
                    probs_for_entropy = probs
                safe_probs = jt.clamp(probs_for_entropy, min_v=1e-8)
                entropy_terms = -(safe_probs * jt.log(safe_probs))
                if op_masks is not None:
                    entropy_terms = entropy_terms * op_masks
                entropy = entropy_terms.sum(dim=-1)
                if len(actions.shape) == 1:
                    # K=1 时，actions 是 [B]，直接 gather 出每个旧 action 的概率后取 log。
                    selected = probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
                    log_probs = jt.log(jt.clamp(selected, min_v=1e-8))
                else:
                    # Top-K 时 actions 为 [B, K]，需要计算整组 skill 的联合 log probability。
                    # 这里必须和 rollout 时 select_action 记录 old_log_prob 的计算口径一致，
                    # 否则 ratio = exp(new - old) 就没有可比性。
                    log_probs = self._compute_topk_log_prob(logits, actions)
                topk_entropy, topk_mass, topk_bin_entropy = self._compute_topk_stats(logits, actions)
                return log_probs, values, entropy, {
                    "topk_entropy": topk_entropy,
                    "topk_mass": topk_mass,
                    "topk_bin_entropy": topk_bin_entropy,
                }

            def compute_ppo_loss(self, batch: dict[str, Any], returns: np.ndarray, advantages: np.ndarray):
                # compute_ppo_loss 是训练阶段的核心入口。
                # 输入 batch 来自 PPOBuffer.get_batch()，里面保存的是 rollout 时的旧轨迹；
                # returns/advantages 通常由 PPOBuffer.compute_returns_and_advantages() 计算。
                n = len(batch["states"])

                # op_embs 是动态动作空间，不同样本候选 skill 数可能不同。
                # pad_op_embeddings 会把它们 padding 成 [B, max_ops, op_dim]，
                # 同时生成 op_masks/new_masks，防止 padding 位置参与 softmax。
                # 这个 padding/mask 不是 Jittor 独有概念，PyTorch 版也需要；
                # 但 Jittor 迁移时必须把它整理成规则 batch tensor，才能一次性计算 PPO loss。
                # 注意：max_ops 不是在第一个 span 前向执行时预先知道的。
                # rollout 阶段是一个 span 一个 span 地 select_action，每个 span 直接使用自己的
                # [num_ops_this_span, op_dim] 候选 skill，不需要 padding。
                # 只有等当前 PPO batch 的轨迹都收集完，进入反向训练时，才根据这一批
                # batch["op_embs"] 动态计算 max_ops=max(num_ops)，再 padding + mask。
                op_padded, op_masks, new_masks = pad_op_embeddings(
                    batch["op_embs"], batch.get("new_op_masks")
                )

                # 将 PPOBuffer 中保存的 numpy/list 轨迹字段转换成 Jittor jt.Var。
                # 这些变量对应 PyTorch baseline 中的 torch.Tensor，是 Jittor 迁移的核心边界。
                # 从这里开始，policy_loss/value_loss/entropy_loss 都应保持为 Jittor tensor，
                # 直到函数返回 total_loss 给外部 optimizer.step(total_loss)。
                state_embs = jt.array(np.asarray(batch["states"], dtype=np.float32))
                op_embs = jt.array(op_padded)
                self._assert_state_dim(state_embs)
                self._assert_op_dim(op_embs)
                op_masks_t = jt.array(op_masks)
                new_masks_t = jt.array(new_masks) if new_masks is not None else None

                # actions/log_probs/values 是 rollout 时旧策略记录的行为和数值基准。
                actions = jt.array(np.asarray(batch["actions"], dtype=np.int64))
                old_log_probs = jt.array(np.asarray(batch["log_probs"], dtype=np.float32))
                old_values = jt.array(np.asarray(batch["values"], dtype=np.float32))

                # returns/advantages 来自 PPOBuffer.rewards/values/dones 的 GAE 计算。
                # returns_t 训练 critic，advantages_t 决定 policy 更新方向。
                # advantages_t > 0：提高旧动作概率；
                # advantages_t < 0：降低旧动作概率。
                returns_t = jt.array(np.asarray(returns, dtype=np.float32))
                advantages_t = jt.array(np.asarray(advantages, dtype=np.float32))

                # 用当前 Jittor controller 重新评估旧 actions，得到新策略概率和值函数预测。
                new_log_probs, values, entropy, topk_stats = self.evaluate_actions(
                    state_embs, op_embs, actions, op_masks_t, new_masks_t
                )

                # PPO clipped objective:
                # ratio = 新策略选择同一动作组的概率 / 旧策略选择同一动作组的概率。
                # clamp 限制策略更新幅度，避免一次更新把 controller 推得过猛。
                # 这里 old_log_probs 是 rollout 时存入 buffer 的 Python float 再转 jt.Var；
                # new_log_probs 是当前网络参数重新前向得到的 jt.Var。
                # 因此梯度只会流向 new_log_probs 对应的 actor 网络，不会流向 old_log_probs。
                ratio = jt.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_t
                surr2 = jt.clamp(
                    ratio, min_v=1.0 - self.clip_epsilon, max_v=1.0 + self.clip_epsilon
                ) * advantages_t
                policy_loss = -jt.minimum(surr1, surr2).mean()
                if self.vf_clip is not None and self.vf_clip > 0:
                    # value clipping 与 PyTorch PPO 逻辑一致：
                    # 限制 critic 新预测 values 相对 old_values 的变化幅度，
                    # 防止 critic 一次更新过大导致 advantage/return 估计不稳定。
                    value_pred_clipped = old_values + jt.clamp(
                        values - old_values, min_v=-self.vf_clip, max_v=self.vf_clip
                    )
                    value_loss_unclipped = (values - returns_t) ** 2
                    value_loss_clipped = (value_pred_clipped - returns_t) ** 2
                    value_loss = 0.5 * jt.maximum(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((values - returns_t) ** 2).mean()

                # entropy_loss 前面取负号；加入 total_loss 后，最小化 total_loss 会鼓励更高 entropy。
                # 直观上就是避免 actor 太早把概率集中到少数 skill。
                entropy_loss = -entropy.mean()
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # explained_var 是 critic 拟合质量的日志指标，不参与实际优化目标。
                explained_var = 1 - (returns_t - values).var() / (returns_t.var() + 1e-8)
                loss_info = {
                    # loss_info 只用于打印/记录，所以可以安全转成 Python float。
                    # 但 total_loss 必须保持 Jittor tensor 返回，外部 optimizer 才能反向传播。
                    "policy_loss": self._scalar(policy_loss),
                    "value_loss": self._scalar(value_loss),
                    "entropy": self._scalar(entropy.mean()),
                    "topk_entropy": self._scalar(topk_stats["topk_entropy"].mean()),
                    "topk_mass": self._scalar(topk_stats["topk_mass"].mean()),
                    "topk_bin_entropy": self._scalar(topk_stats["topk_bin_entropy"].mean()),
                    "approx_kl": self._scalar((old_log_probs - new_log_probs).mean()),
                    "clip_frac": self._scalar((jt.abs(ratio - 1.0) > self.clip_epsilon).float32().mean()),
                    "explained_variance": self._scalar(explained_var),
                    "value_mean": self._scalar(values.mean()),
                    "return_mean": self._scalar(returns_t.mean()),
                    "advantage_mean": self._scalar(advantages_t.mean()),
                }
                return total_loss, loss_info

        return _Controller()
