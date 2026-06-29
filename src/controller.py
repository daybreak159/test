"""
Controller: Trainable agent that selects operations
Uses PPO (Proximal Policy Optimization) for training with dynamic action space
- Dual-encoder architecture: state_net + op_net with interaction layer
- Actor-Critic: policy head + value head
- On-policy learning with GAE (Generalized Advantage Estimation)
- Clipped surrogate objective for stable updates
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# 记录轨迹的 PPOBuffer，保存 state、action、reward 等信息供 PPO 更新使用。
# 它属于训练数据缓存层，不是神经网络层；PPOController 才是可训练网络主体。
class PPOBuffer:
    """
    Buffer for PPO training
    Stores episode trajectories with states, actions, log_probs, values, rewards

    Supports both single action (K=1) and top-K action selection:
    - actions: List of action indices (single int for K=1, List[int] for K>1)
    - log_probs: Joint log probability of selected action(s)
    """
    def __init__(self):
        # 这里的 buffer 是 PPO 的 on-policy 轨迹缓存，不是 replay buffer。
        # 每一轮 rollout 中 controller 看到一个 state，选择一个或多个 MemSkill，
        # 然后把当时的动作概率和值函数估计一起保存下来，后续 PPO 更新要用
        # "旧策略概率 old_log_prob" 和 "旧 value" 作为对齐基准。
        self.states = []        # state s_t：当前 span/session 与检索到的 memories 融合后的状态向量
        self.op_embs = []       # action space A_t：当前 OperationBank 中所有候选 MemSkill 的 embedding
        self.new_op_masks = []  # 新 skill 探索 mask：标记哪些候选 MemSkill 是 Designer 新增或刚更新的
        self.actions = []       # action a_t：controller 选中的 MemSkill 索引；Top-K 时为 List[int] 按顺序记录选择的一个/多个skill编号
        self.log_probs = []     # old log_prob：旧策略 π_old 在 rollout 时选择该 action/Top-K action 的 log 概率 用于和新概率做比值计算
        self.values = []        # value V(s_t)：critic 对当前状态未来累计 reward 的预测
        self.rewards = []       # reward r_t：当前 step 的过程奖励，episode 结束后会叠加回分配的 QA final reward
        self.dones = []         # done 标记：当前 step 是否是一条 trace/episode 的结束，用于 GAE 边界切断
    # 把每个span经过controller处理的结果写入它的ppo buffer
    def push(self, state_emb: np.ndarray, op_embs: np.ndarray,
             action: Union[int, List[int]], log_prob: float, value: float, reward: float = 0.0,
             new_op_mask: Optional[np.ndarray] = None):
        """
        Store a transition with optional immediate reward (e.g., exploration bonus)

        Args:
            action: Single action index (int) for K=1, or list of action indices for top-K
            log_prob: Joint log probability of selected action(s)
        """
        # state_emb: 当前 span 与检索到的 memories 融合后的状态向量。
        # 它对应 PPO 里的 s_t，是 actor 和 critic 后续共同使用的输入。
        self.states.append(state_emb)

        # op_embs: 当前 operation bank 中所有候选 MemSkill 的向量。
        # 它对应当前 step 的动态动作空间 A_t；不同 step 的候选 skill 数可能不同。
        self.op_embs.append(op_embs)

        if new_op_mask is None:
            new_op_mask = np.zeros(op_embs.shape[0], dtype=np.float32)

        # new_op_mask: 与 op_embs 行一一对应，1 表示该位置是 Designer 新增/更新的 skill。
        # 训练和 rollout 时可用它给新 skill 加探索 bias，避免新 skill 永远没有被验证的机会。
        self.new_op_masks.append(new_op_mask)

        # action: controller 在当前 state 下选中的 skill 下标。
        # 当 action_top_k > 1 时，action 是一个按选择顺序排列的下标列表。
        self.actions.append(action)

        # log_prob: rollout 时旧策略 π_old 选择该 action/Top-K action 的 log 概率。
        # PPO 更新时会与新策略 log_prob 相减，得到 ratio = exp(new - old)。
        self.log_probs.append(log_prob)

        # value: rollout 时 critic 对当前状态 V(s_t) 的估计。
        # 后续 GAE 和 value clipping 都需要这个旧 value 作为基准。
        self.values.append(value)

        # reward: 当前 step 的即时过程奖励，episode 结束后可能再叠加 QA final reward 回分配。
        self.rewards.append(reward)  # Can include shaping rewards like exploration bonus

        # done: 刚 push 进来的普通 step 默认不是 episode 终点。
        # 等整条 trace 结束后，finish_episode 会把最后一个 step 标为 True。
        self.dones.append(False)

    # 将并行采集到的另一个 local_buffer 追加到当前主 buffer。
    # merge 不参与 state_net/op_net/actor/critic 的计算；它只是汇总 rollout 数据。
    # 合并后物理上是一个更长的 step 列表，逻辑上的 episode 边界仍由 dones=True 保留。
    def merge(self, other: 'PPOBuffer'):
        """Merge another buffer into this one (for parallel episode collection)"""
        self.states.extend(other.states)
        self.op_embs.extend(other.op_embs)
        self.new_op_masks.extend(other.new_op_masks)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.values.extend(other.values)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
    # 处理延迟奖励：一整条 trace/episode 结束后，最终 QA/F1 reward 才出现。
    # 该函数将 final_reward 写回当前 episode 的最后一步，或按权重分摊到多个 span。
    def finish_episode(self, final_reward: float, redistribute: bool = True,
                       redistribution_decay: float = 0.9,
                       final_reward_last_ratio: float = 0.6):
        """
        Mark episode as finished and handle final reward.

        Args:
            final_reward: The delayed reward (e.g., QA performance)
            redistribute: If True, spread final reward across all steps in the episode
                         using exponential decay. This helps with credit assignment
                         in long horizon settings.
            redistribution_decay: Decay factor for reward redistribution (0-1).
                                 Higher values give more credit to later steps.
            final_reward_last_ratio: Portion of final_reward added directly to the last
                                     step when redistribute is True. Remainder is
                                     redistributed across all steps.
        """
        # 如果 buffer 中没有任何奖励记录，说明 episode 为空，直接返回。
        if len(self.rewards) == 0:
            return

        # 寻找当前 episode 的起始位置。
        # PPOBuffer 可能包含多个 episode 的数据，通过倒序查找上一个 done=True 的位置来确定。实现了
        episode_start = 0
        for i in range(len(self.dones) - 1, -1, -1):
            if self.dones[i]:
                episode_start = i + 1  # 当前 episode 从上一个 episode 结束后的第一个 step 开始。
                break

        # 计算当前 episode 的长度。
        episode_length = len(self.rewards) - episode_start

        # 如果开启奖励重新分配（redistribute=True）且 episode 长度大于1。
        if redistribute and episode_length > 1:
            # 确保分配给最后一个步骤的奖励比例在 [0, 1] 之间。
            final_reward_last_ratio = max(0.0, min(final_reward_last_ratio, 1.0))
            # 计算直接分配给最后一个步骤的奖励值。
            last_reward = final_reward * final_reward_last_ratio
            # 计算需要被重新分配到整个 episode 的剩余奖励值。
            remaining_reward = final_reward - last_reward

            # 如果还有剩余奖励需要分配。
            if remaining_reward != 0.0:
                # MemSkill 的 QA/F1 分数通常在一整条 trace 处理结束后才得到，
                # 这属于延迟奖励。为了让前面的 span 也能收到学习信号，
                # 这里把一部分 final_reward 按指数衰减分配回 episode 内的每一步。
                # Redistribute remaining reward across all steps with exponential decay
                # Later steps get more credit than earlier steps
                # 计算每个步骤的权重，使用指数衰减，离终点越近的步骤权重越高。
                weights = np.array([redistribution_decay ** (episode_length - 1 - i)
                                   for i in range(episode_length)])
                # 将权重归一化，确保所有权重之和为1。
                weights = weights / weights.sum()

                # 遍历当前 episode 的所有步骤，并根据权重分配奖励。
                for i, w in enumerate(weights):
                    self.rewards[episode_start + i] += remaining_reward * w

            # 将预留的 last_reward 添加到最后一个步骤的奖励上。
            self.rewards[-1] += last_reward
        else:
            # 如果不开启重新分配，或 episode 只有一个步骤，则将所有最终奖励都加到最后一个步骤上。
            # Original behavior: add all final reward to last step
            self.rewards[-1] += final_reward

        # 将最后一个步骤的 done 标记设置为 True，表示这个 episode 到此结束。
        self.dones[-1] = True
    # 基于已经写入 buffer 的 rewards/values/dones 计算 PPO 训练目标。
    # returns 用来训练 critic，advantages 用来决定 actor 策略更新方向。
    def compute_returns_and_advantages(self, gamma: float = 0.99,
                                        gae_lambda: float = 0.95,
                                        last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages
        Correctly handles multiple episodes in the buffer by resetting GAE at episode boundaries.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value estimate for the state after last step (0 if episode ended)
        Returns:
            returns: [N] array of discounted returns
            advantages: [N] array of GAE advantages
        """
        n = len(self.rewards)
        if n == 0:
            return np.array([]), np.array([])

        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute GAE with proper episode boundary handling
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            # GAE 从后往前递推：后一时刻的 value 与 advantage 会影响当前时刻。
            # 遇到 done=True 时 next_non_terminal 变成 0，避免跨 episode 串奖励。
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # Non-terminal mask: 0 if done (episode ended), 1 otherwise
            next_non_terminal = 1.0 - dones[t]

            # TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

            # GAE: reset last_gae when episode ends (next_non_terminal = 0)
            # This prevents advantage from bleeding across episode boundaries
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        # Returns = advantages + values
        returns = advantages + values

        return returns, advantages

    def get_batch(self) -> Dict:
        """
        Get all data as a batch dict

        For top-K actions:
        - If actions are lists (K>1), returns 'actions' as 2D array [N, K]
        - If actions are ints (K=1), returns 'actions' as 1D array [N] for backward compatibility
        """
        # Determine if we have top-K actions (list) or single actions (int)
        if len(self.actions) > 0 and isinstance(self.actions[0], (list, np.ndarray)):
            # Top-K case: convert to 2D array [N, K]
            actions_array = np.array(self.actions)
        else:
            # Single action case: 1D array [N]
            actions_array = np.array(self.actions)

        return {
            # states/op_embs 仍保持 list，是因为 op_embs 的第一维 num_ops 可能随 step 变化。
            # 真正进入 loss 前，compute_ppo_loss 会把 op_embs padding 成统一 batch tensor。
            'states': self.states,
            'op_embs': self.op_embs,
            'new_op_masks': self.new_op_masks,

            # actions 已经在这里统一成 ndarray；Top-K 为 [N, K]，单动作为 [N]。
            'actions': actions_array,

            # log_probs/values 是 rollout 当时旧策略和旧 critic 的数值记录。
            # PPO 不是只看当前网络输出，而要比较 "旧策略行为" 和 "新策略概率"。
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),

            # rewards/dones 用于从后往前计算 Return 和 GAE Advantage。
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones)
        }

    def clear(self):
        """Clear the buffer"""
        # 一个 PPO 更新周期结束后清空缓存，下一轮 rollout 重新收集 on-policy 数据。
        self.states = []
        self.op_embs = []
        self.new_op_masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.states)


class PPOController(nn.Module):
    """
    PPO Controller with Actor-Critic architecture for operation selection
    - Dual-encoder: state_net + op_net for handling dynamic action space
    - Actor (Policy): outputs action logits for each operation
    - Critic (Value): outputs state value V(s)
    - Uses clipped surrogate objective for stable policy updates
    - Supports top-K action selection (action_top_k parameter)
    """
    def __init__(self, state_dim: int = 768, op_dim: int = 768,
                 hidden_dim: int = 256, device: str = 'cuda',
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, vf_clip: float = 0.0,
                 new_action_p_min: float = 0.0, new_action_delta_max: float = 0.0,
                 action_top_k: int = 1):
        super(PPOController, self).__init__()

        self.state_dim = state_dim
        self.op_dim = op_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.vf_clip = vf_clip
        self.new_action_p_min = new_action_p_min
        self.new_action_delta_max = new_action_delta_max
        self.new_action_bias_scale = 0.0
        self.action_top_k = action_top_k  # Number of top actions to select per step

        # State encoder network (shared backbone)
        # 输入 state_embedding 的维度通常是 2 * embedding_dim：
        # [当前 session/span embedding || 检索 memories 融合 embedding]。
        # 这里不是大语言模型，而是一个两层 MLP，把外部文本编码器得到的向量
        # 投影到 controller 自己的 hidden_dim 空间。
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Operation encoder network
        # 每个 MemSkill/operation 也先由文本编码器得到 op_embedding，
        # 再通过 op_net 投影到同一个 hidden_dim 空间。
        # 这样 action space 可以动态变化：新增 skill 只要有文本描述和 embedding，
        # 就可以被同一套 op_net 评分，而不需要改最后一层分类头大小。
        self.op_net = nn.Sequential(
            nn.Linear(op_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: computes action logits from [state_h, op_h] pairs
        # actor_head 不是输出固定类别，而是对每个 (state, candidate_skill) pair 单独打分。
        # 所以候选 skill 数量可以是动态的，输出 logits 形状为 [batch, num_ops]。
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output score for each (state, op) pair
        )

        # Critic head: computes state value from state_h only
        # critic 只看当前 state，不看具体 action，用来估计 V(s)：
        # 在当前记忆状态下继续按当前策略选技能，预期能得到多少最终 reward。
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.to(device)

    def encode_state(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Encode state"""
        # 输入通常是 [B, state_dim]；输出是 [B, hidden_dim]。
        # 这里只做可训练 MLP 投影，不负责文本 embedding。
        return self.state_net(state_embedding)

    def encode_ops(self, op_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode operations"""
        # 输入通常是 [B, num_ops, op_dim]；输出是 [B, num_ops, hidden_dim]。
        # num_ops 可以随样本变化；batch 训练时会先 padding 到 max_ops。
        return self.op_net(op_embeddings)

    def get_action_logits(self, state_h: torch.Tensor, op_h: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute action logits for each operation
        state_h: [B, hidden_dim]
        op_h: [B, num_ops, hidden_dim]
        mask: [B, num_ops] - 1 for valid ops, 0 for padding
        Returns: [B, num_ops] logits
        """
        batch_size = state_h.shape[0]
        num_ops = op_h.shape[1]

        # Expand state to match ops
        # 一个 state 要和所有候选 operation 分别配对打分。
        # state_h: [B, H] -> [B, num_ops, H]，然后与 op_h 拼接。
        state_expanded = state_h.unsqueeze(1).expand(-1, num_ops, -1)  # [B, num_ops, hidden_dim]

        # Concatenate
        combined = torch.cat([state_expanded, op_h], dim=-1)  # [B, num_ops, hidden_dim*2]

        # Compute logits
        # actor_head 对每个 pair 输出一个标量，表示该 skill 在当前 state 下的偏好分数。
        # 这个分数还不是概率，后续会经过 softmax / Categorical / Top-K 采样。
        logits = self.actor_head(combined).squeeze(-1)  # [B, num_ops]

        # Apply mask (set padded positions to large negative value)
        # 训练 batch 中不同样本的候选 skill 数可能不同，需要 padding 到同一长度。
        # mask=0 的位置是补齐出来的假 operation，设为 -inf 后 softmax 概率接近 0。
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        return logits

    def get_value(self, state_h: torch.Tensor) -> torch.Tensor:
        # critic 分支只基于 state_h 估计 V(s)，不拼接 skill。
        # 它评估的是“当前状态整体预期收益”，不是某个具体 action 的分数。
        """
        Compute state value
        state_h: [B, hidden_dim]
        Returns: [B] values
        """
        return self.critic_head(state_h).squeeze(-1)

    def set_new_action_bias_scale(self, bias_scale: float):
        """Set current bias scale for new-action exploration."""
        # trainer 在 Designer 更新 SkillBank 后会临时调大该系数，给新 skill 探索机会。
        # 随训练步数推进，bias_scale 可以逐步衰减回 0。
        self.new_action_bias_scale = float(bias_scale)

    def _apply_new_action_bias(self, logits: torch.Tensor,
                               new_op_mask: torch.Tensor) -> torch.Tensor:
        # new_op_mask 标记 Designer 新增/更新的 skill。
        # 该函数只在有新 skill 且配置开启探索 bias 时改变 logits。
        if new_op_mask is None:
            return logits
        if self.new_action_p_min <= 0.0 or self.new_action_delta_max <= 0.0:
            return logits
        if self.new_action_bias_scale <= 0.0:
            return logits

        if logits.dim() == 1:
            logits_in = logits.unsqueeze(0)
            mask_in = new_op_mask.unsqueeze(0)
        else:
            logits_in = logits
            mask_in = new_op_mask

        mask_in = mask_in.float()

        mask_sum = mask_in.sum(dim=-1)
        has_new = mask_sum > 0
        if not torch.any(has_new):
            return logits

        with torch.no_grad():
            # Designer 新增/修改 skill 后，系统希望 controller 在短期内给这些新 skill
            # 一点探索机会。这里通过提高 new_op_mask 对应位置的 logits 来实现，
            # 但这个 bias 不参与梯度学习，只是 rollout/评估时的探索修正。
            probs = torch.softmax(logits_in, dim=-1)
            p_new = (probs * mask_in).sum(dim=-1)
            target = float(self.new_action_p_min)
            need_bias = has_new & (p_new < target)
            if not torch.any(need_bias):
                return logits
            eps = 1e-8
            safe_p = torch.where(need_bias, p_new, torch.ones_like(p_new))
            delta = torch.log(target / (safe_p + eps))
            delta = torch.where(need_bias, delta, torch.zeros_like(delta))
            delta = torch.clamp(delta, min=0.0, max=float(self.new_action_delta_max))
            delta = delta * float(self.new_action_bias_scale)

        logits_out = logits_in + delta.unsqueeze(-1) * mask_in
        if logits.dim() == 1:
            return logits_out[0]
        return logits_out

    def _compute_topk_log_prob(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute joint log probability for top-K action selection (without replacement).

        For Gumbel-top-k sampling (sampling K items without replacement), the joint
        probability is:
            P(a1, a2, ..., aK) = p(a1) × p(a2)/(1-p(a1)) × p(a3)/(1-p(a1)-p(a2)) × ...

        In log form:
            log P = Σ_i [log p(ai) - log(1 - Σ_{j<i} p(aj))]

        This correctly matches the Gumbel-top-k sampling distribution used in forward().

        Args:
            logits: [B, num_ops] or [num_ops] action logits
            actions: [B, K] or [K] indices of selected actions

        Returns:
            [B] or scalar joint log probabilities
        """
        single_batch = logits.dim() == 1
        if single_batch:
            logits = logits.unsqueeze(0)  # [1, num_ops]
            actions = actions.unsqueeze(0)  # [1, K]

        batch_size = logits.shape[0]
        k = actions.shape[1]

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)  # [B, num_ops]

        # Gather probs for selected actions in order
        # Top-K 是按顺序选择的一组 action，例如 [insert, update, delete]。
        # PPO 中需要这整组动作在旧策略/新策略下的联合 log_prob，而不是单个动作概率。
        selected_probs = torch.gather(probs, dim=-1, index=actions)  # [B, K]

        # Compute log probability for without-replacement sampling
        # log P = Σ_i [log p(ai) - log(1 - Σ_{j<i} p(aj))]
        joint_log_prob = torch.zeros(batch_size, device=logits.device)
        remaining_prob = torch.ones(batch_size, device=logits.device)  # 1 - cumsum of selected probs
        eps = 1e-8

        for i in range(k):
            p_i = torch.clamp(selected_probs[:, i], min=eps)
            # log P(ai | a1,...,ai-1 selected) = log(p_i / remaining_prob)
            # = log(p_i) - log(remaining_prob)
            # without replacement 的含义是：前面已经选中的 skill 不会再被选一次，
            # 所以后续动作的概率要除以 remaining_prob 重新归一化。
            denom = torch.clamp(remaining_prob, min=eps)
            joint_log_prob = joint_log_prob + torch.log(p_i) - torch.log(denom)
            remaining_prob = torch.clamp(remaining_prob - p_i, min=eps)

        if single_batch:
            return joint_log_prob[0]  # scalar
        return joint_log_prob

    def _compute_topk_stats(self, logits: torch.Tensor,
                            actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-K diagnostics from the action probabilities.

        Returns:
            topk_entropy: [B] entropy of normalized top-K probabilities
            topk_mass: [B] total probability mass assigned to top-K actions
            topk_bin_entropy: [B] binary entropy between top-K and tail mass
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        probs = torch.softmax(logits, dim=-1)  # [B, num_ops]
        selected_probs = torch.gather(probs, dim=-1, index=actions)  # [B, K]

        mass_topk = selected_probs.sum(dim=-1)  # [B]
        eps = 1e-8
        safe_mass = mass_topk.clamp(min=eps)
        normalized = (selected_probs / safe_mass.unsqueeze(-1)).clamp(min=eps)
        topk_entropy = -(normalized * torch.log(normalized)).sum(dim=-1)

        m = mass_topk.clamp(min=eps, max=1 - eps)
        topk_bin_entropy = -(m * torch.log(m) + (1 - m) * torch.log(1 - m))

        return topk_entropy, mass_topk, topk_bin_entropy

    def forward(self, state_embedding: torch.Tensor,
                op_embeddings: torch.Tensor,
                deterministic: bool = False,
                new_op_mask: Optional[torch.Tensor] = None
                ) -> Tuple[Union[int, List[int]], float, float]:
        """
        Forward pass with action selection (supports top-K)

        Args:
            state_embedding: [state_dim]
            op_embeddings: [num_ops, op_dim]
            deterministic: If True, always select top actions by logits
            new_op_mask: Optional mask for new action bias

        Returns:
            - action: int (if K=1) or List[int] (if K>1) selected action indices
            - log_prob: float, joint log probability of selected action(s)
            - value: float, state value estimate
        """
        # forward 是 rollout 阶段的动作选择入口。
        # 每次处理一个当前 span/state，所以原始输入没有 batch 维度。
        # Add batch dimension
        state_emb = state_embedding.unsqueeze(0)  # [1, state_dim]
        op_embs = op_embeddings.unsqueeze(0)  # [1, num_ops, op_dim]

        # Encode
        # rollout 阶段只处理当前一个 state，所以先补 batch 维度。
        state_h = self.encode_state(state_emb)  # [1, hidden_dim]
        op_h = self.encode_ops(op_embs)  # [1, num_ops, hidden_dim]

        # Get action logits and value
        logits = self.get_action_logits(state_h, op_h)  # [1, num_ops]
        value = self.get_value(state_h)  # [1]

        logits = logits[0]  # [num_ops]
        value = value[0]  # scalar
        if new_op_mask is not None:
            if not torch.is_tensor(new_op_mask):
                new_op_mask = torch.tensor(new_op_mask, dtype=torch.float32, device=logits.device)
            else:
                new_op_mask = new_op_mask.to(logits.device)
            logits = self._apply_new_action_bias(logits, new_op_mask)

        num_ops = logits.shape[0]
        k = min(self.action_top_k, num_ops)  # Can't select more than available ops

        if k == 1:
            # action_top_k=1 时退化成普通离散动作选择。
            # Categorical 会基于 logits 内部做 softmax，不需要手动先算概率。
            # Original single-action behavior
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits).item()
            else:
                action = dist.sample().item()

            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            return action, log_prob, value.item()
        else:
            # Top-K action selection
            if deterministic:
                # Select top-K by logits
                _, top_k_indices = torch.topk(logits, k)
                actions = top_k_indices.tolist()
            else:
                # 训练 rollout 用 Gumbel-Top-K 在不放回条件下采样 K 个 skill。
                # 它比直接取 top-k 更有探索性，也能给 PPO 计算一组动作的联合概率。
                # Sample K actions without replacement using Gumbel-top-k trick
                # This provides proper sampling from the categorical distribution
                gumbel = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)
                perturbed = logits + gumbel
                _, top_k_indices = torch.topk(perturbed, k)
                actions = top_k_indices.tolist()

            # Compute joint log probability
            actions_tensor = torch.tensor(actions, device=self.device)
            log_prob = self._compute_topk_log_prob(logits, actions_tensor).item()

            return actions, log_prob, value.item()

    def evaluate_actions(self, state_embs: torch.Tensor, op_embs: torch.Tensor,
                         actions: torch.Tensor, op_masks: Optional[torch.Tensor] = None,
                         new_op_masks: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate actions for PPO update (supports top-K actions)

        Args:
            state_embs: [B, state_dim]
            op_embs: [B, max_ops, op_dim]
            actions: [B] for single action or [B, K] for top-K actions
            op_masks: [B, max_ops] - 1 for valid, 0 for padding

        Returns:
            log_probs: [B] log probabilities of taken action(s)
            values: [B] state values
            entropy: [B] policy entropy
            topk_stats: dict with top-K diagnostics (entropy/mass/bin-entropy)
        """
        # evaluate_actions 是 PPO 更新入口，不负责采样新 action。
        # 它重新评估 buffer 中旧 action 在当前参数下的 log_prob/value/entropy。
        # Encode
        state_h = self.encode_state(state_embs)  # [B, hidden_dim]
        op_h = self.encode_ops(op_embs)  # [B, max_ops, hidden_dim]

        # Get logits and values
        logits = self.get_action_logits(state_h, op_h, mask=op_masks)  # [B, max_ops]
        values = self.get_value(state_h)  # [B]
        if new_op_masks is not None:
            logits = self._apply_new_action_bias(logits, new_op_masks)

        # Compute log probs and entropy
        # PPO 更新阶段不是重新采样动作，而是评估 rollout 时已经发生的 actions
        # 在当前新策略下的 log_prob，并与 buffer 中保存的 old_log_prob 做比值。
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()

        # Handle single action [B] vs top-K actions [B, K]
        if actions.dim() == 1:
            # Single action case (K=1)
            log_probs = dist.log_prob(actions)
        else:
            # Top-K case: compute joint log probability
            log_probs = self._compute_topk_log_prob(logits, actions)

        with torch.no_grad():
            topk_entropy, topk_mass, topk_bin_entropy = self._compute_topk_stats(
                logits.detach(), actions
            )
        topk_stats = {
            'topk_entropy': topk_entropy,
            'topk_mass': topk_mass,
            'topk_bin_entropy': topk_bin_entropy,
        }

        return log_probs, values, entropy, topk_stats
    def compute_ppo_loss(self, batch: Dict, returns: np.ndarray,
                         advantages: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss with value function clipping
        Args:
            batch: dict with states, op_embs, actions, log_probs, values (old)
            returns: [N] array of returns
            advantages: [N] array of advantages
        Returns:
            total_loss: combined loss
            loss_info: dict with individual loss components
        """
        # batch 中每一项对应一个 rollout step；n 是本次 minibatch 的 step 数。
        n = len(batch['states'])

        # Handle variable number of operations (dynamic action space)
        op_dim = batch['op_embs'][0].shape[1]
        max_ops = max(op.shape[0] for op in batch['op_embs'])

        # Pad op_embs and create masks
        # 动态 action space 的 batch 化处理：每个样本候选 operation 数量可能不同，
        # 因此 padding 到 max_ops，并用 op_masks 标记哪些位置是真实 operation。
        op_embs_padded = np.zeros((n, max_ops, op_dim), dtype=np.float32)
        op_masks = np.zeros((n, max_ops), dtype=np.float32)
        new_op_masks_padded = None
        if 'new_op_masks' in batch:
            new_op_masks_padded = np.zeros((n, max_ops), dtype=np.float32)

        for i, op in enumerate(batch['op_embs']):
            n_ops = op.shape[0]
            op_embs_padded[i, :n_ops] = op
            op_masks[i, :n_ops] = 1.0
            if new_op_masks_padded is not None:
                new_mask = batch['new_op_masks'][i]
                if new_mask is not None:
                    new_op_masks_padded[i, :n_ops] = new_mask

        # Convert to tensors
        # states: [N, state_dim]，来自 PPOBuffer.states，是 controller.state_net 的输入。
        state_embs = torch.FloatTensor(np.array(batch['states'])).to(self.device)

        # op_embs: [N, max_ops, op_dim]，来自 PPOBuffer.op_embs，padding 后进入 op_net。
        # op_masks: [N, max_ops]，标记哪些位置是真实 skill，哪些只是 padding。
        op_embs = torch.FloatTensor(op_embs_padded).to(self.device)
        op_masks = torch.FloatTensor(op_masks).to(self.device)
        new_op_masks = None
        if new_op_masks_padded is not None:
            # new_op_masks: [N, max_ops]，标记新 skill 的位置，用于 _apply_new_action_bias。
            new_op_masks = torch.FloatTensor(new_op_masks_padded).to(self.device)

        # actions: rollout 时真正执行过的 action/Top-K actions。
        # PPO 更新不会重新采样，而是评估这些旧 actions 在当前新策略下的概率。
        actions = torch.LongTensor(batch['actions']).to(self.device)

        # old_log_probs / old_values: rollout 时旧策略和旧 critic 的输出。
        # 它们分别用于 PPO ratio 和 value clipping/GAE 对齐。
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        old_values = torch.FloatTensor(batch['values']).to(self.device)

        # returns_t / advantages_t: 由 rewards、values、dones 经过 GAE 计算得到。
        # advantages_t 决定 policy 方向，returns_t 决定 critic 拟合目标。
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # NOTE: Advantages should already be normalized at the full-batch level
        # in trainer.py before minibatch splitting. Do NOT normalize again here,
        # as per-minibatch normalization introduces inconsistent scales.

        # Evaluate actions
        new_log_probs, values, entropy, topk_stats = self.evaluate_actions(
            state_embs, op_embs, actions, op_masks, new_op_masks
        )

        # Policy loss (clipped surrogate objective)
        # ratio = 新策略选择同一动作组的概率 / 旧策略选择同一动作组的概率。
        # PPO clipping 限制 ratio 不要偏离 1 太多，避免一次更新把策略推得过猛。
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (optional clipping)
        # value_loss 训练 critic_head，使 V(s) 接近 return。
        # 如果开启 vf_clip，也会像 policy 一样限制 value 更新幅度，提升稳定性。
        if self.vf_clip is not None and self.vf_clip > 0:
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.vf_clip, self.vf_clip
            )
            value_loss_unclipped = (values - returns_t) ** 2
            value_loss_clipped = (value_pred_clipped - returns_t) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((values - returns_t) ** 2).mean()

        # Entropy bonus (encourage exploration)
        # entropy_loss 前面有负号，加入 total_loss 时会鼓励策略分布保持一定熵，
        # 防止 controller 过早固定选择少数几个 skill。
        entropy_loss = -entropy.mean()

        # Total loss
        # 最终优化目标 = policy 改进 + value 预测 + exploration bonus。
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # explained_variance 是诊断指标，用来粗略观察 critic 的拟合质量。
        # 它不参与梯度，只写入日志。
        # Compute explained variance for value function quality
        with torch.no_grad():
            explained_var = 1 - (returns_t - values).var() / (returns_t.var() + 1e-8)

        loss_info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'topk_entropy': topk_stats['topk_entropy'].mean().item(),
            'topk_mass': topk_stats['topk_mass'].mean().item(),
            'topk_bin_entropy': topk_stats['topk_bin_entropy'].mean().item(),
            'approx_kl': ((old_log_probs - new_log_probs).mean()).item(),
            'clip_frac': ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item(),
            'explained_variance': explained_var.item(),
            'value_mean': values.mean().item(),
            'return_mean': returns_t.mean().item(),
            'advantage_mean': advantages_t.mean().item(),
        }

        return total_loss, loss_info


# Backward compatibility alias
Controller = PPOController


class BaseTextEncoder:
    # 文本编码器包装层：负责把 span/memory/skill description 转成 dense embedding。
    # 该部分通常是冻结的外部模型推理，不属于 PPOController 的可训练 MLP 路径。
    """
    Base text encoder that holds the shared state encoder backbone.
    This class can be shared between StateEncoder and OpEncoder to avoid
    loading multiple copies of the same model.
    """
    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = 'cuda',
                 encode_batch_size: int = 64, use_flash_attn: bool = True):
        # BaseTextEncoder 负责把自然语言文本转成 dense embedding。
        # 注意：这里可能使用 SentenceTransformer / HuggingFace AutoModel，
        # 它们属于外部文本编码器，不是我们迁移到 Jittor 的 PPOController 网络。
        # 训练时 controller 接收到的 state_embedding/op_embedding 已经是 numpy 向量。
        self.model_name = model_name
        self.device = device
        self.encode_batch_size = encode_batch_size
        self.use_flash_attn = bool(use_flash_attn)
        model_name_lower = model_name.lower()
        self._use_qwen_embedding = "qwen3-embedding" in model_name_lower
        self._use_sentence_transformer = (
            model_name.startswith("sentence-transformers/")
            or self._use_qwen_embedding
        )

        # 根据 model_name 选择文本编码后端：
        # 1. sentence-transformers/... 或 qwen3-embedding -> SentenceTransformer.encode
        # 2. 其他 HuggingFace 模型 -> AutoTokenizer + AutoModel + mean pooling
        if self._use_sentence_transformer:
            if self._use_qwen_embedding:
                # Qwen3-Embedding 这类模型可能需要特殊 dtype / flash attention 设置。
                # CPU 或不支持 bf16 的环境下会回退到 float16。
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
                model_kwargs = {"torch_dtype": torch_dtype}
                if self.use_flash_attn:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                tokenizer_kwargs = {"padding_side": "left"}
                if str(device).startswith("cuda"):
                    self.model = SentenceTransformer(
                        model_name,
                        device=device,
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs=tokenizer_kwargs,
                    )
                else:
                    model_kwargs["device_map"] = "auto"
                    self.model = SentenceTransformer(
                        model_name,
                        device=device,
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs=tokenizer_kwargs,
                    )
            else:
                self.model = SentenceTransformer(model_name, device=device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            self.tokenizer = None
        else:
            # 非 SentenceTransformer 路径使用 HuggingFace AutoTokenizer + AutoModel，
            # 后续通过 mean pooling 把 token-level hidden states 聚合成句向量。
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embedding from token embeddings"""
        # AutoModel 输出的是每个 token 的 hidden state。
        # mean pooling 用 attention_mask 去掉 padding token，再对有效 token 求平均，
        # 得到一个固定长度的文本向量。
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode text(s) using the underlying model with batch processing.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Override default batch size (default: self.encode_batch_size)

        Returns:
            numpy array of embeddings
        """
        # 统一成 list，便于批量编码；单条文本最终也会返回一个 embedding 矩阵/向量。
        if isinstance(texts, str):
            texts = [texts]

        # 空输入返回 [0, embedding_dim]，避免上游空 memory 列表导致编码器报错。
        if len(texts) == 0:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        if batch_size is None:
            batch_size = self.encode_batch_size

        if self._use_sentence_transformer:
            # SentenceTransformer 直接返回 sentence embeddings，格式是 numpy。
            return self.model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
        else:
            # Process in batches to avoid OOM
            # AutoModel 路径需要自己 tokenize、前向推理、mean pooling。
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                with torch.no_grad():
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=4096,
                        return_tensors='pt'
                    )
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    outputs = self.model(**encoded)
                    embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
                    all_embeddings.append(embeddings.cpu().numpy())
            return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]


class StateEncoder:
    """
    Encodes session + retrieved memories into state embedding.
    Supports HuggingFace encoder backbones and SentenceTransformer models.

    Can optionally use a shared BaseTextEncoder to avoid loading multiple model copies.
    """
    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = 'cuda',
                 fusion_mode: str = "mean", fusion_tau: float = 1.0,
                 base_encoder: Optional['BaseTextEncoder'] = None,
                 encode_batch_size: int = 64, use_flash_attn: bool = True):
        # StateEncoder 把当前 span/session 和检索到的 memories 编成 controller 的 state。
        # 输出维度是 2 * embedding_dim，因为最终会拼接：
        # [session_emb || memory_emb]。
        self.fusion_mode = fusion_mode
        self.fusion_tau = fusion_tau

        if base_encoder is not None:
            # Use shared encoder
            self._base_encoder = base_encoder
            self.model_name = base_encoder.model_name
            self.device = base_encoder.device
        else:
            # Create own encoder
            self._base_encoder = BaseTextEncoder(
                model_name=model_name,
                device=device,
                encode_batch_size=encode_batch_size,
                use_flash_attn=use_flash_attn
            )
            self.model_name = model_name
            self.device = device

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the encoder"""
        return self._base_encoder.embedding_dim

    def _encode_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) using the underlying model"""
        return self._base_encoder.encode(texts)

    def _fuse_memory_embeddings(self, session_emb: np.ndarray, memory_embs: np.ndarray,
                                fusion_mode: str, fusion_tau: float) -> np.ndarray:
        # 多条 retrieved memories 需要融合成一个 memory_emb。
        # 默认 mean 表示简单平均；sim_weighted 表示按与当前 session 的相似度加权。
        mem_arr = np.asarray(memory_embs)
        if mem_arr.ndim == 1:
            mem_arr = mem_arr.reshape(1, -1)

        if fusion_mode == "mean":
            # 默认融合方式：把多条 retrieved memory 的 embedding 简单平均成一个 memory_emb。
            return np.mean(mem_arr, axis=0)
        if fusion_mode == "sim_weighted":
            # sim_weighted 会先做 L2 归一化，再算 cosine similarity，
            # tau 是温度参数：越小越偏向最相似的 memory，越大越平均。
            sess = session_emb.astype(np.float32)
            mem = mem_arr.astype(np.float32)
            sess_norm = sess / (np.linalg.norm(sess) + 1e-8)
            mem_norms = np.linalg.norm(mem, axis=1, keepdims=True)
            mem_norm = mem / (mem_norms + 1e-8)
            sims = np.dot(mem_norm, sess_norm)
            tau = max(float(fusion_tau), 1e-8)
            sims = sims / tau
            sims = sims - np.max(sims)
            weights = np.exp(sims).astype(np.float32)
            weights = weights / (weights.sum() + 1e-8)
            return np.sum(mem * weights[:, None], axis=0)

        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def encode(self, session_text: str, retrieved_memories: List[str],
               session_embedding: Optional[np.ndarray] = None,
               memory_embeddings: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
               fusion_mode: Optional[str] = None,
               fusion_tau: Optional[float] = None) -> np.ndarray:
        """
        Encode state from session and retrieved memories.
        Strategy: Encode session and memories separately, then concatenate.
        If precomputed embeddings are provided, reuse them to skip redundant encoding.
        fusion_mode controls how memory embeddings are combined.

        Returns: state embedding vector [session_emb || memory_emb]
        """
        if fusion_mode is None:
            fusion_mode = self.fusion_mode
        if fusion_tau is None:
            fusion_tau = self.fusion_tau

        # Encode current session if not provided
        # trainer 中为了减少重复计算，可能会传入预先算好的 session_embedding。
        # 如果没有传入，这里才调用文本编码器。
        if session_embedding is None:
            session_emb = self._encode_texts(session_text)
            if session_emb.ndim == 2:
                session_emb = session_emb[0]
        else:
            session_emb = session_embedding
            if session_emb.ndim == 2:
                session_emb = session_emb[0]

        memory_emb = None
        mem_arr = None
        if memory_embeddings is not None:
            if isinstance(memory_embeddings, list):
                if len(memory_embeddings) == 0:
                    memory_embeddings = None
                else:
                    memory_embeddings = np.vstack(memory_embeddings)

            if memory_embeddings is not None:
                mem_arr = np.asarray(memory_embeddings)
                if mem_arr.size == 0:
                    mem_arr = None

        if mem_arr is None:
            if len(retrieved_memories) == 0:
                # No memories retrieved, use zero vector for memory part
                # 如果当前没有检索到任何记忆，memory 部分用零向量占位，
                # 这样 state_dim 仍然固定为 2 * embedding_dim。
                embedding_dim = session_emb.shape[0]
                memory_emb = np.zeros(embedding_dim, dtype=np.float32)
            else:
                # Encode each retrieved memory separately and average
                # 如果没有预计算 memory embeddings，则对每条 memory 分别编码。
                mem_arr = self._encode_texts(retrieved_memories)

        if memory_emb is None:
            memory_emb = self._fuse_memory_embeddings(
                session_emb=session_emb,
                memory_embs=mem_arr,
                fusion_mode=fusion_mode,
                fusion_tau=fusion_tau
            )

        # Concatenate session and memory embeddings
        # 这是 controller.py 中 state_dim 翻倍的来源：
        # 当前文本表示 + 相关历史记忆表示。
        state_emb = np.concatenate([session_emb, memory_emb], axis=0)

        return state_emb


class OpEncoder:
    """
    Encodes operation descriptions into embeddings.
    Supports HuggingFace encoder backbones and SentenceTransformer models.

    Can optionally use a shared BaseTextEncoder to avoid loading multiple model copies.
    """
    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = 'cuda',
                 base_encoder: Optional['BaseTextEncoder'] = None,
                 encode_batch_size: int = 64, use_flash_attn: bool = True):
        # OpEncoder 把每个 MemSkill 的 short description / operation description
        # 编成 op_embedding。controller 后续会对每个 op_embedding 单独打分。
        if base_encoder is not None:
            # Use shared encoder
            self._base_encoder = base_encoder
            self.model_name = base_encoder.model_name
            self.device = base_encoder.device
        else:
            # Create own encoder
            self._base_encoder = BaseTextEncoder(
                model_name=model_name,
                device=device,
                encode_batch_size=encode_batch_size,
                use_flash_attn=use_flash_attn
            )
            self.model_name = model_name
            self.device = device

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the encoder"""
        return self._base_encoder.embedding_dim

    def encode(self, op_descriptions: List[str]) -> np.ndarray:
        """
        Encode operation descriptions
        Returns: [num_ops, op_dim]
        """
        # 一次性编码当前 operation bank 的所有候选 skill，
        # 输出矩阵每一行对应一个候选 MemSkill。
        return self._base_encoder.encode(op_descriptions)

    def encode_single(self, op_description: str) -> np.ndarray:
        """Encode single operation"""
        # Designer 新增或修改单个 skill 时，可用这个函数更新该 skill 的 embedding。
        emb = self._base_encoder.encode(op_description)
        if emb.ndim == 2:
            return emb[0]
        return emb
