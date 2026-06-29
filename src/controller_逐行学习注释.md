# controller.py 逐行学习注释

> 对应源码：`E:\一堆报告\新芽计划\MemSkill\src\controller.py`  
> 目的：按源码顺序理解 Controller 的完整执行思路。本文是学习版注释，不替代源码；源码中也已补充关键中文注释。

## 0. 文件整体定位

`controller.py` 实际包含三层内容：

1. `PPOBuffer`：PPO 训练数据缓存，保存 rollout 过程中每个 span 的状态、动作、旧概率、value、reward 和 done。
2. `PPOController`：真正的可训练策略网络，包含 `state_net`、`op_net`、`actor_head`、`critic_head`、Top-K action、PPO loss。
3. `BaseTextEncoder / StateEncoder / OpEncoder`：文本向量化模块，把 span、retrieved memories、skill description 转成数值 embedding。

源码中 `SentenceTransformer / AutoModel / AutoTokenizer` 只属于第 3 层文本编码，不属于 PPOController 的训练核心。

---

## 1. import 部分

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
```

含义：

- `torch`：原始 PyTorch 版 Controller 的张量计算、自动求导和分布采样。
- `torch.nn as nn`：搭建 MLP 网络，例如 `Linear`、`ReLU`、`Sequential`。
- `numpy`：缓存 rollout 数据、padding 变长 `op_embs`、计算 GAE。
- `typing`：给函数参数和返回值加类型提示。
- `SentenceTransformer`：当 embedding model 是 sentence-transformers 或 qwen3-embedding 路径时，用 `.encode()` 得到句向量。
- `AutoTokenizer / AutoModel`：当 embedding model 是普通 HuggingFace 模型时，用 tokenizer + transformer + mean pooling 得到句向量。

关键边界：

```text
文本编码器输出 embedding
→ PPOController 接收 embedding 做策略训练
```

---

## 2. PPOBuffer：PPO 轨迹缓存

### 2.1 `__init__`

`PPOBuffer` 不是神经网络，它只是保存训练数据。每个列表长度对应 rollout 中的 step 数。一个 step 通常对应一个 span 的 Controller 决策。

字段含义：

- `states`：当前 span/session 与 retrieved memories 融合后的状态向量，即 PPO 里的 `s_t`。
- `op_embs`：当前 step 的候选 MemSkill embeddings，即动态 action space。
- `new_op_masks`：标记哪些候选 skill 是 Designer 新增或刚更新的 skill。
- `actions`：Controller 选中的 skill index；Top-K 时是一个 index 列表。
- `log_probs`：rollout 时旧策略选择该 action/Top-K action 的 log probability。
- `values`：rollout 时 critic 对当前状态的 value 估计。
- `rewards`：当前 step 的 reward，可能先是过程奖励，episode 结束后再叠加 final reward。
- `dones`：episode 边界标记；`True` 表示当前 step 是某条 trace/episode 的结束。

### 2.2 `push(...)`

`push` 在 rollout 阶段调用。每处理一个 span，Controller 选择 action 后，就把这一步的训练证据写入 buffer。

输入：

```text
state_emb       当前 span 的状态向量
op_embs         当前候选 skills 的向量矩阵
action          选中的 skill index 或 Top-K indices
log_prob        旧策略选择该 action 的 log probability
value           critic 对当前 state 的 value 估计
reward          当前 step 的即时奖励，默认 0
new_op_mask     新 skill 探索 mask
```

执行逻辑：

1. 把 `state_emb` append 到 `states`。
2. 把 `op_embs` append 到 `op_embs`。
3. 如果没有传 `new_op_mask`，就创建全 0 mask。
4. 记录 action、log_prob、value、reward。
5. 默认 `done=False`，等整条 episode 结束时由 `finish_episode` 改最后一步。

### 2.3 `merge(other)`

`merge` 用在并行 episode 收集之后。它不是网络计算，也不是 PPO loss，它只是把另一个 `PPOBuffer` 的列表追加到当前 buffer。

典型流程：

```text
ThreadPoolExecutor 并行运行多个 _run_single_episode
每个 episode 生成一个 local_buffer
主线程调用 ppo_buffer.merge(local_buffer)
合并成一个大 batch
```

合并后：

```text
物理上：一个更长的 step 列表
逻辑上：仍然是多条 episode
边界：由 dones=True 保留
```

如果有 n 个 buffer，每个 x 个 step，合并后 step 数约为 `n * x`；如果 episode 长度不同，就是 `x1 + x2 + ... + xn`。

### 2.4 `finish_episode(...)`

MemSkill 的最终奖励往往在整条 trace 处理完后才得到，比如 QA F1、LLM judge 或 success rate。这个 final reward 是延迟奖励。

该函数做两件事：

1. 把 final reward 写回当前 episode。
2. 把最后一个 step 的 `done` 设为 `True`。

如果 `redistribute=True` 且 episode 长度大于 1：

- 一部分 reward 直接给最后一步。
- 剩余 reward 按指数衰减权重分配给当前 episode 的每一步。
- 越靠后的 step 权重越高。

这样前面的 span 也能收到训练信号。

### 2.5 `compute_returns_and_advantages(...)`

该函数从后往前计算 PPO 需要的：

- `returns`：用于训练 critic 的累计收益目标。
- `advantages`：实际收益相对 critic 预测的超额部分，用于 actor policy update。

核心公式：

```text
delta = reward_t + gamma * next_value * next_non_terminal - value_t
advantage_t = delta + gamma * lambda * next_non_terminal * advantage_{t+1}
return_t = advantage_t + value_t
```

`next_non_terminal = 1 - done_t`。如果遇到 `done=True`，GAE 会在 episode 边界切断，避免不同 trace 的奖励串起来。

### 2.6 `get_batch()`

把 buffer 中的 list 打包成 dict，供 `compute_ppo_loss` 使用。

注意：

- `states` 和 `op_embs` 仍保留 list。
- 因为不同 step 的 `op_embs` 第一维 `num_ops` 可能不同。
- 真正 padding 成 `[B, max_ops, op_dim]` 是在 `compute_ppo_loss` 里做。

### 2.7 `clear()` 和 `__len__`

- `clear()`：清空当前 buffer，下一轮 rollout 重新收集 on-policy 数据。
- `__len__()`：返回当前 buffer 中 step 数，即 `len(self.states)`。

---

## 3. PPOController：可训练策略网络

`PPOController` 是这个文件的核心神经网络。它采用 actor-critic 结构，并支持动态 action space 和 Top-K action。

### 3.1 `__init__` 参数

关键参数：

- `state_dim`：输入 state embedding 维度。
- `op_dim`：输入 skill/operation embedding 维度。
- `hidden_dim`：Controller 内部 MLP 隐藏维度。
- `gamma`：reward 折扣因子。
- `gae_lambda`：GAE 平滑参数。
- `clip_epsilon`：PPO ratio clip 范围。
- `entropy_coef`：entropy bonus 权重。
- `value_coef`：value loss 权重。
- `vf_clip`：value function clipping 参数。
- `new_action_p_min / new_action_delta_max`：新 skill 探索 bias 相关参数。
- `action_top_k`：每一步选择多少个 skill。

### 3.2 `state_net`

```text
state_embedding [B, state_dim]
→ state_net
→ state_h [B, hidden_dim]
```

作用：把当前 span + retrieved memories 融合后的 state embedding 投影到 Controller 的 hidden space。

### 3.3 `op_net`

```text
op_embeddings [B, num_ops, op_dim]
→ op_net
→ op_h [B, num_ops, hidden_dim]
```

作用：把每个候选 MemSkill 的文本 embedding 投影到和 state 一致的 hidden space。

### 3.4 `actor_head`

Actor 不输出固定类别，而是对每个 `(state, skill)` pair 单独打分。

流程：

```text
state_h: [B, hidden_dim]
op_h:    [B, num_ops, hidden_dim]
state_h 扩展为 [B, num_ops, hidden_dim]
concat([state_h, op_h]) -> [B, num_ops, hidden_dim * 2]
actor_head -> logits [B, num_ops]
```

每个 logit 表示当前 state 下某个 skill 的偏好分数。

### 3.5 `critic_head`

```text
state_h [B, hidden_dim]
→ critic_head
→ value [B]
```

critic 只看 state，不看具体 skill。它估计当前状态继续执行后的预期 reward，即 `V(s)`。

### 3.6 `encode_state` / `encode_ops`

这两个函数只是对 `state_net` 和 `op_net` 的简单封装：

- `encode_state`：`state_embedding -> state_h`
- `encode_ops`：`op_embeddings -> op_h`

注意：这里的 “encode” 不是文本编码。文本编码已经由 `StateEncoder / OpEncoder` 完成，这里是可训练 MLP 投影。

### 3.7 `get_action_logits(state_h, op_h, mask=None)`

该函数负责 actor 打分。

输入：

- `state_h`: `[B, hidden_dim]`
- `op_h`: `[B, num_ops, hidden_dim]`
- `mask`: `[B, num_ops]`，真实 skill 为 1，padding skill 为 0。

输出：

- `logits`: `[B, num_ops]`

如果传入 mask，padding 位置会被设成 `-inf`，softmax 后概率接近 0，不会被选中，也不会影响 PPO 概率计算。

### 3.8 `get_value(state_h)`

调用 critic_head 得到 `V(s)`。

### 3.9 `set_new_action_bias_scale(...)` 和 `_apply_new_action_bias(...)`

Designer 新增或修改 skill 后，如果完全按原策略采样，新 skill 可能一直没机会被选中。这个 bias 机制给新 skill 临时提高 logits，让它获得探索机会。

关键点：

- 只在 `new_op_mask` 标记的位置加 bias。
- bias 由 `new_action_p_min`、`new_action_delta_max`、`new_action_bias_scale` 控制。
- PyTorch 版用 `torch.no_grad()`，说明它是探索修正，不作为可学习路径参与梯度。

### 3.10 `_compute_topk_log_prob(logits, actions)`

MemSkill 每一步可以选择 Top-K skills，不是单个 action。因此 PPO 需要整组 action 的联合 log probability。

Top-K 是不放回选择，前面选过的 skill 后面不能再选。

公式含义：

```text
P(a1, a2, ..., aK)
= p(a1) * p(a2)/(1-p(a1)) * p(a3)/(1-p(a1)-p(a2)) * ...
```

log 形式：

```text
log_prob = Σ_i [log p(ai) - log remaining_prob]
```

这个 `joint_log_prob` 会在 PPO ratio 中使用：

```text
ratio = exp(new_joint_log_prob - old_joint_log_prob)
```

### 3.11 `_compute_topk_stats(...)`

这是诊断函数，不是核心训练目标。它统计：

- `topk_entropy`：选中的 Top-K 概率内部的熵。
- `topk_mass`：Top-K actions 占据的总概率质量。
- `topk_bin_entropy`：Top-K mass 与剩余 tail mass 的二元熵。

这些指标主要用于日志和训练观察。

### 3.12 `forward(...)`

`forward` 是 rollout 阶段的动作选择入口。

输入：

```text
state_embedding: [state_dim]
op_embeddings: [num_ops, op_dim]
deterministic: 是否直接选 logits 最大的 skill
new_op_mask: 新 skill 探索 mask
```

执行流程：

1. 给 state 和 ops 加 batch 维度。
2. 用 `state_net` 和 `op_net` 得到 `state_h`、`op_h`。
3. 用 `actor_head` 得到 logits。
4. 用 `critic_head` 得到 value。
5. 如果有新 skill bias，则修正 logits。
6. 若 `action_top_k == 1`：用 Categorical 采样或 argmax。
7. 若 `action_top_k > 1`：用 deterministic top-k 或 Gumbel-Top-K 采样。
8. 计算 action/Top-K action 的 log_prob。
9. 返回 action、log_prob、value。

### 3.13 `evaluate_actions(...)`

这是 PPO 更新阶段的动作评估入口。它不重新采样动作，而是重新评估 buffer 中旧 action 在当前参数下的概率。

输入：

```text
state_embs: [B, state_dim]
op_embs: [B, max_ops, op_dim]
actions: [B] 或 [B, K]
op_masks: [B, max_ops]
new_op_masks: [B, max_ops]
```

输出：

```text
new_log_probs: 当前新策略对旧 actions 的 log_prob
values: 当前 critic 的 value 预测
entropy: 当前策略分布熵
topk_stats: Top-K 诊断指标
```

PPO 更新需要比较：

```text
old_log_probs: rollout 时保存
new_log_probs: evaluate_actions 重新计算
```

### 3.14 `compute_ppo_loss(batch, returns, advantages)`

这是 PPOController 的训练核心。

执行流程：

1. 读取 batch 中的 `states / op_embs / actions / log_probs / values`。
2. 对变长 `op_embs` 做 padding，生成 `op_masks`。
3. 转成 torch tensor。
4. 调用 `evaluate_actions` 得到 `new_log_probs / values / entropy`。
5. 计算 PPO ratio：

```text
ratio = exp(new_log_probs - old_log_probs)
```

6. 计算 clipped policy loss：

```text
surr1 = ratio * advantages
surr2 = clamp(ratio) * advantages
policy_loss = -mean(min(surr1, surr2))
```

7. 计算 value loss：让 critic 的 `values` 接近 `returns`。
8. 计算 entropy loss：鼓励策略保持探索。
9. 汇总：

```text
total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

10. 返回 loss 和日志指标。

---

## 4. Controller alias

```python
Controller = PPOController
```

这是向后兼容写法。旧代码如果 import `Controller`，实际拿到的是 `PPOController`。

---

## 5. BaseTextEncoder：统一文本编码后端

`BaseTextEncoder` 负责把文本转成 embedding。它可能使用两条路径之一。

### 5.1 初始化判断

```text
如果 model_name 以 sentence-transformers/ 开头
或 model_name 中包含 qwen3-embedding
→ 使用 SentenceTransformer

否则
→ 使用 AutoTokenizer + AutoModel
```

### 5.2 SentenceTransformer 路径

```text
texts -> self.model.encode(...) -> numpy embeddings
```

这种模型本身就封装了句向量输出。

### 5.3 AutoModel 路径

```text
texts
→ tokenizer
→ AutoModel
→ token hidden states
→ mean pooling
→ numpy embeddings
```

普通 HuggingFace 模型只输出 token-level hidden states，所以需要 `_mean_pooling` 聚合成句向量。

### 5.4 `_mean_pooling(...)`

该函数用 `attention_mask` 忽略 padding token，只对有效 token 的 hidden states 求平均。

---

## 6. StateEncoder：构造 Controller state

`StateEncoder` 把当前 session/span 和 retrieved memories 组合成一个 state embedding。

### 6.1 输入

```text
session_text: 当前 span/session 文本
retrieved_memories: 检索到的相关 memories
session_embedding: 可选预计算 session embedding
memory_embeddings: 可选预计算 memory embeddings
fusion_mode: memory 融合方式
fusion_tau: sim_weighted 的温度参数
```

### 6.2 `_fuse_memory_embeddings(...)`

如果有多条 retrieved memories，需要融合成一个 memory embedding。

支持两种方式：

1. `mean`：简单平均。
2. `sim_weighted`：根据 memory 与 session 的 cosine similarity 加权平均。

### 6.3 `encode(...)`

完整流程：

1. 得到 `session_emb`。
2. 如果没有 retrieved memories，memory 部分用零向量占位。
3. 如果有 memory embeddings，则融合成一个 `memory_emb`。
4. 拼接：

```text
state_emb = [session_emb || memory_emb]
```

因此 `state_dim = 2 * embedding_dim`。

---

## 7. OpEncoder：构造候选 skill embeddings

`OpEncoder` 把每个 MemSkill/operation description 编成 `op_embedding`。

### 7.1 `encode(op_descriptions)`

输入一组 skill descriptions，输出：

```text
[num_ops, op_dim]
```

每一行对应一个候选 skill。

### 7.2 `encode_single(op_description)`

用于 Designer 新增或修改单个 skill 后，单独更新这个 skill 的 embedding。

---

## 8. 完整执行主线

从训练角度看，完整流程是：

```text
当前 trace/span
→ MemoryBank retrieval
→ StateEncoder 得到 state_embedding
→ OpEncoder 得到 op_embeddings
→ PPOController.forward 选择 action/Top-K actions
→ Executor 执行 selected skills 并更新 MemoryBank
→ 记录 state/action/log_prob/value/reward 到 PPOBuffer
→ episode 结束后 finish_episode 写入 final reward
→ update_controller 调 compute_returns_and_advantages
→ compute_ppo_loss 重新评估旧 actions
→ loss.backward + optimizer.step 更新 Controller
```

从代码边界看：

```text
BaseTextEncoder / StateEncoder / OpEncoder
负责文本 -> embedding

PPOBuffer
负责保存 rollout 训练证据

PPOController
负责可训练策略网络、动作选择、PPO loss
```

---

## 9. 学习建议顺序

明早建议按这个顺序读：

1. 先看 `PPOBuffer.__init__`，理解每个字段保存什么。
2. 看 `push / finish_episode / compute_returns_and_advantages`，理解 reward 和 advantage 怎么形成。
3. 看 `PPOController.__init__`，理解四个网络：`state_net / op_net / actor_head / critic_head`。
4. 看 `get_action_logits`，理解动态 SkillBank 为什么不需要固定分类头。
5. 看 `forward`，理解 rollout 时如何选 skill。
6. 看 `evaluate_actions`，理解 PPO 更新时为什么要重新算 new_log_prob。
7. 看 `compute_ppo_loss`，理解最终 loss 怎么构造。
8. 最后看 `BaseTextEncoder / StateEncoder / OpEncoder`，理解文本 embedding 从哪里来。
