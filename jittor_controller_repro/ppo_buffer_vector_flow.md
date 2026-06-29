# MemSkill PPOBuffer 向量与训练字段执行记录

本文档记录 MemSkill 中 `PPOBuffer` 的核心字段在完整训练流程中的来源、写入位置、读取位置和训练作用。它对应论文中 “Controller 选择 MemSkill，并通过 QA / process reward 做 PPO 优化” 这一部分。

本文档重点围绕以下字段展开：

```python
self.states
self.op_embs
self.new_op_masks
self.actions
self.log_probs
self.values
self.rewards
self.dones
```

这些字段不是长期记忆库，也不是技能库，而是 **PPO 训练时的轨迹缓存**。每处理一个 span/session，系统都会把 Controller 当时看到的状态、候选技能、选择结果、旧策略概率、价值预测和奖励写入 buffer。等一条 trace 处理结束后，再根据这些轨迹计算 Return、Advantage 和 PPO loss。

## 1. 总体执行链路

MemSkill 的一条训练 episode 可以理解为一条 conversation trace。该 trace 被切成多个 span/session，每个 span 都会执行一次 Controller 选择。

```text
conversation trace
→ split into sessions/spans
→ 对每个 span:
    session_text
    → session_embedding
    → MemoryBank 检索相关 memories
    → StateEncoder 构造 state_embedding
    → OperationBank 取出候选 MemSkills
    → OpEncoder / OperationBank 提供 op_embeddings
    → Controller 选择 action / Top-K actions
    → Executor 调用 LLM 更新 MemoryBank
    → 计算 process_reward
    → PPOBuffer.push(...)
→ trace 结束:
    用最终 MemoryBank 回答 QA
    → F1 / LLM judge 得到 final_reward
    → PPOBuffer.finish_episode(...)
→ PPO 更新:
    compute_returns_and_advantages(...)
    → compute_ppo_loss(...)
    → optimizer 更新 controller 参数
```

对应主要源码位置：

| 阶段 | 源码位置 | 作用 |
|---|---|---|
| 处理一条 trace | `MemSkill/src/trainer.py:568` | `train_episode` 顺序处理多个 session |
| 处理一个 span/session | `MemSkill/src/trainer.py:669` | `_process_session` 生成状态、选择动作、执行操作、写入 buffer |
| 写入 PPOBuffer | `MemSkill/src/controller.py:40` | `PPOBuffer.push(...)` |
| episode 结束奖励回填 | `MemSkill/src/controller.py:75` | `finish_episode(...)` |
| 计算 Return / Advantage | `MemSkill/src/controller.py:129` | `compute_returns_and_advantages(...)` |
| PPO loss | `MemSkill/src/controller.py:615` | `compute_ppo_loss(...)` |
| 训练更新 | `MemSkill/src/trainer.py:1031` | `update_controller(...)` |

## 2. `states`: 当前状态向量

### 2.1 论文含义

`states` 对应强化学习中的状态 `s_t`。在 MemSkill 中，一个状态表示：

```text
当前 span/session 的语义信息 + 当前 MemoryBank 中检索到的相关 memories
```

它不是原始文本，而是数值向量 `state_embedding`。

如果文本编码器输出维度是 `D`，那么：

```text
session_emb: [D]
memory_emb:  [D]
state_emb:   [2D]
```

### 2.2 生成过程

在 `_process_session` 中，当前 span 先被编码：

```text
session_text
→ self.state_encoder._encode_texts(session_text)
→ session_embedding
```

然后用该向量从 MemoryBank 中检索相关记忆：

```text
memory_bank.retrieve(
    session_embedding,
    use_state_encoder=True,
    return_embeddings=True
)
→ retrieved_memories
→ retrieved_indices
→ retrieved_memory_embeddings
```

最后由 `StateEncoder.encode(...)` 融合为最终状态：

```text
state_embedding = concat(session_emb, fused_memory_emb)
```

相关源码：

| 步骤 | 源码位置 |
|---|---|
| session 批量编码 | `MemSkill/src/trainer.py:594` |
| 单个 session 编码 | `MemSkill/src/trainer.py:685` |
| MemoryBank 检索 | `MemSkill/src/trainer.py:691` |
| state encoder 融合 | `MemSkill/src/trainer.py:698` |
| memory embedding 融合函数 | `MemSkill/src/controller.py:893` |
| 拼接成 state embedding | `MemSkill/src/controller.py:985` |

### 2.3 写入 buffer

`state_embedding` 被传入 `ppo_buffer.push(...)`：

```python
ppo_buffer.push(
    state_emb=state_embedding,
    ...
)
```

在 `PPOBuffer.push` 内部执行：

```python
self.states.append(state_emb)
```

相关源码：

| 步骤 | 源码位置 |
|---|---|
| 调用 `push` | `MemSkill/src/trainer.py:777` |
| append 到 `self.states` | `MemSkill/src/controller.py:53` |

### 2.4 训练时读取

PPO 更新时，`get_batch()` 返回：

```python
'states': self.states
```

随后在 `compute_ppo_loss(...)` 中转换成张量：

```python
state_embs = torch.FloatTensor(np.array(batch['states'])).to(self.device)
```

Jittor 版本中对应：

```python
state_embs = jt.array(np.asarray(batch["states"], dtype=np.float32))
```

### 2.5 在 Controller 中的作用

`states` 进入 `state_net`：

```text
state_embedding
→ state_net
→ state_h
```

然后分两路使用：

```text
state_h + op_h → actor_head → 每个 skill 的 logit
state_h        → critic_head → V(s_t)
```

因此，`states` 是 actor 和 critic 的共同输入。

## 3. `op_embs`: 候选 MemSkill 向量集合

### 3.1 论文含义

`op_embs` 对应当前状态下的动态动作空间 `A_t`。这里的动作不是固定类别，而是 OperationBank 中当前所有候选 MemSkill。

每个 MemSkill 有文本描述，例如：

```text
insert
update
delete
noop
insert_exact_temporal_fact
insert_created_object_detail
```

这些文本描述会被编码成向量。Controller 不直接读 skill 名称，而是读 skill embedding。

### 3.2 生成过程

在 `_process_session` 中，系统先从 OperationBank 取出候选技能：

```python
candidate_ops = self.operation_bank.get_candidate_operations()
```

然后把每个 operation 的 embedding 堆叠起来：

```python
op_embeddings = np.vstack([op.embedding for op in candidate_ops])
```

OperationBank 初始化时会根据基础技能生成 embedding；Designer 新增或更新技能后，也会重新计算对应 embedding。

相关源码：

| 步骤 | 源码位置 |
|---|---|
| 初始化基础 operations | `MemSkill/src/operation_bank.py:121` |
| 为 operations 计算 embeddings | `MemSkill/src/operation_bank.py:143` |
| 返回候选 operations | `MemSkill/src/operation_bank.py:187` |
| 在 trainer 中取候选 operations | `MemSkill/src/trainer.py:707` |
| 堆叠为 `op_embeddings` | `MemSkill/src/trainer.py:711` |

### 3.3 写入 buffer

`op_embeddings` 被传入：

```python
ppo_buffer.push(
    op_embs=op_embeddings,
    ...
)
```

在 `PPOBuffer.push` 中：

```python
self.op_embs.append(op_embs)
```

### 3.4 训练时读取

因为不同 step 的候选技能数量可能不同，所以 PPO 训练时需要 padding：

```python
max_ops = max(op.shape[0] for op in batch['op_embs'])
op_embs_padded = np.zeros((n, max_ops, op_dim), dtype=np.float32)
op_masks = np.zeros((n, max_ops), dtype=np.float32)
```

真实技能位置 `op_masks=1`，补齐位置 `op_masks=0`。这样 batch 可以统一成：

```text
op_embs:  [batch_size, max_ops, op_dim]
op_masks: [batch_size, max_ops]
```

### 3.5 在 Controller 中的作用

`op_embs` 进入 `op_net`：

```text
op_embeddings
→ op_net
→ op_h
```

然后每个 skill 与同一个 state 配对：

```text
[state_h, op_h_i]
→ actor_head
→ logit_i
```

这就是 MemSkill 支持动态 skill bank 的关键：Controller 不是固定输出 `N` 个类别，而是对每个候选 skill 单独计算分数。

## 4. `new_op_masks`: 新技能探索标记

### 4.1 论文含义

`new_op_masks` 用来标记哪些候选 MemSkill 是 Designer 新增或刚刚更新的技能。

Designer 产生新 skill 后，如果完全按旧策略采样，新 skill 可能很少被选中，也就没有机会被验证。因此代码提供了一个新技能探索 bias。

### 4.2 生成过程

在 `_process_session` 中：

```python
new_op_indices = self.operation_bank.get_new_action_indices(candidate_ops)
```

如果存在新技能，则构造 mask：

```python
new_op_mask = np.zeros(len(candidate_ops), dtype=np.float32)
new_op_mask[new_op_indices] = 1.0
```

否则为 `None`。

相关源码：

| 步骤 | 源码位置 |
|---|---|
| 设置新 operation 名称 | `MemSkill/src/operation_bank.py:177` |
| 获取新 operation 下标 | `MemSkill/src/operation_bank.py:181` |
| trainer 中构造 `new_op_mask` | `MemSkill/src/trainer.py:713` |

### 4.3 写入 buffer

`new_op_mask` 被传入 `ppo_buffer.push(...)`。

如果为 `None`，`PPOBuffer.push` 会转成全 0 mask：

```python
if new_op_mask is None:
    new_op_mask = np.zeros(op_embs.shape[0], dtype=np.float32)
self.new_op_masks.append(new_op_mask)
```

### 4.4 训练时读取

训练时和 `op_embs` 一样需要 padding：

```python
new_op_masks_padded = np.zeros((n, max_ops), dtype=np.float32)
```

之后传入：

```python
self.evaluate_actions(..., new_op_masks)
```

在 Controller 中，如果新技能概率过低，会通过 `_apply_new_action_bias(...)` 提高对应 logits。

注意：这个 bias 是探索辅助，不是一个单独的可训练参数。

## 5. `actions`: Controller 选择的技能下标

### 5.1 论文含义

`actions` 对应强化学习动作 `a_t`。在 MemSkill 中，动作就是 Controller 选择的 MemSkill。

如果 `action_top_k=1`：

```text
actions = 2
```

如果 `action_top_k=3`：

```text
actions = [2, 0, 4]
```

这里存的是候选 skill 的下标，不是 skill 名称。真实名称需要通过 `candidate_ops[idx]` 反查。

### 5.2 生成过程

Controller 对每个候选 skill 产生 logit 后，按策略选择 action。

PyTorch 分支：

```python
action_idx, log_prob, value = self.controller(
    state_tensor,
    op_tensor,
    deterministic=False,
    new_op_mask=new_op_mask
)
```

Jittor 分支：

```python
action_idx, log_prob, value = self.controller.select_action(
    state_embedding,
    op_embeddings,
    deterministic=False,
    new_op_mask=new_op_mask
)
```

Top-K 时，训练 rollout 使用 Gumbel-Top-K 做不放回采样：

```text
logits + gumbel noise
→ top-k indices
→ actions
```

相关源码：

| 步骤 | 源码位置 |
|---|---|
| Controller 选择 action | `MemSkill/src/trainer.py:720` |
| PyTorch `forward` | `MemSkill/src/controller.py:483` |
| Gumbel-Top-K 采样 | `MemSkill/src/controller.py:545` |
| Jittor `select_action` | `MemSkill/jittor_controller_repro/models/jittor_controller.py:171` |

### 5.3 写入 buffer

```python
self.actions.append(action)
```

Top-K 时，`action` 是列表。`get_batch()` 会把它整理成二维数组：

```text
[N, K]
```

### 5.4 在系统执行中的作用

`actions` 不仅用于 PPO 训练，也决定 Executor 收到哪些技能：

```python
selected_ops = [candidate_ops[idx] for idx in action_idx]
```

然后：

```python
self.executor.execute_operation(
    operation=selected_ops,
    session_text=session_text,
    retrieved_memories=retrieved_memories
)
```

因此，`actions` 是 Controller 和 LLM Executor 之间的连接点。

### 5.5 训练时读取

PPO 更新时不是重新采样 action，而是评估旧 action 在新策略下的概率：

```python
new_log_probs, values, entropy, topk_stats = self.evaluate_actions(...)
```

Top-K 时使用 `_compute_topk_log_prob(...)` 计算整组 action 的联合 log probability。

## 6. `log_probs`: 旧策略动作概率

### 6.1 论文含义

`log_probs` 对应：

```text
log π_old(a_t | s_t)
```

也就是 rollout 当时旧策略选择该 action 或 Top-K action 组合的 log 概率。

PPO 必须保存旧策略概率，因为更新时要计算：

```text
ratio = exp(log π_new(a_t|s_t) - log π_old(a_t|s_t))
```

### 6.2 生成过程

在 Controller 选择 action 后，同时计算 log probability。

单动作时：

```python
dist = torch.distributions.Categorical(logits=logits)
log_prob = dist.log_prob(action)
```

Top-K 时：

```python
log_prob = self._compute_topk_log_prob(logits, actions_tensor)
```

Top-K 联合概率采用不放回采样：

```text
log P(a1, a2, ..., aK)
= Σ_i [log p(ai) - log(1 - Σ_{j<i} p(aj))]
```

相关源码：

| 步骤 | 源码位置 |
|---|---|
| Top-K log probability | `MemSkill/src/controller.py:398` |
| rollout 返回 `log_prob` | `MemSkill/src/controller.py:554` |
| 写入 step log | `MemSkill/src/trainer.py:750` |
| 写入 buffer | `MemSkill/src/trainer.py:781` |

### 6.3 写入 buffer

```python
self.log_probs.append(log_prob)
```

### 6.4 训练时读取

在 `compute_ppo_loss(...)` 中：

```python
old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
```

然后和新策略概率计算 ratio：

```python
ratio = torch.exp(new_log_probs - old_log_probs)
```

这就是 PPO clipped loss 的核心。

## 7. `values`: Critic 的状态价值预测

### 7.1 论文含义

`values` 对应：

```text
V_old(s_t)
```

它是 rollout 当时 critic 对当前状态未来累计 reward 的预测。

### 7.2 生成过程

Controller 在选择 action 的同时会计算：

```python
value = self.get_value(state_h)
```

其中：

```text
state_embedding
→ state_net
→ state_h
→ critic_head
→ V(s_t)
```

相关源码：

| 步骤 | 源码位置 |
|---|---|
| critic 网络定义 | `MemSkill/src/controller.py:292` |
| value 计算函数 | `MemSkill/src/controller.py:341` |
| rollout 中计算 value | `MemSkill/src/controller.py:511` |
| 写入 step log | `MemSkill/src/trainer.py:751` |

### 7.3 写入 buffer

```python
self.values.append(value)
```

### 7.4 训练时读取

`values` 有两个用途。

第一，用于计算 GAE：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

源码中：

```python
values = np.array(self.values)
delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
```

第二，用于 value loss 或 value clipping：

```python
old_values = torch.FloatTensor(batch['values']).to(self.device)
```

如果开启 `vf_clip`，旧 value 会作为裁剪基准。

## 8. `rewards`: 过程奖励与最终 QA 奖励

### 8.1 论文含义

`rewards` 对应 PPO 的每步奖励 `r_t`。

MemSkill 的 reward 有两类：

```text
1. process reward：每个 span 执行后立即获得
2. final QA reward：整条 trace 处理结束后，通过 QA/F1/LLM judge 获得
```

### 8.2 process reward 的来源

Executor 执行后，会输出实际 memory 操作结果，例如：

```text
INSERT / UPDATE / DELETE / NOOP
```

系统会比较：

```text
Controller 选择的 selected_types
Executor 实际执行成功的 actual_types
```

如果二者匹配，则给正 reward；如果部分重叠，则给部分 reward；如果完全不匹配或解析失败，则给负 reward。

相关源码：

| 步骤 | 源码位置 |
|---|---|
| 构造 process reward meta | `MemSkill/src/trainer.py:1698` |
| 计算 process reward | `MemSkill/src/trainer.py:1715` |
| 正负奖励规则 | `MemSkill/src/trainer.py:1739` |
| 完全不匹配惩罚 | `MemSkill/src/trainer.py:1756` |
| 当前 step 写入 reward | `MemSkill/src/trainer.py:768` |

### 8.3 写入 buffer

```python
immediate_reward = process_reward
ppo_buffer.push(..., reward=immediate_reward)
```

在 `PPOBuffer.push` 中：

```python
self.rewards.append(reward)
```

### 8.4 final QA reward 的来源

一条 trace 的所有 session 处理完后，系统用最终 MemoryBank 回答训练问题：

```text
最终 MemoryBank
→ 检索与 question 相关的 memories
→ LLM 回答 question
→ 与标准答案比较
→ 得到 F1 或 LLM judge reward
```

相关源码：

| 步骤 | 源码位置 |
|---|---|
| episode 结束后 QA 评估 | `MemSkill/src/trainer.py:627` |
| 构造 QA prompt 并调用 LLM | `MemSkill/src/trainer.py:944` |
| 计算 F1 / LLM judge | `MemSkill/src/trainer.py:812` |
| 得到 `avg_reward` | `MemSkill/src/trainer.py:852` |

### 8.5 final reward 回分配

`final_reward` 不是某一步直接产生的，而是整条 trace 的最终表现。因此代码用 `finish_episode(...)` 把它回填到 `rewards`。

如果开启 `redistribute=True`，则按指数衰减权重分配到 episode 内每一步：

```text
later steps receive larger weights
earlier steps receive smaller weights
```

源码：

```python
self.rewards[episode_start + i] += remaining_reward * w
self.rewards[-1] += last_reward
```

如果不开启回分配，则全部加到最后一步：

```python
self.rewards[-1] += final_reward
```

### 8.6 训练时读取

`rewards` 被用于计算 Return 和 Advantage：

```python
returns, advantages = ppo_buffer.compute_returns_and_advantages(...)
```

最终影响：

```text
policy_loss: 哪些 action 应该被强化或削弱
value_loss: critic 对 return 的拟合
```

## 9. `dones`: episode 边界

### 9.1 论文含义

`dones` 标记当前 step 是否是一条 trace/episode 的结束。

它的作用不是给 Controller 输入，而是用于 GAE 计算时切断不同 episode 之间的奖励传播。

### 9.2 写入过程

每个 step 刚写入 buffer 时：

```python
self.dones.append(False)
```

一条 trace 结束后，`finish_episode(...)` 会执行：

```python
self.dones[-1] = True
```

### 9.3 训练时作用

在 GAE 中：

```python
next_non_terminal = 1.0 - dones[t]
```

如果 `dones[t] = True`，则：

```text
next_non_terminal = 0
```

这会阻止下一条 episode 的 value 或 advantage 影响当前 episode。

对应源码：

```python
delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_gae
```

## 10. 字段之间的依赖关系

这些字段不是孤立的，它们共同组成 PPO 的训练样本：

```text
states + op_embs + new_op_masks
→ 重新计算 new_log_probs, values, entropy

actions + old log_probs
→ 计算 PPO ratio

rewards + old values + dones
→ 计算 returns, advantages

returns + values
→ value_loss

advantages + ratio
→ policy_loss

entropy
→ exploration bonus
```

最终 loss：

```text
total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

## 11. PyTorch 与 Jittor 对应关系

本项目的 Jittor 复现主要对齐 `PPOController` 的训练计算。字段在 PyTorch 和 Jittor 中的角色一致，只是张量后端不同。

| 字段 | PyTorch 位置 | Jittor 位置 | 迁移方式 |
|---|---|---|---|
| `states` | `torch.FloatTensor(np.array(batch['states']))` | `jt.array(np.asarray(batch["states"]))` | 输入 state tensor |
| `op_embs` | padding 后转 `torch.FloatTensor` | padding 后转 `jt.array` | 动态 action space 输入 |
| `new_op_masks` | padding 后转 `torch.FloatTensor` | padding 后转 `jt.array` | 新 skill bias mask |
| `actions` | `torch.LongTensor(batch['actions'])` | `jt.array(..., int64)` | 旧动作索引 |
| `log_probs` | `old_log_probs` | `old_log_probs` | PPO ratio 基准 |
| `values` | `old_values` | `old_values` | GAE/value clipping 基准 |
| `rewards` | NumPy 中计算 returns/advantages | NumPy 中计算 returns/advantages | 无梯度，可保留 |
| `dones` | NumPy 中切断 episode | NumPy 中切断 episode | 无梯度，可保留 |

Jittor 复现文件：

```text
MemSkill/jittor_controller_repro/models/jittor_controller.py
```

其中主要对应：

```text
state_net
op_net
actor_head
critic_head
_compute_topk_log_prob
select_action
evaluate_actions
compute_ppo_loss
```

## 12. 一句话总结

`PPOBuffer` 中的这些字段共同把 MemSkill 的 LLM 系统执行过程转成可训练的强化学习数据：

```text
文本 trace 和 memory 更新结果
→ 数值状态、动作、奖励、旧策略概率和值函数预测
→ Return / Advantage
→ PPO loss
→ 更新 Controller 参数
```

因此，Jittor 迁移的核心价值不是迁移 LLM、Executor 或 Designer，而是把这些轨迹字段进入神经网络训练后的张量计算、自动求导和参数更新过程，用 Jittor 重新实现并与 PyTorch 对齐。
