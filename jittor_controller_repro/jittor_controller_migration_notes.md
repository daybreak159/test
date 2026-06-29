# Jittor Controller 迁移适配说明

本文档对应代码：

- `jittor_controller_repro/models/jittor_controller.py`
- 目标：说明 PyTorch 原版 `PPOController` 迁移到 Jittor 时，代码里做了哪些框架适配。

这里不重复讲 PPO 的完整算法原理，而是重点解释：

- 哪些地方是 Jittor 专门需要的写法；
- 哪些地方是为了替代 PyTorch 的 API；
- 哪些地方保持了原算法语义不变；
- 哪些点可以作为 PPT 中“Jittor 复现/适配亮点”来讲。

---

## 1. 整体定位

`jittor_controller.py` 只复现 MemSkill 系统中的 Controller 网络部分。

它不负责：

- 切分 trace / span；
- 生成文本 embedding；
- 维护 MemoryBank；
- 维护 OperationBank；
- 调用 LLM executor；
- 计算 QA reward。

它负责：

- 接收 trainer 已经准备好的 `state_embedding` 和 `op_embeddings`；
- 用 Jittor 网络计算 actor logits 和 critic value；
- rollout 时选择 Top-K skill；
- 记录 old log_prob / old value；
- PPO 更新时重新计算 new log_prob / value / entropy；
- 返回 Jittor tensor 形式的 `total_loss`，供 `optimizer.step(total_loss)` 反向传播。

一句话：

> 这个文件是 PyTorch PPOController 的 Jittor 后端实现，重点是把 Controller 的前向选择和 PPO loss 迁移到 Jittor tensor 计算路径里。

---

## 2. 延迟导入 Jittor

对应代码：

```python
def _require_jittor():
    import jittor as jt
    from jittor import nn
```

### 适配点

Jittor 是可选后端，所以代码没有在文件顶层直接：

```python
import jittor as jt
```

而是写成 `_require_jittor()`，只有真正构造 `JittorPPOController` 时才导入。

### 为什么这样做

如果用户使用 PyTorch backend，不应该因为没有安装 Jittor 而导致整个项目导入失败。

所以这里的适配逻辑是：

```text
普通脚本导入 controller 文件
        ↓
不立即 import jittor
        ↓
只有选择 controller_backend=jittor 时
        ↓
才 import jittor / jittor.nn
```

### PPT 可写

> 通过延迟导入机制，将 Jittor 后端作为可选实现接入原训练框架，避免影响 PyTorch 原流程。

---

## 3. 外层 Factory + 内部 Jittor nn.Module

对应代码：

```python
class JittorPPOController:
    def __new__(...):
        jt, nn = _require_jittor()

        class _Controller(nn.Module):
            ...

        return _Controller()
```

### 适配点

外层 `JittorPPOController` 本身不是一个真正的 Jittor 网络。

真正的网络是内部的：

```python
class _Controller(nn.Module)
```

然后 `__new__` 直接返回 `_Controller()`。

### 为什么这样做

这样可以让 trainer 侧保持类似原来的调用方式：

```python
self.controller = JittorPPOController(**controller_kwargs)
```

但实际拿到的是一个 Jittor `nn.Module`。

这相当于做了一层后端适配：

```text
trainer 仍然按 Controller 使用
        ↓
JittorPPOController 内部返回 Jittor nn.Module
        ↓
不大规模改 trainer 的控制逻辑
```

### PPT 可写

> 通过 Factory 包装，将 Jittor `nn.Module` 封装成与原 PPOController 兼容的接口，减少 trainer 侧侵入式修改。

---

## 4. 网络层从 PyTorch nn.Module 迁移到 Jittor nn.Module

对应代码：

```python
self.state_net = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
)
```

```python
self.op_net = nn.Sequential(...)
self.actor_head = nn.Sequential(...)
self.critic_head = nn.Sequential(...)
```

### 适配点

PyTorch 原版中是：

```python
torch.nn.Module
torch.nn.Sequential
torch.nn.Linear
torch.nn.ReLU
```

Jittor 版替换为：

```python
jittor.nn.Module
jittor.nn.Sequential
jittor.nn.Linear
jittor.nn.ReLU
```

### 网络语义保持不变

这部分不是算法创新，而是框架迁移。

四个网络分支保持和原 PPOController 一致：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `state_net` | `state_embedding` | `state_h` | 编码当前 span / memory 状态 |
| `op_net` | `op_embeddings` | `op_h` | 编码候选 skill |
| `actor_head` | `[state_h, op_h_i]` | `logit_i` | 给每个 skill 打分 |
| `critic_head` | `state_h` | `V(s)` | 预测当前状态价值 |

### Jittor 相关细节

`op_embeddings` 的形状是：

```text
[B, num_ops, op_dim]
```

Jittor 的 `nn.Linear` 可以作用在最后一维，因此 `op_net` 可以直接处理三维 tensor：

```text
[B, num_ops, op_dim]
        ↓ Linear
[B, num_ops, hidden_dim]
```

### PPT 可写

> 网络结构保持原 PPOController 语义不变，将 PyTorch `nn.Module` 系列替换为 Jittor `nn.Module` 系列，并保持 actor/critic 双分支。

---

## 5. 输入边界：numpy/list 转成 Jittor jt.Var

对应代码一：rollout 阶段 `select_action`

```python
if not hasattr(state_embedding, "shape") or not hasattr(state_embedding, "numpy"):
    state_embedding = jt.array(np.asarray(state_embedding, dtype=np.float32))

if not hasattr(op_embeddings, "shape") or not hasattr(op_embeddings, "numpy"):
    op_embeddings = jt.array(np.asarray(op_embeddings, dtype=np.float32))
```

对应代码二：PPO 训练阶段 `compute_ppo_loss`

```python
state_embs = jt.array(np.asarray(batch["states"], dtype=np.float32))
op_embs = jt.array(op_padded)
actions = jt.array(np.asarray(batch["actions"], dtype=np.int64))
old_log_probs = jt.array(np.asarray(batch["log_probs"], dtype=np.float32))
old_values = jt.array(np.asarray(batch["values"], dtype=np.float32))
returns_t = jt.array(np.asarray(returns, dtype=np.float32))
advantages_t = jt.array(np.asarray(advantages, dtype=np.float32))
```

### 适配点

原训练框架中的 `PPOBuffer` 保存的是 Python list / numpy array。

PyTorch 原版会转成：

```python
torch.FloatTensor(...)
torch.LongTensor(...)
```

Jittor 版改成：

```python
jt.array(...)
```

### 为什么要转

Jittor 网络只能对 Jittor tensor 建立计算图。

如果继续保留 numpy array：

- 不能进入 Jittor `nn.Linear`；
- 不能参与 Jittor 自动求导；
- 不能作为 `optimizer.step(loss)` 的计算图来源。

所以这里是迁移的关键边界：

```text
trainer / PPOBuffer: numpy/list
        ↓
jt.array(...)
        ↓
Jittor Controller 前向与 loss 计算
```

### dtype 适配

代码中显式使用：

```python
dtype=np.float32
dtype=np.int64
```

原因是：

- embedding / reward / value / advantage 是浮点数，用 `float32`；
- action index 是下标，用 `int64`；
- 避免 numpy 默认 `float64` 带来不必要的类型转换。

### PPT 可写

> 在 Controller 输入边界统一将 numpy/list 轨迹数据转换为 `jt.Var`，保证后续 actor、critic 与 PPO loss 均处于 Jittor 自动求导路径。

---

## 6. rollout 阶段和 PPO 训练阶段的张量形状不同

### rollout 阶段

对应代码：

```python
state_emb = state_embedding.unsqueeze(0)
op_embs = op_embeddings.unsqueeze(0)
```

单个 span 前向时，输入本来是：

```text
state_embedding: [state_dim]
op_embeddings:  [num_ops_this_span, op_dim]
```

加 batch 维度后：

```text
state_emb: [1, state_dim]
op_embs:   [1, num_ops_this_span, op_dim]
```

这个阶段是一个 span 一个 span 地执行，所以不需要 padding。

### PPO 训练阶段

对应代码：

```python
op_padded, op_masks, new_masks = pad_op_embeddings(...)
```

训练时要把多个 span 组成 batch。

但不同 span 的候选 skill 数可能不同：

```text
span0: [6, op_dim]
span1: [8, op_dim]
span2: [5, op_dim]
```

所以需要 padding 成：

```text
op_padded: [B, max_ops, op_dim]
op_masks:  [B, max_ops]
```

### 重要澄清

`max_ops` 不是在第一个 span 前向执行时提前知道的。

它是在当前 PPO batch 的轨迹都收集完成后，根据 `batch["op_embs"]` 动态计算：

```python
max_ops = max(op.shape[0] for op in op_embs)
```

所以流程是：

```text
rollout: 每个 span 直接用自己的 num_ops
        ↓
轨迹写入 PPOBuffer
        ↓
PPO 更新: 从已收集 batch 中计算 max_ops
        ↓
padding + mask
        ↓
批量计算 loss
```

### PPT 可写

> 前向选择阶段按单个 span 动态执行；PPO 更新阶段才根据已收集 batch 的最大候选数进行 padding，并用 mask 屏蔽补齐位置。

---

## 7. actor 打分中的 broadcast + concat

对应代码：

```python
state_expanded = state_h.unsqueeze(1).broadcast((batch_size, num_ops, self.hidden_dim))
combined = jt.concat([state_expanded, op_h], dim=-1)
logits = self.actor_head(combined).squeeze(-1)
```

### 适配点

Jittor 版显式使用：

```python
unsqueeze
broadcast
jt.concat
squeeze
```

完成每个 `(state, skill)` pair 的打分。

### 形状变化

```text
state_h: [B, H]
op_h:    [B, num_ops, H]
```

先把 `state_h` 扩展到每个 skill：

```text
state_expanded: [B, num_ops, H]
```

再拼接：

```text
combined: [B, num_ops, 2H]
```

最后 actor 输出：

```text
logits: [B, num_ops]
```

### 为什么要这样写

actor_head 的输入是：

```text
当前状态表示 + 某个候选 skill 表示
```

所以每个 skill 都要和同一个 state 拼接一次。

这部分在 Jittor 中通过 tensor broadcast 和 concat 完成，避免为每个 skill 写 Python 循环。

### PPT 可写

> 使用 Jittor broadcast + concat 将一个 state 与多个候选 skill 组成批量 pair，一次性计算所有 skill logits。

---

## 8. mask 机制：jt.where + -1e9

对应代码：

```python
neg_inf = jt.full(logits.shape, -1.0e9)
logits = jt.where(mask == 0, neg_inf, logits)
```

### 适配点

PyTorch 原版常用：

```python
logits = logits.masked_fill(mask == 0, float("-inf"))
```

Jittor 版使用：

```python
jt.where(mask == 0, neg_inf, logits)
```

完成同样逻辑。

### 作用

padding 出来的假 skill 不应该参与 softmax。

所以把它们的 logit 设为极小值：

```text
真实 skill:    正常 logit
padding skill: -1e9
```

softmax 后：

```text
padding skill 概率 ≈ 0
```

### 它如何阻止 padding skill 参与训练

不是通过显式写：

```python
requires_grad = False
```

而是通过概率路径截断：

```text
padding logit = -1e9
        ↓
softmax probability ≈ 0
        ↓
不会被采样为 action
        ↓
log_prob / entropy / policy loss 中贡献约为 0
```

同时 entropy 计算时又额外做了一次 mask：

```python
probs_for_entropy = probs * op_masks
norm = jt.clamp(probs_for_entropy.sum(dim=-1, keepdims=True), min_v=1e-8)
probs_for_entropy = probs_for_entropy / norm
```

这保证 entropy 只统计真实 skill。

### PPT 可写

> 将 PyTorch `masked_fill` 逻辑迁移为 Jittor `jt.where`，通过 `-1e9` logits 使 padding skill 在 softmax 后概率接近 0，从概率路径中排除无效动作。

---

## 9. Python float 与 Jittor tensor 的边界：_scalar

对应代码：

```python
def _scalar(value):
    return float(np.asarray(value.numpy()).reshape(-1)[0])
```

### 适配点

Jittor 计算结果是 `jt.Var`。

但这些地方需要 Python 标量：

- 写入 PPOBuffer 的 `old_log_prob`;
- 写入 PPOBuffer 的 `old_value`;
- 打印日志；
- 保存 `loss_info`。

所以要从 Jittor tensor 转回：

```text
jt.Var
  ↓ value.numpy()
numpy array
  ↓ float(...)
Python float
```

### 关键注意

`_scalar` 只能用于离开计算图的位置。

例如：

```python
loss_info["policy_loss"] = self._scalar(policy_loss)
```

是可以的，因为日志不需要梯度。

但不能这样写：

```python
return self._scalar(total_loss), loss_info
```

因为 `total_loss` 必须保持 Jittor tensor，外部才能：

```python
optimizer.step(total_loss)
```

### PPT 可写

> 日志与 buffer 记录处将 `jt.Var` 转为 Python 标量；训练主路径中的 `total_loss` 保持 Jittor tensor，确保可以反向传播。

---

## 10. 新 skill 探索偏置：jt.where / jt.clamp / stop_grad

对应代码：

```python
probs = nn.softmax(logits_in, dim=-1)
p_new = (probs * mask_in).sum(dim=-1)
need_bias = (p_new < target) & has_new
safe_p = jt.where(need_bias, p_new, jt.ones_like(p_new))
delta = jt.log(target / (safe_p + 1e-8))
delta = jt.where(need_bias, delta, jt.zeros_like(delta))
delta = jt.clamp(delta, min_v=0.0, max_v=float(self.new_action_delta_max))
delta = delta.stop_grad()
out = logits_in + delta.unsqueeze(-1) * mask_in
```

### 适配点

这部分用 Jittor tensor 操作实现新 skill bias：

- `nn.softmax` 计算当前概率；
- `jt.where` 做条件选择；
- `jt.log` 计算需要补的 logit 偏置；
- `jt.clamp` 限制偏置上限；
- `stop_grad` 阻断 bias 本身的梯度。

### 作用

如果 Designer 产生了新 skill，但当前策略几乎不选它，那么新 skill 没有机会被验证。

所以代码给新 skill 的 logits 加一个临时偏置，让它们有最小概率质量。

### 为什么要 stop_grad

这个 bias 是训练策略之外的人为探索机制，不是网络应该学习的参数。

所以：

```python
delta = delta.stop_grad()
```

意思是：

```text
允许 delta 改变本次 action preference
但不允许梯度沿 delta 反向传播
```

对应 PyTorch 中的：

```python
with torch.no_grad()
```

或：

```python
delta.detach()
```

### PPT 可写

> 新 skill 探索偏置通过 Jittor tensor 条件计算完成，并使用 `stop_grad` 保持其为非学习路径，只影响采样倾向，不改变 PPO 参数学习目标。

---

## 11. Top-K 联合概率：替代 Categorical/log_prob

对应代码：

```python
probs = nn.softmax(logits, dim=-1)
selected_probs = probs.gather(dim=-1, index=actions)
prefix_selected = jt.cumsum(selected_probs, dim=-1) - selected_probs
remaining = jt.clamp(1.0 - prefix_selected, min_v=eps)
safe_selected = jt.clamp(selected_probs, min_v=eps)
joint = (jt.log(safe_selected) - jt.log(remaining)).sum(dim=-1)
```

### 适配点

PyTorch 中单动作概率通常可以用：

```python
torch.distributions.Categorical(logits=logits).log_prob(action)
```

但 Jittor 版这里没有依赖等价的 Distribution 封装，而是手写：

```text
softmax + gather + log
```

### 为什么还要 cumsum

因为这里不是单个 action，而是 Top-K action。

并且 Top-K 是无放回选择。

如果选中：

```text
[a1, a2, a3]
```

联合概率不是简单：

```text
p(a1) * p(a2) * p(a3)
```

而是：

```text
p(a1) * p(a2)/(1-p(a1)) * p(a3)/(1-p(a1)-p(a2))
```

Jittor 版用：

```python
jt.cumsum(selected_probs, dim=-1)
```

一次性得到每个位置前面已经选走的概率质量。

### 这部分的迁移价值

这部分是 Jittor 版中比较值得讲的适配点：

```text
原本可以逐个 Top-K action 循环计算
        ↓
改为 jt.cumsum / jt.clamp / jt.log 的张量化批量计算
        ↓
减少 Python 控制流
        ↓
更适合 Jittor JIT 和批量训练
```

### PPT 可写

> Jittor 版不依赖 PyTorch `Categorical.log_prob`，而是显式使用 `softmax + gather + log` 构造动作概率；Top-K 联合概率进一步用 `jt.cumsum` 张量化计算，保持 PPO ratio 的计算口径一致。

---

## 12. Top-K 采样：jt.topk 与 Gumbel-Top-K

对应代码：

```python
if deterministic:
    _, indices = jt.topk(logits, k)
```

```python
u = jt.random(logits.shape)
gumbel = -jt.log(-jt.log(jt.clamp(u, min_v=1e-8, max_v=1.0 - 1e-8)))
_, indices = jt.topk(logits + gumbel, k)
```

### 适配点

确定性选择：

```text
jt.topk(logits, k)
```

随机探索选择：

```text
jt.random + Gumbel noise + jt.topk
```

### 为什么用 Gumbel

Gumbel-Top-K 可以把“按概率随机选 Top-K”转成：

```text
logits + 随机噪声
        ↓
topk
```

这样采样过程也可以尽量保持在 Jittor tensor 操作中，而不是依赖 Python 循环。

### 注意

采样本身不是 PPO 反传的主要路径。

PPO 真正训练时会用 `evaluate_actions` 重新计算旧 action 在当前策略下的概率。

但 rollout 阶段需要采样动作，所以这里仍然要用 Jittor 实现。

### PPT 可写

> Top-K 采样使用 `jt.topk`；随机探索用 Jittor tensor 形式的 Gumbel-Top-K，实现采样逻辑与 Jittor 后端一致。

---

## 13. evaluate_actions：不重新采样，只重算概率

对应代码：

```python
state_h = self.encode_state(state_embs)
op_h = self.encode_ops(op_embs)
logits = self.get_action_logits(state_h, op_h, mask=op_masks)
values = self.get_value(state_h)
...
log_probs = self._compute_topk_log_prob(logits, actions)
```

### 适配点

`evaluate_actions` 是 PPO 更新阶段的 Jittor 前向入口。

它接收：

- 已经 padding 的 `op_embs`;
- mask；
- rollout 时保存的旧 `actions`;
- 当前网络参数。

它输出：

- `new_log_probs`;
- `values`;
- `entropy`;
- `topk_stats`。

### 关键逻辑

这里不会重新选 action。

而是：

```text
旧 action 保持不变
        ↓
用当前 Jittor Controller 重新计算这些旧 action 的概率
        ↓
得到 new_log_prob
        ↓
和 old_log_prob 计算 PPO ratio
```

### Jittor 适配意义

这一段要求所有计算保持在 Jittor tensor 内：

```text
state_embs / op_embs / actions
        ↓
Jittor forward
        ↓
new_log_probs / values / entropy
```

否则后面的 policy loss 和 value loss 无法反向传播。

### PPT 可写

> PPO 更新阶段不重新采样 action，而是在 Jittor 中重算旧 action 的当前概率，形成 `new_log_prob`，用于和 buffer 中的 `old_log_prob` 构造 ratio。

---

## 14. PPO loss：使用 Jittor 数学算子重写

对应代码：

```python
ratio = jt.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages_t
surr2 = jt.clamp(ratio, min_v=1.0 - self.clip_epsilon, max_v=1.0 + self.clip_epsilon) * advantages_t
policy_loss = -jt.minimum(surr1, surr2).mean()
```

```python
value_loss = 0.5 * ((values - returns_t) ** 2).mean()
```

```python
entropy_loss = -entropy.mean()
total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
```

### 适配点

PyTorch 中通常使用：

```python
torch.exp
torch.clamp
torch.min
torch.max
tensor.mean()
```

Jittor 版对应替换为：

```python
jt.exp
jt.clamp
jt.minimum
jt.maximum
tensor.mean()
```

### 梯度流向

`old_log_probs` 来自 rollout buffer，是旧策略记录，转成 Jittor tensor 后只是常量基准。

`new_log_probs` 来自当前 Jittor 网络前向。

因此：

```text
ratio = exp(new_log_probs - old_log_probs)
```

梯度主要流向：

```text
new_log_probs → actor_head / state_net / op_net
```

不会流向：

```text
old_log_probs
```

critic 则通过：

```python
values - returns_t
```

训练：

```text
critic_head / state_net
```

### total_loss 必须保持 Jittor tensor

函数最后：

```python
return total_loss, loss_info
```

其中：

- `loss_info` 可以转成 Python float；
- `total_loss` 必须保持 Jittor tensor。

因为外部训练代码需要：

```python
optimizer.step(total_loss)
```

### PPT 可写

> PPO loss 中的 `exp / clamp / minimum / maximum` 全部替换为 Jittor tensor 算子，`total_loss` 保持 `jt.Var` 返回，直接接入 Jittor optimizer 反向传播。

---

## 15. value clipping：jt.maximum / jt.clamp

对应代码：

```python
value_pred_clipped = old_values + jt.clamp(
    values - old_values, min_v=-self.vf_clip, max_v=self.vf_clip
)
value_loss_unclipped = (values - returns_t) ** 2
value_loss_clipped = (value_pred_clipped - returns_t) ** 2
value_loss = 0.5 * jt.maximum(value_loss_unclipped, value_loss_clipped).mean()
```

### 适配点

这是 PyTorch PPO 中 value clipping 的 Jittor 写法。

核心替换：

```text
torch.clamp  → jt.clamp
torch.maximum → jt.maximum
```

### 作用

限制 critic 新预测值相对旧预测值变化过大，提升 PPO 更新稳定性。

### PPT 可写

> value clipping 保持 PPO 原语义，通过 `jt.clamp` 和 `jt.maximum` 实现 critic 更新幅度约束。

---

## 16. 日志指标也用 Jittor 算完后再转 float

对应代码：

```python
explained_var = 1 - (returns_t - values).var() / (returns_t.var() + 1e-8)
```

```python
"clip_frac": self._scalar((jt.abs(ratio - 1.0) > self.clip_epsilon).float32().mean())
```

### 适配点

即使是日志指标，也先用 Jittor tensor 算：

- `explained_variance`;
- `approx_kl`;
- `clip_frac`;
- `value_mean`;
- `return_mean`;
- `advantage_mean`。

最后再：

```python
self._scalar(...)
```

转成 Python float。

### 为什么这样写

这样可以复用已经在 Jittor 中的中间变量，不需要频繁在 Jittor / numpy 之间来回转换。

### PPT 可写

> 训练日志复用 Jittor 中间张量计算，最后仅在记录阶段转为 Python 标量，减少框架边界转换。

---

## 17. 全文件 Jittor 适配点总表

| 代码位置 | PyTorch 原思路 | Jittor 版适配 | 作用 |
|---|---|---|---|
| `_require_jittor` | 顶层 import torch | 延迟 import jittor | Jittor 后端可选接入 |
| `__new__` | 直接定义 `nn.Module` | Factory 返回内部 `_Controller(nn.Module)` | 保持 trainer 接口兼容 |
| 网络定义 | `torch.nn.*` | `jittor.nn.*` | 迁移 actor/critic 网络 |
| 输入转换 | `torch.FloatTensor / LongTensor` | `jt.array(np.asarray(...))` | 进入 Jittor 计算图 |
| actor pair 构造 | torch broadcast/concat | `unsqueeze + broadcast + jt.concat` | 批量计算 state-skill 打分 |
| mask | `masked_fill(-inf)` | `jt.where(mask == 0, -1e9, logits)` | 屏蔽 padding skill |
| 标量导出 | `.item()` | `.numpy() -> float` | 日志和 buffer 记录 |
| 新 skill bias | `no_grad/detach` | `stop_grad` | 探索偏置不进入学习路径 |
| 单动作概率 | `Categorical.log_prob` | `softmax + gather + log` | 显式构造 log_prob |
| Top-K 联合概率 | 可循环计算 | `jt.cumsum` 张量化 | 批量计算 joint log_prob |
| 随机 Top-K | PyTorch sample/topk | `jt.random + Gumbel + jt.topk` | Jittor 采样路径 |
| PPO ratio | `torch.exp` | `jt.exp` | 新旧策略概率比 |
| PPO clip | `torch.clamp` | `jt.clamp` | 限制策略更新幅度 |
| policy loss | `torch.min` | `jt.minimum` | clipped surrogate objective |
| value clipping | `torch.max` | `jt.maximum` | critic 稳定训练 |
| 反传入口 | `loss.backward()` | `optimizer.step(total_loss)` | Jittor 优化器接口 |

---

## 18. 可以放进 PPT 的核心总结

### 一句话版本

> 我们将 MemSkill 的 PPOController 从 PyTorch 迁移到 Jittor，重点适配了动态 SkillBank 的 batch 化、Top-K 动作概率重算、mask 屏蔽无效 skill，以及 PPO loss 的 Jittor tensor 反向传播路径。

### 三点版本

1. **输入与网络迁移**

   将 buffer 中的 numpy/list 轨迹数据转换为 `jt.Var`，并使用 Jittor `nn.Module` 复现 actor/critic 网络。

2. **动态候选 skill 适配**

   rollout 阶段按单个 span 动态选择；PPO 更新阶段对不同候选数的 skill embedding 做 padding，并用 `jt.where + -1e9` mask 屏蔽 padding skill。

3. **Top-K 与 PPO loss 张量化**

   使用 `softmax + gather + log` 替代 PyTorch `Categorical.log_prob`，并用 `jt.cumsum` 计算 Top-K 联合概率；最终 `total_loss` 保持 Jittor tensor，接入 Jittor optimizer。

### 更适合答辩的版本

> Jittor 迁移的核心不是改变 PPO 算法，而是让原本依赖 PyTorch tensor、distribution 和 optimizer 的 Controller 训练路径，在 Jittor 中完整闭环：从 embedding 输入、动态 skill mask、Top-K joint log probability，到 PPO clipped loss 和 optimizer 更新，全部保持在 Jittor tensor 计算图内。

---

## 19. 需要避免的表述

下面这些说法不建议写进 PPT：

- “Jittor 版重新设计了 PPO 算法。”
- “mask 是 Jittor 独有机制。”
- “Jittor 版一定端到端更快。”
- “padding skill 被显式设置为不参与梯度。”

更准确的说法是：

- PPO 算法语义保持一致；
- mask 是动态动作空间 batch 化所需，PyTorch/Jittor 都需要；
- Jittor 版主要优化 Controller 计算路径，不代表端到端 API 调用一定加速；
- padding skill 通过 `-1e9 logits -> softmax 约为 0` 从概率路径中被排除。

