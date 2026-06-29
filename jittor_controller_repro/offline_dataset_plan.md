# MemSkill Controller 离线训练数据集设计

## 1. 设计目标

在线训练可以证明 Jittor controller 已经接入 MemSkill 完整闭环，但在线 API、LLM judge 和随机采样会让 PyTorch/Jittor 的 reward 曲线不完全公平。因此需要制作一份固定的离线训练数据集，用于课程报告中的严格对齐实验。

离线数据集的目标是把一次真实 MemSkill 在线训练过程固化为 controller 可直接消费的数值轨迹，让 PyTorch 原版 `PPOController` 和 Jittor 迁移版 `PPOController` 在完全相同的数据上计算 loss、反向传播和更新参数。

## 2. 数据来源

优先使用已经生成的在线 step records：

- `jittor_controller_repro/runs/online_torch_step_records.jsonl`
- `jittor_controller_repro/runs/online_jittor_step_records.jsonl`

这两份数据来自真实 LoCoMo 样本、真实 MemSkill executor、真实 API 调用和真实 QA reward。后续也可以追加更多 trace，扩大数据规模。

## 3. 推荐字段

离线 `.npz` 数据集建议包含以下字段：

| 字段 | 含义 | 用途 |
|---|---|---|
| `state_embeddings` | 当前 span 与相关 memory 融合后的状态向量 | controller 状态输入 |
| `op_embeddings` | 当前 skill bank 中每个 MemSkill 的向量表示 | controller 候选 action 输入 |
| `actions` | 当步选中的 Top-K skill 下标 | PPO action log_prob 计算 |
| `old_log_probs` | rollout 时旧策略给该 action 的 log probability | PPO ratio 计算 |
| `old_values` | rollout 时 critic 预测值 | value 对齐和诊断 |
| `returns` | 延迟 reward 回传后的累计回报 | value loss 目标 |
| `advantages` | return 与 value 的差值或 GAE 结果 | policy loss 权重 |
| `op_masks` | 候选 skill 是否有效 | 屏蔽 padding 或无效 skill |
| `new_op_masks` | 新增/演化 skill 的 mask | 保留原论文 action 约束逻辑 |
| `process_rewards` | 每一步 executor 过程奖励 | 分析 reward 构成 |
| `episode_ids` | 所属 trace/episode | 后续按 episode 切分训练/测试 |

文本字段如 `executor_prompt`、`executor_response`、`candidate_ops` 可以保存在旁路 JSONL 中，用于 PPT 展示和错误分析；正式训练 `.npz` 只保留数值张量字段。

## 4. 生成流程

1. 在线运行原 MemSkill 或 Jittor-MemSkill，开启 step record dump。
2. 从 JSONL 中读取每个 step 的 embedding、action、log_prob、value、reward。
3. 按 episode 分组，重新计算或校验 `returns` 和 `advantages`。
4. 对不同长度的 operation bank 做 padding，并生成 `op_masks`。
5. 保存为 `.npz`，同时导出一个 compact metadata JSON，记录数据来源、API 模型、时间、trace 数量和字段维度。

## 5. 对齐实验方案

离线数据集生成后，分别运行：

```bash
python -m jittor_controller_repro.train_torch \
  --trace jittor_controller_repro/runs/api_cached_trace.npz \
  --log jittor_controller_repro/runs/api_torch_train.jsonl

python -m jittor_controller_repro.train_jittor \
  --trace jittor_controller_repro/runs/api_cached_trace.npz \
  --log jittor_controller_repro/runs/api_jittor_train.jsonl
```

核心对齐指标：

- `total_loss`
- `policy_loss`
- `value_loss`
- `entropy`
- `approx_kl`
- `clip_frac`
- 训练耗时

如果固定初始化权重，还可以额外做单步数值对齐：比较 logits、value、action log probability 和 PPO loss。

## 6. PPT 叙事位置

离线数据集适合作为 PPT 的扩展贡献：

1. 在线阶段证明“Jittor controller 可以接入原 MemSkill 闭环”。
2. 离线阶段解决“在线 API 随机性导致难以公平对齐”的问题。
3. 固定缓存数据让 PyTorch/Jittor 在同一批 controller evidence 上训练，从而满足课程要求中的训练脚本、测试脚本、loss 曲线、性能 log 与 PyTorch 对齐。

最终汇报时可以把在线训练作为系统演示，把离线缓存训练作为主实验结果。

