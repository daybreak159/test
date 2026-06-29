# 最后一部分 PPT 计划草稿：实验验证、训练结果与总结思考

## 总体主线

最后 9 页围绕一个问题展开：

> 我们不是只把代码跑起来，而是从“真实闭环训练”和“离线可控验证”两个层面证明 Jittor Controller 复现有效。

整体顺序：

```text
验证目标
→ 实验设置
→ 在线闭环
→ 在线结果
→ 行为对齐
→ 离线数据集
→ 模块 Benchmark
→ 性能结果
→ 总结思考
```

## 第 1 页：实验验证目标

标题：

```text
· 实验验证：Jittor 复现是否真正接入 MemSkill 训练闭环
```

核心内容：

```text
在线实验：验证完整系统能跑通
离线实验：隔离 API 影响，验证 Controller 本身
模块测试：观察 Jittor 在 PPOController 中的计算表现
```

视觉建议：

```text
Online Full Loop
        ↓
Offline Trace Replay
        ↓
Controller-only Benchmark
```

页内结论：

```text
实验验证分为系统级闭环和模块级对照，分别回答“能不能跑”和“迁移是否可靠”。
```

## 第 2 页：实验环境与数据设置

标题：

```text
· 实验设置：LoCoMo 小规模真实训练 + 离线轨迹复现
```

建议表格字段：

```text
数据集：LoCoMo10
后端：PyTorch / Jittor
Controller：PPOController
Top-K：3
Reward：F1
Embedding：Qwen3-Embedding-0.6B
GPU：RTX 4070 Laptop
```

如果完整小规模流程跑完，再补：

```text
outer-epochs = 3
inner-epochs = 5
batch-size = 4
ppo-epochs = 2
```

页内结论：

```text
实验保持数据、动作空间、奖励指标一致，只切换 Controller 后端。
```

## 第 3 页：在线完整闭环实验

标题：

```text
· 在线实验：Jittor Controller 接入真实训练流程
```

核心流程：

```text
Trace / Span
    ↓
Jittor Controller 选择 Top-K Skill
    ↓
Executor 更新 MemoryBank
    ↓
QA 评估产生 Reward
    ↓
PPO Loss 更新 Controller
```

页内结论：

```text
Jittor 版本已经不是单独跑 loss，而是进入了 MemSkill 的真实训练闭环。
```

视觉建议：

横向闭环流程图，风格延续前面的 Controller / Reward 页。

## 第 4 页：在线训练结果

标题：

```text
· 在线结果：Jittor 与 PyTorch 行为保持同量级
```

当前 one-batch 可放：

```text
PyTorch reward = 0.41479
Jittor reward = 0.41527

entropy 基本一致
topk_entropy 基本一致
topk_mass 基本一致
policy_loss 均接近 0
```

页内结论：

```text
在线结果说明 Jittor 迁移没有破坏 Controller 的策略分布与 PPO 更新逻辑。
```

后续替换空间：

如果完整小规模流程跑完，本页替换成 reward / value_loss / entropy 曲线。

## 第 5 页：PyTorch 与 Jittor 训练行为对齐

标题：

```text
· 行为对齐：Skill 选择分布没有明显偏移
```

建议图表：

```text
PyTorch selected skills:
update / delete / insert / noop

Jittor selected skills:
update / insert / noop / delete
```

再放 Executor action 分布：

```text
INSERT / UPDATE / DELETE
```

页内结论：

```text
Jittor Controller 产生的 Skill 选择行为与 PyTorch 原实现处于相同分布范围。
```

注意：

这里不说“完全一致”，因为在线实验包含 API 和采样差异。

## 第 6 页：离线轨迹数据集设计

标题：

```text
· 离线数据集：把在线训练固化为可复现实验
```

核心流程：

```text
真实在线训练
    ↓
记录 controller_trace_records.jsonl
    ↓
提取 state / op_embs / action / log_prob / value / reward
    ↓
保留 return / advantage
    ↓
保存为 offline trace .npz
```

可标注文件：

```text
locomo_torch_real_cached_trace.npz
119 steps
state_dim = 2048
op_dim = 1024
action_top_k = 3
```

页内结论：

```text
离线轨迹去除了 API 和 LLM 随机性，让 PyTorch / Jittor 可以在同一批证据上比较。
```

## 第 7 页：Controller-only Benchmark 设计

标题：

```text
· 模块测试：只比较 PPOController 的计算路径
```

测试对象：

```text
forward / select_action
evaluate_actions
compute_ppo_loss
optimizer step
```

页内结论：

```text
端到端训练受 API 影响较大，因此模块加速需要单独观察 Controller。
```

视觉建议：

中间放 Controller 结构图，旁边标出四个计时点。

## 第 8 页：Controller-only 性能结果

标题：

```text
· 性能结果：Jittor 在批量动作评估阶段优势明显
```

真实 LoCoMo cached trace 结果：

```text
forward:          1.00x
evaluate_actions: 5.97x
compute_loss:     0.96x
optimizer step:   1.24x
total epoch:      1.19x
```

页内结论：

```text
Jittor 的主要优势体现在批量 evaluate_actions 与训练 step，整体 Controller-only epoch 约提升 1.19x。
```

注意表述：

```text
这里是 Controller-only epoch，不是端到端 MemSkill 总训练耗时。
```

## 第 9 页：总结思考与后续工作

标题：

```text
· 总结思考：复现重点从“跑通”走向“可验证”
```

总结三点：

```text
1. Jittor Controller 已接入真实 MemSkill 训练闭环
2. 离线轨迹让 PyTorch / Jittor 对比更加可控
3. 模块测试显示 Jittor 在批量动作评估上有优化空间
```

局限：

```text
完整论文级指标仍需要更大数据和更长训练
端到端耗时主要受 API / Executor 影响
Jittor 首次运行存在 JIT 编译开销
```

未来工作：

```text
扩大 LoCoMo 训练规模
加入 Designer 演化实验
继续优化动态 SkillBank 的 batch 组织方式
```

页内结论：

```text
本阶段完成了 Jittor Controller 的真实闭环复现，并建立了可复现实验验证路径。
```

## 已有实验支撑材料

在线 one-batch 对齐：

```text
PyTorch reward = 0.41479
Jittor reward = 0.41527
entropy / topk_entropy / topk_mass 基本一致
```

真实 LoCoMo cached trace benchmark：

```text
forward:          PyTorch 0.1264s / Jittor 0.1266s
evaluate_actions: PyTorch 0.0510s / Jittor 0.0085s
compute_loss:     PyTorch 0.0418s / Jittor 0.0436s
optimizer step:   PyTorch 0.0280s / Jittor 0.0227s
total epoch:      PyTorch 0.2500s / Jittor 0.2093s
```

相关文件：

```text
jittor_controller_repro/runs/controller_benchmark_summary.md
jittor_controller_repro/runs/locomo_torch_real_cached_trace.npz
jittor_controller_repro/controller_benchmark.py
jittor_controller_repro/data/collect_api_traces.py
```
