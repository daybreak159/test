# MemSkillJittor

本仓库用于新芽计划第三阶段汇报：基于 **Jittor** 复现论文 **MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents** 中的可训练 Controller 模块，并将其接回原 MemSkill 的在线记忆构建流程。

> 说明：本仓库重点复现的是 Controller 的神经网络与 PPO 训练路径。Executor 和 Designer 主要由 LLM 驱动，因此保留原 MemSkill 实现，通过 bridge 与 Jittor Controller 对接。

## 1. 复现范围

MemSkill 整体包含三个核心模块：

| 模块 | 原论文中的作用 |
|---|---|
| Controller | 根据当前状态从 SkillBank 中选择 Top-K memory skills，并通过 PPO 学习选择策略 |
| Executor | 根据选中的 skills 调用 LLM 生成 memory actions，并更新 MemoryBank |
| Designer | 根据失败案例调用 LLM 分析并调整 SkillBank |

本仓库重点复现 Controller 的 Jittor 训练路径；Executor 和 Designer 仍沿用原 MemSkill 流程，并通过 bridge 与 Jittor Controller 对接。

Jittor 版本实现内容包括：

- 状态编码网络
- 技能编码网络
- 策略打分网络
- 价值估计网络
- 动态 SkillBank 的 padding 与 mask
- Top-K skill action 采样
- Top-K 动作联合概率重算
- PPO clipped loss
- value loss 与 entropy 项
- Jittor optimizer 更新
- checkpoint 保存
- 与原 MemSkill 在线训练流程的 bridge 接入

## 2. 目录结构

```text
MemskillJittor/
├── README.md
├── main.py
├── src/
│   ├── controller.py
│   ├── trainer.py
│   ├── executor.py
│   ├── designer.py
│   └── data_processing/
├── jittor_controller_repro/
│   ├── models/jittor_controller.py
│   ├── adapter/
│   ├── baselines/
│   ├── data/
│   ├── tests/
│   ├── train_jittor.py
│   ├── train_torch.py
│   ├── controller_benchmark.py
│   ├── plot_logs.py
│   └── runs/
├── scripts/
│   ├── run_jittor_one_batch_debug.sh
│   ├── run_jittor_locomo_full_small_designer.sh
│   ├── run_jittor_locomo_next_outer_epoch.sh
│   ├── run_torch_locomo_full_small_designer.sh
│   └── run_torch_locomo_next_outer_epoch.sh
├── data/
│   ├── locomo10.json
│   └── locomo10_one.json
└── assets/
    └── figures/
```

## 3. 环境配置

推荐环境：

```text
Python >= 3.10
Jittor >= 1.3.11
PyTorch CUDA 环境，用于原版 baseline 对齐
OpenAI-compatible API，用于在线 Executor / Designer / QA 评估
```

参考环境：

```text
Python: 3.10+
Jittor: 1.3.11.0
GPU: CUDA-capable NVIDIA GPU
Compiler: gcc/g++ version compatible with the installed CUDA toolkit
```

如果 Jittor 首次编译 CUDA kernel 时提示 `nvcc` 与系统 `gcc/g++` 版本不匹配，请安装与当前 CUDA 版本兼容的编译器，并在运行前显式指定编译器路径。例如：

```bash
export CC=/path/to/compatible/gcc
export CXX=/path/to/compatible/g++
export cc_path=/path/to/compatible/g++
export cache_name=jittor_gpu_cache
export DISABLE_MULTIPROCESSING=1
```

如果只做 CPU smoke test，可以关闭 nvcc：

```bash
export nvcc_path=''
```

## 4. API 与本地模型缓存

在线 MemSkill 流程需要 LLM API。请在本地 `.env` 或脚本环境变量中配置：

```bash
MEMSKILL_MODEL=<chat-model-name>
MEMSKILL_DESIGNER_MODEL=<designer-model-name>
MEMSKILL_API_BASE=<openai-compatible-base-url>
MEMSKILL_API_KEY=<api-key>
```

不要提交真实 API Key。

如果已经提前下载好 embedding 模型，可以开启 HuggingFace / Transformers 离线模式，减少训练时的网络依赖：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
```

## 5. 数据准备

小规模在线训练使用 LoCoMo 数据：

```text
data/locomo10.json
data/locomo10_one.json
```

其中：

- `locomo10_one.json` 用于快速调试；
- `locomo10.json` 用于小规模完整流程训练。

本项目受 API 调用成本和显存限制，目标是验证 Jittor Controller 可以跑通完整训练闭环，并与 PyTorch 原流程在训练记录和行为分布上保持相近，而不是复现论文最终大规模指标。

## 6. 训练脚本

### 6.1 Jittor one-batch debug

```bash
./scripts/run_jittor_one_batch_debug.sh
```

输出目录：

```text
jittor_controller_repro/runs/locomo_jittor_one_batch_debug/
```

这个脚本用于快速验证 Jittor Controller 是否能够接入原 MemSkill 边界，消费真实 state / skill embedding，输出 Top-K skills，并产生 PPO 更新记录。

### 6.2 Jittor 完整小规模训练

为了避免长时间运行中断，完整训练按 outer epoch 分轮执行：

```bash
./scripts/run_jittor_locomo_next_outer_epoch.sh 1
./scripts/run_jittor_locomo_next_outer_epoch.sh 2
./scripts/run_jittor_locomo_next_outer_epoch.sh 3
```

底层脚本：

```bash
./scripts/run_jittor_locomo_full_small_designer.sh
```

输出目录：

```text
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/
```

主要输出：

```text
train_epoch_1.log
train_epoch_2.log
train.log
metrics.csv
metrics.jsonl
controller_trace_records.jsonl
reward_curve.png
policy_loss_curve.png
value_loss_curve.png
entropy_curve.png
selected_skill_stats.png
memory_action_stats.png
training_summary.md
```

### 6.3 PyTorch 原版对齐训练

PyTorch baseline 使用原版 Controller，并尽量保持与 Jittor 版本一致的小规模设置：

```bash
./scripts/run_torch_locomo_next_outer_epoch.sh 1
./scripts/run_torch_locomo_next_outer_epoch.sh 2
./scripts/run_torch_locomo_next_outer_epoch.sh 3
```

底层脚本：

```bash
./scripts/run_torch_locomo_full_small_designer.sh
```

输出目录：

```text
jittor_controller_repro/runs/locomo_torch_full_small_designer_epochwise/
```

## 7. 测试脚本

单元测试：

```bash
pytest jittor_controller_repro/tests
```

Bridge smoke test：

```bash
python -m jittor_controller_repro.test_bridges
```

Top-K 联合概率测试：

```bash
pytest jittor_controller_repro/tests/test_topk_logprob.py
```

Jittor Controller forward 测试：

```bash
pytest jittor_controller_repro/tests/test_jittor_controller_forward.py
```

## 8. 离线 Controller 训练

在线训练会受到 API、LLM 输出和采样随机性的影响。为了更稳定地比较 PyTorch 与 Jittor Controller，可以将轨迹缓存为固定 `.npz`，再分别运行两种后端。

生成合成 trace：

```bash
python -m jittor_controller_repro.data.synthetic_generator \
  --output jittor_controller_repro/runs/synthetic_trace.npz \
  --n-steps 128 \
  --state-dim 128 \
  --op-dim 128 \
  --action-top-k 3 \
  --seed 42
```

分别训练 PyTorch 与 Jittor：

```bash
python -m jittor_controller_repro.train_torch \
  --trace jittor_controller_repro/runs/synthetic_trace.npz \
  --log jittor_controller_repro/runs/torch_train.jsonl

python -m jittor_controller_repro.train_jittor \
  --trace jittor_controller_repro/runs/synthetic_trace.npz \
  --log jittor_controller_repro/runs/jittor_train.jsonl
```

绘制 loss 曲线：

```bash
python -m jittor_controller_repro.plot_logs \
  --torch-log jittor_controller_repro/runs/torch_train.jsonl \
  --jittor-log jittor_controller_repro/runs/jittor_train.jsonl \
  --metric total_loss \
  --output jittor_controller_repro/runs/total_loss_curve.png
```

## 9. 训练曲线与结果展示

### 9.1 PyTorch / Jittor 离线 loss 对齐

下图展示固定离线 trace 上的 total loss 曲线。它用于说明在去除 API 和 LLM 随机性后，Jittor Controller 与 PyTorch 原版 Controller 可以产生相近的训练趋势。

![offline total loss](assets/figures/offline_total_loss_curve.png)

### 9.2 Jittor 在线训练曲线

在线训练曲线用于展示 Jittor Controller 已经接入真实 MemSkill 流程，并能够产生 PPO 训练信号。

Value Loss：

![online value loss](assets/figures/online_jittor_value_loss_curve.png)

Policy Loss：

![online policy loss](assets/figures/online_jittor_policy_loss_curve.png)

需要说明的是，在线训练中 Executor 和 Designer 都涉及 LLM 调用，因此 reward、memory action 数量和 loss 曲线不会与 PyTorch 逐点完全一致。这里重点验证的是训练流程可运行、指标数量级合理、PPO 更新信号正常。

### 9.3 技能选择与记忆操作分布

Controller 选择的 skill 分布：

![selected skill stats](assets/figures/selected_skill_stats.png)

Executor 最终执行的 memory action 分布：

![memory action stats](assets/figures/memory_action_stats.png)

这些图用于说明 Jittor Controller 没有坍缩到单一技能，而是能够在真实在线流程中调用多种候选 skill，并驱动原 Executor 生成有效 memory actions。

## 10. 在线训练指标示例

| backend | step | reward | policy_loss | value_loss | entropy | num_steps |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 0 | 0.229167 | -0.000253 | 0.004511 | 1.386294 | 19 |
| PyTorch | 1 | 0.309524 | -0.000064 | 0.008984 | 1.386294 | 19 |
| PyTorch | 2 | 0.241071 | -0.000044 | 0.003785 | 1.386293 | 19 |
| Jittor | 0 | 0.205357 | -0.000105 | 0.003178 | 1.386295 | 19 |
| Jittor | 1 | 0.142857 | -0.000154 | 0.000934 | 1.386294 | 19 |
| Jittor | 2 | 0.238095 | -0.000028 | 0.004407 | 1.386294 | 19 |

这些数据说明两种实现都完成了在线 rollout、reward 回传、PPO loss 计算和 Controller 更新。由于在线过程包含 LLM 输出和采样随机性，本表不用于证明严格数值相等，而用于展示功能对齐和训练信号正常。

## 11. Controller-only 性能测试

端到端 MemSkill 训练包含 LLM API、Retriever、Executor 和 QA 评估，无法直接体现 Controller 模块迁移后的性能。因此我们额外做了 Controller-only benchmark，只比较：

- forward / select_action
- evaluate_actions
- compute_ppo_loss
- optimizer step

运行：

```bash
python -m jittor_controller_repro.controller_benchmark
```

### 11.1 Synthetic Trace

| Metric | PyTorch | Jittor | Jittor / PyTorch | Speedup |
|---|---:|---:|---:|---:|
| forward_sec_mean | 0.233940 | 0.226198 | 0.967 | 1.034x |
| evaluate_sec_mean | 0.070696 | 0.014009 | 0.198 | 5.046x |
| loss_sec_mean | 0.038953 | 0.040351 | 1.036 | 0.965x |
| train_step_sec_mean | 0.024638 | 0.020121 | 0.817 | 1.224x |
| epoch_sec_mean | 0.370885 | 0.307038 | 0.828 | 1.208x |

### 11.2 Real LoCoMo Cached Trace

| Metric | PyTorch | Jittor | Jittor / PyTorch | Speedup |
|---|---:|---:|---:|---:|
| forward_sec_mean | 0.126375 | 0.126555 | 1.001 | 0.999x |
| evaluate_sec_mean | 0.051021 | 0.008546 | 0.168 | 5.970x |
| loss_sec_mean | 0.041755 | 0.043633 | 1.045 | 0.957x |
| train_step_sec_mean | 0.027974 | 0.022659 | 0.810 | 1.235x |
| epoch_sec_mean | 0.249953 | 0.209275 | 0.837 | 1.194x |

结论：

- Jittor 在 `evaluate_actions` 阶段表现出明显优势；
- 真实 LoCoMo cached trace 上，Controller-only 单 epoch 约 `1.19x`；
- 端到端在线训练受 API 和 LLM 调用影响较大，因此整体加速不明显，优势主要体现在 Controller 局部计算。

## 12. 实现细节

### 12.1 动态 SkillBank

训练过程中 SkillBank 数量会变化。Jittor Controller 会将候选 skill embedding padding 成规则 batch tensor，并用 mask 避免 padding 位置参与 softmax 和 PPO 概率计算。

### 12.2 Top-K 联合概率

Top-K skills 组成一个有顺序的动作。PPO 更新阶段需要在新策略下重新计算同一组 Top-K skills 的联合概率。

Jittor 版本将部分前缀概率计算改写为 `jt.cumsum`，减少 Python 循环，并让概率计算保持在 Jittor 张量化路径中。

### 12.3 为什么不把 Executor 和 Designer 改成 Jittor

Executor 和 Designer 主要是 LLM 驱动：

- Executor 根据选中的 skill 生成具体 memory action；
- Designer 根据 hard cases 反思并修改或新增 skill。

这两个模块不是主要的可训练神经网络路径。因此，本次复现重点放在 Controller，而不是将所有流程都改写成 Jittor。

## 13. 当前局限

- 当前实验使用 LoCoMo 小规模设置，受 API 成本和显存限制，没有复现论文最终大规模指标；
- 在线训练结果会受到 LLM 输出、API 延迟和采样随机性的影响；
- 端到端加速并不明显，性能优势主要体现在 Controller 局部计算；
- README 中的曲线和日志用于说明功能对齐、训练信号和性能趋势，不应解读为 Jittor 版本全面优于 PyTorch。

## 14. 提交检查清单

- [x] 环境配置说明
- [x] 数据准备说明
- [x] Jittor 训练脚本说明
- [x] PyTorch baseline 脚本说明
- [x] 测试脚本说明
- [x] 训练日志说明
- [x] loss 曲线展示
- [x] PyTorch / Jittor 对齐结果
- [x] 性能测试结果
- [x] 当前局限说明

## 15. 致谢

本仓库基于原 MemSkill 实现进行课程复现，重点补充 Jittor Controller、训练脚本、测试脚本、实验日志和可视化结果。
