# MemSkill Jittor Controller 复现说明

本目录开源的是 MemSkill 中可训练 `PPOController` 的 Jittor 复现版本，并保留了与原 PyTorch Controller 对齐的训练脚本、测试脚本、实验日志、loss 曲线和性能日志。

复现范围聚焦在 Controller 与 PPO 更新路径：

- `state_net` / `op_net` 编码 MLP
- actor-critic 打分与 value 预测
- 动态 SkillBank 的 padding + mask
- Top-K skill 选择与 joint log probability
- PPO clipped objective、value loss、entropy bonus
- Jittor optimizer update 与 checkpoint 保存

没有迁移到 Jittor 的部分包括 LLM Executor、Designer、SentenceTransformer/Qwen embedding、MemoryBank 数据结构和 QA 评估。这些模块仍沿用原 MemSkill 实现，Jittor Controller 通过 bridge / trainer 集成进入原流程。

## 目录索引

| 内容 | 路径 |
|---|---|
| Jittor Controller 主实现 | `jittor_controller_repro/models/jittor_controller.py` |
| PyTorch baseline 适配 | `jittor_controller_repro/baselines/original_torch_runner.py` |
| Executor / Designer bridge | `jittor_controller_repro/adapter/` |
| 离线 trace schema 与生成 | `jittor_controller_repro/data/` |
| Jittor / PyTorch 离线训练入口 | `jittor_controller_repro/train_jittor.py`, `jittor_controller_repro/train_torch.py` |
| Controller-only benchmark | `jittor_controller_repro/controller_benchmark.py` |
| 对齐与单元测试 | `jittor_controller_repro/tests/`, `jittor_controller_repro/test_bridges.py` |
| 在线训练脚本 | `scripts/run_jittor_*.sh`, `scripts/run_torch_*.sh` |
| 训练与性能日志 | `jittor_controller_repro/runs/` |

## 环境配置

### 1. 基础环境

从 MemSkill 仓库根目录运行：

```bash
cd /path/to/MemSkill
```

Jittor 复现需要：

- Python 3.10+
- Jittor
- PyTorch baseline 环境
- CUDA GPU 环境可选；无 GPU 时可切换 CPU 模式
- 原 MemSkill 的 LLM API 配置与 embedding/retriever 依赖

本地实验使用过的 Jittor GPU 环境：

```text
Python: /home/wsy/jittor_envs/stage3-jittor/bin/python
Jittor: 1.3.11.0
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Compiler: /home/wsy/local/gcc12/usr/bin/g++-12
Jittor cache_name: gpu_gcc12
```

CUDA 12.2 与系统 g++ 13 可能不匹配，因此脚本中显式指定了 gcc/g++ 12：

```bash
export CC=/home/wsy/local/gcc12/usr/bin/gcc-12
export CXX=/home/wsy/local/gcc12/usr/bin/g++-12
export cc_path=/home/wsy/local/gcc12/usr/bin/g++-12
export cache_name=gpu_gcc12
export DISABLE_MULTIPROCESSING=1
```

如果只做 CPU 测试：

```bash
export nvcc_path=''
```

### 2. API 与模型配置

在线 LoCoMo 训练需要在仓库根目录准备 `.env`：

```bash
MEMSKILL_MODEL=<chat-model-name>
MEMSKILL_DESIGNER_MODEL=<designer-model-name>
MEMSKILL_API_BASE=<openai-compatible-base-url>
MEMSKILL_API_KEY=<api-key>
# 如果网关使用 OpenAI Responses API，请设置：
MEMSKILL_WIRE_API=responses
```

不要把真实 API Key 提交到公开仓库。

为了减少网络依赖，完整训练脚本默认使用本地 HuggingFace 缓存的 Qwen embedding：

```text
/home/wsy/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

脚本中默认开启：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
```

## 数据准备

### 1. 在线 LoCoMo 小规模数据

在线完整流程使用：

```text
data/locomo10.json
```

资源受限时，也可使用：

```text
data/locomo10_one.json
```

本项目的完整对齐实验使用 LoCoMo10 小规模设置，目的是验证 Jittor Controller 可以跑通完整训练闭环，并与 PyTorch 版本在同量级指标上对齐；不是复现论文最终大规模指标。

### 2. 合成离线 trace

不依赖 API 和 encoder 的离线数据可由脚本生成：

```bash
python -m jittor_controller_repro.data.synthetic_generator \
  --output jittor_controller_repro/runs/synthetic_trace.npz \
  --n-steps 128 \
  --state-dim 128 \
  --op-dim 128 \
  --action-top-k 3 \
  --seed 42
```

### 3. 在线记录转离线 trace

如果已经有在线 `controller_trace_records.jsonl`，可以转换为固定离线 trace：

```bash
python -m jittor_controller_repro.data.collect_api_traces \
  --input-records path/to/controller_trace_records.jsonl \
  --output jittor_controller_repro/runs/api_cached_trace.npz
```

干跑验证：

```bash
python -m jittor_controller_repro.data.collect_api_traces \
  --dry-run-synthetic \
  --output jittor_controller_repro/runs/api_cached_trace.npz
```

## 训练脚本

### 1. Jittor one-batch debug

快速验证 Jittor Controller 能接入原 MemSkill 训练边界：

```bash
./scripts/run_jittor_one_batch_debug.sh
```

输出目录：

```text
jittor_controller_repro/runs/locomo_jittor_one_batch_debug/
```

### 2. Jittor 完整小规模训练

完整小规模配置接近论文训练流程：LoCoMo10、`batch-size=4`、`inner-epochs=5`、`ppo-epochs=2`、`action-top-k=3`、Designer enabled。

按 outer epoch 分轮运行，避免长时间训练中断：

```bash
./scripts/run_jittor_locomo_next_outer_epoch.sh 1
./scripts/run_jittor_locomo_next_outer_epoch.sh 2
./scripts/run_jittor_locomo_next_outer_epoch.sh 3
```

底层脚本：

```text
scripts/run_jittor_locomo_full_small_designer.sh
```

默认输出目录：

```text
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/
```

关键输出：

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
checkpoints/
api_cache/
```

### 3. PyTorch 对齐训练

PyTorch baseline 使用原 Controller，训练配置与 Jittor 脚本保持一致：

```bash
./scripts/run_torch_locomo_next_outer_epoch.sh 1
./scripts/run_torch_locomo_next_outer_epoch.sh 2
./scripts/run_torch_locomo_next_outer_epoch.sh 3
```

底层脚本：

```text
scripts/run_torch_locomo_full_small_designer.sh
```

默认输出目录：

```text
jittor_controller_repro/runs/locomo_torch_full_small_designer_epochwise/
```

### 4. 离线 Controller 训练

同一份 `.npz` trace 可分别用于 PyTorch 和 Jittor：

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

## 测试脚本

### 1. 单元测试与 bridge 测试

```bash
export nvcc_path=''
python -m pytest -s -q \
  jittor_controller_repro/test_bridges.py \
  jittor_controller_repro/tests
```

覆盖内容：

- checkpoint IO
- Jittor forward
- PPOBuffer
- Top-K log probability
- Executor / Designer bridge

### 2. 数值对齐测试

将 PyTorch linear weights 拷贝到 Jittor Controller，并比较 log probability、value、entropy 与 PPO loss：

```bash
python -m jittor_controller_repro.eval_parity \
  --trace jittor_controller_repro/runs/synthetic_trace.npz \
  --hidden-dim 64
```

### 3. 在线评估 demo

使用训练后 checkpoint 在固定 trace 上输出 selected skills、memory actions 与 final memory bank：

```bash
python -m jittor_controller_repro.eval_online_jittor \
  --checkpoint checkpoints/online_jittor_demo/online-jittor-demo_epoch_final.pt \
  --trace jittor_controller_repro/runs/api_cached_trace.npz \
  --output-dir jittor_controller_repro/runs/online_eval_demo \
  --executor-mode mock
```

示例产物目录：

```text
jittor_controller_repro/runs/eval_demo_smoke/
```

包含：

```text
selected_skills.jsonl
memory_actions.jsonl
final_memory_bank.json
demo_trace_report.md
```

### 4. 官方 eval-only 配置验证

论文原 LoCoMo 测试脚本中，eval-only 阶段与训练阶段的动作数量不同：训练常用 `--action-top-k 3`，测试脚本使用 `--action-top-k 7`。本仓库按该配置验证训练后 checkpoint 能否接入在线 MemoryBank 构建流程：

```text
--eval-only
--action-top-k 7
--session-mode fixed-length
--chunk-size 512
--chunk-overlap 64
--mem-top-k-eval 20
--reward-metric llm_judge
```

如果 API 网关使用 Responses API，需要设置：

```bash
export MEMSKILL_WIRE_API=responses
```

本地验证记录：

| Backend | checkpoint | run dir | eval 结果 | API cache | Executor action 分布 | memory bank |
|---|---|---|---|---:|---|---|
| Jittor | `locomo-jittor-full-small-designer-epochwise_epoch_final.pt` | `runs/eval_jittor_checkpoint_official_topk7/` | 完成第 1 条 test sample 的 `50/50` 个 fixed-length session；第 2 条 sample 跑到 `4/65` 后为节省 API 成本手动停止 | 54 | `INSERT: 97`, `UPDATE: 86` | 已保存 `memory_locomo_sample_conv-49_...topk_7...pkl` |
| PyTorch | `locomo-torch-full-small-designer-epochwise_epoch_final.pt` | `runs/eval_torch_checkpoint_official_topk7/` | 完成第 1 条 test sample 的 `50/50` 个 fixed-length session；第 2 条 sample 跑到 `2/65` 后为节省 API 成本手动停止 | 89 | `INSERT: 329`, `UPDATE: 74`, `NOOP: 3` | 已保存 `memory_locomo_sample_conv-49_...topk_7_retry_b8.pkl` |

说明：

- 这里验证的是训练后 checkpoint 能按官方 eval 参数完成在线 MemoryBank 构建，不声称复现论文最终大规模测试分数。
- `result.json` 没有生成，是因为完整 test set 的第二条 sample 和后续 QA scoring 未继续运行。
- PyTorch 首次使用 `encode-batch-size=64` 时在 `39/50` 处遇到 CUDA unknown error；重试时将 `encode-batch-size` 降回训练时的 `8` 后完成第一条 sample。
- 两个后端的 API 返回均为正常结构化 memory actions，没有出现 HTML 返回或大面积解析失败。

### 5. conv-49 / QA20 小规模 QA 跑分

为了进一步验证保存下来的 MemoryBank 是否能用于下游问答，我们在同一条 LoCoMo 测试 trace 上做了一个小规模 QA 评估：

```text
测试样本: conv-49
问题数量: 前 20 个 QA
评估方式: 复用各自已构建的 conv-49 MemoryBank，再进行 QA answer + LLM judge
Judge 模型: gpt-5.5
```

结果如下：

| Backend | checkpoint | MemoryBank 条数 | F1 | LLM Judge |
|---|---|---:|---:|---:|
| Jittor | `locomo-jittor-full-small-designer-epochwise_epoch_final.pt` | 83 | 0.5332 | 0.6750 |
| PyTorch | `locomo-torch-full-small-designer-epochwise_epoch_final.pt` | 180 | 0.6051 | 0.8000 |

这组结果说明 Jittor checkpoint 可以完成从 MemoryBank 构建到 QA 评估的完整推理链路，但它不是严格的同 SkillBank 对齐测试。原因是 Executor 和 Designer 仍由在线 LLM 驱动，两次训练结束时 Designer 演化出的 skill 不同：

| Backend | Designer 生成的新 skill | 影响 |
|---|---|---|
| Jittor | `capture_participation_event` | 更偏向捕捉参与活动、演讲、志愿、公开出现等事件 |
| PyTorch | `capture_visual_details` | 更偏向捕捉照片、物体、场景等视觉细节 |

因此，测试阶段 Executor 接收到的 skill prompt 并不完全相同，最终 MemoryBank 条数和 QA 分数会受到不同 Designer 产物的影响。这里的结论应理解为：Jittor 版本能够接入完整推理与 QA 评估流程，并取得有效结果；但上述 QA 分数不用于声称 Jittor 与 PyTorch 在最终测试集上严格等价。

## 与 PyTorch 对齐的实验 Log

已整理的一轮完整小规模对齐归档：

```text
jittor_controller_repro/runs/compare_jittor_torch_full_small_epoch1_20260624/
```

目录内容：

| 文件 | 说明 |
|---|---|
| `README.md` | 对齐实验摘要 |
| `raw_metrics/jittor_metrics.csv` | Jittor 原始训练指标 |
| `raw_metrics/torch_metrics.csv` | PyTorch 原始训练指标 |
| `logs/jittor_key_events.txt` | Jittor 关键日志摘录 |
| `logs/torch_key_events.txt` | PyTorch 关键日志摘录 |
| `checkpoint_index/checkpoints.txt` | checkpoint 路径和大小 |
| `derived/comparison_metrics.csv` | 每个 inner epoch 的对比表 |
| `derived/summary.json` | 机器可读摘要 |

核心结果：

| Backend | metric rows | reward sequence | reward mean | final reward | trace records | checkpoints | Designer changes |
|---|---:|---|---:|---:|---:|---:|---|
| Jittor | 5 | 0.3670, 0.3711, 0.3794, 0.3063, 0.3931 | 0.3634 | 0.3931 | 482 | 2 | refined insert/update; added `capture_participation_event` |
| PyTorch | 5 | 0.4217, 0.3564, 0.3942, 0.3425, 0.3363 | 0.3702 | 0.3363 | 512 | 2 | refined insert; added `capture_visual_details` |

PPO 更新信号：

| Backend | policy_loss mean | value_loss mean | entropy mean | approx_kl mean | clip_frac mean | return_mean mean |
|---|---:|---:|---:|---:|---:|---:|
| Jittor | -0.00001503 | 0.020824 | 1.386294 | -0.00000588 | 0.0000 | 0.192293 |
| PyTorch | -0.00003529 | 0.015475 | 1.386294 | 0.00001003 | 0.0000 | 0.206896 |

说明：

- 两个后端都完成了 `5` 个 inner epoch。
- 两个后端都保存 checkpoint。
- 两个后端都完成 Designer evolution。
- 在线 API、LLM 输出、随机采样和 Designer 演化会带来差异，因此该实验用于证明闭环可运行和指标同量级，不用于声称逐数值完全一致。

## Loss 曲线与可视化

Jittor 在线完整小规模训练曲线：

```text
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/reward_curve.png
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/policy_loss_curve.png
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/value_loss_curve.png
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/entropy_curve.png
```

训练过程可视化：

```text
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/selected_skill_stats.png
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/memory_action_stats.png
```

离线 loss 曲线：

```text
jittor_controller_repro/runs/total_loss_curve.png
jittor_controller_repro/runs/value_loss_curve.png
```

在线训练阶段历史曲线：

```text
jittor_controller_repro/runs/online_reward_curve.png
jittor_controller_repro/runs/online_policy_loss_curve.png
jittor_controller_repro/runs/online_value_loss_curve.png
jittor_controller_repro/runs/online_entropy_curve.png
```

## 性能 Log

端到端 MemSkill 训练包含 LLM API、Retriever、Executor、QA 评估等外部耗时，不能直接体现 Controller 框架迁移后的模块性能。因此我们另外做了 Controller-only benchmark，只比较：

- `forward/select_action`
- `evaluate_actions`
- `compute_ppo_loss`
- `optimizer step`

性能摘要：

```text
jittor_controller_repro/runs/controller_benchmark_summary.md
```

原始性能日志：

```text
jittor_controller_repro/runs/controller_benchmark_locomo_real_gpu/
jittor_controller_repro/runs/controller_benchmark_synthetic_fullbatch_gpu/
```

真实 LoCoMo cached trace 结果：

| 指标 | PyTorch | Jittor | Jittor / PyTorch | 加速比 |
|---|---:|---:|---:|---:|
| forward_sec_mean | 0.126375 | 0.126555 | 1.001 | 0.999x |
| evaluate_sec_mean | 0.051021 | 0.008546 | 0.168 | 5.970x |
| loss_sec_mean | 0.041755 | 0.043633 | 1.045 | 0.957x |
| train_step_sec_mean | 0.027974 | 0.022659 | 0.810 | 1.235x |
| epoch_sec_mean | 0.249953 | 0.209275 | 0.837 | 1.194x |

Synthetic full-batch GPU 结果：

| 指标 | PyTorch | Jittor | Jittor / PyTorch | 加速比 |
|---|---:|---:|---:|---:|
| forward_sec_mean | 0.233940 | 0.226198 | 0.967 | 1.034x |
| evaluate_sec_mean | 0.070696 | 0.014009 | 0.198 | 5.046x |
| loss_sec_mean | 0.038953 | 0.040351 | 1.036 | 0.965x |
| train_step_sec_mean | 0.024638 | 0.020121 | 0.817 | 1.224x |
| epoch_sec_mean | 0.370885 | 0.307038 | 0.828 | 1.208x |

结论：

- Jittor Controller 在固定离线轨迹上可以独立完成 PPO 训练路径。
- `evaluate_actions` 阶段在真实 LoCoMo cached trace 上约 `5.97x`。
- 真实 LoCoMo cached trace 的 Controller-only epoch 约 `1.19x`。
- 端到端训练总体时间仍受 API 与 LLM 输出影响，不能据此声称完整系统大幅加速。

## 完整闭环 Jittor 自进化记录

除一轮 Jittor/PyTorch 对齐外，还保留了 Jittor 接入完整 MemSkill self-evolution 的记录：

```text
jittor_controller_repro/runs/full_flow_jittor_designer/summary.md
jittor_controller_repro/runs/full-flow-jittor-evolve2/summary.md
```

其中 `full-flow-jittor-evolve2` 完成了小规模两阶段自进化：

```text
初始 SkillBank
→ Jittor PPOController 选择 Top-K skills
→ Executor 更新 MemoryBank
→ QA reward
→ PPO 更新 Controller
→ Designer 演化 SkillBank
→ 下一阶段在新版 SkillBank 上继续训练
```

该运行中，Designer 第一次新增 skill 后 reward 下降，系统触发 rollback，并基于负反馈生成更窄的新 skill。这说明 Jittor Controller 已接入 MemSkill 的 closed-loop self-evolution，而不是孤立的 toy controller。

## 资源受限说明

当前开源记录没有声称复现论文最终大规模指标。原因：

- 完整 LoCoMo 训练依赖 LLM API、QA 评估和 Designer，多轮运行时间较长。
- API 输出、LLM judge、随机采样和 Designer 演化会导致在线训练结果不可逐数值复现。
- 本地 GPU/时间资源有限，因此主对齐结果采用 LoCoMo10 小规模训练和固定离线 Controller benchmark。

在资源受限条件下，本仓库提供了三层证据：

1. **系统可运行**：Jittor Controller 能接入原 MemSkill 完整在线训练流程。
2. **结果同量级**：小规模 LoCoMo 训练中，Jittor 与 PyTorch 都完成 PPO 更新、checkpoint 保存和 Designer evolution。
3. **模块性能可测**：固定离线 trace 上，Controller-only benchmark 给出 PyTorch/Jittor 的可复现实验日志和性能日志。

## 复现检查清单

- [x] 环境配置说明
- [x] 数据准备脚本
- [x] Jittor 训练脚本
- [x] PyTorch 对齐训练脚本
- [x] 测试脚本
- [x] 与 PyTorch 对齐的实验 Log
- [x] Controller-only 性能 Log
- [x] 训练过程 Log
- [x] Loss / reward / entropy 曲线
- [x] Memory action 与 selected skill 可视化
- [x] 资源受限条件下的小规模训练效果说明
