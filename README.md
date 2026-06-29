# MemSkillJittor

Jittor reproduction of the trainable Controller module in **MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents**.

This repository is prepared for the Jittor reproduction stage of the 新芽计划 report. The reproduction focuses on the neural and PPO-based part of MemSkill: the Controller. The LLM-driven Executor and Designer remain compatible with the original MemSkill pipeline.

## 1. Reproduction Scope

MemSkill contains three major modules:

| Module | Role in MemSkill | Jittor reproduction status |
|---|---|---|
| Controller | Selects Top-K memory skills and learns the policy with PPO | Reimplemented in Jittor |
| Executor | Uses LLM calls to generate memory actions and update MemoryBank | Reused from original MemSkill |
| Designer | Uses LLM feedback analysis to refine or propose skills | Reused from original MemSkill |

The Jittor implementation covers:

- state encoding network
- skill encoding network
- actor scoring network
- critic value network
- dynamic SkillBank padding and masks
- Top-K skill action sampling
- joint probability recomputation for selected Top-K skills
- PPO clipped loss
- value loss and entropy term
- optimizer update and checkpoint saving
- bridge back to the original MemSkill online training flow

## 2. Repository Structure

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
└── data/
    ├── locomo10.json
    └── locomo10_one.json
```

## 3. Environment Setup

### 3.1 Python Environment

Recommended:

```bash
python >= 3.10
jittor >= 1.3.11
torch with CUDA support for PyTorch baseline
```

The local Jittor GPU environment used in this reproduction:

```text
Python: /home/wsy/jittor_envs/stage3-jittor/bin/python
Jittor: 1.3.11.0
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Compiler: gcc/g++ 12
Jittor cache_name: gpu_gcc12
```

CUDA 12.2 may fail with system gcc/g++ 13. The training scripts therefore use gcc/g++ 12:

```bash
export CC=/home/wsy/local/gcc12/usr/bin/gcc-12
export CXX=/home/wsy/local/gcc12/usr/bin/g++-12
export cc_path=/home/wsy/local/gcc12/usr/bin/g++-12
export cache_name=gpu_gcc12
export DISABLE_MULTIPROCESSING=1
```

For CPU-only smoke tests:

```bash
export nvcc_path=''
```

### 3.2 API and Embedding Settings

Online MemSkill training requires an OpenAI-compatible API configuration:

```bash
MEMSKILL_MODEL=<chat-model-name>
MEMSKILL_DESIGNER_MODEL=<designer-model-name>
MEMSKILL_API_BASE=<openai-compatible-base-url>
MEMSKILL_API_KEY=<api-key>
```

Do not commit real API keys.

To avoid repeated network downloads, the experiment uses local HuggingFace cache for Qwen embedding:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 4. Data Preparation

The small-scale online experiments use LoCoMo data:

```text
data/locomo10.json
data/locomo10_one.json
```

`locomo10_one.json` is used for quick smoke tests. `locomo10.json` is used for small-scale full-flow comparison.

The purpose of the small-scale setting is to verify:

- the Jittor Controller can run inside the real MemSkill online pipeline
- PPO training signals are generated correctly
- Jittor and PyTorch produce comparable logs and behavior distributions

It is not intended to reproduce the final large-scale paper metrics.

## 5. Training Scripts

### 5.1 Jittor One-Batch Debug

```bash
./scripts/run_jittor_one_batch_debug.sh
```

Main output:

```text
jittor_controller_repro/runs/locomo_jittor_one_batch_debug/
```

This verifies that the Jittor Controller can consume real state/skill embeddings, select Top-K skills, and return PPO update records.

### 5.2 Jittor Full Small-Scale Online Training

```bash
./scripts/run_jittor_locomo_next_outer_epoch.sh 1
./scripts/run_jittor_locomo_next_outer_epoch.sh 2
./scripts/run_jittor_locomo_next_outer_epoch.sh 3
```

Underlying script:

```bash
./scripts/run_jittor_locomo_full_small_designer.sh
```

Main output:

```text
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/
```

Important artifacts:

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

### 5.3 PyTorch Baseline Training

```bash
./scripts/run_torch_locomo_next_outer_epoch.sh 1
./scripts/run_torch_locomo_next_outer_epoch.sh 2
./scripts/run_torch_locomo_next_outer_epoch.sh 3
```

Underlying script:

```bash
./scripts/run_torch_locomo_full_small_designer.sh
```

Main output:

```text
jittor_controller_repro/runs/locomo_torch_full_small_designer_epochwise/
```

The PyTorch run uses the original Controller logic and the same small-scale LoCoMo setting, so that the Jittor logs can be compared with the original implementation.

## 6. Testing Scripts

Unit tests:

```bash
pytest jittor_controller_repro/tests
```

Bridge smoke test:

```bash
python -m jittor_controller_repro.test_bridges
```

Top-K log probability test:

```bash
pytest jittor_controller_repro/tests/test_topk_logprob.py
```

Jittor Controller forward test:

```bash
pytest jittor_controller_repro/tests/test_jittor_controller_forward.py
```

## 7. Offline Controller Training

The offline trace path is used to remove LLM/API randomness and compare Controller behavior on fixed cached data.

Synthetic trace generation:

```bash
python -m jittor_controller_repro.data.synthetic_generator \
  --output jittor_controller_repro/runs/synthetic_trace.npz \
  --n-steps 128 \
  --state-dim 128 \
  --op-dim 128 \
  --action-top-k 3 \
  --seed 42
```

Train PyTorch and Jittor on the same trace:

```bash
python -m jittor_controller_repro.train_torch \
  --trace jittor_controller_repro/runs/synthetic_trace.npz \
  --log jittor_controller_repro/runs/torch_train.jsonl

python -m jittor_controller_repro.train_jittor \
  --trace jittor_controller_repro/runs/synthetic_trace.npz \
  --log jittor_controller_repro/runs/jittor_train.jsonl
```

Plot loss curves:

```bash
python -m jittor_controller_repro.plot_logs \
  --torch-log jittor_controller_repro/runs/torch_train.jsonl \
  --jittor-log jittor_controller_repro/runs/jittor_train.jsonl \
  --metric total_loss \
  --output jittor_controller_repro/runs/total_loss_curve.png
```

## 8. Training Logs and Curves

### 8.1 Online Small-Scale Training Curves

The following files record online training curves:

```text
jittor_controller_repro/runs/online_reward_curve.png
jittor_controller_repro/runs/online_policy_loss_curve.png
jittor_controller_repro/runs/online_value_loss_curve.png
jittor_controller_repro/runs/online_entropy_curve.png
```

These curves are used to verify that both PyTorch and Jittor complete online rollout, reward feedback, PPO loss computation, and Controller update.

### 8.2 Example Online Metrics

| backend | step | reward | policy_loss | value_loss | entropy | num_steps |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 0 | 0.229167 | -0.000253 | 0.004511 | 1.386294 | 19 |
| PyTorch | 1 | 0.309524 | -0.000064 | 0.008984 | 1.386294 | 19 |
| PyTorch | 2 | 0.241071 | -0.000044 | 0.003785 | 1.386293 | 19 |
| Jittor | 0 | 0.205357 | -0.000105 | 0.003178 | 1.386295 | 19 |
| Jittor | 1 | 0.142857 | -0.000154 | 0.000934 | 1.386294 | 19 |
| Jittor | 2 | 0.238095 | -0.000028 | 0.004407 | 1.386294 | 19 |

Because online Executor and Designer depend on LLM calls, exact action counts and rewards are not expected to be identical. The goal is to verify comparable training signal ranges and successful integration into the full MemSkill loop.

### 8.3 Behavior Distribution Artifacts

```text
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/selected_skill_stats.png
jittor_controller_repro/runs/locomo_jittor_full_small_designer_epochwise/memory_action_stats.png
```

These figures record:

- Controller selected skill distribution
- Executor memory action distribution

They are used to show that the Jittor Controller does not collapse to one action type and can drive the original Executor to produce valid memory updates.

## 9. Controller-Only Benchmark

End-to-end MemSkill training includes LLM API, retrieval, Executor and QA evaluation, so it does not directly reflect the Controller module speed. The controller-only benchmark freezes the trace and compares:

- forward / select_action
- evaluate_actions
- compute_ppo_loss
- optimizer step

Run:

```bash
python -m jittor_controller_repro.controller_benchmark
```

### 9.1 Synthetic Trace Result

| Metric | PyTorch | Jittor | Jittor / PyTorch | Speedup |
|---|---:|---:|---:|---:|
| forward_sec_mean | 0.233940 | 0.226198 | 0.967 | 1.034x |
| evaluate_sec_mean | 0.070696 | 0.014009 | 0.198 | 5.046x |
| loss_sec_mean | 0.038953 | 0.040351 | 1.036 | 0.965x |
| train_step_sec_mean | 0.024638 | 0.020121 | 0.817 | 1.224x |
| epoch_sec_mean | 0.370885 | 0.307038 | 0.828 | 1.208x |

### 9.2 Real LoCoMo Cached Trace Result

| Metric | PyTorch | Jittor | Jittor / PyTorch | Speedup |
|---|---:|---:|---:|---:|
| forward_sec_mean | 0.126375 | 0.126555 | 1.001 | 0.999x |
| evaluate_sec_mean | 0.051021 | 0.008546 | 0.168 | 5.970x |
| loss_sec_mean | 0.041755 | 0.043633 | 1.045 | 0.957x |
| train_step_sec_mean | 0.027974 | 0.022659 | 0.810 | 1.235x |
| epoch_sec_mean | 0.249953 | 0.209275 | 0.837 | 1.194x |

Interpretation:

- Jittor shows clear speedup in `evaluate_actions`.
- End-to-end Controller-only epoch speedup on real cached LoCoMo trace is about `1.19x`.
- Full MemSkill online training is dominated by API and LLM calls, so total wall-clock speedup is not expected to be large.

## 10. Implementation Notes

### 10.1 Dynamic SkillBank

The number of candidate skills may change during training. The Jittor Controller therefore pads candidate skill embeddings to a regular batch tensor and uses masks to exclude padding positions from softmax and PPO probability computation.

### 10.2 Top-K Joint Probability

The selected Top-K skills form an ordered action. During PPO update, the controller recomputes the joint probability of the same selected skills under the new policy.

The Jittor version rewrites part of the prefix probability computation with `jt.cumsum`, reducing Python-side iterative updates and keeping the probability path inside Jittor tensor computation.

### 10.3 Why Executor and Designer Are Not Rewritten in Jittor

Executor and Designer are primarily LLM-driven modules:

- Executor turns selected skills into concrete memory actions.
- Designer analyzes hard cases and refines or proposes skills.

They are not the main trainable neural network path. Therefore, this reproduction focuses on the Controller, while keeping Executor and Designer compatible with the original MemSkill workflow.

## 11. Known Limitations

- The current experiments use small-scale LoCoMo settings due to limited API budget and GPU memory.
- The results do not claim to reproduce the final large-scale paper metrics.
- Online reward and memory actions are affected by LLM output randomness.
- End-to-end speedup is limited by API calls and Executor/Designer latency.
- The clearest acceleration evidence is in the Controller-only benchmark, especially `evaluate_actions`.

## 12. Submission Checklist

- [x] Environment configuration documented
- [x] Data preparation documented
- [x] Jittor training scripts documented
- [x] PyTorch baseline scripts documented
- [x] Test scripts documented
- [x] Training logs documented
- [x] Loss curve paths documented
- [x] PyTorch/Jittor alignment results documented
- [x] Performance logs documented
- [x] Known limitations documented

## 13. Acknowledgement

This repository is based on the original MemSkill implementation and adds a Jittor reproduction of the trainable Controller module for course/report submission.
