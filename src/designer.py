"""
Designer: Evolves operation bank based on failure case analysis
"""
import json
import re
import sys
import os
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_repair import repair_json
from llm_utils import get_llm_response_via_api
from prompts.designer_prompts import (
    DESIGNER_ANALYSIS_PROMPT,
    DESIGNER_REFLECTION_PROMPT,
    DESIGNER_REFINEMENT_PROMPT,
)
from src.operation_bank import Operation


@dataclass
class DesignerCase:
    """
    保存一条给 Designer 分析用的 QA 评估记录。

    它不是原始数据集中的一条裸 QA，而是 trainer._evaluate_qa() 在评估后
    把原始 QA、当前 memory 检索结果、模型回答、评分结果和训练上下文打包出来的结构。
    Designer 后续会从这些记录中筛选失败样本，分析 memory skill 应该如何新增或修改。
    """
    # -----------------------------
    # 原始 QA 相关字段
    # -----------------------------
    # 当前问题的唯一 ID。通常由 conversation_id 和原始 qa_idx 拼出来，
    # 例如 "conv-26_2"，表示 conv-26 这个样本里的第 2 条 QA。
    # CaseCollector 会优先用它作为去重 key，判断同一个问题是否重复失败。
    query_id: str

    # 问题文本，来自原始 qa_item['question']。
    # Designer 聚类时当前主要用这个字段做 embedding，所以它决定失败 case 的语义分组。
    question: str

    # 标准答案，来自 evaluator.get_ground_truth(qa)。
    # 如果原始答案是 list，会在 trainer._evaluate_qa() 中拼成字符串再放进来。
    ground_truth: str

    # 支撑标准答案的证据位置，LoCoMo 中常见形式如 ["D1:9", "D1:11"]。
    # 这个字段主要用于分析和调试，帮助定位答案在原始对话中的来源。
    evidence: Optional[str] = None

    # 问题类别，数据集相关字段；LoCoMo 中表示不同问题类型。
    # 例如某些类别更偏事实、时间、聚合或多跳问题。
    category: Optional[int] = None

    # -----------------------------
    # 当前评估时的 memory 相关字段
    # -----------------------------
    # 回答这道题时，整个 memory_bank 的序列化快照。
    # 实际结构来自 memory_bank.to_dict()，通常包含 memories、retriever_name、top_k、timestep。
    # 注意：注解写成 Optional[List[Dict]]，但当前实现实际传入的是一个 dict 快照。
    memory_bank_snapshot: Optional[List[Dict]] = None

    # 用 question 检索 memory_bank 后返回的 top-k memory 文本列表。
    # Designer prompt 里会展示这些 memory，用来判断是“没存好”还是“检索没检到”。
    retrieved_memories: Optional[List[str]] = None

    # retrieved_memories 对应在 memory_bank.memories 中的下标。
    # 例如 retrieved_indices=[24,42,1] 表示返回的三条 memory 来自 memory_bank 的这些位置。
    retrieved_indices: Optional[List[int]] = None

    # -----------------------------
    # 模型回答和评分结果
    # -----------------------------
    # 回答 LLM 基于 retrieved_memories 对 question 生成的答案。
    # 这是和 ground_truth 对比后判断失败原因的核心字段。
    prediction: str = ""

    # 当前 QA 是否被认为答对。
    # F1 模式下由 f1_score >= designer_f1_threshold 得到；
    # llm_judge 模式下由 llm_judge_score == 1.0 得到。
    # CaseCollector.add_case() 会直接丢弃 is_correct=True 的 case。
    is_correct: bool = False

    # prediction 和 ground_truth 的 F1 分数。
    # reward_metric 不是 llm_judge 时，它既用于训练 reward，也用于判断是否进入失败池。
    f1_score: float = 0.0

    # LLM judge 给出的回答质量分数，通常是 0.0、0.5 或 1.0。
    # 只有 reward_metric == "llm_judge" 时，它才直接决定 is_correct。
    llm_judge_score: float = 0.0

    # -----------------------------
    # 训练上下文字段
    # -----------------------------
    # 当前 QA 所属的 conversation/sample ID，例如 "conv-26"。
    # 它和 qa_idx 一起组成 query_id，方便跨 epoch 追踪同一道题。
    conversation_id: Optional[str] = None

    # 收集这条 case 时所在的全局 inner epoch。
    # trainer 中一般用 outer_epoch * inner_epochs + inner_epoch 计算。
    # CaseCollector 会根据它做 rolling window 剪枝，只保留近期失败。
    epoch: int = 0

    # 当前 conversation/session 内的训练 step。
    # 它用于定位这条失败 case 是在 memory 构建到哪个阶段后产生的。
    step: int = 0

    # 同一个 query_id/question 在失败池中重复失败的次数。
    # 重复失败越多，sample_cases_for_analysis() 里重要性越高，更容易被抽给 Designer 分析。
    fail_count: int = 1

    def to_dict(self) -> Dict:
        """把 DesignerCase 转成普通 dict，方便写入 checkpoint 或 snapshot。"""
        return {
            'query_id': self.query_id,
            'question': self.question,
            'ground_truth': self.ground_truth,
            'evidence': self.evidence,
            'category': self.category,
            'memory_bank_snapshot': self.memory_bank_snapshot,
            'retrieved_memories': self.retrieved_memories,
            'retrieved_indices': self.retrieved_indices,
            'prediction': self.prediction,
            'is_correct': self.is_correct,
            'f1_score': self.f1_score,
            'llm_judge_score': self.llm_judge_score,
            'conversation_id': self.conversation_id,
            'epoch': self.epoch,
            'step': self.step,
            'fail_count': self.fail_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DesignerCase':
        """从 checkpoint/snapshot 里保存的 dict 恢复 DesignerCase。"""
        return cls(
            query_id=data.get('query_id', ''),
            question=data.get('question', ''),
            ground_truth=data.get('ground_truth', ''),
            evidence=data.get('evidence'),
            category=data.get('category'),
            memory_bank_snapshot=data.get('memory_bank_snapshot'),
            retrieved_memories=data.get('retrieved_memories'),
            retrieved_indices=data.get('retrieved_indices'),
            prediction=data.get('prediction', ''),
            is_correct=data.get('is_correct', False),
            f1_score=data.get('f1_score', 0.0),
            llm_judge_score=data.get('llm_judge_score', 0.0),
            conversation_id=data.get('conversation_id'),
            epoch=data.get('epoch', 0),
            step=data.get('step', 0),
            fail_count=data.get('fail_count', 1)
        )

    def get_embedding_text(self) -> str:
        """返回用于失败 case 聚类的文本表示。"""
        # 当前只用 question 做聚类文本，所以 cluster 更偏向把“问题类型相似”的失败放在一起。
        # 下面被注释掉的逻辑说明：也可以把 retrieved_memories 拼进去，让聚类同时考虑检索现场。
        text = self.question
        # if self.retrieved_memories:
        #     text += " " + " ".join(self.retrieved_memories[:3])  # Top 3 memories
        return text


class CaseCollector:
    """
    Collects failure cases during training using a rolling pool.
    """

    def __init__(self, failure_window_epochs: int = 20,
                 failure_pool_size: int = 200,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            failure_window_epochs: Rolling window size in global inner epochs
            failure_pool_size: Maximum number of failures kept in the pool
            logger: Logger instance for output
        """
        self.logger = logger or logging.getLogger('AgenticMemory')

        # Failure pool (rolling window)
        self.failure_pool: Dict[str, DesignerCase] = {}
        self.failure_window_epochs = max(0, int(failure_window_epochs))
        self.failure_pool_size = max(0, int(failure_pool_size))
        self.latest_epoch: Optional[int] = None
        self._lock = threading.RLock()

    def _case_key(self, case: DesignerCase) -> str:
        if case.query_id:
            return str(case.query_id)
        return case.question.strip().lower()

    def _prune_failure_pool(self, current_epoch: Optional[int]):
        if current_epoch is None:
            return

        with self._lock:
            if self.failure_window_epochs > 0:
                cutoff = current_epoch - self.failure_window_epochs
                stale_keys = [key for key, case in self.failure_pool.items()
                              if case.epoch < cutoff]
                for key in stale_keys:
                    del self.failure_pool[key]

            if self.failure_pool_size > 0 and len(self.failure_pool) > self.failure_pool_size:
                sorted_keys = sorted(
                    self.failure_pool.keys(),
                    key=lambda k: (self.failure_pool[k].epoch, self.failure_pool[k].fail_count),
                    reverse=True
                )
                keep = set(sorted_keys[:self.failure_pool_size])
                for key in list(self.failure_pool.keys()):
                    if key not in keep:
                        del self.failure_pool[key]

    def add_case(self, case: DesignerCase):
        """Add a failure case to the rolling pool"""
        # 流程第 6 步：CaseCollector 只保留失败 case。
        # _evaluate_qa() 会把所有参与评估的 QA 都打包成 DesignerCase；
        # 如果当前 case 已经答对，就直接丢弃，不进入 Designer 的失败池。
        if case.is_correct:
            return

        with self._lock:
            # key 优先使用 query_id，例如 conv-26_2；没有 query_id 时退化为归一化后的 question。
            # 这样同一道题跨多轮重复失败时，会累积到同一个池子条目上。
            key = self._case_key(case)
            existing = self.failure_pool.get(key)
            if existing is not None:
                # 同一个问题再次失败，说明这是一个稳定失败模式，因此增加 fail_count。
                existing.fail_count += 1
                # 保留最新一轮的预测、分数、检索结果和 memory snapshot。
                # Designer 后续分析时会看到最新现场，同时 fail_count 记录它重复失败的次数。
                existing.prediction = case.prediction
                existing.f1_score = case.f1_score
                existing.llm_judge_score = case.llm_judge_score
                existing.retrieved_memories = case.retrieved_memories
                existing.retrieved_indices = case.retrieved_indices
                existing.memory_bank_snapshot = case.memory_bank_snapshot
                existing.epoch = case.epoch
                existing.step = case.step
                existing.conversation_id = case.conversation_id
            else:
                # 第一次看到这个失败问题时，把它作为新的失败案例放入 rolling pool。
                case.fail_count = max(int(getattr(case, 'fail_count', 1)), 1)
                self.failure_pool[key] = case

            # latest_epoch 用于 rolling window 剪枝，保证失败池只关注最近一段训练。
            self.latest_epoch = case.epoch
            # 每次加入/更新 case 后都剪枝，避免失败池无限增长。
            self._prune_failure_pool(self.latest_epoch)

    def get_all_cases(self) -> List[DesignerCase]:
        """Get all collected cases"""
        with self._lock:
            # 取出失败池前再按最新 epoch 剪枝一次，确保 Designer 看到的是有效窗口内的 case。
            if self.latest_epoch is not None:
                self._prune_failure_pool(self.latest_epoch)
            # 返回 DesignerCase 对象列表，供 prepare_evolution() 继续过滤、聚类和抽样。
            return list(self.failure_pool.values())

    def clear(self, reset_pool: bool = False):
        """Optionally clear the failure pool."""
        if reset_pool:
            with self._lock:
                self.failure_pool = {}
                self.latest_epoch = None

    def to_dict(self) -> Dict:
        """Serialize the rolling failure pool state."""
        with self._lock:
            return {
                'failure_pool': {
                    key: case.to_dict() for key, case in self.failure_pool.items()
                },
                'latest_epoch': self.latest_epoch
            }

    def load_dict(self, data: Optional[Dict]):
        """Restore the rolling failure pool state in-place."""
        if not isinstance(data, dict):
            return

        failure_pool = data.get('failure_pool', {})
        latest_epoch = data.get('latest_epoch', None)

        restored_pool: Dict[str, DesignerCase] = {}
        if isinstance(failure_pool, dict):
            for key, case_data in failure_pool.items():
                if not isinstance(case_data, dict):
                    continue
                restored_pool[str(key)] = DesignerCase.from_dict(case_data)

        with self._lock:
            self.failure_pool = restored_pool
            self.latest_epoch = latest_epoch
            if self.latest_epoch is not None:
                self._prune_failure_pool(self.latest_epoch)


@dataclass
class EvolutionSnapshot:
    """
    保存某个 Designer 演化阶段的 OperationBank 快照。

    它记录的是“这一版 skill bank 长什么样、这一版跑出了多少 reward、
    以及它是由哪一次 Designer 修改得到的”。SnapshotManager 会用这些
    快照比较不同版本的 skill bank，决定哪一版是当前 best，以及失败演化
    应该如何反馈给下一轮 Designer。
    """
    # 当前快照对应第几个演化阶段，从 0 开始递增。
    # stage_id=0 通常是初始 skill bank 或第一段训练后保存的基线版本。
    stage_id: int

    # 当前 OperationBank 的序列化结果，来自 operation_bank.to_dict()。
    # 里面保存所有 skill/operation 的 name、description、instruction_template、
    # update_type、meta_info 和 embedding 等信息。
    operation_bank_dict: Dict

    # 这一阶段的平均 reward。
    # trainer.add_snapshot() 传入时通常取本阶段后半/最后一部分 step 的 reward，
    # 用它来判断当前 skill bank 是否优于历史 best snapshot。
    avg_reward: float

    # 产生当前 skill bank 版本的 Designer 修改结果。
    # 它来自上一轮 set_pending_evolution_result() 暂存的 evolution_result。
    # 如果是初始快照或没有发生修改，这里可能是 None。
    evolution_result: Optional[Dict] = None

    # 本阶段 Designer 做分析时使用的失败样本列表，已经序列化成 dict。
    # 如果后续回滚到这个 best snapshot，可以复用这些 case 构造下一轮 prompt。
    analysis_cases: Optional[List[Dict]] = None

    def to_dict(self) -> Dict:
        """把快照转成普通 dict，方便保存到 checkpoint。"""
        return {
            'stage_id': self.stage_id,
            'operation_bank_dict': self.operation_bank_dict,
            'avg_reward': self.avg_reward,
            'evolution_result': self.evolution_result,
            'analysis_cases': self.analysis_cases
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvolutionSnapshot':
        """从 checkpoint 里的 dict 恢复 EvolutionSnapshot 对象。"""
        return cls(
            stage_id=data['stage_id'],
            operation_bank_dict=data['operation_bank_dict'],
            avg_reward=data['avg_reward'],
            evolution_result=data.get('evolution_result'),
            analysis_cases=data.get('analysis_cases')
        )


class EvolutionSnapshotManager:
    """
    Manages operation bank snapshots across evolution stages.

    Key responsibilities:
    1. Store snapshots with their associated rewards
    2. Track the best performing snapshot
    3. Track consecutive failures (no improvement)
    4. Generate feedback for LLM prompts (comparing with BEST snapshot, not previous)
    5. Provide rollback to best snapshot when needed
    6. Accumulate failed evolution attempts for comprehensive feedback

    Important: Since we always evolve from the BEST snapshot, feedback should compare
    current stage's reward with the best snapshot's reward, not the previous snapshot.

    When consecutive stages fail, they all evolve from the same best snapshot.
    We accumulate all failed evolution attempts so the LLM knows what approaches
    have already been tried and failed.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化演化快照管理器，负责记录各阶段 skill bank 的版本和效果。"""
        self.logger = logger or logging.getLogger('AgenticMemory')

        # Ordered list of snapshots (by stage_id)
        self.snapshots: List[EvolutionSnapshot] = []

        # Best snapshot tracking
        self.best_snapshot_idx: int = -1  # Index into snapshots list

        # Early stopping tracking
        self.consecutive_no_improvement: int = 0

        # Total evolution count
        self.total_evolves: int = 0

        # Pending evolution result: stores the evolve result that will affect the next stage
        # This is set after evolve, and consumed when adding the next snapshot
        self.pending_evolution_result: Optional[Dict] = None

        # Accumulated failed evolution attempts since last best snapshot
        # Each entry is a Dict with: evolution_result, reward_achieved
        # Cleared when a new best is found
        self.failed_evolution_attempts: List[Dict] = []

    def add_snapshot(self, operation_bank, avg_reward: float,
                     analysis_cases: Optional[List[Dict]] = None) -> bool:
        """
        Add a new snapshot and determine if it's the new best.

        中文说明：
        在一个 evolution stage 跑完后调用。它会保存当前 OperationBank 的快照，
        把上一轮 pending_evolution_result 和当前 avg_reward 绑定起来，
        然后判断当前版本是否超过历史 best。

        The evolution_result is taken from pending_evolution_result (set after previous evolve).
        The analysis_cases should be passed directly (cases collected during this stage).

        When a new best is found:
        - Clear failed_evolution_attempts (fresh start)

        When NOT a new best:
        - Add the evolution attempt to failed_evolution_attempts for feedback

        Args:
            operation_bank: OperationBank object to snapshot
            avg_reward: Average reward for this stage
            analysis_cases: Serialized analysis cases collected during this stage (optional)

        Returns:
            True if this snapshot is the new best, False otherwise
        """
        # stage_id 按保存顺序递增，用来标识第几个 skill bank 版本。
        stage_id = len(self.snapshots)

        # Use pending evolution result (from previous evolve)
        # pending_evolution_result 是上一次 Designer 修改 skill 后暂存的结果；
        # 当前 stage 跑完才知道这次修改带来的 avg_reward，所以在这里消费它。
        evolution_result = self.pending_evolution_result
        self.pending_evolution_result = None  # Consume it

        # 保存当前 OperationBank 的完整序列化状态、reward、修改来源和分析样本。
        snapshot = EvolutionSnapshot(
            stage_id=stage_id,
            operation_bank_dict=operation_bank.to_dict(),
            avg_reward=avg_reward,
            evolution_result=evolution_result,
            analysis_cases=analysis_cases
        )
        self.snapshots.append(snapshot)

        # Check if this is the new best
        is_new_best = False
        if self.best_snapshot_idx < 0:
            # First snapshot is automatically the best
            self.best_snapshot_idx = 0
            self.consecutive_no_improvement = 0
            is_new_best = True
            self.failed_evolution_attempts = []  # Clear on new best
            self.logger.info(f"[SnapshotManager] First snapshot (stage {stage_id}), "
                           f"avg_reward={avg_reward:.4f}")
        else:
            best_reward = self.snapshots[self.best_snapshot_idx].avg_reward
            if avg_reward > best_reward:
                # New best found
                self.best_snapshot_idx = stage_id
                self.consecutive_no_improvement = 0
                is_new_best = True
                self.failed_evolution_attempts = []  # Clear on new best
                self.logger.info(f"[SnapshotManager] New best at stage {stage_id}, "
                               f"avg_reward={avg_reward:.4f} (was {best_reward:.4f})")
            else:
                # No improvement - accumulate failed attempt
                self.consecutive_no_improvement += 1
                if evolution_result is not None:
                    self.failed_evolution_attempts.append({
                        'stage_id': stage_id,
                        'evolution_result': evolution_result,
                        'reward_achieved': avg_reward,
                        'best_reward': best_reward
                    })
                self.logger.info(f"[SnapshotManager] Stage {stage_id} avg_reward={avg_reward:.4f} "
                               f"did not beat best ({best_reward:.4f}). "
                               f"Consecutive no improvement: {self.consecutive_no_improvement}, "
                               f"Total failed attempts: {len(self.failed_evolution_attempts)}")

        return is_new_best

    def set_pending_evolution_result(self, evolution_result: Dict):
        """
        Store the evolution result to be associated with the next snapshot.
        Called after evolve completes.

        中文说明：
        Designer 刚修改完 skill 时，还不知道效果好不好；这里先把“改了什么”
        暂存起来。下一轮 stage 跑完后，add_snapshot() 会把它和新 reward 绑定。
        """
        self.pending_evolution_result = evolution_result

    def set_latest_snapshot_analysis_cases(self, analysis_cases: List[Dict]):
        """
        Set the analysis_cases for the latest (most recently added) snapshot.

        中文说明：
        prepare_evolution() 选出一批失败样本后，把它们挂到最新 snapshot 上。
        如果这个 snapshot 后来成为 best，后续回滚到它时可以复用同一批分析样本。

        This is called after prepare_evolution() to save the analysis cases
        that were used for this stage. These saved cases can be reused when
        consecutive stages fail and we need to evolve from the same best snapshot.

        Args:
            analysis_cases: List of serialized DesignerCase dicts
        """
        if len(self.snapshots) > 0:
            self.snapshots[-1].analysis_cases = analysis_cases

    def get_best_snapshot(self) -> Optional[EvolutionSnapshot]:
        """返回当前 reward 最高的 snapshot；如果还没有任何快照则返回 None。"""
        if self.best_snapshot_idx < 0 or len(self.snapshots) == 0:
            return None
        return self.snapshots[self.best_snapshot_idx]

    def get_latest_snapshot(self) -> Optional[EvolutionSnapshot]:
        """返回最近一次保存的 snapshot；常用于查看刚跑完的 stage。"""
        if len(self.snapshots) == 0:
            return None
        return self.snapshots[-1]

    def get_previous_snapshot(self) -> Optional[EvolutionSnapshot]:
        """返回最新 snapshot 的前一个版本；历史不足两个版本时返回 None。"""
        if len(self.snapshots) < 2:
            return None
        return self.snapshots[-2]

    def should_early_stop(self, patience: int) -> bool:
        """根据连续未提升次数判断是否触发早停。"""
        return self.consecutive_no_improvement >= patience

    def increment_evolve_count(self):
        """Designer 每尝试完成一次演化后调用，累计总演化次数。"""
        self.total_evolves += 1

    def should_stop_evolving(self, max_evolves: int, patience: int) -> bool:
        """
        Check if evolution should stop.

        中文说明：
        综合两个条件判断是否停止 Designer 演化：
        1. 总演化次数达到 max_evolves；
        2. 连续 patience 次没有超过 best reward。

        Args:
            max_evolves: Maximum number of evolution cycles
            patience: Early stop patience

        Returns:
            True if should stop, False otherwise
        """
        if self.total_evolves >= max_evolves:
            self.logger.info(f"[SnapshotManager] Reached max evolves ({max_evolves})")
            return True
        if self.should_early_stop(patience):
            self.logger.info(f"[SnapshotManager] Early stopping: no improvement for {patience} consecutive evolves")
            return True
        return False

    def generate_feedback(self) -> Optional[Dict]:
        """
        Generate feedback about the previous evolution's effect.

        中文说明：
        根据最新 snapshot 和 best snapshot 的 reward 差异，生成结构化反馈。
        这个反馈会告诉下一轮 Designer：上一次修改是正向提升还是负向下降，
        改了哪些 skill，以及当前连续未提升次数是多少。

        IMPORTANT: Since we always evolve from the BEST snapshot, feedback compares
        the current stage's reward with the BEST snapshot's reward (not previous snapshot).

        The evolution_result stored in the current snapshot tells us what modification
        was made to the best snapshot to produce this current state.

        Returns:
            Dict with feedback information, or None if no evolution has occurred yet
        """
        if len(self.snapshots) < 2:
            # First evolution, no feedback available
            return None

        curr_snapshot = self.snapshots[-1]  # Current snapshot (just added)
        best_snapshot = self.get_best_snapshot()

        # If current IS the best, we need to compare with the previous best
        # But since we just updated best_snapshot_idx, we need the pre-update best
        # Actually, if current is new best, then feedback is positive
        # If current is not best, then best_snapshot is the one we evolved from

        # The evolution_result in curr_snapshot describes what we did to get here
        # This modification was applied to the best snapshot (before this stage)
        evolution_result = curr_snapshot.evolution_result

        if evolution_result is None:
            # No evolution was applied (first stage or evolution was skipped)
            return None

        # Compare current reward with best reward
        # Note: If current became the new best, best_snapshot_idx now points to current
        # So we need to check if current IS the best
        is_current_best = (self.best_snapshot_idx == curr_snapshot.stage_id)

        if is_current_best:
            # Current is new best - find previous best to compare against
            # Look for the best among snapshots before current
            prev_best_reward = 0.0
            prev_best_stage = -1
            for snap in self.snapshots[:-1]:  # Exclude current
                if snap.avg_reward > prev_best_reward:
                    prev_best_reward = snap.avg_reward
                    prev_best_stage = snap.stage_id
            base_reward = prev_best_reward
            base_stage = prev_best_stage
        else:
            # Current is not best - compare with actual best
            base_reward = best_snapshot.avg_reward
            base_stage = best_snapshot.stage_id

        reward_diff = curr_snapshot.avg_reward - base_reward
        is_improvement = reward_diff > 0

        # Describe what changes were made (from the evolution result)
        changes_description = self._describe_evolution_changes(evolution_result)

        feedback = {
            'is_improvement': is_improvement,
            'base_reward': base_reward,  # The reward of the snapshot we evolved from
            'base_stage_id': base_stage,
            'curr_reward': curr_snapshot.avg_reward,
            'reward_diff': reward_diff,
            'best_reward': best_snapshot.avg_reward,
            'curr_stage_id': curr_snapshot.stage_id,
            'changes_description': changes_description,
            'consecutive_no_improvement': self.consecutive_no_improvement
        }

        return feedback

    def _describe_evolution_changes(self, evolution_result: Optional[Dict],
                                      detailed: bool = False) -> str:
        """
        Generate human-readable description of what changed in an evolution.

        中文说明：
        把 evolution_result 里的 add_new/refine_existing/no_change 等结构化结果
        转成适合写进 prompt 或日志的人类可读文本。detailed=True 时会展开
        instruction_template 的 before/after 对比。

        Args:
            evolution_result: The evolution result dict
            detailed: If True, include full before/after comparison (for refinement prompt).
                      If False, include only summary (for analysis prompt).
        """
        if evolution_result is None:
            return "No changes were made (first stage or no evolution applied)"

        def describe_change(change: Dict, detailed_view: bool) -> str:
            change_action = str(change.get('action', 'unknown')).lower().strip()

            if change_action == 'add_new':
                new_op = change.get('new_operation', {})
                op_name = new_op.get('name', 'unknown')
                op_type = new_op.get('update_type', 'unknown')
                op_desc = new_op.get('description', '')
                reasoning = new_op.get('reasoning', '')

                if detailed_view:
                    instruction_template = new_op.get('instruction_template', '')
                    return (f"**Added new operation: '{op_name}'** (type: {op_type})\n"
                           f"- **Description:** {op_desc}\n"
                           f"- **Instruction Template:**\n```\n{instruction_template}\n```\n"
                           f"- **Reasoning:** {reasoning}")
                return (f"Added new operation '{op_name}' (type: {op_type}). "
                       f"Description: {op_desc[:100]}... Reasoning: {reasoning}")

            if change_action == 'refine_existing':
                refined_op = change.get('refined_operation', {})
                op_name = refined_op.get('name', 'unknown')
                changes = refined_op.get('changes', {})
                changed_fields = list(changes.keys())
                reasoning = refined_op.get('reasoning', '')

                if detailed_view:
                    original_op = change.get('original_operation', {})
                    comparison_text = f"**Refined operation: '{op_name}'**\n"
                    comparison_text += f"- **Changed fields:** {changed_fields}\n"
                    comparison_text += f"- **Reasoning:** {reasoning}\n\n"

                    if 'description' in changes:
                        old_desc = original_op.get('description', '[not available]')
                        new_desc = changes['description']
                        comparison_text += f"**Description change:**\n"
                        comparison_text += f"- BEFORE: {old_desc}\n"
                        comparison_text += f"- AFTER: {new_desc}\n\n"

                    if 'instruction_template' in changes:
                        old_template = original_op.get('instruction_template', '[not available]')
                        new_template = changes['instruction_template']
                        comparison_text += f"**Instruction template change:**\n"
                        comparison_text += f"- BEFORE:\n```\n{old_template}\n```\n"
                        comparison_text += f"- AFTER:\n```\n{new_template}\n```\n"

                    return comparison_text
                return (f"Refined operation '{op_name}', changed: {changed_fields}. "
                       f"Reasoning: {reasoning}")

            if change_action == 'no_change':
                return f"No changes applied. Reasoning: {change.get('reasoning', 'N/A')}"

            return f"Unknown action: {change_action}"

        changes = evolution_result.get('changes')
        if isinstance(changes, list) and len(changes) > 0:
            descriptions = [describe_change(change, detailed) for change in changes]
            if detailed:
                return "\n\n".join(descriptions)
            return " | ".join(descriptions)

        action = evolution_result.get('action', 'unknown')
        if action == 'add_new':
            return describe_change(
                {'action': 'add_new', 'new_operation': evolution_result.get('new_operation', {})},
                detailed
            )
        if action == 'refine_existing':
            return describe_change(
                {
                    'action': 'refine_existing',
                    'refined_operation': evolution_result.get('refined_operation', {}),
                    'original_operation': evolution_result.get('original_operation', {})
                },
                detailed
            )
        if action == 'no_change':
            reasoning = evolution_result.get('reasoning', 'N/A')
            return f"No changes applied. Reasoning: {reasoning}"
        if action == 'multi':
            return "Multiple changes applied."

        return f"Unknown action: {action}"

    def format_feedback_for_prompt(self, detailed: bool = False) -> str:
        """
        Format feedback as text to include in the analysis prompt.

        中文说明：
        把 generate_feedback() 的结构化结果格式化成 Stage 1 analysis prompt 里的文本。
        作用是提醒 LLM：上一轮修改是否有效、当前 best reward 是多少、
        哪些失败修改不要重复尝试。

        Since we always evolve from the best snapshot, the feedback compares
        the current stage's performance against the best snapshot's performance.

        When there are multiple consecutive failures, all failed attempts are
        listed so the LLM knows what approaches have already been tried.

        Args:
            detailed: If True, include full instruction_template changes and
                      detailed failed attempts (useful for analysis prompts).
        """
        feedback = self.generate_feedback()

        if feedback is None:
            return ""

        changes_description = feedback['changes_description']
        if detailed:
            curr_snapshot = self.get_latest_snapshot()
            if curr_snapshot and curr_snapshot.evolution_result:
                changes_description = self._describe_evolution_changes(
                    curr_snapshot.evolution_result, detailed=True
                )

        if feedback['is_improvement']:
            effect_text = (f"POSITIVE: Reward improved from {feedback['base_reward']:.4f} "
                          f"(stage {feedback['base_stage_id']}) to {feedback['curr_reward']:.4f} "
                          f"(+{feedback['reward_diff']:.4f})")
            guidance = "Since the previous change was beneficial, consider building upon it or making similar improvements."
            failed_attempts_text = ""
        else:
            effect_text = (f"NEGATIVE: Reward decreased from {feedback['base_reward']:.4f} "
                          f"(stage {feedback['base_stage_id']}) to {feedback['curr_reward']:.4f} "
                          f"({feedback['reward_diff']:.4f})")
            guidance = ("Since the previous change was NOT beneficial, the operation bank has been "
                       "rolled back to the best performing version. Please try a DIFFERENT approach "
                       "and avoid similar modifications.")

            # Include all accumulated failed attempts
            failed_attempts_text = self._format_failed_attempts(detailed=detailed)

        prompt_text = f"""
## Previous Evolution Feedback
The last evolution (applied to the best-performing operation bank) had the following effect:
- **Effect**: {effect_text}
- **What was changed**: {changes_description}
- **Current best reward**: {feedback['best_reward']:.4f}
- **Consecutive stages without improvement**: {feedback['consecutive_no_improvement']}

{guidance}
{failed_attempts_text}
"""
        return prompt_text

    def _format_failed_attempts(self, detailed: bool = False,
                                  exclude_current: bool = True) -> str:
        """
        Format all accumulated failed evolution attempts for the prompt.
        This helps the LLM avoid repeating failed approaches.

        中文说明：
        把 failed_evolution_attempts 中累计的失败修改尝试整理成 prompt 文本。
        当连续几次 Designer 修改都没有提升 reward 时，这段文本会告诉 LLM
        “这些方向已经试过且失败了，不要重复”。

        Args:
            detailed: If True, include full before/after comparison for each attempt.
            exclude_current: If True, exclude the most recent failed attempt (current snapshot)
                             since it's already described separately in "What was changed".
        """
        if not self.failed_evolution_attempts:
            return ""

        # Get attempts to format (exclude current if requested)
        attempts_to_format = self.failed_evolution_attempts
        if exclude_current and len(attempts_to_format) > 0:
            # Exclude the last one (current snapshot's failed attempt)
            attempts_to_format = attempts_to_format[:-1]

        if not attempts_to_format:
            return ""

        lines = ["\n### Previously Failed Approaches (DO NOT repeat these)",
                 "The following modifications have already been tried and did NOT improve performance:"]

        for i, attempt in enumerate(attempts_to_format, 1):
            evolution_result = attempt.get('evolution_result', {})
            reward_achieved = attempt.get('reward_achieved', 0.0)
            best_reward = attempt.get('best_reward', 0.0)
            stage_id = attempt.get('stage_id', 'unknown')

            change_desc = self._describe_evolution_changes(evolution_result, detailed=detailed)
            lines.append(f"\n**Failed Attempt {i} (Stage {stage_id}):**")
            lines.append(f"- Result: Reward {reward_achieved:.4f} (best was {best_reward:.4f})")
            lines.append(f"- Change Details:\n{change_desc}")

        lines.append("\n**IMPORTANT:** Avoid similar modifications to the ones listed above.")
        return "\n".join(lines)

    def format_evolution_feedback_for_refinement(self) -> str:
        """
        Format detailed evolution feedback for the refinement prompt.

        中文说明：
        给 Stage 2 refinement prompt 使用的详细反馈版本。
        相比 format_feedback_for_prompt()，这里更强调具体改动的 before/after，
        让 LLM 在提出下一次 skill 修改时能避开失败方案或延续成功方案。

        This includes full before/after comparison of operation changes,
        which helps the LLM understand exactly what was tried and failed.
        The analysis prompt can request a similar detailed view when needed.
        """
        feedback = self.generate_feedback()

        if feedback is None:
            return ""

        if feedback['is_improvement']:
            effect_text = (f"POSITIVE: Reward improved from {feedback['base_reward']:.4f} "
                          f"(stage {feedback['base_stage_id']}) to {feedback['curr_reward']:.4f} "
                          f"(+{feedback['reward_diff']:.4f})")
            guidance = ("The previous change was BENEFICIAL. You may build upon it or "
                       "explore similar improvements.")
            failed_attempts_text = ""
        else:
            effect_text = (f"NEGATIVE: Reward decreased from {feedback['base_reward']:.4f} "
                          f"(stage {feedback['base_stage_id']}) to {feedback['curr_reward']:.4f} "
                          f"({feedback['reward_diff']:.4f})")

            # Include detailed failed attempts (excluding current, which is shown in "What Was Changed")
            failed_attempts_text = self._format_failed_attempts(detailed=True)

            # Adjust guidance based on whether there are previous failed attempts
            if failed_attempts_text:
                # There are previous failed attempts to review
                guidance = ("The previous change was NOT beneficial. The operation bank has been "
                           "rolled back to the best performing version. You MUST try a DIFFERENT "
                           "approach. Review the current change above AND the previously failed "
                           "attempts below, and avoid similar modifications.")
            else:
                # Only the current attempt failed (first failure)
                guidance = ("The previous change was NOT beneficial. The operation bank has been "
                           "rolled back to the best performing version. You MUST try a DIFFERENT "
                           "approach. Review the current change shown above and avoid similar modifications.")

        # Get detailed description of the last change
        curr_snapshot = self.get_latest_snapshot()
        detailed_changes = ""
        if curr_snapshot and curr_snapshot.evolution_result:
            detailed_changes = self._describe_evolution_changes(
                curr_snapshot.evolution_result, detailed=True
            )

        prompt_text = f"""
## Evolution Feedback (IMPORTANT - Read carefully before proposing changes)

### Last Evolution Effect
- **Effect**: {effect_text}
- **Current best reward**: {feedback['best_reward']:.4f}
- **Consecutive stages without improvement**: {feedback['consecutive_no_improvement']}

### What Was Changed (Detailed)
{detailed_changes}

### Guidance
{guidance}
{failed_attempts_text}
"""
        return prompt_text

    def to_dict(self) -> Dict:
        """把 SnapshotManager 的完整状态转成 dict，方便写入 checkpoint。"""
        return {
            'snapshots': [s.to_dict() for s in self.snapshots],
            'best_snapshot_idx': self.best_snapshot_idx,
            'consecutive_no_improvement': self.consecutive_no_improvement,
            'total_evolves': self.total_evolves,
            'pending_evolution_result': self.pending_evolution_result,
            'failed_evolution_attempts': self.failed_evolution_attempts
        }

    @classmethod
    def from_dict(cls, data: Dict, logger: Optional[logging.Logger] = None) -> 'EvolutionSnapshotManager':
        """从 checkpoint 里的 dict 恢复 SnapshotManager 状态。"""
        manager = cls(logger=logger)
        manager.snapshots = [EvolutionSnapshot.from_dict(s) for s in data.get('snapshots', [])]
        manager.best_snapshot_idx = data.get('best_snapshot_idx', -1)
        manager.consecutive_no_improvement = data.get('consecutive_no_improvement', 0)
        manager.total_evolves = data.get('total_evolves', 0)
        manager.pending_evolution_result = data.get('pending_evolution_result', None)
        manager.failed_evolution_attempts = data.get('failed_evolution_attempts', [])
        return manager


class Designer:
    """
    Designer evolves the operation bank based on failure case analysis.
    Uses clustering to identify patterns in failure cases and proposes improvements.
    """

    def __init__(self, args,
                 collect_epochs_before_designer: int = 5,
                 num_clusters: int = 5,
                 samples_per_cluster: int = 3,
                 f1_threshold: float = 0.5,
                 failure_window_epochs: int = 20,
                 failure_pool_size: int = 200,
                 encoder=None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            args: Training arguments
            collect_epochs_before_designer: Legacy (unused; rolling failure pool is always active)
            num_clusters: Number of clusters for failure case analysis
            samples_per_cluster: Number of cases to sample from each cluster
            f1_threshold: F1 threshold for success/failure classification
            encoder: Shared BaseTextEncoder instance (optional, avoids loading multiple models)
            logger: Logger instance for output
        """
        self.args = args
        self.designer_model = getattr(args, "designer_model", None) or args.model
        self.num_clusters = num_clusters
        self.samples_per_cluster = samples_per_cluster
        self.f1_threshold = f1_threshold
        self.logger = logger or logging.getLogger('AgenticMemory')

        # Initialize case collector
        self.case_collector = CaseCollector(
            failure_window_epochs=failure_window_epochs,
            failure_pool_size=failure_pool_size,
            logger=self.logger
        )

        # Encoder for clustering (use shared encoder if provided)
        self._encoder = encoder

    def _get_encoder(self):
        """Get encoder for clustering (lazy init if not provided)"""
        if self._encoder is None:
            from src.controller import BaseTextEncoder
            model_name = getattr(self.args, 'state_encoder', 'allenai/longformer-base-4096')
            device = getattr(self.args, 'device', 'cuda')
            encode_batch_size = getattr(self.args, 'encode_batch_size', 64)
            self._encoder = BaseTextEncoder(
                model_name=model_name, device=device, encode_batch_size=encode_batch_size
            )
        return self._encoder

    def _call_llm_with_retry(self, prompt: str, max_tokens: int, tau: float) -> str:
        max_rounds = int(getattr(self.args, "round", 1) or 1)
        max_rounds = max(1, max_rounds)
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_rounds + 1):
            try:
                response, _, _ = get_llm_response_via_api(
                    prompt=prompt,
                    LLM_MODEL=self.designer_model,
                    base_url=self.args.api_base,
                    api_key=self.args.api_key,
                    MAX_TOKENS=max_tokens,
                    TAU=tau
                )
                return response
            except Exception as exc:
                last_exc = exc
                self.logger.warning(
                    f"[Designer] LLM call failed ({attempt}/{max_rounds}): {exc}"
                )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM call failed with unknown error")

    def filter_and_deduplicate(self, cases: List[DesignerCase]) -> List[DesignerCase]:
        """只保留失败案例，并按 question 去重。"""
        # cases 来自训练过程中的 QA 评估结果，里面既有答对的，也有答错的。
        # Designer 的目标是分析失败模式，所以先只保留 is_correct=False 的样本。
        failure_cases = [c for c in cases if not c.is_correct]
        self.logger.info(f"[Designer] Filtered to {len(failure_cases)} failure cases (from {len(cases)} total)")

        if len(failure_cases) == 0:
            return []

        # 同一个 question 可能在多轮训练中重复失败。
        # 这里用归一化后的 question 作为 key，只保留一个代表案例，避免 LLM prompt 被重复问题占满。
        case_map = {}

        for case in failure_cases:
            normalized_q = case.question.strip().lower()
            existing = case_map.get(normalized_q)
            # 如果重复问题出现多次，保留 fail_count 更高的版本。
            # fail_count 高说明这个问题反复失败，更能代表稳定的缺陷模式。
            if existing is None or case.fail_count > existing.fail_count:
                case_map[normalized_q] = case

        unique_cases = list(case_map.values())
        self.logger.info(f"[Designer] Deduplicated to {len(unique_cases)} unique failure cases")
        return unique_cases

    def cluster_cases(self, cases: List[DesignerCase]) -> Dict[int, List[DesignerCase]]:
        """用 embedding + KMeans 将失败案例聚成若干类。"""
        if len(cases) == 0:
            return {}

        if len(cases) <= self.num_clusters:
            # 失败案例数量不超过目标聚类数时，没有必要再做 KMeans。
            # 直接把每个 case 当成一个 cluster，保证后续采样逻辑仍然统一。
            return {i: [case] for i, case in enumerate(cases)}

        # 聚类时不直接用完整 DesignerCase，而是把 case 转成文本。
        # 当前 get_embedding_text() 主要使用 question，因此聚类结果更偏向“问题类型”相似性。
        encoder = self._get_encoder()
        texts = [case.get_embedding_text() for case in cases]
        embeddings = encoder.encode(texts)

        # 对失败案例向量做 KMeans，把相似失败问题分到同一组。
        # 这样后面给 LLM 的案例不会只集中在某一种失败类型上。
        from sklearn.cluster import KMeans

        n_clusters = min(self.num_clusters, len(cases))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 将 KMeans 输出的 cluster label 重新整理成 {cluster_id: [cases]}。
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(cases[i])

        self.logger.info(f"[Designer] Clustered {len(cases)} cases into {n_clusters} clusters")
        for label, cluster_cases in clusters.items():
            self.logger.info(f"  Cluster {label}: {len(cluster_cases)} cases")

        return dict(clusters)

    def _case_importance(self, case: DesignerCase) -> float:
        """计算失败案例的重要程度，用于从每个 cluster 中挑代表样本。"""
        # 如果 reward_metric 是 llm_judge，就用 judge 分数衡量失败严重度；
        # 否则默认用 F1，分数越低代表失败越严重。
        metric = getattr(self.args, "reward_metric", "f1")
        if metric == "llm_judge":
            severity = 1.0 - float(case.llm_judge_score)
        else:
            severity = 1.0 - float(case.f1_score)

        severity = max(0.0, min(1.0, severity))
        # 同一个问题失败次数越多，说明它越稳定地暴露系统缺陷。
        # log1p 用来放大重复失败的影响，但避免 fail_count 过大时权重爆炸。
        fail_count = max(1, int(getattr(case, "fail_count", 1)))
        return float(severity * np.log1p(fail_count))

    def sample_cases_for_analysis(self, clusters: Dict[int, List[DesignerCase]]) -> List[DesignerCase]:
        """从每个 cluster 中挑选代表性失败案例，供 LLM Designer 分析。"""
        sampled_cases = []

        for cluster_id, cluster_cases in clusters.items():
            if len(cluster_cases) == 0:
                continue
            # 每个 cluster 内部按照重要程度排序。
            # 重要程度由失败严重度和重复失败次数共同决定。
            sorted_cases = sorted(cluster_cases, key=self._case_importance, reverse=True)
            # 每类最多取 samples_per_cluster 个，保证 LLM prompt 覆盖多种失败模式。
            sampled_cases.extend(sorted_cases[:self.samples_per_cluster])

        # 理论目标样本数 = 聚类数 * 每类样本数。
        # 如果有些 cluster 太小导致没取满，就从剩余案例中按重要程度补齐。
        target_total = self.num_clusters * self.samples_per_cluster
        if len(sampled_cases) < target_total:
            selected_keys = set()
            for case in sampled_cases:
                key = case.query_id or case.question.strip().lower()
                selected_keys.add(key)

            # 收集尚未被选中的候选案例。
            remaining = []
            for cluster_cases in clusters.values():
                for case in cluster_cases:
                    key = case.query_id or case.question.strip().lower()
                    if key not in selected_keys:
                        remaining.append(case)

            # 从剩余案例中继续选最重要的，直到达到目标数量或没有更多案例。
            remaining_sorted = sorted(remaining, key=self._case_importance, reverse=True)
            sampled_cases.extend(remaining_sorted[:target_total - len(sampled_cases)])

        self.logger.info(f"[Designer] Sampled {len(sampled_cases)} cases for analysis")
        return sampled_cases

    def prepare_analysis_cases(self, cases: List[DesignerCase]) -> List[DesignerCase]:
        """
        Full pipeline: filter, deduplicate, cluster, and sample cases.

        Args:
            cases: All collected cases

        Returns:
            Sampled failure cases for LLM analysis
        """
        # 流程第 7 步的前半段：先只保留失败样本，并按 question 去重。
        # add_case() 已经倾向于只收失败 case，但这里再过滤一次，保证外部传入混合 case 时也安全。
        unique_failures = self.filter_and_deduplicate(cases)

        if len(unique_failures) == 0:
            self.logger.info("[Designer] No failure cases to analyze")
            return []

        # 按问题语义聚类，避免 prompt 里的样本全部来自同一种失败。
        # cluster_cases() 当前主要用 question 文本做 embedding。
        clusters = self.cluster_cases(unique_failures)

        # 从每个 cluster 中挑选代表性样本。
        # sample_cases_for_analysis() 会优先选择 F1/Judge 分数低且 fail_count 高的 case。
        sampled_cases = self.sample_cases_for_analysis(clusters)

        # 返回值就是后续 build_analysis_prompt() 会展示给 Designer LLM 的 failure cases。
        return sampled_cases

    def _normalize_cases_for_prompt(self, cases: List[Any]) -> List[DesignerCase]:
        if not cases:
            return []

        normalized_cases = []
        for case in cases:
            if isinstance(case, DesignerCase):
                normalized_cases.append(case)
            elif isinstance(case, dict):
                normalized_cases.append(DesignerCase.from_dict(case))

        return normalized_cases

    def _format_operation_bank_description(self, operation_bank) -> str:
        ops = operation_bank.get_all_operations()
        op_descriptions = []
        for op in ops:
            op_desc = f"- **{op.name}** (type: {op.update_type})\n"
            op_desc += f"  Description: {op.description}"
            op_descriptions.append(op_desc)
        return "\n".join(op_descriptions)

    def _format_failure_cases_details(self, cases: List[DesignerCase]) -> str:
        case_details = []
        for i, case in enumerate(cases):
            case_str = f"### Case {i + 1}\n"
            case_str += f"**Question:** {case.question}\n"
            case_str += f"**Expected Answer:** {case.ground_truth}\n"
            case_str += f"**System Prediction:** {case.prediction}\n"

            if case.retrieved_memories:
                case_str += f"**Retrieved Memories ({len(case.retrieved_memories)}):**\n"
                for j, mem in enumerate(case.retrieved_memories[:20]):
                    mem_preview = mem
                    case_str += f"  {j + 1}. {mem_preview}\n"
            else:
                case_str += "**Retrieved Memories:** None\n"

            case_details.append(case_str)

        return "\n".join(case_details)

    def _get_max_changes(self) -> int:
        max_changes = getattr(self.args, "designer_max_changes", 1)
        try:
            max_changes = int(max_changes)
        except (TypeError, ValueError):
            max_changes = 1
        if max_changes < 1:
            max_changes = 1
        return max_changes

    def build_analysis_prompt(self, cases: List[DesignerCase], operation_bank,
                               evolution_feedback: str = "") -> str:
        """
        Stage 1: Build analysis prompt from sampled failure cases.

        Args:
            cases: Sampled failure cases
            operation_bank: Current operation bank
            evolution_feedback: Formatted feedback from previous evolution (optional)

        Returns:
            Formatted prompt for LLM analysis
        """
        cases = self._normalize_cases_for_prompt(cases)
        operation_bank_description = self._format_operation_bank_description(operation_bank)
        failure_cases_details = self._format_failure_cases_details(cases)

        new_skill_hint = ""
        if getattr(self.args, "designer_new_skill_hint", False):
            new_skill_hint = ("Note: If failures indicate a capability gap, it is encouraged to recommend "
                              "adding a new skill.")

        # Format the analysis prompt
        prompt = DESIGNER_ANALYSIS_PROMPT.format(
            operation_bank_description=operation_bank_description,
            evolution_feedback=evolution_feedback,
            num_failure_cases=len(cases),
            failure_cases_details=failure_cases_details,
            new_skill_hint=new_skill_hint,
            max_changes=self._get_max_changes()
        )

        return prompt

    def build_analysis_prompt_from_saved_cases(self, saved_cases: List[Dict], operation_bank,
                                                evolution_feedback: str = "") -> str:
        """
        Build analysis prompt from previously saved (serialized) analysis cases.

        This is used when consecutive stages fail and we need to evolve from
        the same best snapshot, using the best snapshot's saved analysis_cases.

        Args:
            saved_cases: List of serialized DesignerCase dicts (from snapshot.analysis_cases)
            operation_bank: Current operation bank
            evolution_feedback: Formatted feedback from previous evolution (optional)

        Returns:
            Formatted prompt for LLM analysis
        """
        return self.build_analysis_prompt(saved_cases, operation_bank, evolution_feedback)

    def build_reflection_prompt(self, analysis_feedback: str, cases: List[Any], operation_bank,
                                 evolution_feedback: str = "", reflection_round: int = 2,
                                 reflection_round_total: int = 2) -> str:
        """
        Stage 1b: Build reflection prompt to critique and improve analysis.

        Args:
            analysis_feedback: JSON analysis from prior round
            cases: Failure cases (DesignerCase objects or serialized dicts)
            operation_bank: Current operation bank
            evolution_feedback: Formatted feedback from previous evolution (optional)
            reflection_round: Current reflection round (1-based)
            reflection_round_total: Total reflection rounds

        Returns:
            Formatted prompt for LLM reflection
        """
        cases = self._normalize_cases_for_prompt(cases)
        operation_bank_description = self._format_operation_bank_description(operation_bank)
        failure_cases_details = self._format_failure_cases_details(cases)

        new_skill_hint = ""
        if getattr(self.args, "designer_new_skill_hint", False):
            new_skill_hint = ("Note: If failures indicate a capability gap, it is encouraged to recommend "
                              "adding a new skill.")

        prompt = DESIGNER_REFLECTION_PROMPT.format(
            analysis_feedback=analysis_feedback,
            operation_bank_description=operation_bank_description,
            evolution_feedback=evolution_feedback,
            num_failure_cases=len(cases),
            failure_cases_details=failure_cases_details,
            reflection_round=reflection_round,
            reflection_round_total=reflection_round_total,
            new_skill_hint=new_skill_hint,
            max_changes=self._get_max_changes()
        )

        return prompt

    def build_refinement_prompt(self, analysis_feedback: str, operation_bank,
                                 evolution_feedback: str = "") -> str:
        """
        Stage 2: Build refinement prompt based on analysis feedback.

        Args:
            analysis_feedback: JSON analysis from stage 1
            operation_bank: Current operation bank
            evolution_feedback: Detailed evolution feedback (optional)

        Returns:
            Formatted prompt for operation refinement
        """
        # Build full operation bank details
        ops = operation_bank.get_all_operations()
        op_full_details = []
        for op in ops:
            op_detail = f"### {op.name}\n"
            op_detail += f"- **Type:** {op.update_type}\n"
            op_detail += f"- **Description:** {op.description}\n"
            op_detail += f"- **Instruction Template:**\n```\n{op.instruction_template}\n```\n"
            op_full_details.append(op_detail)

        operation_bank_full = "\n".join(op_full_details) if op_full_details else "(No evolvable operations available)"

        new_skill_hint = ""
        if getattr(self.args, "designer_new_skill_hint", False):
            new_skill_hint = ("Note: If you see a clear capability gap, it is encouraged to propose "
                              "a new operation (skill).")

        max_changes = self._get_max_changes()

        # Format the refinement prompt
        prompt = DESIGNER_REFINEMENT_PROMPT.format(
            analysis_feedback=analysis_feedback,
            operation_bank_full=operation_bank_full,
            evolution_feedback=evolution_feedback,
            new_skill_hint=new_skill_hint,
            max_changes=max_changes
        )

        return prompt

    def prepare_evolution(self, operation_bank, evolution_feedback: str = "") -> Optional[Dict]:
        """
        Prepare for evolution: collect cases, cluster, and build analysis prompt.
        This is the expensive step that should only be done once.

        Args:
            operation_bank: OperationBank object
            evolution_feedback: Formatted feedback from previous evolution (optional)

        Returns:
            Dict with analysis_prompt, analysis_cases, and evolution_feedback, or None if no cases to analyze
        """
        # 流程第 7 步入口：从 CaseCollector 的 rolling failure pool 中取出所有候选失败案例。
        # 这些 case 是 trainer._evaluate_qa() 打包并通过 add_case() 保存下来的。
        all_cases = self.case_collector.get_all_cases()

        if len(all_cases) == 0:
            self.logger.info("[Designer] No cases collected, skipping evolution")
            return None

        # 对失败池做二次整理：过滤、按 question 去重、聚类、从每类抽样。
        # 输出 analysis_cases 是真正会进入 Designer LLM prompt 的代表性失败样本。
        analysis_cases = self.prepare_analysis_cases(all_cases)

        if len(analysis_cases) == 0:
            self.logger.info("[Designer] No failure cases after filtering, skipping evolution")
            return None

        # 把代表性失败样本、当前 operation_bank 和上一轮演化反馈拼成 Stage 1 分析 prompt。
        # 这里还没有调用 LLM，只是准备 run_evolution() 所需的输入数据。
        analysis_prompt = self.build_analysis_prompt(
            analysis_cases, operation_bank, evolution_feedback=evolution_feedback
        )

        # prepared_data 是 run_evolution() 的直接输入：
        # - analysis_prompt: 交给 LLM 分析失败模式
        # - analysis_cases: 保存本轮实际使用了哪些 DesignerCase，方便快照/复用
        # - evolution_feedback: 上一轮 skill 修改效果反馈
        return {
            'analysis_prompt': analysis_prompt,
            'analysis_cases': analysis_cases,
            'evolution_feedback': evolution_feedback
        }

    def run_evolution(self, operation_bank, prepared_data: Dict,
                       evolution_feedback_for_refinement: str = "") -> Dict:
        """
        Run the two-stage evolution using prepared data.
        This can be retried without redoing the expensive preparation step.

        Args:
            operation_bank: OperationBank object
            prepared_data: Dict from prepare_evolution() containing analysis_prompt and analysis_cases
            evolution_feedback_for_refinement: Detailed evolution feedback for refinement prompt

        Returns:
            evolution_result: dict with action, analysis, and operation changes
        """
        analysis_prompt = prepared_data['analysis_prompt']
        analysis_cases = prepared_data.get('analysis_cases') or []
        evolution_feedback = prepared_data.get('evolution_feedback', "")

        reflection_cycles = getattr(self.args, "designer_reflection_cycles", 3)
        try:
            reflection_cycles = int(reflection_cycles)
        except (TypeError, ValueError):
            reflection_cycles = 1
        if reflection_cycles < 1:
            reflection_cycles = 1

        # =====================================================================
        # Stage 1: Analysis
        # =====================================================================
        self.logger.info(f"[Designer] Stage 1: Analyzing failure cases (round 1/{reflection_cycles})...")

        analysis_response = self._call_llm_with_retry(
            prompt=analysis_prompt,
            max_tokens=2048,
            tau=0.0
        )
        self.logger.info(f"[Designer] Stage 1 round 1 complete. Analysis: {analysis_response}")

        if reflection_cycles > 1:
            for cycle in range(2, reflection_cycles + 1):
                self.logger.info(f"[Designer] Stage 1: Reflection round {cycle}/{reflection_cycles}...")
                reflection_prompt = self.build_reflection_prompt(
                    analysis_response,
                    analysis_cases,
                    operation_bank,
                    evolution_feedback=evolution_feedback,
                    reflection_round=cycle,
                    reflection_round_total=reflection_cycles
                )
                analysis_response = self._call_llm_with_retry(
                    prompt=reflection_prompt,
                    max_tokens=2048,
                    tau=0.0
                )
                self.logger.info(f"[Designer] Stage 1 reflection round {cycle} complete. Analysis: {analysis_response}")

        # =====================================================================
        # Stage 2: Refinement
        # =====================================================================
        self.logger.info("[Designer] Stage 2: Proposing operation improvements...")
        refinement_prompt = self.build_refinement_prompt(
            analysis_response, operation_bank,
            evolution_feedback=evolution_feedback_for_refinement
        )

        refinement_response = self._call_llm_with_retry(
            prompt=refinement_prompt,
            max_tokens=4096,
            tau=0.0
        )
        self.logger.info(f"[Designer] Stage 2 complete. Response: {refinement_response}")

        # Parse refinement response
        evolution_result = self._parse_refinement_response(refinement_response)
        evolution_result['stage1_analysis'] = analysis_response

        return evolution_result

    def _parse_refinement_response(self, response: str) -> Dict:
        """
        Parse Stage 2 refinement response.

        Expected format:
        - action: "apply_changes" | "add_new" | "refine_existing" | "no_change"
        - changes: [...] (if action is apply_changes)
        - new_operation: {...} (if action is add_new)
        - refined_operation: {...} (if action is refine_existing)
        - reasoning/summary: optional text
        """
        try:
            # First, strip markdown code block markers if present
            # LLM may return ```json ... ``` or ``` ... ```
            cleaned_response = response.strip()

            # Remove opening code block markers (```json, ```JSON, ```, etc.)
            # Match ```json or ```JSON or just ``` at the start
            cleaned_response = re.sub(r'^```(?:json|JSON)?\s*\n?', '', cleaned_response)
            # Match ``` at the end
            cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)

            # Try to extract JSON from response
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                self.logger.warning("[Designer] No JSON found in refinement response")
                return {
                    'action': 'no_change',
                    'stage2_result': None,
                    'reasoning': 'Failed to parse response: no JSON found',
                    'raw_response': response
                }

            json_str = cleaned_response[json_start:json_end]

            # Fix backtick-quoted strings (LLM sometimes uses ` instead of " for string values)
            # This is a common issue where LLM outputs JavaScript template literals instead of JSON strings
            # Pattern: replace `: ` (colon followed by backtick) with `: "`
            # and replace backtick at end of value (before comma, newline, or closing brace) with double quote
            json_str = re.sub(r':\s*`', ': "', json_str)  # `: ` -> `: "`
            json_str = re.sub(r'`\s*([,}\]])', r'"\1', json_str)  # `` before ,}] -> `"` before ,}]
            json_str = re.sub(r'`\s*$', '"', json_str)  # trailing backtick -> quote

            repaired_json = repair_json(json_str)
            result = json.loads(repaired_json)

            # Allow list-only outputs by wrapping into a dict.
            if isinstance(result, list):
                result = {'action': 'apply_changes', 'changes': result}

            # Guard against non-dict JSON (e.g., null, string)
            if not isinstance(result, dict):
                self.logger.warning(f"[Designer] Unexpected JSON type: {type(result).__name__}, expected dict")
                return {
                    'action': 'no_change',
                    'changes': [],
                    'stage2_result': None,
                    'reasoning': f'Unexpected JSON type: {type(result).__name__}',
                    'raw_response': response
                }

            changes = []
            top_reasoning = ""

            if isinstance(result.get('changes'), list):
                changes = result.get('changes', [])
                top_reasoning = str(result.get('summary') or result.get('reasoning') or '')
            else:
                # Normalize action to lowercase (LLM may output "Add_New", "ADD_NEW", etc.)
                action = str(result.get('action', 'no_change')).lower().strip()
                if action == 'add_new':
                    changes = [{
                        'action': 'add_new',
                        'new_operation': result.get('new_operation', {})
                    }]
                    top_reasoning = result.get('new_operation', {}).get('reasoning', '')
                elif action == 'refine_existing':
                    changes = [{
                        'action': 'refine_existing',
                        'refined_operation': result.get('refined_operation', {})
                    }]
                    top_reasoning = result.get('refined_operation', {}).get('reasoning', '')
                elif 'new_operation' in result or 'refined_operation' in result:
                    if 'new_operation' in result:
                        changes = [{
                            'action': 'add_new',
                            'new_operation': result.get('new_operation', {})
                        }]
                        top_reasoning = result.get('new_operation', {}).get('reasoning', '')
                    else:
                        changes = [{
                            'action': 'refine_existing',
                            'refined_operation': result.get('refined_operation', {})
                        }]
                        top_reasoning = result.get('refined_operation', {}).get('reasoning', '')
                else:
                    return {
                        'action': 'no_change',
                        'changes': [],
                        'stage2_result': result,
                        'reasoning': result.get('reasoning', 'No changes proposed')
                    }

            normalized_changes = []
            for change in changes:
                if not isinstance(change, dict):
                    continue
                change_action = str(change.get('action', '')).lower().strip()
                if not change_action:
                    if 'new_operation' in change:
                        change_action = 'add_new'
                    elif 'refined_operation' in change:
                        change_action = 'refine_existing'
                if change_action == 'add_new':
                    normalized_changes.append({
                        'action': 'add_new',
                        'new_operation': change.get('new_operation', {})
                    })
                elif change_action == 'refine_existing':
                    normalized_changes.append({
                        'action': 'refine_existing',
                        'refined_operation': change.get('refined_operation', {})
                    })

            if len(normalized_changes) == 0:
                return {
                    'action': 'no_change',
                    'changes': [],
                    'stage2_result': result,
                    'reasoning': result.get('reasoning', 'No changes proposed')
                }

            action = 'multi' if len(normalized_changes) > 1 else normalized_changes[0]['action']
            return {
                'action': action,
                'changes': normalized_changes,
                'stage2_result': result,
                'reasoning': top_reasoning
            }

        except json.JSONDecodeError as e:
            self.logger.error(f"[Designer] Failed to parse refinement JSON: {e}")
            return {
                'action': 'no_change',
                'stage2_result': None,
                'reasoning': f'JSON parse error: {str(e)}',
                'raw_response': response
            }

    def apply_evolution(self, operation_bank, evolution_result: Dict) -> bool:
        """
        Apply evolution result to operation bank.

        Handles the two-stage evolution format:
        - changes: list of add_new/refine_existing items
        - action: "multi" | "add_new" | "refine_existing" | "no_change"

        Args:
            operation_bank: OperationBank object
            evolution_result: dict from run_evolution()

        Returns:
            True if changes were applied, False otherwise
        """
        changes = evolution_result.get('changes')
        if not isinstance(changes, list):
            changes = []
            action = evolution_result.get('action', 'no_change')
            if action == 'add_new':
                changes = [{'action': 'add_new', 'new_operation': evolution_result.get('new_operation', {})}]
            elif action == 'refine_existing':
                changes = [{'action': 'refine_existing', 'refined_operation': evolution_result.get('refined_operation', {})}]

        refine_only = bool(getattr(self.args, 'designer_refine_only', False))
        if refine_only and changes:
            filtered_changes = []
            dropped = 0
            for change in changes:
                if not isinstance(change, dict):
                    continue
                change_action = str(change.get('action', '')).lower().strip()
                if not change_action:
                    if 'refined_operation' in change:
                        change_action = 'refine_existing'
                    elif 'new_operation' in change:
                        change_action = 'add_new'
                if change_action == 'refine_existing':
                    filtered_changes.append(change)
                else:
                    dropped += 1
            if dropped:
                self.logger.info(f"[Designer] Refine-only enabled; skipped {dropped} add_new change(s).")
            changes = filtered_changes

        max_changes = self._get_max_changes()
        if len(changes) > max_changes:
            self.logger.warning(f"[Designer] Received {len(changes)} changes; truncating to {max_changes}")
            changes = changes[:max_changes]

        new_op_names = []
        applied_changes = []

        if not changes:
            evolution_result['action'] = 'no_change'
            evolution_result['changes'] = []
            self.logger.info(f"[Designer] No changes applied. Reason: {evolution_result.get('reasoning', 'N/A')}")
            return False

        for change in changes:
            if not isinstance(change, dict):
                continue

            change_action = str(change.get('action', '')).lower().strip()
            if change_action == 'add_new':
                new_op_data = change.get('new_operation', {})
                if not new_op_data:
                    self.logger.warning("[Designer] No new_operation data provided")
                    continue

                try:
                    required_fields = ['name', 'description', 'instruction_template', 'update_type']
                    for field in required_fields:
                        if field not in new_op_data:
                            self.logger.warning(f"[Designer] Missing required field '{field}' in new operation")
                            raise ValueError("missing required field")

                    update_type = str(new_op_data.get('update_type', '')).lower().strip()
                    new_op_data['update_type'] = update_type
                    if update_type not in ['insert', 'update']:
                        self.logger.warning(f"[Designer] Invalid update_type: {update_type} (only 'insert' and 'update' are allowed)")
                        raise ValueError("invalid update_type")

                    template = new_op_data['instruction_template']
                    if not isinstance(template, str) or not template.strip():
                        self.logger.warning("[Designer] Rejected new operation: instruction_template is empty")
                        raise ValueError("empty template")
                    if '{session_text}' in template or '{retrieved_memories}' in template:
                        self.logger.warning("[Designer] Rejected new operation: instruction_template must not include context placeholders")
                        raise ValueError("template has placeholders")

                    new_op = Operation(
                        name=new_op_data['name'],
                        description=new_op_data['description'],
                        instruction_template=new_op_data['instruction_template'],
                        update_type=new_op_data['update_type'],
                        meta_info={
                            'usage_count': 0,
                            'avg_reward': 0.0,
                            'recent_rewards': [],
                            'recent_usage_ema': 0.0,
                            'created_at': 'designer',
                            'last_modified': 'designer'
                        }
                    )
                    if not operation_bank.add_operation(new_op):
                        self.logger.warning(f"[Designer] Add operation rejected: {new_op.name}")
                        continue
                    new_op_names.append(new_op.name)
                    applied_changes.append(change)
                    self.logger.info(f"[Designer] Added new operation: {new_op.name}")
                    self.logger.info(f"[Designer] Reasoning: {new_op_data.get('reasoning', evolution_result.get('reasoning', 'N/A'))}")
                except Exception as e:
                    self.logger.error(f"[Designer] Failed to add new operation: {e}")
                    continue

            elif change_action == 'refine_existing':
                refined_op_data = change.get('refined_operation', {})
                if not refined_op_data:
                    self.logger.warning("[Designer] No refined_operation data provided")
                    continue

                try:
                    op_name = refined_op_data.get('name')
                    if not op_name:
                        self.logger.warning("[Designer] Missing operation name for refinement")
                        raise ValueError("missing op name")

                    existing_op = None
                    for op in operation_bank.get_all_operations():
                        if op.name == op_name:
                            existing_op = op
                            break

                    if existing_op is None:
                        self.logger.warning(f"[Designer] Operation '{op_name}' not found in bank")
                        raise ValueError("operation not found")

                    original_operation = {
                        'name': existing_op.name,
                        'description': existing_op.description,
                        'instruction_template': existing_op.instruction_template,
                        'update_type': existing_op.update_type
                    }
                    change['original_operation'] = original_operation

                    changes_payload = refined_op_data.get('changes', {})
                    if not changes_payload:
                        self.logger.warning("[Designer] No changes specified for refinement")
                        raise ValueError("no changes")

                    if 'instruction_template' in changes_payload:
                        template = changes_payload['instruction_template']
                        if not isinstance(template, str) or not template.strip():
                            self.logger.warning("[Designer] Rejected refinement: instruction_template is empty")
                            raise ValueError("empty template")
                        if '{session_text}' in template or '{retrieved_memories}' in template:
                            self.logger.warning("[Designer] Rejected refinement: instruction_template must not include context placeholders")
                            raise ValueError("template has placeholders")

                    operation_bank.update_operation(op_name, **changes_payload)
                    new_op_names.append(op_name)
                    applied_changes.append(change)
                    self.logger.info(f"[Designer] Refined operation: {op_name}")
                    self.logger.info(f"[Designer] Changes: {list(changes_payload.keys())}")
                    self.logger.info(f"[Designer] Reasoning: {refined_op_data.get('reasoning', evolution_result.get('reasoning', 'N/A'))}")
                except Exception as e:
                    self.logger.error(f"[Designer] Failed to refine operation: {e}")
                    continue
            else:
                self.logger.warning(f"[Designer] Unknown change action: {change_action}")

        if not applied_changes:
            evolution_result['action'] = 'no_change'
            evolution_result['changes'] = []
            self.logger.info(f"[Designer] No changes applied. Reason: {evolution_result.get('reasoning', 'N/A')}")
            return False

        evolution_result['changes'] = applied_changes
        applied_action = 'multi' if len(applied_changes) > 1 else applied_changes[0]['action']
        evolution_result['action'] = applied_action

        # Update operation bank with new operation names for exploration bias
        if new_op_names:
            operation_bank.set_new_operation_names(new_op_names)
            self.logger.info(f"[Designer] Set new operation names for exploration: {new_op_names}")

        self.logger.info(f"\n[Designer] Evolution Summary:")
        self.logger.info(f"  Action: {applied_action}")
        self.logger.info(f"  Operations affected: {new_op_names if new_op_names else 'None'}")

        return True
