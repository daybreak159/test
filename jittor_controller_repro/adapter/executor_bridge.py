"""Bridge from a Jittor controller action to the original MemSkill Executor.

The executor itself is an LLM/prompt module, so it is intentionally kept in the
original Python implementation.  This bridge makes the integration boundary
explicit: Jittor predicts action indices, this module maps them back to
Operation objects, calls the executor, applies memory updates, and returns a
JSON-serializable record for logs/PPT demos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


@dataclass
class ExecutorBridgeRecord:
    """Serializable record for one Jittor-controller-to-executor step."""

    action_indices: list[int]
    selected_ops: list[str]
    executor_result_types: list[str]
    executor_success: list[bool]
    apply_success: bool | None
    memory_before_count: int | None
    memory_after_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JittorExecutorBridge:
    """Adapt Jittor controller Top-K actions to the original MemSkill Executor."""

    def __init__(self, operation_bank: Any, executor: Any):
        self.operation_bank = operation_bank
        self.executor = executor

    @staticmethod
    def normalize_action_indices(action_idx: int | Sequence[int]) -> list[int]:
        if isinstance(action_idx, (list, tuple)):
            return [int(idx) for idx in action_idx]
        return [int(action_idx)]

    def selected_operations(self, action_idx: int | Sequence[int]) -> list[Any]:
        """Map controller action indices to current OperationBank entries."""

        candidate_ops = self.operation_bank.get_candidate_operations()
        if not candidate_ops:
            return []
        selected = []
        for idx in self.normalize_action_indices(action_idx):
            if idx < 0 or idx >= len(candidate_ops):
                raise IndexError(
                    f"Controller action index {idx} out of range for "
                    f"{len(candidate_ops)} candidate operations"
                )
            selected.append(candidate_ops[idx])
        return selected

    @staticmethod
    def operation_names(operations: Iterable[Any]) -> list[str]:
        names = []
        seen = set()
        for op in operations:
            name = getattr(op, "name", None)
            if name and name not in seen:
                names.append(str(name))
                seen.add(name)
        return names

    @staticmethod
    def memory_count(memory_bank: Any) -> int | None:
        memories = getattr(memory_bank, "memories", None)
        if memories is None:
            return None
        try:
            return len(memories)
        except TypeError:
            return None

    def execute(
        self,
        action_idx: int | Sequence[int],
        session_text: str,
        retrieved_memories: list[str],
        memory_bank: Any | None = None,
        retrieved_indices: list[int] | None = None,
        apply_to_memory: bool = True,
    ) -> tuple[list[Any], ExecutorBridgeRecord]:
        """Execute selected MemSkills and optionally apply results to MemoryBank."""

        selected_ops = self.selected_operations(action_idx)
        action_indices = self.normalize_action_indices(action_idx)
        memory_before = self.memory_count(memory_bank) if memory_bank is not None else None

        exec_results = self.executor.execute_operation(
            operation=selected_ops,
            session_text=session_text,
            retrieved_memories=retrieved_memories,
        )

        apply_success = None
        if apply_to_memory and memory_bank is not None:
            apply_success = self.executor.apply_to_memory_bank(
                results=exec_results,
                memory_bank=memory_bank,
                retrieved_indices=retrieved_indices or [],
                operation_name=self.operation_names(selected_ops),
            )

        memory_after = self.memory_count(memory_bank) if memory_bank is not None else None
        record = ExecutorBridgeRecord(
            action_indices=action_indices,
            selected_ops=self.operation_names(selected_ops),
            executor_result_types=[str(getattr(r, "action_type", "")) for r in exec_results],
            executor_success=[bool(getattr(r, "success", False)) for r in exec_results],
            apply_success=apply_success,
            memory_before_count=memory_before,
            memory_after_count=memory_after,
        )
        return exec_results, record
