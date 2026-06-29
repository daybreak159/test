"""Lightweight tests for the Jittor-to-MemSkill bridge helpers."""

from __future__ import annotations

from types import SimpleNamespace

from jittor_controller_repro.adapter.designer_bridge import JittorDesignerBridge
from jittor_controller_repro.adapter.executor_bridge import JittorExecutorBridge


class DummyOperationBank:
    def __init__(self):
        self.new_operation_names = {"new_insert"}
        self.ops = [
            SimpleNamespace(
                name="insert",
                description="old insert",
                instruction_template="insert facts",
                update_type="insert",
            ),
            SimpleNamespace(
                name="new_insert",
                description="new insert",
                instruction_template="insert exact facts",
                update_type="insert",
            ),
        ]

    def get_candidate_operations(self):
        return list(self.ops)

    def get_all_operations(self):
        return list(self.ops)


class DummyExecutor:
    def execute_operation(self, operation, session_text, retrieved_memories):
        return [SimpleNamespace(action_type="INSERT", success=True)]

    def apply_to_memory_bank(self, results, memory_bank, retrieved_indices, operation_name=None):
        memory_bank.memories.append("new memory")
        return True


def test_executor_bridge_maps_topk_actions_and_records_memory_delta():
    bridge = JittorExecutorBridge(DummyOperationBank(), DummyExecutor())
    memory_bank = SimpleNamespace(memories=["old memory"])

    _, record = bridge.execute(
        action_idx=[1, 0],
        session_text="remember this",
        retrieved_memories=[],
        memory_bank=memory_bank,
        retrieved_indices=[],
    )

    assert record.selected_ops == ["new_insert", "insert"]
    assert record.executor_result_types == ["INSERT"]
    assert record.apply_success is True
    assert record.memory_before_count == 1
    assert record.memory_after_count == 2


def test_designer_bridge_diff_detects_updated_skill():
    bank = DummyOperationBank()
    before = JittorDesignerBridge.operation_snapshot(bank)
    bank.ops[0].description = "refined insert"
    after = JittorDesignerBridge.operation_snapshot(bank)

    diff = JittorDesignerBridge.diff(
        before=before,
        after=after,
        operation_bank=bank,
        evolution_result={"action": "refine_existing", "reasoning": "more exact"},
        applied=True,
    )

    assert diff.updated == ["insert"]
    assert diff.added == []
    assert diff.new_action_names == ["new_insert"]
