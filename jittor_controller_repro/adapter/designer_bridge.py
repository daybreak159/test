"""Bridge from the original MemSkill Designer to Jittor-controller runs.

Designer is also an LLM/prompt module, not a differentiable tensor model.  This
bridge keeps the original implementation, but exposes the SkillBank evolution
boundary as explicit records: what skills existed before, what the Designer
proposed, what changed, and which new/updated skills should be visible to the
next Jittor controller rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SkillBankDiff:
    """Serializable operation-bank diff around one Designer evolution."""

    before_names: list[str]
    after_names: list[str]
    added: list[str]
    removed: list[str]
    updated: list[str]
    new_action_names: list[str]
    evolution_action: str
    applied: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JittorDesignerBridge:
    """Run original Designer evolution and summarize changes for Jittor runs."""

    def __init__(self, designer: Any):
        self.designer = designer

    @staticmethod
    def operation_snapshot(operation_bank: Any) -> dict[str, dict[str, Any]]:
        snapshot = {}
        for op in operation_bank.get_all_operations():
            snapshot[str(op.name)] = {
                "description": getattr(op, "description", ""),
                "instruction_template": getattr(op, "instruction_template", ""),
                "update_type": getattr(op, "update_type", ""),
            }
        return snapshot

    @classmethod
    def diff(
        cls,
        before: dict[str, dict[str, Any]],
        after: dict[str, dict[str, Any]],
        operation_bank: Any,
        evolution_result: dict[str, Any],
        applied: bool,
    ) -> SkillBankDiff:
        before_names = sorted(before.keys())
        after_names = sorted(after.keys())
        before_set = set(before_names)
        after_set = set(after_names)
        common = before_set & after_set
        updated = sorted(
            name for name in common
            if before.get(name, {}) != after.get(name, {})
        )
        return SkillBankDiff(
            before_names=before_names,
            after_names=after_names,
            added=sorted(after_set - before_set),
            removed=sorted(before_set - after_set),
            updated=updated,
            new_action_names=sorted(getattr(operation_bank, "new_operation_names", set())),
            evolution_action=str(evolution_result.get("action", "unknown")),
            applied=bool(applied),
            reasoning=str(evolution_result.get("reasoning", "")),
        )

    def evolve(
        self,
        operation_bank: Any,
        evolution_feedback: str = "",
        evolution_feedback_for_refinement: str = "",
        prepared_data: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], SkillBankDiff]:
        """Run prepare/evolve/apply and return a SkillBank diff."""

        before = self.operation_snapshot(operation_bank)
        if prepared_data is None:
            prepared_data = self.designer.prepare_evolution(
                operation_bank,
                evolution_feedback=evolution_feedback,
            )

        if prepared_data is None:
            evolution_result = {
                "action": "no_change",
                "reasoning": "No cases to analyze",
                "changes": [],
            }
            applied = False
        else:
            evolution_result = self.designer.run_evolution(
                operation_bank=operation_bank,
                prepared_data=prepared_data,
                evolution_feedback_for_refinement=evolution_feedback_for_refinement,
            )
            applied = self.designer.apply_evolution(operation_bank, evolution_result)

        after = self.operation_snapshot(operation_bank)
        return evolution_result, self.diff(
            before,
            after,
            operation_bank,
            evolution_result,
            applied,
        )

    def summarize_applied_result(
        self,
        operation_bank: Any,
        evolution_result: dict[str, Any],
    ) -> SkillBankDiff:
        """Summarize an evolution result that was already applied by Trainer."""

        after = self.operation_snapshot(operation_bank)
        before = dict(after)
        for change in evolution_result.get("changes", []) or []:
            if not isinstance(change, dict):
                continue
            if change.get("action") == "add_new":
                data = change.get("new_operation", {}) or {}
                before.pop(str(data.get("name", "")), None)
        return self.diff(
            before,
            after,
            operation_bank,
            evolution_result,
            applied=bool(evolution_result.get("action") != "no_change"),
        )
