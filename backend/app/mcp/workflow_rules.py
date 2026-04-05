"""
Phase 11.3 deterministic workflow rules.

These rules define the only approved multi-step MCP sequences. Planning is
data-driven and does not rely on prompt text or model output.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.mcp.enums import ToolRole


@dataclass(frozen=True)
class WorkflowStepRule:
    step_name: str
    tool_name: str
    critical: bool = True


@dataclass(frozen=True)
class WorkflowRule:
    workflow_name: str
    description: str
    allowed_roles: frozenset[ToolRole]
    steps: tuple[WorkflowStepRule, ...]
    continue_on_non_critical_failure: bool = False


WORKFLOW_RULES: dict[str, WorkflowRule] = {
    "student_review_assist": WorkflowRule(
        workflow_name="student_review_assist",
        description=(
            "Runs a bounded student review assist workflow: summary first, then "
            "structure guidance."
        ),
        allowed_roles=frozenset({ToolRole.STUDENT, ToolRole.ADMIN}),
        steps=(
            WorkflowStepRule(
                step_name="summarise_submission",
                tool_name="student.summariser.v1",
                critical=True,
            ),
            WorkflowStepRule(
                step_name="improve_structure",
                tool_name="student.structure_improver.v1",
                critical=False,
            ),
        ),
        continue_on_non_critical_failure=True,
    ),
    "professor_review_assist": WorkflowRule(
        workflow_name="professor_review_assist",
        description=(
            "Runs a bounded professor review assist workflow: rubric evaluation "
            "first, then consistency checking."
        ),
        allowed_roles=frozenset({ToolRole.PROFESSOR, ToolRole.ADMIN}),
        steps=(
            WorkflowStepRule(
                step_name="evaluate_rubric",
                tool_name="professor.rubric_evaluator.v1",
                critical=True,
            ),
            WorkflowStepRule(
                step_name="check_consistency",
                tool_name="professor.consistency_checker.v1",
                critical=False,
            ),
        ),
        continue_on_non_critical_failure=True,
    ),
}


def get_workflow_rule(workflow_name: str) -> WorkflowRule | None:
    return WORKFLOW_RULES.get(workflow_name)


def list_workflow_rules() -> list[WorkflowRule]:
    return list(WORKFLOW_RULES.values())
