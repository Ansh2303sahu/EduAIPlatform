"""Phase 10 prompt builders; student and professor prompts stay separate."""

from app.langchain.prompts.professor import (
    build_professor_prompt,
    build_professor_safe_prompt,
)
from app.langchain.prompts.student import (
    build_student_prompt,
    build_student_safe_prompt,
)

__all__ = [
    "build_student_prompt",
    "build_student_safe_prompt",
    "build_professor_prompt",
    "build_professor_safe_prompt",
]
