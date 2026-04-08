"""Phase 12 graph specifications and compiled graphs for student and professor flows."""

from .professor_graph import (
    PROFESSOR_GRAPH_SPEC,
    get_professor_compiled_graph,
    get_professor_graph_spec,
)
from .professor_generative_graph import get_professor_generative_compiled_graph
from .student_graph import (
    STUDENT_GRAPH_SPEC,
    get_student_compiled_graph,
    get_student_graph_spec,
)
from .student_generative_graph import get_student_generative_compiled_graph

__all__ = [
    "PROFESSOR_GRAPH_SPEC",
    "STUDENT_GRAPH_SPEC",
    "get_professor_compiled_graph",
    "get_professor_generative_compiled_graph",
    "get_professor_graph_spec",
    "get_student_compiled_graph",
    "get_student_generative_compiled_graph",
    "get_student_graph_spec",
]
